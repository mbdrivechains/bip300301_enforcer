//! A sidechain bids the same amount on every block. When blocks come faster
//! than its bids are mined, a bid goes stale: the chain moves past the block
//! the bid commits to before any block includes it. The enforcer wallet then
//! evicts the stale bid and frees its inputs, but the bid itself stays in
//! Bitcoin Core's mempool. A later bid that reuses a freed input conflicts
//! with it, so Core only takes it as an RBF replacement, which a bid of the
//! same amount (the same absolute fee) is not, unless the enforcer has
//! deprioritised the stale bid in Core first.
//!
//! Field report, eCash mainnet with blocks 7-8 s apart: stale BMM bids are not
//! dropped fast enough, so the bids get stuck and BMM stops.
//!
//! This test reproduces and measures that. It does not assume a fix: it fails,
//! printing the per-round table, whenever a same-amount bid placed after a
//! stale round does not make it into the next block. Three phases:
//! * settled: the re-bid waits until the wallet has freed the stale bid's
//!   inputs
//! * immediate: the re-bid goes out the moment the enforcer's ConnectBlock
//!   event reports the block that made the previous bid stale, as a sidechain
//!   reacting to a new tip would
//! * tip-poll: the re-bid goes out the moment the enforcer's chain tip (polled
//!   back to back) moves to that block, the fastest a sidechain can react
//! * forced: as settled, but the test first undoes the enforcer's
//!   deprioritisation of the stale bid in Core, recreating the window in
//!   which the stale bid is still full-fee there. Only the wallet
//!   deprioritising it itself lets the same-amount re-bid replace it
//!
//! Observed (2026-10-30, eCash v1.2.0 regtest, enforcer v1.0.1 + our fixes):
//! the enforcer deprioritises a stale bid in Core (`prioritisetransaction`,
//! -21M BTC) when it connects the block, so a same-amount re-bid that reuses
//! its input replaces it. The failure is a race instead: the validator's tip
//! moves a few ms before the wallet connects the block, and a bid built in
//! that window still sees the stale bid as unconfirmed and spends its change.
//! Core takes that child, but the enforcer rejects it with its stale parent,
//! so no template carries it.

use std::{
    future::Future,
    time::{Duration, Instant},
};

use bip300301_enforcer_lib::proto::{
    self,
    common::{ConsensusHex, ReverseHex},
    mainchain::{
        BlockHeaderInfo, CreateBmmCriticalDataTransactionRequest, CreateNewAddressRequest,
        GetSidechainsRequest, ListUnspentOutputsRequest, ListUnspentOutputsResponse,
        SubscribeEventsRequest, SubscribeEventsResponse, subscribe_events_response::event::Event,
    },
};
use bitcoin::{BlockHash, OutPoint, Txid};
use buffa::MessageField;
use jsonrpsee::{
    core::{client::ClientT as _, params::ObjectParams},
    rpc_params,
};

use crate::{
    integration_test::{propose_sidechain, wait_for_wallet_sync},
    mine::{MiningPolicy, mine_check_block_events},
    setup::{
        DummySidechain, Mode, PostSetup, Sidechain, WAIT_TIMEOUT, generate_empty_block,
        wait_for_tx_in_mempool,
    },
    test_bmm_bid_auction::{
        block_template, block_txids, chain_tip, coinbase_m7_accepts, get_block,
        get_raw_transaction, h_star, template_txids,
    },
};

/// The same bid every round, like a sidechain with a fixed BMM bid.
const BID_SATS: u64 = 5_000;
/// Rounds after the round-0 baseline, as (stale, re-bid) pairs.
const SETTLED_PAIRS: u32 = 5;
const IMMEDIATE_PAIRS: u32 = 10;
const TIP_POLL_PAIRS: u32 = 15;
const FORCED_PAIRS: u32 = 3;
/// `prioritisetransaction` delta the enforcer uses to reject a tx; deltas add
/// up, so adding its negation undoes it.
const NEGATIVE_MAX_SATS: i64 = -(21_000_000 * 100_000_000);
/// How long a bid the enforcer reports as created may take to show up in
/// Bitcoin Core's mempool before the round records it as missing.
const MEMPOOL_WAIT: Duration = Duration::from_secs(10);
/// How long a bid in Core's mempool may take to reach the enforcer's block
/// template before the round mines without it.
const TEMPLATE_WAIT: Duration = Duration::from_secs(20);
/// How long the wallet may take to free a stale bid's inputs.
const EVICT_WAIT: Duration = Duration::from_secs(20);
const POLL: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Bid, then mine through the enforcer.
    Normal,
    /// Bid, then mine an empty block straight from Bitcoin Core.
    Stale,
    /// Bid the moment the enforcer's ConnectBlock event reports the stale
    /// round's block, then mine through the enforcer.
    Immediate,
    /// Bid the moment the enforcer's chain tip moves to the stale round's
    /// block, then mine through the enforcer.
    TipPoll,
    /// As [`Kind::Normal`], after the stale round undid the enforcer's
    /// deprioritisation of the stale bid in Core.
    Forced,
}

impl Kind {
    fn is_fast_rebid(self) -> bool {
        matches!(self, Self::Immediate | Self::TipPoll)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Normal => "settled",
            Self::Stale => "STALE",
            Self::Immediate => "immediate",
            Self::TipPoll => "tip-poll",
            Self::Forced => "forced",
        }
    }
}

/// Poll `check` until it holds or `limit` passes, returning how long it took.
/// A failing check counts as "not yet".
async fn poll_elapsed<Check, Fut>(limit: Duration, mut check: Check) -> Option<Duration>
where
    Check: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<bool>>,
{
    let start = Instant::now();
    loop {
        if matches!(check().await, Ok(true)) {
            return Some(start.elapsed());
        }
        if start.elapsed() >= limit {
            return None;
        }
        tokio::time::sleep(POLL).await;
    }
}

async fn in_core_mempool(post_setup: &PostSetup, txid: &Txid) -> bool {
    post_setup
        .bitcoind_client
        .request::<serde_json::Value, _>("getmempoolentry", rpc_params![txid])
        .await
        .is_ok()
}

/// The enforcer wallet's unspent outputs, with whether each is confirmed.
async fn wallet_utxos(post_setup: &PostSetup) -> anyhow::Result<Vec<(OutPoint, bool)>> {
    let outputs = post_setup
        .wallet_service_client
        .list_unspent_outputs(ListUnspentOutputsRequest::default())
        .await?
        .into_owned()
        .outputs;
    outputs
        .into_iter()
        .map(|output| {
            let txid: Txid = output
                .txid
                .into_option()
                .ok_or_else(|| anyhow::anyhow!("expected `txid` in unspent output"))?
                .decode::<ListUnspentOutputsResponse, _>("txid")?;
            Ok((
                OutPoint {
                    txid,
                    vout: output.vout,
                },
                output.is_confirmed,
            ))
        })
        .collect()
}

/// A bid that went stale while in Core's mempool.
struct StaleBid {
    round: u32,
    txid: Txid,
    inputs: Vec<OutPoint>,
}

#[derive(Default)]
struct BidOutcome {
    grpc_ms: u128,
    /// `Ok(txid)` or the error the enforcer returned.
    grpc: Option<Result<Txid, String>>,
    in_mempool: bool,
    relation: String,
    template_ms: Option<u128>,
    inputs: Vec<OutPoint>,
}

/// Bid [`BID_SATS`] through the enforcer wallet on `prev_bytes`, then see
/// whether Core and the enforcer's template take it.
async fn place_bid(
    post_setup: &PostSetup,
    h: &[u8; 32],
    prev_bytes: ReverseHex,
    tip_height: u32,
    stale_bids: &[StaleBid],
    utxos_before: Option<&[(OutPoint, bool)]>,
) -> anyhow::Result<BidOutcome> {
    let mut outcome = BidOutcome::default();
    let started = Instant::now();
    let grpc = post_setup
        .wallet_service_client
        .create_bmm_critical_data_transaction(CreateBmmCriticalDataTransactionRequest {
            sidechain_id: proto::wrap_u32(DummySidechain::SIDECHAIN_NUMBER.0.into()),
            value_sats: proto::wrap_u64(BID_SATS),
            height: proto::wrap_u32(tip_height),
            critical_hash: MessageField::some(ConsensusHex::encode(h)),
            prev_bytes: MessageField::some(prev_bytes),
        })
        .await;
    outcome.grpc_ms = started.elapsed().as_millis();
    let txid = match grpc {
        Ok(resp) => resp
            .into_owned()
            .txid
            .into_option()
            .and_then(|txid| proto::unwrap_string(txid.hex))
            .ok_or_else(|| anyhow::anyhow!("expected `txid` in CreateBmmCriticalDataTransaction"))?
            .parse::<Txid>()?,
        Err(err) => {
            let msg = err.to_string();
            outcome.relation = stale_bids
                .iter()
                .rev()
                .find(|stale| msg.contains(&stale.txid.to_string()))
                .map_or_else(
                    || "refused".to_owned(),
                    |stale| format!("conflicts w/ stale r{}", stale.round),
                );
            outcome.grpc = Some(Err(msg));
            return Ok(outcome);
        }
    };
    outcome.grpc = Some(Ok(txid));
    outcome.in_mempool = poll_elapsed(MEMPOOL_WAIT, || async {
        Ok(in_core_mempool(post_setup, &txid).await)
    })
    .await
    .is_some();
    if !outcome.in_mempool {
        outcome.relation = "created, not in Core mempool".to_owned();
        return Ok(outcome);
    }
    let tx = get_raw_transaction(post_setup, &txid).await?;
    outcome.inputs = tx.input.iter().map(|input| input.previous_output).collect();
    let inputs = &outcome.inputs;
    outcome.relation = if let Some(stale) = stale_bids
        .iter()
        .rev()
        .find(|stale| inputs.iter().any(|input| input.txid == stale.txid))
    {
        format!("chains on stale r{}", stale.round)
    } else if let Some(stale) = stale_bids
        .iter()
        .rev()
        .find(|stale| inputs.iter().any(|input| stale.inputs.contains(input)))
    {
        format!("replaced stale r{}", stale.round)
    } else {
        match utxos_before {
            Some(utxos)
                if inputs.iter().all(|input| {
                    utxos
                        .iter()
                        .any(|(outpoint, confirmed)| outpoint == input && *confirmed)
                }) =>
            {
                "fresh confirmed input".to_owned()
            }
            Some(_) => "fresh, unconfirmed input".to_owned(),
            None => "fresh input".to_owned(),
        }
    };
    outcome.template_ms = poll_elapsed(TEMPLATE_WAIT, || async {
        Ok(template_txids(&block_template(post_setup).await?).contains(&txid))
    })
    .await
    .map(|elapsed| elapsed.as_millis());
    Ok(outcome)
}

#[derive(Default)]
struct Row {
    round: u32,
    kind: Option<Kind>,
    tip_height: u32,
    /// Confirmed and unconfirmed wallet UTXOs before the bid; not read
    /// before an immediate re-bid.
    utxos: Option<(usize, usize)>,
    /// Stale bids still in Core's mempool, before the bid (after it for an
    /// immediate re-bid).
    stale_in_core: usize,
    bid: BidOutcome,
    /// Stale rounds: ms from `generateblock` until the enforcer reported the
    /// empty block, until its template moved there, and until the wallet
    /// freed the stale bid's inputs.
    tip_ms: Option<u128>,
    template_tip_ms: Option<u128>,
    evict_ms: Option<Option<u128>>,
    /// The test undid the enforcer's deprioritisation of this stale bid.
    delta_undone: bool,
    block_height: u32,
    bid_in_block: bool,
    m7_ours: bool,
}

impl Row {
    fn is_rebid(&self) -> bool {
        self.round > 0
            && matches!(
                self.kind,
                Some(Kind::Normal | Kind::Immediate | Kind::TipPoll | Kind::Forced)
            )
    }

    fn ok(&self) -> bool {
        matches!(self.bid.grpc, Some(Ok(_)))
            && self.bid.in_mempool
            && self.bid_in_block
            && self.m7_ours
    }

    fn line(&self) -> String {
        let ms = |v: Option<u128>| v.map_or_else(|| "never".to_owned(), |v| format!("{v}ms"));
        let grpc = match &self.bid.grpc {
            Some(Ok(txid)) => format!("ok {}", &txid.to_string()[..12]),
            Some(Err(_)) => "REFUSED".to_owned(),
            None => "-".to_owned(),
        };
        let block = if self.kind == Some(Kind::Stale) {
            let mut block = format!(
                "empty block h{} (enforcer tip {}",
                self.block_height,
                ms(self.tip_ms)
            );
            if let Some(template_tip_ms) = self.template_tip_ms {
                block.push_str(&format!(", template {template_tip_ms}ms"));
            }
            if let Some(evict_ms) = self.evict_ms {
                block.push_str(&format!(", wallet freed {}", ms(evict_ms)));
            }
            if self.delta_undone {
                block.push_str(", Core delta undone");
            }
            block.push(')');
            block
        } else {
            format!(
                "h{} bid_in_block={} M7_ours={}{}",
                self.block_height,
                self.bid_in_block,
                self.m7_ours,
                if self.is_rebid() {
                    if self.ok() { " => PASS" } else { " => FAIL" }
                } else if self.ok() {
                    " (baseline ok)"
                } else {
                    " (baseline FAILED)"
                }
            )
        };
        format!(
            "{:>2} {:<9} tip{:<4} utxos {:<7} stale_in_core={}{} grpc={:<15} ({:>3}ms) \
             mempool={:<5} template={:<6} input={:<28} | {}",
            self.round,
            self.kind.map_or("?", Kind::label),
            self.tip_height,
            self.utxos
                .map_or_else(|| "-".to_owned(), |(c, u)| format!("{c}c/{u}u")),
            self.stale_in_core,
            if self.kind.is_some_and(Kind::is_fast_rebid) {
                "*"
            } else {
                " "
            },
            grpc,
            self.bid.grpc_ms,
            self.bid.in_mempool,
            if self.bid.in_mempool {
                ms(self.bid.template_ms)
            } else {
                "-".to_owned()
            },
            self.bid.relation,
            block,
        )
    }
}

fn table(mode: Mode, rows: &[Row]) -> String {
    let mut out = format!(
        "bmm_same_bid_after_stale (mode: {mode}): {BID_SATS} sat bid every round; round 0 \
         baseline, then {SETTLED_PAIRS} (STALE, normal) pairs where the re-bid waits for the \
         wallet to free the stale bid's inputs, then {IMMEDIATE_PAIRS} (STALE, immediate) pairs \
         where the re-bid goes out on the ConnectBlock event, then {TIP_POLL_PAIRS} (STALE, \
         tip-poll) pairs where it goes out as soon as GetChainTip shows the new block, then \
         {FORCED_PAIRS} (STALE, forced) pairs where the test undoes the enforcer's \
         deprioritisation of the stale bid in Core before a settled re-bid. stale_in_core* = \
         counted after the bid.\n"
    );
    for row in rows {
        out.push_str(&row.line());
        out.push('\n');
    }
    for kind in [Kind::Normal, Kind::Immediate, Kind::TipPoll, Kind::Forced] {
        let rebids: Vec<&Row> = rows
            .iter()
            .filter(|row| row.is_rebid() && row.kind == Some(kind))
            .collect();
        let count = |pred: &dyn Fn(&Row) -> bool| rebids.iter().filter(|row| pred(row)).count();
        out.push_str(&format!(
            "{} re-bids after a stale round: {}/{} in the next block; refused {}, created but not \
             in Core's mempool {}, chained on a stale bid {}, replaced a stale bid {}\n",
            kind.label(),
            count(&|row| row.ok()),
            rebids.len(),
            count(&|row| matches!(row.bid.grpc, Some(Err(_)))),
            count(&|row| matches!(row.bid.grpc, Some(Ok(_))) && !row.bid.in_mempool),
            count(&|row| row.bid.relation.starts_with("chains on stale")),
            count(&|row| row.bid.relation.starts_with("replaced stale")),
        ));
    }
    for row in rows {
        if let Some(Err(err)) = &row.bid.grpc {
            out.push_str(&format!("round {} refusal: {err}\n", row.round));
        }
    }
    out
}

/// Fund the enforcer wallet with one UTXO from Bitcoin Core's wallet, as an
/// operator funding a BMM wallet with a single send. Every bid's change then
/// carries the wallet's coins forward, so a re-bid after a stale round has
/// the stale bid's freed input or the stale bid's change to spend.
async fn fund_enforcer_single_utxo(
    post_setup: &mut PostSetup,
    core_addr: &str,
) -> anyhow::Result<()> {
    let _mined: Vec<BlockHash> = post_setup
        .bitcoind_client
        .request("generatetoaddress", rpc_params![101, core_addr])
        .await?;
    let address = post_setup
        .wallet_service_client
        .create_new_address(CreateNewAddressRequest::default())
        .await?
        .into_owned()
        .address;
    // Named, with an explicit `fee_rate`: regtest has no fee history.
    let mut params = ObjectParams::new();
    params.insert("address", address)?;
    params.insert("amount", 1.0)?;
    params.insert("fee_rate", 2)?;
    let funding_txid: Txid = post_setup
        .bitcoind_client
        .request("sendtoaddress", params)
        .await?;
    let () = wait_for_tx_in_mempool(&post_setup.bitcoind_client, &funding_txid).await?;
    let _mined: Vec<BlockHash> = post_setup
        .bitcoind_client
        .request("generatetoaddress", rpc_params![1, core_addr])
        .await?;
    wait_for_wallet_sync(post_setup).await
}

/// * Activate a sidechain and fund the enforcer wallet with one UTXO
/// * Each round, bid [`BID_SATS`] through the enforcer wallet on the current
///   tip. Stale rounds then mine an empty block straight from Bitcoin Core, so
///   the bid goes stale in Core's mempool; other rounds mine through the
///   enforcer and check the block for the bid and its M7
/// * Fail if a bid placed after a stale round does not get into the block
pub async fn test_bmm_same_bid_after_stale(mut post_setup: PostSetup) -> anyhow::Result<()> {
    let mode = post_setup.mode;
    let () = propose_sidechain::<DummySidechain>(&mut post_setup).await?;
    let () = mine_check_block_events::<_, DummySidechain>(
        &mut post_setup,
        6,
        MiningPolicy::VOTE,
        |_, _| Ok(()),
    )
    .await?;
    let sidechains = post_setup
        .validator_service_client
        .get_sidechains(GetSidechainsRequest::default())
        .await?
        .into_owned()
        .sidechains;
    anyhow::ensure!(
        sidechains.len() == 1,
        "expected the sidechain active, got {} active",
        sidechains.len()
    );
    let core_addr: String = post_setup
        .bitcoind_client
        .request("getnewaddress", rpc_params![])
        .await?;
    let () = fund_enforcer_single_utxo(&mut post_setup, &core_addr).await?;
    tracing::info!("Activated the sidechain and funded the enforcer wallet");

    let schedule: Vec<Kind> = std::iter::once(Kind::Normal)
        .chain((0..SETTLED_PAIRS).flat_map(|_| [Kind::Stale, Kind::Normal]))
        .chain((0..IMMEDIATE_PAIRS).flat_map(|_| [Kind::Stale, Kind::Immediate]))
        .chain((0..TIP_POLL_PAIRS).flat_map(|_| [Kind::Stale, Kind::TipPoll]))
        .chain((0..FORCED_PAIRS).flat_map(|_| [Kind::Stale, Kind::Forced]))
        .collect();
    let mut stale_bids = Vec::<StaleBid>::new();
    let mut rows = Vec::<Row>::new();
    // Set by a stale round followed by a fast re-bid: the empty block's proto
    // hash and height, straight from its ConnectBlock event or the tip poll.
    let mut immediate_tip: Option<(ReverseHex, u32)> = None;
    for (round, kind) in schedule.iter().copied().enumerate() {
        let round = u32::try_from(round)?;
        let h = h_star(&format!("same-amount bid, round {round}"));
        let mut row = Row {
            round,
            kind: Some(kind),
            ..Row::default()
        };
        if kind.is_fast_rebid() {
            // Nothing between the event and the bid: no wallet sync wait, no
            // wallet read.
            let (prev_bytes, tip_height) = immediate_tip
                .take()
                .ok_or_else(|| anyhow::anyhow!("a fast re-bid must follow a stale round"))?;
            row.tip_height = tip_height;
            row.bid = place_bid(&post_setup, &h, prev_bytes, tip_height, &stale_bids, None).await?;
        } else {
            let () = wait_for_wallet_sync(&mut post_setup).await?;
            let (prev_bytes, _, tip_height) = chain_tip(&mut post_setup).await?;
            let utxos = wallet_utxos(&post_setup).await?;
            row.tip_height = tip_height;
            row.utxos = Some((
                utxos.iter().filter(|(_, confirmed)| *confirmed).count(),
                utxos.iter().filter(|(_, confirmed)| !*confirmed).count(),
            ));
            for stale in &stale_bids {
                if in_core_mempool(&post_setup, &stale.txid).await {
                    row.stale_in_core += 1;
                }
            }
            row.bid = place_bid(
                &post_setup,
                &h,
                prev_bytes,
                tip_height,
                &stale_bids,
                Some(&utxos),
            )
            .await?;
        }
        if kind.is_fast_rebid() {
            for stale in &stale_bids {
                if in_core_mempool(&post_setup, &stale.txid).await {
                    row.stale_in_core += 1;
                }
            }
        }

        if kind == Kind::Stale {
            let next = schedule.get(round as usize + 1).copied();
            let mut events = post_setup
                .validator_service_client
                .subscribe_events(SubscribeEventsRequest {
                    sidechain_id: proto::wrap_u32(DummySidechain::SIDECHAIN_NUMBER.0.into()),
                })
                .await?;
            let started = Instant::now();
            let empty_block = generate_empty_block(&post_setup.bitcoind_client, &core_addr).await?;
            let (empty_proto_hash, empty_height) = if next == Some(Kind::TipPoll) {
                // Back to back, no sleep: react the moment the validator's
                // tip moves.
                let deadline = Instant::now() + WAIT_TIMEOUT;
                loop {
                    let (proto_hash, hash, height) = chain_tip(&mut post_setup).await?;
                    if hash == empty_block {
                        break (proto_hash, height);
                    }
                    anyhow::ensure!(
                        Instant::now() < deadline,
                        "the enforcer's tip never moved to {empty_block}"
                    );
                }
            } else {
                loop {
                    let message = tokio::time::timeout(WAIT_TIMEOUT, events.message())
                        .await
                        .map_err(|_elapsed| {
                            anyhow::anyhow!("timed out waiting for the event for {empty_block}")
                        })??
                        .ok_or_else(|| anyhow::anyhow!("event stream ended"))?;
                    let resp: SubscribeEventsResponse = message.to_owned_message();
                    let Some(Event::ConnectBlock(connect_block)) =
                        resp.event.into_option().and_then(|inner| inner.event)
                    else {
                        continue;
                    };
                    let header_info = connect_block
                        .header_info
                        .into_option()
                        .ok_or_else(|| anyhow::anyhow!("expected `header_info` in ConnectBlock"))?;
                    let proto_hash = header_info.block_hash.into_option().ok_or_else(|| {
                        anyhow::anyhow!("expected `block_hash` in BlockHeaderInfo")
                    })?;
                    let hash: BlockHash = proto_hash
                        .clone()
                        .decode::<BlockHeaderInfo, _>("block_hash")?;
                    if hash == empty_block {
                        break (proto_hash, header_info.height);
                    }
                }
            };
            drop(events);
            row.tip_ms = Some(started.elapsed().as_millis());
            row.block_height = empty_height;
            if let Some(Ok(txid)) = row.bid.grpc.clone().filter(|_| row.bid.in_mempool) {
                stale_bids.push(StaleBid {
                    round,
                    txid,
                    inputs: row.bid.inputs.clone(),
                });
            }
            if next.is_some_and(Kind::is_fast_rebid) {
                immediate_tip = Some((empty_proto_hash, empty_height));
            } else {
                row.template_tip_ms = poll_elapsed(TEMPLATE_WAIT, || async {
                    Ok(block_template(&post_setup).await?.prev_blockhash == empty_block)
                })
                .await
                .map(|_| started.elapsed().as_millis());
                if row.bid.in_mempool {
                    let post_setup_ref = &post_setup;
                    let inputs = &row.bid.inputs;
                    row.evict_ms = Some(
                        poll_elapsed(EVICT_WAIT, || async move {
                            let unspent = wallet_utxos(post_setup_ref).await?;
                            Ok(inputs
                                .iter()
                                .all(|input| unspent.iter().any(|(outpoint, _)| outpoint == input)))
                        })
                        .await
                        .map(|_| started.elapsed().as_millis()),
                    );
                }
                if next == Some(Kind::Forced)
                    && let Some(Ok(txid)) = row.bid.grpc.clone().filter(|_| row.bid.in_mempool)
                {
                    // Wait for the enforcer's deprioritisation to land, then
                    // undo it: the stale bid is full-fee in Core again.
                    let deprioritized = poll_elapsed(EVICT_WAIT, || async {
                        let entry: serde_json::Value = post_setup
                            .bitcoind_client
                            .request("getmempoolentry", rpc_params![txid])
                            .await?;
                        let base = entry["fees"]["base"].as_f64().unwrap_or(0.0);
                        let modified = entry["fees"]["modified"].as_f64().unwrap_or(0.0);
                        Ok(modified < base)
                    })
                    .await;
                    anyhow::ensure!(
                        deprioritized.is_some(),
                        "the enforcer never deprioritised stale bid {txid} in Core"
                    );
                    let _undone: bool = post_setup
                        .bitcoind_client
                        .request(
                            "prioritisetransaction",
                            rpc_params![txid, 0, -NEGATIVE_MAX_SATS],
                        )
                        .await?;
                    row.delta_undone = true;
                }
            }
        } else {
            let () = mine_check_block_events::<_, DummySidechain>(
                &mut post_setup,
                1,
                MiningPolicy::SILENT,
                |_, _| Ok(()),
            )
            .await?;
            let (_, block_hash, height) = chain_tip(&mut post_setup).await?;
            let block = get_block(&post_setup, &block_hash).await?;
            row.block_height = height;
            row.bid_in_block = matches!(
                &row.bid.grpc,
                Some(Ok(txid)) if block_txids(&block).contains(txid)
            );
            row.m7_ours = coinbase_m7_accepts(&block)?.iter().any(|m7| {
                m7.sidechain_number == DummySidechain::SIDECHAIN_NUMBER
                    && m7.sidechain_block_hash.0 == h
            });
        }
        tracing::info!("{}", row.line());
        rows.push(row);
    }

    let report = table(mode, &rows);
    tracing::info!("\n{report}");
    #[expect(clippy::print_stderr)]
    {
        eprintln!("\n{report}");
    }
    anyhow::ensure!(
        rows[0].ok(),
        "the baseline round 0 bid did not get into a block, so the setup is broken\n{report}"
    );
    let rebids: Vec<&Row> = rows.iter().filter(|row| row.is_rebid()).collect();
    let failed: Vec<&&Row> = rebids.iter().filter(|row| !row.ok()).collect();
    if let Some(first) = failed.first() {
        let reason = match &first.bid.grpc {
            Some(Err(err)) => format!("the enforcer refused the bid: {err}"),
            Some(Ok(_)) if !first.bid.in_mempool => {
                "the enforcer created the bid but it never reached Core's mempool".to_owned()
            }
            Some(Ok(_)) => format!(
                "the bid ({}) was in Core's mempool but not in block h{}",
                first.bid.relation, first.block_height
            ),
            None => "no bid".to_owned(),
        };
        anyhow::bail!(
            "BMM stuck: a same-amount ({BID_SATS} sat) bid placed after a stale round did not \
             get into the next block in {}/{} rounds (mode: {mode}); first was round {} ({}): \
             {reason}\n{report}",
            failed.len(),
            rebids.len(),
            first.round,
            first.kind.map_or("?", Kind::label),
        );
    }
    Ok(())
}
