//! Deep unconfirmed chains in the node's mempool, CPFP, RBF, a stale BMM
//! request the template dry run rejects, and a reorg that hands a block's
//! transactions back to the mempool, with block templates requested after
//! every step. The enforcer must keep serving templates throughout.
//!
//! Regression for the 2026-10-29 eCash mainnet crash: a replayed Bitcoin block
//! (973740, 25-deep chains) in the mempool, and `getblocktemplate` panicked
//! with "Amount subtraction error" in the mempool crate's `Mempool::remove()`,
//! killing the enforcer. The crate keeps a running ancestor-fee total per tx.
//! When a block holding X and its child M is disconnected while M's child D
//! is in the mempool, the node re-adds X and then M, so the mirror inserts M
//! after D. `insert` adds only M's fee to D's ancestor total, never X's. Any
//! later `remove(X)` (proposing X for a template, the dry-run filter dropping
//! it, or X confirming again) subtracts X's fee from D's short total, which
//! underflowed until the subtraction saturated (cusf-enforcer-mempool PR #139).

use std::time::Duration;

use bip300301_enforcer_lib::proto::mainchain::GetSidechainsRequest;
use bitcoin::{BlockHash, Txid};
use jsonrpsee::{core::client::ClientT as _, rpc_params};

use crate::{
    block_verdict::wait_for_enforcer_tip_hash,
    integration_test::propose_sidechain,
    mine::{MiningPolicy, mine_check_block_events},
    setup::{DummySidechain, PostSetup, Sidechain, invalidate_block, wait_until},
    test_bmm_bid_auction::{block_template, craft_core_bid, h_star, template_txids},
};

const CHAINS: usize = 16;
/// The node's ancestor limit.
const DEPTH: usize = 25;
/// The chain head's fee. Far above every descendant's ancestor total
/// without it, so a short total underflows when the head is removed.
const HEAD_FEE_SATS: u64 = 1_000_000;
const LINK_FEE_SATS: u64 = 1_000;
/// Odd chains end in a child paying for its parents (CPFP).
const CPFP_LEAF_FEE_SATS: u64 = 300_000;
/// The first `RBF_CHAINS` even chains get their link at `RBF_LEVEL`
/// replaced, which evicts every descendant of the replaced link. Even chains
/// only: an odd chain's CPFP leaf pays more than the replacement.
const RBF_CHAINS: usize = 3;
const RBF_LEVEL: usize = 12;
const RBF_FEE_SATS: u64 = 50_000;
const GBT_ROUNDS_AFTER_REORG: usize = 10;

struct ChainTx {
    txid: Txid,
    hex: String,
}

/// The enforcer's block template, or the reason it could not serve one. If
/// it died, the panic it logged.
async fn template_or_panic(post_setup: &PostSetup, step: &str) -> anyhow::Result<usize> {
    match block_template(post_setup).await {
        Ok(template) => Ok(template.transactions.len()),
        Err(err) => {
            // Give a dying process a moment to flush its last log lines.
            tokio::time::sleep(Duration::from_secs(1)).await;
            let mut panics = Vec::new();
            for file in ["stdout.txt", "stderr.txt"] {
                let path = post_setup.directories.enforcer_dir.join(file);
                let bytes = std::fs::read(&path).unwrap_or_default();
                let text = String::from_utf8_lossy(&bytes);
                panics.extend(
                    text.lines()
                        .filter(|line| {
                            line.contains("panicked")
                                || line.contains("Amount subtraction")
                                || line.contains("exited unexpectedly")
                        })
                        .map(|line| line.chars().take(400).collect::<String>()),
                );
            }
            anyhow::bail!(
                "the enforcer stopped serving block templates after {step}: {err:#}\n\
                 enforcer log lines: {panics:#?}"
            )
        }
    }
}

/// One link of a chain: spend `(prev_txid, 0)` worth `prev_value` sats,
/// paying `fee` and the rest to `address`.
async fn build_link(
    post_setup: &PostSetup,
    prev_txid: Txid,
    prev_value: u64,
    fee: u64,
    address: &str,
    script_pubkey: &str,
) -> anyhow::Result<(ChainTx, u64)> {
    let value = prev_value - fee;
    let btc = |sats: u64| format!("{}.{:08}", sats / 100_000_000, sats % 100_000_000);
    let inputs = serde_json::json!([{ "txid": prev_txid, "vout": 0 }]);
    let outputs = serde_json::json!([{ address: btc(value) }]);
    let raw: String = post_setup
        .bitcoind_client
        .request("createrawtransaction", rpc_params![inputs, outputs])
        .await?;
    let prevtxs = serde_json::json!([{
        "txid": prev_txid,
        "vout": 0,
        "scriptPubKey": script_pubkey,
        "amount": btc(prev_value),
    }]);
    let signed: serde_json::Value = post_setup
        .bitcoind_client
        .request("signrawtransactionwithwallet", rpc_params![raw, prevtxs])
        .await?;
    anyhow::ensure!(
        signed["complete"].as_bool() == Some(true),
        "failed to sign chain link: {signed}"
    );
    let hex = signed["hex"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("signrawtransactionwithwallet returned no hex"))?
        .to_owned();
    let tx: bitcoin::Transaction = bitcoin::consensus::deserialize(&hex::decode(&hex)?)?;
    Ok((
        ChainTx {
            txid: tx.compute_txid(),
            hex,
        },
        value,
    ))
}

async fn send(post_setup: &PostSetup, hex: &str) -> anyhow::Result<Txid> {
    // `maxfeerate` 0: the chain heads pay a deliberately high fee.
    Ok(post_setup
        .bitcoind_client
        .request("sendrawtransaction", rpc_params![hex, 0])
        .await?)
}

async fn core_mempool(post_setup: &PostSetup) -> anyhow::Result<Vec<Txid>> {
    Ok(post_setup
        .bitcoind_client
        .request("getrawmempool", rpc_params![])
        .await?)
}

/// * Flood the node's mempool with deep chains, level by level as a replay
///   bridge would, requesting a template after every level
/// * Replace a mid-chain link in some chains (RBF)
/// * Mine a block holding the top two links of every chain, put a BMM
///   request on it, then disconnect it: the request is stale, and the node
///   hands the links back to the mempool behind their descendants
/// * Keep requesting templates, then mine through the enforcer and request
///   again
pub async fn test_mempool_deep_chain_reorg(mut post_setup: PostSetup) -> anyhow::Result<()> {
    let mut log = Vec::<String>::new();
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
    anyhow::ensure!(sidechains.len() == 1, "expected the sidechain active");
    let core_addr: String = post_setup
        .bitcoind_client
        .request("getnewaddress", rpc_params![])
        .await?;
    let address_info: serde_json::Value = post_setup
        .bitcoind_client
        .request("getaddressinfo", rpc_params![core_addr.clone()])
        .await?;
    let script_pubkey = address_info["scriptPubKey"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("getaddressinfo returned no scriptPubKey"))?
        .to_owned();
    // CHAINS chain roots plus a few spare mature coinbases for the BMM bid.
    let mined: Vec<BlockHash> = post_setup
        .bitcoind_client
        .request("generatetoaddress", rpc_params![120, core_addr.clone()])
        .await?;
    let tip = *mined
        .last()
        .ok_or_else(|| anyhow::anyhow!("mined no blocks"))?;
    let () = wait_for_enforcer_tip_hash(&post_setup, tip).await?;

    // Build every chain up front, as signed raw txs.
    let unspent: serde_json::Value = post_setup
        .bitcoind_client
        .request("listunspent", rpc_params![1])
        .await?;
    let roots: Vec<(Txid, u32, u64)> = unspent
        .as_array()
        .into_iter()
        .flatten()
        .filter(|utxo| utxo["spendable"].as_bool() == Some(true))
        .filter_map(|utxo| {
            let txid = utxo["txid"].as_str()?.parse().ok()?;
            let vout = u32::try_from(utxo["vout"].as_u64()?).ok()?;
            let sats = bitcoin::Amount::from_btc(utxo["amount"].as_f64()?)
                .ok()?
                .to_sat();
            (sats >= 100_000_000).then_some((txid, vout, sats))
        })
        .take(CHAINS)
        .collect();
    anyhow::ensure!(roots.len() == CHAINS, "only {} mature roots", roots.len());
    let mut chains = Vec::<Vec<ChainTx>>::with_capacity(CHAINS);
    for (chain, (root_txid, root_vout, root_sats)) in roots.into_iter().enumerate() {
        // The head spends the root outpoint, which need not be vout 0.
        let btc = |sats: u64| format!("{}.{:08}", sats / 100_000_000, sats % 100_000_000);
        let head_value = root_sats - HEAD_FEE_SATS;
        let raw: String = post_setup
            .bitcoind_client
            .request(
                "createrawtransaction",
                rpc_params![
                    serde_json::json!([{ "txid": root_txid, "vout": root_vout }]),
                    serde_json::json!([{ core_addr.clone(): btc(head_value) }])
                ],
            )
            .await?;
        let signed: serde_json::Value = post_setup
            .bitcoind_client
            .request("signrawtransactionwithwallet", rpc_params![raw])
            .await?;
        anyhow::ensure!(signed["complete"].as_bool() == Some(true), "head unsigned");
        let head_hex = signed["hex"].as_str().unwrap_or_default().to_owned();
        let head: bitcoin::Transaction = bitcoin::consensus::deserialize(&hex::decode(&head_hex)?)?;
        let mut links = vec![ChainTx {
            txid: head.compute_txid(),
            hex: head_hex,
        }];
        let mut value = head_value;
        for level in 1..DEPTH {
            let fee = if level == DEPTH - 1 && chain % 2 == 1 {
                CPFP_LEAF_FEE_SATS
            } else {
                LINK_FEE_SATS
            };
            let prev = links[level - 1].txid;
            let (link, next_value) =
                build_link(&post_setup, prev, value, fee, &core_addr, &script_pubkey).await?;
            links.push(link);
            value = next_value;
        }
        chains.push(links);
    }
    log.push(format!("built {CHAINS} chains x {DEPTH} links"));

    // Submit level by level, as a replay bridge's hold/retry loop does.
    for level in 0..DEPTH {
        for links in &chains {
            let _txid: Txid = send(&post_setup, &links[level].hex).await?;
        }
        let in_template = template_or_panic(&post_setup, &format!("flood level {level}")).await?;
        if level % 6 == 0 || level == DEPTH - 1 {
            log.push(format!(
                "flood level {level}: template has {in_template} txs"
            ));
        }
    }
    let all: Vec<Txid> = chains.iter().flatten().map(|link| link.txid).collect();
    let () = wait_until("the template to carry every chain link", || async {
        let txids = template_txids(&block_template(&post_setup).await?);
        Ok(all.iter().all(|txid| txids.contains(txid)))
    })
    .await?;
    log.push(format!(
        "flooded: Core mempool {} txs, all in the template",
        core_mempool(&post_setup).await?.len()
    ));

    // RBF a mid-chain link: the replacement drops every descendant.
    for links in chains.iter().step_by(2).take(RBF_CHAINS) {
        let prev_tx: bitcoin::Transaction =
            bitcoin::consensus::deserialize(&hex::decode(&links[RBF_LEVEL - 1].hex)?)?;
        let prev_value = prev_tx.output[0].value.to_sat();
        let (replacement, _) = build_link(
            &post_setup,
            links[RBF_LEVEL - 1].txid,
            prev_value,
            RBF_FEE_SATS,
            &core_addr,
            &script_pubkey,
        )
        .await?;
        let _txid: Txid = send(&post_setup, &replacement.hex).await?;
    }
    let in_template = template_or_panic(&post_setup, "RBF").await?;
    log.push(format!(
        "RBF'd link {RBF_LEVEL} of {RBF_CHAINS} chains: Core mempool {} txs, template {in_template}",
        core_mempool(&post_setup).await?.len()
    ));

    // A block holding each chain's head and second link.
    let block_txids: Vec<String> = chains
        .iter()
        .flat_map(|links| [links[0].txid.to_string(), links[1].txid.to_string()])
        .collect();
    #[derive(serde::Deserialize)]
    struct Generated {
        hash: BlockHash,
    }
    let generated: Generated = post_setup
        .bitcoind_client
        .request("generateblock", rpc_params![core_addr.clone(), block_txids])
        .await?;
    let block = generated.hash;
    let () = wait_for_enforcer_tip_hash(&post_setup, block).await?;
    let in_template = template_or_panic(&post_setup, "the block with the chain heads").await?;
    log.push(format!(
        "mined {block} with {} chain links: template {in_template}",
        2 * CHAINS
    ));
    let bid = craft_core_bid(
        &post_setup,
        DummySidechain::SIDECHAIN_NUMBER,
        5_000,
        &h_star("bid on the block the reorg disconnects"),
        block,
        0,
    )
    .await?;
    let () = wait_until("the BMM request to reach the template", || async {
        Ok(template_txids(&block_template(&post_setup).await?).contains(&bid))
    })
    .await?;
    log.push(format!("BMM request {bid} on {block} is in the template"));

    // Disconnect it: the heads come back behind their descendants, the BMM
    // request is stale.
    let parent: BlockHash = {
        let header: serde_json::Value = post_setup
            .bitcoind_client
            .request("getblockheader", rpc_params![block])
            .await?;
        header["previousblockhash"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("no previousblockhash"))?
            .parse()?
    };
    let () = invalidate_block(&post_setup.bitcoind_client, block).await?;
    let () = wait_for_enforcer_tip_hash(&post_setup, parent).await?;
    log.push(format!(
        "invalidated {block}: Core mempool {} txs",
        core_mempool(&post_setup).await?.len()
    ));
    for round in 0..GBT_ROUNDS_AFTER_REORG {
        let in_template =
            template_or_panic(&post_setup, &format!("the reorg (template {round})")).await?;
        if round == 0 || round == GBT_ROUNDS_AFTER_REORG - 1 {
            log.push(format!("after reorg, template {round}: {in_template} txs"));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let template = block_template(&post_setup).await?;
    anyhow::ensure!(
        !template_txids(&template).contains(&bid),
        "the stale BMM request must not be in the template"
    );

    // Mine through the enforcer: the heads confirm again and leave the mirror.
    let () = mine_check_block_events::<_, DummySidechain>(
        &mut post_setup,
        1,
        MiningPolicy::SILENT,
        |_, _| Ok(()),
    )
    .await?;
    for round in 0..3 {
        let in_template = template_or_panic(
            &post_setup,
            &format!("re-mining the heads (template {round})"),
        )
        .await?;
        log.push(format!(
            "after re-mining, template {round}: {in_template} txs"
        ));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let report = log.join("\n");
    tracing::info!("\n{report}");
    #[expect(clippy::print_stderr)]
    {
        eprintln!("\nmempool_deep_chain_reorg:\n{report}");
    }
    Ok(())
}
