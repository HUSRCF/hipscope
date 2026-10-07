// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Session-cache restores on Flash-Next are byte-identical to a cold prefill.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_SESSION_CACHE_MODEL`.
//!
//! Prompt A (three chunks and a tail) is prefilled cold, capturing a chain of
//! delta snapshots at its three chunk boundaries. Prompt B shares only A's
//! first chunk and is also prefilled cold, so its two-chunk snapshot is a
//! delta over A's first (asserted through the stored bytes). A is then
//! restored through its three-link chain and B through the shared root.
//!
//! Each restored prefill must leave the live state byte-equal to its cold
//! prefill: a SHA-256 of every state part (metadata, every fixed part and the
//! valid rows of every row stream) is compared, so a wrong byte in rows that
//! attention never selects still fails. The AR leg also compares the final
//! logits and 16 greedy ids; the MTP leg repeats the chain and branch on the
//! native MTP route (target plus draft head and its policy) and compares the
//! seed token.

use hipfire_arch_qwen4::bundle::{session_snapshot_bytes, Qwen4Bundle};
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const MODEL_ENV: &str = "HIPFIRE_SESSION_CACHE_MODEL";
const MAX_SEQ: usize = 32768;
const DECODE: usize = 16;

fn prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// SHA-256 of every part of the live state at `position` on `route`: the
/// metadata, each fixed part, then the valid rows of each row stream.
fn state_digests(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    route: SessionRoute,
    position: usize,
) -> Vec<[u8; 32]> {
    let SnapshotParts { meta, layout } =
        SessionState::snapshot_parts(bundle, route, position).expect("state parts");
    gpu.hip.device_synchronize().expect("sync");
    let mut digests = vec![Sha256::digest(&meta).into()];
    let ranges = layout
        .fixed
        .iter()
        .map(|part| (part.buf, part.offset, part.bytes))
        .chain(
            layout
                .rows
                .iter()
                .map(|stream| (stream.buf, 0, stream.rows * stream.row_bytes)),
        );
    for (buf, offset, bytes) in ranges {
        let mut host = vec![0u8; bytes];
        gpu.hip
            .memcpy_dtoh_at(&mut host, buf, offset)
            .expect("download state part");
        digests.push(Sha256::digest(&host).into());
    }
    digests
}

/// Fail with the index of the first state part a restore got wrong.
fn assert_same_state(name: &str, cold: &[[u8; 32]], warm: &[[u8; 32]]) {
    assert_eq!(cold.len(), warm.len(), "{name}: state part count");
    if let Some(part) = (0..cold.len()).find(|&i| cold[i] != warm[i]) {
        panic!(
            "{name}: state part {part} of {} differs from cold (0 = metadata)",
            cold.len()
        );
    }
}

/// Prefill `tokens` reusing `reused` cached tokens and commit. Returns the
/// state digests, the final logits bytes and `DECODE` greedy ids.
fn run(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    tokens: &[u32],
    reused: usize,
) -> (Vec<[u8; 32]>, Vec<u8>, Vec<u32>) {
    bundle
        .prefill_final(gpu, tokens, reused, logits)
        .expect("prefill");
    let digests = state_digests(bundle, gpu, SessionRoute::Ar, tokens.len());
    bundle.session_commit();
    let mut bytes = vec![0u8; logits.byte_size()];
    gpu.hip.device_synchronize().expect("sync");
    gpu.hip
        .memcpy_dtoh(&mut bytes, &logits.buf)
        .expect("download logits");
    let ids = (0..DECODE)
        .map(|_| {
            bundle
                .forward_token_or_argmax(gpu, None, logits)
                .expect("decode")
        })
        .collect();
    (digests, bytes, ids)
}

/// Native MTP prefill of `tokens` reusing `reused` cached tokens, then
/// commit. Returns the state digests (target and draft head) and the seed.
fn run_mtp(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    drafter: &mut Qwen4MtpDrafter,
    tokens: &[u32],
    reused: usize,
) -> (Vec<[u8; 32]>, u32) {
    let seed = drafter
        .mtp_prefill(
            gpu,
            bundle,
            tokens,
            &tokens[reused..],
            reused,
            reused > 0,
            &|| false,
        )
        .expect("MTP prefill");
    let digests = state_digests(bundle, gpu, SessionRoute::Mtp, tokens.len());
    bundle.session_commit();
    (digests, seed)
}

fn stored_bytes(bundle: &Qwen4Bundle) -> u64 {
    bundle
        .session_cache()
        .expect("session cache")
        .stored_bytes()
}

/// Four snapshots (A's three links and B's one) of one chunk each: a full
/// copy of B's two chunks would exceed four chunk-sized snapshots.
fn assert_four_deltas(route: &str, stored: u64, one: u64) {
    println!("{route}: {stored} bytes stored, {one} per one-chunk snapshot");
    assert!(
        stored > 3 * one && stored <= 4 * one,
        "{route}: expected four one-chunk snapshots ({stored} bytes, {one} each)"
    );
}

#[test]
#[ignore = "needs a HIP GPU and HIPFIRE_SESSION_CACHE_MODEL"]
fn restored_prefill_matches_cold_on_flash_next() {
    let model = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    let mut hfq = HfqFile::open(&model).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let receipt = admit_hfqm_artifact(&hfq).expect("admission");
    let mut gpu = Gpu::init().expect("GPU");
    let domain = CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id);
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().expect("mesh");
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .expect("weights");
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )
    .expect("state format");
    let vocab = receipt.config.vocab_size;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        transaction,
        &receipt.placements,
        &mut gpu,
        MAX_SEQ,
        receipt.ple,
        state_format,
        backend,
    )
    .expect("assemble");
    bundle.attach_forward(&mut gpu, MAX_SEQ).expect("forward");
    bundle.attach_session_cache(SessionCache::new(domain, u64::MAX >> 1));
    let chunk = bundle.spec_chunk_rows().expect("chunk rows");
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");
    let a = prompt(0xa, 3 * chunk + 300);
    let mut b = a[..chunk].to_vec();
    b.extend(prompt(0xb, chunk + 200));
    println!(
        "chunk={chunk} state={state_format:?} backend={}",
        backend.name()
    );

    let one = |mtp| {
        session_snapshot_bytes(&bundle.config, state_format, mtp, chunk).expect("snapshot bytes")
    };
    let (one_ar, one_mtp) = (one(false), one(true));

    let cold_a = run(&mut bundle, &mut gpu, &logits, &a, 0);
    let cold_b = run(&mut bundle, &mut gpu, &logits, &b, 0);
    assert_four_deltas("AR", stored_bytes(&bundle), one_ar);
    for (name, tokens, cold, links) in [("A", &a, &cold_a, 3), ("B", &b, &cold_b, 2)] {
        let reused = bundle.session_plan(tokens, SessionRoute::Ar);
        assert_eq!(
            reused,
            links * chunk,
            "plan must offer {name}'s {links}-chunk snapshot"
        );
        let (digests, warm_logits, warm_ids) = run(&mut bundle, &mut gpu, &logits, tokens, reused);
        println!(
            "AR {name} cold ids {:?}\nAR {name} warm ids {warm_ids:?}",
            cold.2
        );
        assert_same_state(&format!("AR {name}"), &cold.0, &digests);
        assert!(
            warm_logits == cold.1,
            "AR {name}: restored final logits differ from cold"
        );
        assert_eq!(warm_ids, cold.2, "AR {name}");
    }

    bundle.attach_mtp(&mut gpu, MAX_SEQ).expect("MTP head");
    let mut drafter = Qwen4MtpDrafter::new(3, MAX_SEQ, None);
    let before = stored_bytes(&bundle);
    let cold_a = run_mtp(&mut bundle, &mut gpu, &mut drafter, &a, 0);
    let cold_b = run_mtp(&mut bundle, &mut gpu, &mut drafter, &b, 0);
    assert_four_deltas("MTP", stored_bytes(&bundle) - before, one_mtp);
    for (name, tokens, cold, links) in [("A", &a, &cold_a, 3), ("B", &b, &cold_b, 2)] {
        let reused = bundle.session_plan(tokens, SessionRoute::Mtp);
        assert_eq!(
            reused,
            links * chunk,
            "MTP plan must offer {name}'s {links}-chunk snapshot"
        );
        let (digests, seed) = run_mtp(&mut bundle, &mut gpu, &mut drafter, tokens, reused);
        println!("MTP {name} seed cold {} warm {seed}", cold.1);
        assert_same_state(&format!("MTP {name}"), &cold.0, &digests);
        assert_eq!(seed, cold.1, "MTP {name} seed");
    }

    gpu.free_tensor(logits).expect("free logits");
    bundle.free_gpu(&mut gpu).expect("free bundle");
}
