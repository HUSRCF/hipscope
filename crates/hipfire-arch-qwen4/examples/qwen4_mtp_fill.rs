// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native-MTP prompt-fill digest probe: runs an `mtp_prefill` per prompt
//! length and writes the target/MTP state family digests, the pending hidden
//! row, the seed and one draft window as JSON. Run it once per
//! `HIPFIRE_QWEN4_MTP_BATCHED_FILL` value (the route is read once per process)
//! and `diff` the outputs: the batched fill must match the per-row fill byte
//! for byte.
//!
//! Without WARM_CHUNKS (or with 0) every fill is cold. With WARM_CHUNKS = K,
//! each LEN is the suffix of a prefix-cache hit that restores the checkpoint
//! after K whole prefill chunks and fills from that `start_pos`.
//!
//!   cargo run --release -p hipfire-arch-qwen4 --features reference-parity \
//!     --example qwen4_mtp_fill -- MODEL OUT.json MAX_SEQ LEN[,LEN...] [WARM_CHUNKS]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 && args.len() != 5 {
        eprintln!("usage: qwen4_mtp_fill MODEL OUT.json MAX_SEQ LEN[,LEN...] [WARM_CHUNKS]");
        std::process::exit(2);
    }
    let max_seq: usize = args[2].parse().expect("MAX_SEQ");
    let lengths: Vec<usize> = args[3]
        .split(',')
        .map(|length| length.parse().expect("LEN"))
        .collect();
    let warm_chunks: usize = args
        .get(4)
        .map_or(0, |chunks| chunks.parse().expect("WARM_CHUNKS"));
    match hipfire_arch_qwen4::state_parity::run_mtp_fill_digest(
        std::path::Path::new(&args[0]),
        &lengths,
        max_seq,
        warm_chunks,
    ) {
        Ok(report) => {
            std::fs::write(&args[1], serde_json::to_string_pretty(&report).unwrap())
                .expect("write output");
        }
        Err(error) => {
            eprintln!("qwen4_mtp_fill: {error}");
            std::process::exit(1);
        }
    }
}
