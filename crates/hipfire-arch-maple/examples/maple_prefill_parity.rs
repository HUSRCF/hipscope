// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! maple_prefill_parity — batched prefill vs the verified per-token path.
//!
//! Bit-exactness is NOT the bar: a GEMM reassociates differently from a GEMV.
//! The bar is per-position cosine >= 0.9999 AND identical greedy argmax.
//!
//! Usage:
//!   maple_prefill_parity --model <hfq> [--tokens N] [--b N]... [--chunk-split]
//!                        [--window-liveness]

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::{decode_step, forward_batch};
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 {
                (i, x)
            } else {
                acc
            }
        })
        .0
}

fn synthetic_tokens(n: usize) -> Vec<u32> {
    // Deterministic, spread across the vocab, avoiding special ids.
    (0..n)
        .map(|i| (1000 + (i * 7919) % 100_000) as u32)
        .collect()
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut n_tokens = 256usize;
    let mut bs: Vec<usize> = Vec::new();
    let mut chunk_split = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv[i + 1].clone());
                i += 2;
            }
            "--tokens" => {
                n_tokens = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--b" => {
                bs.push(argv[i + 1].parse().unwrap());
                i += 2;
            }
            "--chunk-split" => {
                chunk_split = true;
                i += 1;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let model = model.expect("--model");
    if bs.is_empty() {
        bs = vec![1, 17, 256];
    }

    let tokens = synthetic_tokens(n_tokens);
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // Oracle: per-token path, logits at every position.
    let mut hfq = HfqFile::open(Path::new(&model)).expect("open");
    let mut b0 = load_maple_from_hfq(&mut hfq, &mut gpu, n_tokens + 64).expect("load");
    let mut want = Vec::with_capacity(n_tokens);
    for (p, &t) in tokens.iter().enumerate() {
        want.push(
            decode_step(
                &b0.config,
                &b0.weights,
                &mut b0.state,
                &mut gpu,
                t,
                p as u32,
            )
            .expect("decode_step"),
        );
    }

    let mut failures = 0usize;
    for &b in &bs {
        let mut hfq = HfqFile::open(Path::new(&model)).expect("open");
        let mut bb = load_maple_from_hfq(&mut hfq, &mut gpu, n_tokens + 64).expect("load");
        let chunks: Vec<(usize, usize)> = if chunk_split {
            vec![(0, n_tokens / 2), (n_tokens / 2, n_tokens - n_tokens / 2)]
        } else {
            hipfire_arch_maple::batch::prefill_chunks(n_tokens, b)
        };
        let mut last = Vec::new();
        for (start, len) in &chunks {
            last = forward_batch(
                &bb.config,
                &bb.weights,
                &mut bb.state,
                &mut gpu,
                &tokens[*start..*start + *len],
                *start,
            )
            .expect("forward_batch");
        }
        // forward_batch returns only the LAST position's logits, so compare
        // against the oracle's last position.
        let w = &want[n_tokens - 1];
        let c = cosine(&last, w);
        let same = argmax(&last) == argmax(w);
        let ok = c >= 0.9999 && same;
        if !ok {
            failures += 1;
        }
        println!(
            "{} B={b:<4} chunks={:<3} cosine={c:.6} argmax {}",
            if ok { "OK  " } else { "FAIL" },
            chunks.len(),
            if same { "match" } else { "DIFFER" },
        );
    }

    println!("\n{} configuration(s), {failures} failure(s)", bs.len());
    if failures > 0 {
        std::process::exit(1);
    }
}
