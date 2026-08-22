// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! maple_prefill_parity — batched prefill vs the verified per-token path.
//!
//! Bit-exactness is NOT the bar: the batched path is a WMMA GEMM over F16
//! activations (codebook deltas computed in `_Float16`, F32 accumulation —
//! see `kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip` ~L89-136),
//! while the per-token oracle is an F32 scalar GEMV. A ~1e-3 relative
//! per-GEMM difference compounds over 24 layers; this is arithmetic, not a
//! defect, and is the same shape as the shipped cohere2moe Q8 prefill path.
//!
//! The HARD bar is **identical greedy argmax at every comparison point**
//! (every `forward_batch` call — it returns only the last token's logits per
//! call, so each chunk boundary is one point). Cosine is a REPORTED metric
//! with a floor set from measurement on real text, not from wishful
//! bit-parity (see `COSINE_FLOOR` below for the measured numbers).
//!
//! DEFAULT input is real tokenized prose (see `real_tokens`), because
//! synthetic OOD input (`--synthetic`) makes logits flat and hypersensitive
//! to perturbation, which exaggerates the arithmetic gap above and is not
//! representative of how this path is actually exercised.
//!
//! KNOWN OPEN FINDING (2026-08-22): on real text with multiple sequential
//! chunks (`--b 1`, `--b 17`), the hard bar currently FAILS — argmax
//! mismatches at a minority of comparison points. `--b 256` (single chunk)
//! and `--chunk-split` (two chunks) pass. See `.superpowers/sdd/task-4-report.md`
//! and `COSINE_FLOOR` below for full numbers. This is NOT masked by the bar.
//!
//! Usage:
//!   maple_prefill_parity --model <hfq> [--tokens N] [--b N]... [--chunk-split]
//!                        [--synthetic]

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::{decode_step, forward_batch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

/// Cosine floor for the REPORTED metric (not the hard bar — see module docs).
///
/// Measured 2026-08-22 on real tokenized prose (200 tokens), gfx1151,
/// release, per-comparison-point cosine (two runs; GPU reduction order is
/// not fully deterministic call-to-call, so figures are given as ranges):
///   B=1     (200 chained single-token chunks): min 0.897-0.900  mean ~0.994
///   B=17    (12 chunks):                        min 0.984-0.984  mean ~0.996
///   B=256   (1 chunk, whole prompt):             min 0.980-0.996
///   chunk-split (2 chunks):                      min 0.998-0.999
/// The single-chunk case (B=256, one comparison point) matches the
/// "~0.98-0.99 over 24 layers" arithmetic reasoning below almost exactly.
/// The lower B configurations dip further because chunk N's KV cache is
/// itself the OUTPUT of the WMMA-batched path for chunk N-1: with many
/// sequential chunks the F16-vs-F32 gap doesn't just compound over 24
/// layers once, it compounds again at every chunk boundary through the
/// cached K/V. That is still the same root cause (F16 input to the WMMA
/// GEMM), just applied repeatedly, not a second defect.
///
/// This floor (0.85) sits with margin below the worst measured real-text
/// minimum (~0.897), so a normal run reliably passes it while a genuine
/// collapse (an unrelated ~152k-wide logit vector sits near cosine 0) still
/// trips it. It is deliberately NOT 0.9999: that bar is unachievable by
/// construction because the batched path's WMMA GEMM consumes F16
/// activations (codebook deltas computed in `_Float16`, F32 accumulate)
/// while the oracle's GEMV is F32 scalar throughout, so a ~1e-3 relative
/// per-GEMM difference is expected and compounds as described above.
///
/// Cosine is NOT the hard bar (see module docs) — argmax-at-every-point is.
/// On this same real-text data, argmax MISMATCHED at some comparison points
/// for B=1 (~7% of points) and B=17 (~25% of points) — see
/// `.superpowers/sdd/task-4-report.md`. That is reported as a real, open
/// finding, not hidden by loosening this floor or the hard bar.
const COSINE_FLOOR: f64 = 0.85;

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

/// OUT-OF-DISTRIBUTION stress case. Deterministic ids scattered across the
/// vocab, avoiding special ids. On this input the model's logits are flat
/// and hypersensitive to perturbation, which exaggerates the F16-vs-F32
/// arithmetic gap described in the module docs — this is a stress test, NOT
/// a representative measurement, and must never be reported as if it were.
fn synthetic_tokens(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (1000 + (i * 7919) % 100_000) as u32)
        .collect()
}

/// Realistic prose, tokenized through the model's own tokenizer and
/// repeated/extended to reach `n` tokens. This is the DEFAULT input: it puts
/// the model in-distribution, which is what the batched prefill path is
/// actually used for (chat prompts), unlike `synthetic_tokens`.
fn real_tokens(tokenizer: &Tokenizer, n: usize) -> Vec<u32> {
    const PASSAGE: &str = "The history of computing is a story of \
        abstraction: each generation of engineers built a new layer that let \
        the next generation stop thinking about the one below it. Transistors \
        gave way to logic gates, logic gates to instruction sets, instruction \
        sets to operating systems, and operating systems to the sprawling \
        libraries and frameworks that most software is now assembled from. \
        Time-series databases are a narrower case of the same pattern: they \
        exist because general-purpose relational engines make you pay, on \
        every single query, for flexibility that time-ordered, append-mostly \
        data almost never needs. A database that assumes rows arrive roughly \
        in timestamp order, that a query is usually bounded by a time range, \
        and that most columns are numeric can partition, compress, and index \
        far more aggressively than one that has to stay correct for arbitrary \
        update-heavy workloads. The interesting engineering is in how much of \
        that specialization can be exposed to the user as convenience — \
        SAMPLE BY, ASOF JOIN, LATEST ON — rather than as constraints they have \
        to work around by hand. ";
    let mut text = String::new();
    while tokenizer.encode(&text).len() < n {
        text.push_str(PASSAGE);
    }
    let mut toks = tokenizer.encode(&text);
    toks.truncate(n);
    toks
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut n_tokens = 256usize;
    let mut bs: Vec<usize> = Vec::new();
    let mut chunk_split = false;
    let mut synthetic = false;
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
            "--synthetic" => {
                synthetic = true;
                i += 1;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let model = model.expect("--model");
    if bs.is_empty() {
        bs = vec![1, 17, 256];
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // Oracle: per-token path, logits at every position.
    let mut hfq = HfqFile::open(Path::new(&model)).expect("open");

    let tokens = if synthetic {
        println!(
            "*** --synthetic: OUT-OF-DISTRIBUTION stress case (arbitrary ids \
             scattered across the vocab). Flat, hypersensitive logits \
             EXAGGERATE the F16-vs-F32 arithmetic gap. NOT a representative \
             measurement — do not compare these numbers against the real-text \
             run. ***"
        );
        synthetic_tokens(n_tokens)
    } else {
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
        println!("input: real tokenized prose (in-distribution)");
        real_tokens(&tokenizer, n_tokens)
    };
    assert_eq!(
        tokens.len(),
        n_tokens,
        "token generator produced wrong count"
    );

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

        // Every `forward_batch` CALL is one comparison point: it returns
        // only the LAST token's logits for that chunk, compared against the
        // oracle's logits at that same absolute position. The hard bar is
        // argmax match at EVERY point, not just the final chunk.
        let mut points = 0usize;
        let mut mismatches = 0usize;
        let mut cosines: Vec<f64> = Vec::with_capacity(chunks.len());
        for (start, len) in &chunks {
            let last = forward_batch(
                &bb.config,
                &bb.weights,
                &mut bb.state,
                &mut gpu,
                &tokens[*start..*start + *len],
                *start,
            )
            .expect("forward_batch");
            let w = &want[*start + *len - 1];
            let c = cosine(&last, w);
            let same = argmax(&last) == argmax(w);
            points += 1;
            cosines.push(c);
            if !same {
                mismatches += 1;
            }
        }

        let min_c = cosines.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_c = cosines.iter().sum::<f64>() / cosines.len() as f64;
        // Hard bar: argmax must match at EVERY comparison point.
        let argmax_ok = mismatches == 0;
        // Reported floor, not the hard bar (see COSINE_FLOOR docs).
        let cosine_ok = min_c >= COSINE_FLOOR;
        let ok = argmax_ok && cosine_ok;
        if !ok {
            failures += 1;
        }
        println!(
            "{} B={b:<4} chunks={:<3} points={points:<3}  argmax {}/{points} match [{}]  \
             cosine min={min_c:.6} mean={mean_c:.6} floor={COSINE_FLOOR} [{}]",
            if ok { "OK  " } else { "FAIL" },
            chunks.len(),
            points - mismatches,
            if argmax_ok { "PASS" } else { "FAIL" },
            if cosine_ok { "PASS" } else { "FAIL" },
        );
    }

    println!("\n{} configuration(s), {failures} failure(s)", bs.len());
    if failures > 0 {
        std::process::exit(1);
    }
}
