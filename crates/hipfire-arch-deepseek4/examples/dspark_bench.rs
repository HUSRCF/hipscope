// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
//! dspark_bench: A/B benchmark harness for the DSpark vs MTP spec-decode
//! drafters, driving the EXACT generic `Speculator` path the daemon uses
//! (`Deepseek4DsparkDrafter` / `Deepseek4MtpDrafter` behind `MtpSpeculator`).
//!
//! Loads the DeepSeek-V4-Flash trunk + `-dspark` sidecar, builds a
//! `Deepseek4Bundle` (the `SpecTarget`), constructs the requested speculator,
//! prefills a chat-framed prompt, then runs the generic prefill→step decode
//! loop greedily for `--max` tokens, reporting:
//!   - decode tok/s (TG only, fresh-process, post-warm)
//!   - acceptance τ = committed_tokens / windows (avg tokens emitted per window)
//!   - draft accept fraction = accepted / proposed
//!   - the decoded text (for human coherence eyeball)
//!
//! Drafter selection (matches the loader's `HIPFIRE_DEEPSEEK4_DSPARK` gate):
//!   HIPFIRE_DEEPSEEK4_DSPARK=0  → MTP drafter; otherwise DSpark (if sidecar).
//!
//! ENV:
//!   HIPFIRE_DEEPSEEK4_MODEL   trunk HFQ path (default ~/.hipfire/models/deepseek-v4-flash.mq2lloyd)
//!   HIPFIRE_DEEPSEEK4_PROMPT  prompt text (default a fixed sentence)
//!   HIPFIRE_DEEPSEEK4_MAX     max decode tokens (default 160)
//!   HIPFIRE_DEEPSEEK4_WARMUP  throwaway warmup tokens before the timed run (default 24)
//!   HIPFIRE_DEEPSEEK4_DSPARK  =0 forces MTP; else DSpark
//!   HIPFIRE_DEEPSEEK4_BENCH_RAW=1  base completion (no chat framing)

use hipfire_arch_deepseek4::dspark_speculator::build_deepseek4_dspark_speculator;
use hipfire_arch_deepseek4::mtp_speculator::build_deepseek4_mtp_speculator;
use hipfire_arch_deepseek4::{Deepseek4Bundle, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{PrefillOutcome, Speculator};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn no_abort() -> bool {
    false
}

/// Run the generic spec loop for `max` tokens. Returns
/// (generated_tokens, windows, drafts_proposed, drafts_accepted).
#[allow(clippy::too_many_arguments)]
fn decode_loop(
    spec: &mut dyn Speculator,
    bundle: &mut Deepseek4Bundle,
    gpu: &mut Gpu,
    first_token: u32,
    start_pos: usize,
    max: usize,
    eos: u32,
    raw: bool,
) -> Result<(Vec<u32>, u64, u64, u64), String> {
    let mut generated: Vec<u32> = Vec::with_capacity(max);
    let mut position = start_pos;
    let mut seed = first_token;
    let mut windows: u64 = 0;
    let mut proposed: u64 = 0;
    let mut accepted: u64 = 0;

    // The first token is the prefill's argmax; it is emitted as the seed of the
    // first window's continuation (the daemon emits it before stepping).
    if !raw && first_token == eos {
        return Ok((generated, windows, proposed, accepted));
    }
    generated.push(first_token);

    while generated.len() < max {
        let step = spec.step(gpu, bundle, position, seed, &generated, None, 0.0)?;
        windows += 1;
        proposed += step.proposed as u64;
        accepted += step.accepted as u64;
        let mut hit_eos = false;
        for &t in step.emit.iter() {
            if generated.len() >= max {
                break;
            }
            if !raw && t == eos {
                hit_eos = true;
                break;
            }
            generated.push(t);
        }
        position += step.emit.len();
        seed = step.next_seed;
        if hit_eos || (!raw && seed == eos) {
            break;
        }
    }
    Ok((generated, windows, proposed, accepted))
}

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_DEEPSEEK4_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.hipfire/models/deepseek-v4-flash.mq2lloyd",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let prompt = std::env::var("HIPFIRE_DEEPSEEK4_PROMPT")
        .unwrap_or_else(|_| "Explain in three sentences why the sky is blue.".to_string());
    let max: usize = std::env::var("HIPFIRE_DEEPSEEK4_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(160);
    let warmup: usize = std::env::var("HIPFIRE_DEEPSEEK4_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let raw = std::env::var("HIPFIRE_DEEPSEEK4_BENCH_RAW").ok().as_deref() == Some("1");

    eprintln!("Loading DeepSeek V4 trunk from {path}...");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found in HFQ metadata: {e:?}"))?;

    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok = lookup_id("<｜User｜>");
    let asst_tok = lookup_id("<｜Assistant｜>");
    let eos_tok = lookup_id("<｜end▁of▁sentence｜>").unwrap_or(tokenizer.eos_id);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let state = DeepseekV4State::new(&cfg)?;

    let dspark_enabled = weights.dspark.is_some()
        && std::env::var("HIPFIRE_DEEPSEEK4_DSPARK").ok().as_deref() != Some("0");
    let block = if let Some(d) = weights.dspark.as_ref() {
        d.cfg.block_size
    } else {
        5
    };
    eprintln!(
        "Drafter: {} (block={}) dspark_present={}",
        if dspark_enabled { "DSpark" } else { "MTP" },
        block,
        weights.dspark.is_some()
    );

    let mut bundle = Deepseek4Bundle {
        config: cfg.clone(),
        weights,
        state,
        eos_tok,
    };

    let ctx_cap = cfg.max_position_embeddings;
    let mut spec: Box<dyn Speculator> = if dspark_enabled {
        build_deepseek4_dspark_speculator(block, ctx_cap)
    } else {
        let k: usize = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(block);
        build_deepseek4_mtp_speculator(k, ctx_cap)
    };

    // Build the prompt tokens (chat-framed unless raw).
    let mut prompt_tokens: Vec<u32> = Vec::new();
    if raw {
        prompt_tokens.extend(tokenizer.encode(&prompt));
    } else {
        if let Some(b) = bos_tok {
            prompt_tokens.push(b);
        }
        if let Some(u) = user_tok {
            prompt_tokens.push(u);
        }
        prompt_tokens.extend(tokenizer.encode(&prompt));
        if let Some(a) = asst_tok {
            prompt_tokens.push(a);
        }
    }
    let prompt_md5 = format!("{:x}", md5ish(&prompt_tokens));
    eprintln!(
        "prompt: {prompt:?} -> {} tokens (token-md5 {prompt_md5})",
        prompt_tokens.len()
    );

    // ── WARMUP: full prefill + short throwaway decode (JIT + DPM ramp). ──
    {
        let outcome = spec
            .prefill(
                &mut gpu,
                &mut bundle,
                &prompt_tokens,
                &prompt_tokens,
                0,
                false,
                None,
                &no_abort,
            )
            .map_err(|e| format!("warmup prefill: {e}"))?;
        let first = match outcome {
            PrefillOutcome::Ready { first_token } => first_token,
            PrefillOutcome::Aborted => return Err("warmup prefill aborted".into()),
        };
        let _ = decode_loop(
            spec.as_mut(),
            &mut bundle,
            &mut gpu,
            first,
            prompt_tokens.len(),
            warmup,
            eos_tok,
            raw,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("warmup sync: {e:?}"))?;
    }

    // ── TIMED RUN: fresh prefill, then timed decode. ──
    spec.reset(&mut gpu);
    bundle.state.reset();
    let outcome = spec
        .prefill(
            &mut gpu,
            &mut bundle,
            &prompt_tokens,
            &prompt_tokens,
            0,
            false,
            None,
            &no_abort,
        )
        .map_err(|e| format!("prefill: {e}"))?;
    let first = match outcome {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("prefill aborted".into()),
    };
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("pre-timer sync: {e:?}"))?;

    let t0 = Instant::now();
    let (generated, windows, proposed, accepted) = decode_loop(
        spec.as_mut(),
        &mut bundle,
        &mut gpu,
        first,
        prompt_tokens.len(),
        max,
        eos_tok,
        raw,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("post-timer sync: {e:?}"))?;
    let dt = t0.elapsed().as_secs_f64();

    let n = generated.len();
    let tok_s = if dt > 0.0 { n as f64 / dt } else { 0.0 };
    // τ = avg tokens emitted per window (committed.len()), the standard accept-len.
    let tau = if windows > 0 {
        n as f64 / windows as f64
    } else {
        0.0
    };
    let accept_frac = if proposed > 0 {
        accepted as f64 / proposed as f64
    } else {
        0.0
    };

    let text = tokenizer.decode(&generated);
    println!("=== dspark_bench ===");
    println!(
        "drafter={} block={} prompt_md5={} prompt_tokens={}",
        if dspark_enabled { "DSpark" } else { "MTP" },
        block,
        prompt_md5,
        prompt_tokens.len()
    );
    println!(
        "tokens={n} time={dt:.3}s tok/s={tok_s:.2} | windows={windows} tau={tau:.3} accept={accept_frac:.3} (proposed={proposed} accepted={accepted})"
    );
    println!("--- decoded ({n} tokens) ---");
    println!("{text}");
    println!("--- token ids ---");
    println!("{generated:?}");

    spec.free(&mut gpu);
    Ok(())
}

/// Cheap stable hash of the token sequence — NOT cryptographic, just a stable
/// fingerprint so two runs can confirm byte-identical prompt tokenization.
fn md5ish(toks: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
