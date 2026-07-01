// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen3_dspark_parity: GPU-vs-CPU numeric parity for the Qwen3-8B DSpark
//! drafter forward (Task 9 gate).
//!
//! ## What is validated
//!
//! Checks (a)–(d) against the CPU reference produced by
//! `/home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py`:
//!
//!   (a) `main_x` = `hidden_norm(fc(main_hidden))` — cosine ≥ 0.999
//!   (b) `x_head_out` = post-final-norm block hidden `[block, dim]` — cosine ≥ 0.999
//!   (c) markov greedy token sequence — token-identical to CPU
//!   (d) confidence logits (pre-sigmoid) — cosine ≥ 0.999
//!
//! ## Inputs
//!
//! Fixed synthetic `main_hidden[5*4096]`:
//!   `main_hidden[i] = sin(i * 0.013) * 0.5`   (same as deepseek4 parity harness)
//! Fixed `seed = 12345`, `seed_pos = 42`, `block = 7`.
//!
//! The CPU reference files are expected in
//!   `/home/bjoern/dspark-work/qwen3_parity_refs/`
//! (raw binary: *.f32bin = F32 LE, *.i32bin = I32 LE).
//!
//! ## CPU reference
//!
//! Run BEFORE this binary:
//! ```
//! cd /home/bjoern/hipfire
//! nix develop --command bash -c '
//!   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(find /nix/store -maxdepth 1 -path "*gcc-15*-lib" | head -1)/lib:/nix/store/6v5hbaxvndmaf21rfyryxpn1xjkljrid-zlib-1.3.2/lib"
//!   export PYTHONPATH="/home/bjoern/dspark-work/DeepSpec:$PYTHONPATH"
//!   /home/bjoern/hipfire/.venv/bin/python3 /home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py
//! '
//! ```
//!
//! ## RoPE fix (Task 9)
//!
//! The parity gate confirmed that block positions must follow `create_position_ids`
//! exactly: block slot i gets position `seed_pos + i` (0-indexed), NOT `seed_pos+1+i`.
//! Both Q and K block positions are `[seed_pos, ..., seed_pos+block-1]`, matching
//! `apply_rotary_pos_emb`'s `cos[..., -q_len:, :]` slice behaviour.
//!
//! ## Usage
//! ```
//! source scripts/gpu-lock.sh && gpu_acquire dspark-qwen3
//! cargo build --release -p hipfire-arch-llama --example qwen3_dspark_parity
//! ./target/release/examples/qwen3_dspark_parity [path-to-qwen3-8b-dspark.hfq] [refs-dir]
//! gpu_release
//! ```

use hipfire_arch_llama::dspark_body::{
    dspark_qwen3_block_forward, load_qwen3_dspark, Qwen3DsparkScratch,
};
use hipfire_runtime::dspark_core::{main_proj_ingest, noise_block_ids, run_heads};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

/// Fixed test parameters (must match qwen3_dspark_cpu_ref.py).
const SEED_TOK: u32 = 12345;
const SEED_POS: usize = 42;
const BLOCK: usize = 7;
const N_TARGETS: usize = 5;

fn main() -> Result<(), String> {
    let hfq_path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3-8b-dspark.hfq",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let refs_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/bjoern/dspark-work/qwen3_parity_refs".into());

    eprintln!("opening {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path)).map_err(|e| format!("open: {e:?}"))?;
    hfq.drop_mmap();

    eprintln!("initialising GPU");
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU ready (arch={})", gpu.arch_caps.arch());

    // ── Load sidecar ──────────────────────────────────────────────────────────
    let (dspark_weights, assets) =
        load_qwen3_dspark(&hfq, &mut gpu)?.ok_or("load_qwen3_dspark: no dspark_* in metadata")?;

    let cfg = &dspark_weights.cfg;
    let dim = assets.config.dim;
    let vocab = assets.weights.output.m;
    eprintln!(
        "loaded: dim={dim} vocab={vocab} block_size={} markov_rank={} enable_confidence={}",
        cfg.block_size, cfg.markov_rank, cfg.enable_confidence
    );

    // ── Load CPU reference files ───────────────────────────────────────────────
    let refs = Path::new(&refs_dir);
    let cpu_main_hidden = load_f32bin(refs.join("main_hidden.f32bin"))?;
    let cpu_main_x = load_f32bin(refs.join("main_x.f32bin"))?;
    let cpu_x_head = load_f32bin(refs.join("x_head_out.f32bin"))?;
    let cpu_markov_i32 = load_i32bin(refs.join("markov_tokens.i32bin"))?;
    let cpu_confidence = load_f32bin(refs.join("confidence_logits.f32bin"))?;

    let cpu_markov: Vec<u32> = cpu_markov_i32.iter().map(|&t| t as u32).collect();
    eprintln!(
        "refs loaded: main_hidden={} main_x={} x_head={} markov={} conf={}",
        cpu_main_hidden.len(),
        cpu_main_x.len(),
        cpu_x_head.len(),
        cpu_markov.len(),
        cpu_confidence.len()
    );

    // Verify expected sizes.
    let concat_w = N_TARGETS * dim;
    if cpu_main_hidden.len() != concat_w {
        return Err(format!(
            "main_hidden: expected {concat_w} got {}",
            cpu_main_hidden.len()
        ));
    }
    if cpu_main_x.len() != dim {
        return Err(format!("main_x: expected {dim} got {}", cpu_main_x.len()));
    }
    if cpu_x_head.len() != BLOCK * dim {
        return Err(format!(
            "x_head: expected {} got {}",
            BLOCK * dim,
            cpu_x_head.len()
        ));
    }
    if cpu_markov.len() != BLOCK {
        return Err(format!("markov: expected {BLOCK} got {}", cpu_markov.len()));
    }
    if cpu_confidence.len() != BLOCK {
        return Err(format!(
            "confidence: expected {BLOCK} got {}",
            cpu_confidence.len()
        ));
    }

    // Upload fixed main_hidden to GPU.
    let main_hidden_dev = upload_f32(&mut gpu, &cpu_main_hidden)?;

    // ── Check (a): main_proj_ingest = hidden_norm(fc(main_hidden)) ────────────
    let main_x_dev = gpu
        .alloc_tensor(&[dim], DType::F32)
        .map_err(|e| format!("alloc main_x: {e:?}"))?;
    main_proj_ingest(&mut gpu, &dspark_weights, &main_hidden_dev, &main_x_dev)?;
    let gpu_main_x = gpu
        .download_f32(&main_x_dev)
        .map_err(|e| format!("d2h main_x: {e:?}"))?;
    let check_a = parity_stats("(a) main_x", &gpu_main_x, &cpu_main_x, 0.999, None);

    // ── Check (b): x_head_out from dspark_qwen3_block_forward ─────────────────
    // Allocate as [BLOCK, dim] (2-D) so run_heads can infer hidden = dim via shape.last().
    let scratch = Qwen3DsparkScratch::new(&mut gpu, &assets.config, BLOCK)
        .map_err(|e| format!("Qwen3DsparkScratch::new: {e}"))?;
    let x_head_dev = gpu
        .alloc_tensor(&[BLOCK, dim], DType::F32)
        .map_err(|e| format!("alloc x_head: {e:?}"))?;
    let block_ids = noise_block_ids(cfg, SEED_TOK);
    dspark_qwen3_block_forward(
        &mut gpu,
        &assets.weights,
        &assets.config,
        &main_x_dev,
        &block_ids,
        SEED_POS,
        BLOCK,
        &scratch,
        &x_head_dev,
    )?;
    let gpu_x_head = gpu
        .download_f32(&x_head_dev)
        .map_err(|e| format!("d2h x_head: {e:?}"))?;
    let check_b = parity_stats("(b) x_head_out", &gpu_x_head, &cpu_x_head, 0.999, None);

    // ── Check (c) + (d): run_heads → markov tokens + confidence ───────────────
    let stage_norm_ref = &assets.weights.output_norm;

    // The lm_head in LlamaWeights is a WeightTensor whose buf.dtype=Raw (upload_raw
    // always sets Raw), but the actual weight data is F16 (WeightTensor.gpu_dtype=F16).
    // run_heads dispatches on GpuTensor.dtype; we shallow_clone the buf and override
    // dtype=F16 so the correct WMMA GEMM kernel is selected.
    // NOTE: Task 10's speculator builder will supply a properly-typed GpuTensor
    // directly (mirroring the deepseek4 path); the parity harness does this manually.
    let mut lm_head_f16 = assets.weights.output.buf.shallow_clone();
    lm_head_f16.dtype = DType::F16;
    lm_head_f16.shape = vec![vocab];

    let draft = run_heads(
        &mut gpu,
        &dspark_weights,
        stage_norm_ref,
        &lm_head_f16,
        &x_head_dev,
        SEED_TOK, // prev_token before the window
        BLOCK,
        vocab,
    )?;
    let gpu_markov = &draft.tokens;
    let gpu_confidence = &draft.confidence;

    // Token-identical check (c).
    let tokens_match = gpu_markov == &cpu_markov;
    let first_mismatch = gpu_markov
        .iter()
        .zip(cpu_markov.iter())
        .enumerate()
        .find(|(_, (g, c))| g != c);

    // Cosine check (d) on confidence logits.
    let check_d = parity_stats(
        "(d) confidence logits",
        gpu_confidence,
        &cpu_confidence,
        0.999,
        None,
    );

    // ── Print report ──────────────────────────────────────────────────────────
    println!("\nQwen3-8B DSpark GPU-vs-CPU parity (block={BLOCK} seed_pos={SEED_POS} seed_tok={SEED_TOK}):");
    println!(
        "  {:<32} {:>8} {:>12} {:>10}  {}",
        "check", "n", "max_abs", "cosine", "verdict"
    );
    for c in [&check_a, &check_b, &check_d] {
        println!(
            "  {:<32} {:>8} {:>12.3e} {:>10.6}  {}",
            c.name,
            c.n,
            c.max_abs,
            c.cosine,
            if c.pass { "PASS" } else { "FAIL" }
        );
    }
    // Token check.
    let tok_verdict = if tokens_match { "PASS" } else { "FAIL" };
    println!(
        "  {:<32} {:>8}                           {tok_verdict}",
        "(c) markov tokens", BLOCK
    );
    if !tokens_match {
        if let Some((i, (g, c))) = first_mismatch {
            println!("    first mismatch at slot {i}: GPU={g} CPU={c}");
        }
        println!("  GPU tokens: {gpu_markov:?}");
        println!("  CPU tokens: {cpu_markov:?}");
    } else {
        println!("  token sequence: {:?}", gpu_markov);
    }

    // Free.
    let _ = gpu.free_tensor(main_hidden_dev);
    let _ = gpu.free_tensor(main_x_dev);
    scratch.free_gpu(&mut gpu);
    let _ = gpu.free_tensor(x_head_dev);

    let all_pass = check_a.pass && check_b.pass && tokens_match && check_d.pass;
    if all_pass {
        println!("\nPARITY PASS — Qwen3 DSpark GPU forward matches CPU reference");
        Ok(())
    } else {
        Err("PARITY FAIL — see above for first diverging check".into())
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

struct ParityCheck {
    name: &'static str,
    n: usize,
    max_abs: f32,
    cosine: f32,
    pass: bool,
}

fn parity_stats(
    name: &'static str,
    gpu: &[f32],
    cpu: &[f32],
    cosine_threshold: f32,
    max_abs_threshold: Option<f32>,
) -> ParityCheck {
    let n = gpu.len().min(cpu.len());
    let (mut dot, mut ng, mut nc, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for i in 0..n {
        let (g, c) = (gpu[i], cpu[i]);
        max_abs = max_abs.max((g - c).abs());
        dot += g as f64 * c as f64;
        ng += g as f64 * g as f64;
        nc += c as f64 * c as f64;
    }
    let cosine = if ng > 0.0 && nc > 0.0 {
        (dot / (ng.sqrt() * nc.sqrt())) as f32
    } else {
        0.0
    };
    let pass_cosine = cosine >= cosine_threshold;
    let pass_abs = max_abs_threshold.map(|t| max_abs <= t).unwrap_or(true);
    ParityCheck {
        name,
        n,
        max_abs,
        cosine,
        pass: pass_cosine && pass_abs,
    }
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[v.len()], DType::F32)
        .map_err(|e| format!("alloc: {e:?}"))?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| format!("htod: {e:?}"))?;
    Ok(t)
}

fn load_f32bin(path: impl AsRef<Path>) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: file size {} not divisible by 4",
            path.as_ref().display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut v = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
    }
    Ok(v)
}

fn load_i32bin(path: impl AsRef<Path>) -> Result<Vec<i32>, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: file size {} not divisible by 4",
            path.as_ref().display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut v = vec![0i32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
    }
    Ok(v)
}
