//! Phase 1 smoke test: run one forward pass through A3B (or any qwen3_5_moe
//! HFQ) and verify logits are finite. Exercises both the MoE FFN helper and
//! the new DeltaNetMoe/FullAttnMoe match arms in forward_scratch_layers.
//!
//! Usage:
//!   cargo run --release --features deltanet --example a3b_smoke_forward -- \
//!       ~/.hipfire/models/qwen3.5-35b-a3b.mq4
//!
//!   # Optional: generate N tokens greedily to probe short-term stability.
//!   HIPFIRE_SMOKE_STEPS=8 cargo run --release ...

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use engine::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: a3b_smoke_forward <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let n_steps: usize = std::env::var("HIPFIRE_SMOKE_STEPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    assert!(config.num_experts > 0, "this smoke test expects a MoE model");

    eprintln!("A3B config: dim={}, layers={}, experts={}, top_k={}, moe_inter={}, shared_inter={}",
        config.dim, config.n_layers, config.num_experts, config.num_experts_per_tok,
        config.moe_intermediate_size, config.shared_expert_intermediate_size);

    eprintln!("Loading weights ...");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Loaded {} layers.", weights.layers.len());

    let kv_seq = 256usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv cache alloc");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state alloc");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch alloc");

    // Pick a benign starting token — use the BOS-like `<|im_start|>` id if we
    // can, otherwise fall back to token 0. The exact token doesn't matter for
    // a finite-logits smoke test; we just need SOMETHING.
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let start_tok = {
        let enc = tokenizer.encode("Hello");
        enc.first().copied().unwrap_or(0)
    };
    eprintln!("Starting token: {start_tok}");

    eprintln!("\n=== forward_scratch pos=0 (cold) ===");
    let t0 = std::time::Instant::now();
    qwen35::forward_scratch(
        &mut gpu, &weights, &config, start_tok, 0,
        &mut kv_cache, &mut dn_state, &scratch,
    ).expect("forward_scratch failed");
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let elapsed = t0.elapsed();

    // ─── Correctness gates ──────────────────────────────────────────────
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &logits {
        if v.is_nan() { n_nan += 1; }
        else if v.is_infinite() { n_inf += 1; }
        else {
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
    }
    eprintln!("  logits.len = {}", logits.len());
    eprintln!("  finite range: [{:.4}, {:.4}]", min_v, max_v);
    eprintln!("  NaNs: {n_nan}  Infs: {n_inf}");
    assert!(n_nan == 0, "NaN in logits — forward path is broken");
    assert!(n_inf == 0, "Inf in logits — forward path is broken");

    // Top-5
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top-5 token ids: {:?}", &indexed[..5]);
    let argmax = indexed[0].0;
    eprintln!("  argmax = {argmax}  (elapsed: {:?})", elapsed);

    // ─── Optional multi-step decode ────────────────────────────────────
    if n_steps > 1 {
        eprintln!("\n=== decoding {} more tokens greedily ===", n_steps - 1);
        let mut next = argmax;
        let mut timings = Vec::with_capacity(n_steps.saturating_sub(1));
        for step in 1..n_steps {
            let t0 = std::time::Instant::now();
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, next, step,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch failed");
            let l = gpu.download_f32(&scratch.logits).expect("download");
            timings.push(t0.elapsed());
            let has_nan = l.iter().any(|v| v.is_nan() || v.is_infinite());
            assert!(!has_nan, "NaN/Inf at step {step}");
            next = llama::argmax(&l);
            let decoded = tokenizer.decode(&[next]);
            eprintln!("  step {step:2} -> {next:6} '{}'  ({} µs)",
                decoded.replace('\n', "\\n"), timings.last().unwrap().as_micros());
        }

        // Steady-state summary — ignore the first 2 steps (graph capture
        // and KV cache warm-up throw off early measurements).
        let settled: Vec<_> = timings.iter().skip(2).collect();
        if !settled.is_empty() {
            let sum: u128 = settled.iter().map(|d| d.as_micros()).sum();
            let avg_us = sum / settled.len() as u128;
            let tok_per_s = 1_000_000.0 / avg_us as f64;
            eprintln!("\nsteady-state decode (n={}): avg {} µs/tok = {:.1} tok/s",
                settled.len(), avg_us, tok_per_s);
        }
    }

    eprintln!("\n=== SMOKE TEST PASSED ===");
}
