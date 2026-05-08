//! eval_hipfire — KLD eval for hipfire quant variants against a BF16 reference.
//!
//! Loads a hipfire model, reads the slice (or pre-tokenized tokens), reads
//! the BF16 reference in hipfire β format (HFKLDR), runs forward inference
//! chunk-by-chunk over the matched eval tokens, computes per-token KLD via
//! a top-K-of-reference approximation, bins per-sequence, emits HFKSEQ
//! output that `kld_reduce.py` aggregates.
//!
//! Usage:
//!   eval_hipfire --model <path-to-hfq-model> \
//!                --ref   <path-to-hipfire-β-ref> \
//!                --output <path-to-output.kldseq> \
//!                [--variant <name>=auto-from-model-path] \
//!                [--arch <name>=auto-from-gpu] \
//!                [--kv-mode <mode>=asym3]
//!
//! Output: HFKSEQ format (see kldref_format.py) — per-sequence (mean, p99)
//! KLD as fp64 pairs.
//!
//! Plan: docs/plans/issue-113-quant-quality-eval.md (rev-3.2).

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    struct Args {
        model: PathBuf,
        ref_path: PathBuf,
        output: PathBuf,
        kv_mode: String,
    }
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut kv_mode = "asym3".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref"   => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4") {
                    eprintln!("--kv-mode must be one of: q8 asym2 asym3 asym4 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire --model <path> --ref <path> --output <path> [--kv-mode asym3]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let args = Args {
        model: model.expect("--model required"),
        ref_path: ref_path.expect("--ref required"),
        output: output.expect("--output required"),
        kv_mode,
    };

    // -------- eval-mode env vars (must precede Gpu::init / forward) --------
    // Per plan §"Eval-mode hipfire flags": force OFF for prompt normalize +
    // graph capture; record kv-mode in env for downstream tooling. Logged so
    // a user reading the run output sees the override explicitly.
    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
    }
    eprintln!("eval_hipfire: forced HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE={}", args.kv_mode);

    // -------- ref sha256 sanity (M1) --------
    verify_ref_sha256(&args.ref_path);

    // -------- load model --------
    let mut hfq = HfqFile::open(&args.model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("eval_hipfire: arch={} model={}", gpu.arch, args.model.display());
    // gfx12 Lloyd kernels are gated by HIPFIRE_LLOYD_GFX12 (see PR #195).
    // Set if running on gfx12; harmless on other arches.
    if gpu.arch.starts_with("gfx12") {
        unsafe { std::env::set_var("HIPFIRE_LLOYD_GFX12", "1"); }
        eprintln!("eval_hipfire: arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
    }
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    // -------- read reference (HFKLDR β) header + tokens --------
    let ref_file = File::open(&args.ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);

    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic: {magic:?}"); std::process::exit(2);
    }
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("read ref header");
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let ref_n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let top_k = u16::from_le_bytes(hdr[16..18].try_into().unwrap()) as usize;
    let _flags = u16::from_le_bytes(hdr[18..20].try_into().unwrap());
    if version != 1 {
        eprintln!("unsupported ref version {version}"); std::process::exit(2);
    }
    if ref_n_vocab != config.vocab_size {
        eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", config.vocab_size);
        std::process::exit(2);
    }
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let total_scored = scored_per_chunk * n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B"
    );

    // Read tokens (n_ctx * n_chunk u32s).
    let n_tokens = n_ctx * n_chunk;
    let mut tokens_raw = vec![0u8; n_tokens * 4];
    ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
    let tokens: Vec<u32> = tokens_raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // -------- KV cache + DeltaNet state + scratch --------
    let kv_max = n_ctx + 16;
    let mut kv_cache = match args.kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        other => panic!("unknown --kv-mode: {other}"),
    };
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    // -------- per-chunk loop --------
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut p99_kld_per_seq:  Vec<f64> = Vec::with_capacity(n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;

    for c in 0..n_chunk {
        // Re-create DeltaNet state per chunk (cheap, GPU mallocs zero buffers).
        // Avoids needing a `reset` method on DeltaNetState. KvCache positions
        // are passed explicitly via `pos` to forward_scratch — overwriting
        // from position 0 each chunk is sufficient.
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

        let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
        let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);

        for pos in 0..(n_ctx - 1) {
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, chunk_tokens[pos], pos,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward");

            // Score logit-positions [n_ctx/2, n_ctx-2], predicting tokens at
            // [n_ctx/2 + 1, n_ctx-1]. Yields exactly `n_ctx - 1 - n_ctx/2`
            // scored positions per chunk, matching what build_kld_ref wrote.
            //
            // Off-by-one was previously `n_ctx/2 + 1`, which under-read 1
            // ref block per chunk and shifted every comparison by 1 logit
            // position. Caught by the consolidated review's C1 finding.
            let scoring_start = n_ctx / 2;
            if pos < scoring_start {
                continue;
            }

            // Read corresponding reference block.
            ref_in.read_exact(&mut block_buf).expect("read ref block");

            // Parse β block: u32 indices[K] | f32 log_probs[K] | f32 residual | f32 pad
            let mut top_indices: Vec<u32> = Vec::with_capacity(top_k);
            let mut top_log_probs: Vec<f32> = Vec::with_capacity(top_k);
            for j in 0..top_k {
                top_indices.push(u32::from_le_bytes(block_buf[j * 4..j * 4 + 4].try_into().unwrap()));
            }
            let lp_off = top_k * 4;
            for j in 0..top_k {
                top_log_probs.push(f32::from_le_bytes(block_buf[lp_off + j * 4..lp_off + j * 4 + 4].try_into().unwrap()));
            }
            let resid_off = top_k * 8;
            let sum_p_residual = f32::from_le_bytes(block_buf[resid_off..resid_off + 4].try_into().unwrap());
            // pad: ignored

            // Download candidate logits at this position.
            let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");

            // KLD via top-K-of-reference approximation (fp64 throughout).
            // Compute candidate's log-Z = log Σ exp(logit_i)
            let mut max_logit = f32::NEG_INFINITY;
            for &v in cand_logits.iter() { if v > max_logit { max_logit = v; } }
            let mut sum_exp = 0.0f64;
            for &v in cand_logits.iter() {
                sum_exp += ((v - max_logit) as f64).exp();
            }
            let log_z = (max_logit as f64) + sum_exp.ln();

            // KLD = top-K-of-reference contribution + residual cross-term.
            //
            // Top-K contribution:
            //   Σ_{i in top_K_P_ref} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
            //
            // Residual cross-term (per rev-3.3):
            //   sum_p_residual_ref * (log sum_p_residual_ref - log sum_p_residual_cand)
            // where sum_p_residual_cand = 1 - Σ_{i in ref_top_K} P_cand(i).
            // Assumes the ref-tail and cand-tail miss similarly. Reduces bias on
            // flat-distribution tokens (~1% of all tokens have residual >17% per
            // Step 1.6's empirical p99). See plan §"GGUF anchor architecture
            // (rev-3.3)" — residual-term enhancement.
            let mut kld_token = 0.0f64;
            let mut sum_p_cand_at_ref_top = 0.0f64;
            for j in 0..top_k {
                let ref_idx = top_indices[j] as usize;
                if ref_idx >= cand_logits.len() {
                    eprintln!("warn: ref idx {ref_idx} >= n_vocab {}", cand_logits.len());
                    continue;
                }
                let log_p_ref = top_log_probs[j] as f64;
                let log_p_cand = (cand_logits[ref_idx] as f64) - log_z;
                let p_ref = log_p_ref.exp();
                let p_cand = log_p_cand.exp();
                kld_token += p_ref * (log_p_ref - log_p_cand);
                sum_p_cand_at_ref_top += p_cand;
            }
            // Residual cross-term. Only meaningful when both residuals are > 0.
            let sum_p_residual_ref = sum_p_residual as f64;
            let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
            if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
                kld_token += sum_p_residual_ref
                    * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
            }
            // Clamp small-negative roundoff to zero (KLD is ≥ 0 in theory).
            if kld_token < 0.0 && kld_token > -1e-6 { kld_token = 0.0; }

            chunk_klds.push(kld_token);
            total_scored_done += 1;

            if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = total_scored_done as f64 / elapsed.max(1e-9);
                eprint!(
                    "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                    c + 1, n_chunk, total_scored_done, total_scored, pct, rate
                );
            }
        }

        // Per-chunk aggregates
        if chunk_klds.is_empty() {
            mean_kld_per_seq.push(0.0);
            p99_kld_per_seq.push(0.0);
            continue;
        }
        let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
        let mut sorted = chunk_klds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p99 = sorted[p99_idx];
        mean_kld_per_seq.push(mean);
        p99_kld_per_seq.push(p99);
    }
    eprintln!();
    eprintln!(
        "eval_hipfire: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s)",
        t0.elapsed().as_secs_f64(),
        total_scored_done as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );

    // -------- write HFKSEQ output --------
    let out_file = File::create(&args.output).expect("create output");
    let mut out = BufWriter::new(out_file);
    out.write_all(b"HFKSEQ\0\0").unwrap();
    out.write_all(&1u32.to_le_bytes()).unwrap();             // version
    out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap(); // n_chunk
    out.write_all(&0u32.to_le_bytes()).unwrap();             // reserved
    for (m, p) in mean_kld_per_seq.iter().zip(p99_kld_per_seq.iter()) {
        out.write_all(&m.to_le_bytes()).unwrap();
        out.write_all(&p.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();

    let overall_mean: f64 = mean_kld_per_seq.iter().copied().sum::<f64>() / mean_kld_per_seq.len() as f64;
    eprintln!("eval_hipfire: slice-mean KLD = {:.6} ({}-chunk mean of per-chunk means)", overall_mean, n_chunk);
    eprintln!("eval_hipfire: wrote {}", args.output.display());
}

/// Verify the reference file's sha256 against
/// `<repo>/benchmarks/quality-baselines/harness/manifest.json`.
///
/// Layout assumption: ref lives at `.../refs/<name>.kldref.bin`,
/// manifest at `.../harness/manifest.json` (sibling to refs/).
///
/// If the manifest entry is absent, warn and continue (developer
/// pre-upload state). If sha256 disagrees, abort.
#[cfg(feature = "deltanet")]
fn verify_ref_sha256(ref_path: &std::path::Path) {
    use std::process::Command;
    let manifest_path = match ref_path.parent().and_then(|p| p.parent()) {
        Some(p) => p.join("harness").join("manifest.json"),
        None => {
            eprintln!("warning: cannot locate harness/manifest.json; skipping ref sha256 check");
            return;
        }
    };
    if !manifest_path.exists() {
        eprintln!("warning: {} missing; skipping ref sha256 check", manifest_path.display());
        return;
    }
    let manifest_file = std::fs::File::open(&manifest_path)
        .expect("open manifest.json");
    let manifest: serde_json::Value = serde_json::from_reader(manifest_file)
        .expect("parse manifest.json");
    let ref_name = ref_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let expected = manifest
        .get("references")
        .and_then(|r| r.get(ref_name))
        .and_then(|r| r.get("sha256"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let expected = match expected {
        Some(s) => s,
        None => {
            eprintln!("warning: no manifest entry / sha256 for {ref_name}; skipping check");
            return;
        }
    };
    eprintln!("eval_hipfire: computing sha256 of {} ...", ref_path.display());
    let out = Command::new("sha256sum")
        .arg(ref_path)
        .output()
        .expect("invoke sha256sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty sha256sum output");
    if actual != expected {
        eprintln!("ERROR: ref sha256 mismatch for {}", ref_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("eval_hipfire: verified ref sha256 = {actual}");
}
