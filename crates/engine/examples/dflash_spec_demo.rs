//! dflash_spec_demo: end-to-end speculative decoding demo.
//!
//! Loads a Qwen3.5 target (.hfq) + a matching DFlash draft (.hfq, arch=20),
//! tokenizes a prompt, seeds target_hidden, and runs
//! `spec_step_dflash` in a loop until N tokens committed or an EOS is hit.
//! Prints tokens as they commit, plus final stats (accept rate, tok/s).
//!
//! Usage:
//!   dflash_spec_demo --target <target.hfq> --draft <draft.hfq> \
//!                    --prompt "Hello" [--max 64] [--ctx 512]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use engine::hfq::HfqFile;
    use engine::speculative::{
        self, DeltaNetSnapshot, HiddenStateRingBuffer, ModelSlot, ModelSlotConfig, SpecStats,
    };
    use engine::tokenizer::Tokenizer;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: dflash_spec_demo --target <target.hfq> --draft <draft.hfq> \\\n                             --prompt \"Hello\" [--max 64] [--ctx 512]"
        );
        std::process::exit(1);
    }
    let mut target_path: Option<String> = None;
    let mut draft_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut ctx_capacity: usize = 512;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                target_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--draft" => {
                draft_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--prompt" => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--max" => {
                max_tokens = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                ctx_capacity = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let target_path = target_path.expect("--target required");
    let draft_path = draft_path.expect("--draft required");
    let prompt = prompt.expect("--prompt required");

    eprintln!("=== dflash_spec_demo ===");
    eprintln!("target: {target_path}");
    eprintln!("draft:  {draft_path}");

    // ── Init GPU ──────────────────────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    // ── Load draft ────────────────────────────────────────────────────
    let draft_hfq = HfqFile::open(Path::new(&draft_path)).expect("open draft");
    let draft_cfg = DflashConfig::from_hfq(&draft_hfq).expect("parse DflashConfig");
    eprintln!(
        "draft: layers={} hidden={} heads={} kv_heads={} block={} target_layers={:?}",
        draft_cfg.n_layers,
        draft_cfg.hidden,
        draft_cfg.n_heads,
        draft_cfg.n_kv_heads,
        draft_cfg.block_size,
        draft_cfg.target_layer_ids,
    );
    let t0 = Instant::now();
    let draft_weights = DflashWeights::load(&mut gpu, &draft_hfq, &draft_cfg).expect("load draft");
    eprintln!("draft loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let mut draft_scratch = DflashScratch::new(&mut gpu, &draft_cfg, draft_cfg.block_size, ctx_capacity)
        .expect("alloc draft scratch");

    // ── Load target ───────────────────────────────────────────────────
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = ctx_capacity + draft_cfg.block_size + 16;
    let t1 = Instant::now();
    let mut target =
        ModelSlot::load(&mut gpu, Path::new(&target_path), "target", slot_cfg).expect("load target");
    eprintln!("target loaded in {:.2}s", t1.elapsed().as_secs_f64());

    // ── Check vocab compatibility ─────────────────────────────────────
    assert_eq!(
        target.config.vocab_size, draft_cfg.vocab_size,
        "target vocab ({}) != draft vocab ({})",
        target.config.vocab_size, draft_cfg.vocab_size
    );

    let tokenizer: Tokenizer = target.load_tokenizer().expect("target tokenizer");
    let prompt_tokens = tokenizer.encode(&prompt);
    eprintln!("prompt: {:?}", prompt);
    eprintln!("prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // ── Hidden ring buffer + snapshot + target_hidden_host ────────────
    let mut hidden_rb = HiddenStateRingBuffer::new(
        &mut gpu,
        target.config.n_layers,
        draft_cfg.num_extract(),
        draft_cfg.hidden,
        ctx_capacity + draft_cfg.block_size,
    )
    .expect("alloc hidden_rb");

    let mut target_snap = DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state).expect("snap");
    let mut target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_cfg.num_extract() * draft_cfg.hidden);

    // ── Prefill: seed target_hidden via per-token forward_with_hidden ──
    eprintln!("seeding target_hidden from prompt ({} tokens)...", prompt_tokens.len());
    let t2 = Instant::now();
    speculative::seed_target_hidden_from_prompt(
        &mut gpu,
        &mut target,
        &mut hidden_rb,
        &mut target_hidden_host,
        &prompt_tokens,
    )
    .expect("seed target hidden");
    eprintln!("prefill (per-token) in {:.2}s", t2.elapsed().as_secs_f64());

    // ── Initial seed_token: target's greedy pick after prefill ───────
    // Target state is at position `prompt_len` after seed_target_hidden_from_prompt.
    // Its scratch.logits at this point corresponds to the LAST prompt token's output —
    // i.e., the prediction for position prompt_len. Argmax = first emitted token.
    let first_logits = gpu.download_f32(&target.scratch.logits).expect("download logits");
    let first_token = first_logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0;

    // ── Decode loop ───────────────────────────────────────────────────
    let mut emitted: Vec<u32> = vec![first_token];
    let mut position: usize = prompt_tokens.len();
    let mut seed_token: u32 = first_token;
    let mut stats = SpecStats::new(draft_cfg.block_size);
    let eos_id: u32 = tokenizer.eos_id;

    eprintln!("decoding (max {max_tokens} tokens, block_size {})...", draft_cfg.block_size);

    let t_decode = Instant::now();
    while emitted.len() < max_tokens {
        if position + draft_cfg.block_size >= ctx_capacity {
            eprintln!("hit ctx_capacity {}; stopping", ctx_capacity);
            break;
        }
        let step = speculative::spec_step_dflash(
            &mut gpu,
            &mut target,
            &draft_weights,
            &draft_cfg,
            &mut draft_scratch,
            &mut hidden_rb,
            &mut target_hidden_host,
            &mut target_snap,
            position,
            seed_token,
        )
        .expect("spec step");
        stats.record(&step);

        // `step.committed[0]` is the seed_token (already emitted). Emit [1..].
        for (&tok, _) in step.committed.iter().skip(1).zip(0..) {
            emitted.push(tok);
        }

        // Advance position + pick next seed (= bonus_token).
        position += step.accepted + 1;
        seed_token = step.bonus_token;

        // Stop on EOS.
        if step.committed.iter().skip(1).any(|&t| t == eos_id) {
            eprintln!("eos");
            break;
        }
    }
    let elapsed = t_decode.elapsed().as_secs_f64();
    let tok_s = emitted.len() as f64 / elapsed;

    // ── Report ────────────────────────────────────────────────────────
    let text = tokenizer.decode(&emitted);
    eprintln!("--- OUTPUT ---");
    println!("{text}");
    eprintln!("--------------");
    eprintln!(
        "emitted: {} tokens in {:.2}s  ({:.2} tok/s)",
        emitted.len(),
        elapsed,
        tok_s
    );
    eprintln!(
        "cycles: {}  committed: {}  accepted: {}  τ={:.3}  mean_committed={:.3}",
        stats.cycles,
        stats.committed_tokens,
        stats.accepted_tokens,
        stats.tau(),
        stats.mean_committed(),
    );
    let accept_rate = if stats.cycles > 0 {
        stats.accepted_tokens as f32 / (stats.cycles * (draft_cfg.block_size - 1)) as f32
    } else {
        0.0
    };
    eprintln!("accept_rate (accepted / (cycles × (B-1))): {accept_rate:.3}");
    eprintln!(
        "histogram: {:?}",
        stats.acceptance_hist.iter().enumerate().collect::<Vec<_>>()
    );
}
