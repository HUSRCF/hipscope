// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF-native greedy generation (`HIPFIRE_GGUF_NATIVE=1`, dense Qwen3.5 / 3.8
//! GGUF on gfx1201): prefill each prompt, then greedy-decode `--max-tokens`
//! tokens on the ggml decode route. Prints the token ids and the decoded text
//! of every prompt and, with `--out`, writes the ids (one prompt per line) so
//! two fresh processes can be compared byte for byte.
//!
//! Usage: gguf_native_gen <model.gguf> [--prompt TEXT]... [--max-tokens N]
//!        [--chat] [--out ids.txt] [--time]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_loader::gguf_native::GgufNativeModel;
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gguf_native_gen <model.gguf> [--prompt TEXT]... [--max-tokens N] [--chat] [--out ids.txt] [--time]");
        std::process::exit(1);
    }
    let model_path = std::path::PathBuf::from(&args[1]);
    let mut prompts: Vec<String> = Vec::new();
    let mut max_tokens = 64usize;
    let mut chat = false;
    let mut out_path: Option<String> = None;
    let mut timing = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                prompts.push(args[i + 1].clone());
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().expect("--max-tokens integer");
                i += 2;
            }
            "--chat" => {
                chat = true;
                i += 1;
            }
            "--out" => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--time" => {
                timing = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    if prompts.is_empty() {
        prompts = vec![
            "The capital of France is".to_string(),
            "def fibonacci(n):".to_string(),
            "A hash map is a data structure that".to_string(),
        ];
    }

    let model = GgufNativeModel::open(&model_path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_gguf(model.gguf())
        .unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gguf_native_gen: arch={} model={}", gpu.arch, model_path.display());
    let config = model.hparams.config.clone();
    let t_load = Instant::now();
    let weights = model.load_qwen35_weights(&mut gpu).expect("load GGUF weights");
    eprintln!("weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let mut out: Option<std::fs::File> = out_path
        .as_ref()
        .map(|p| std::fs::File::create(p).expect("create --out"));
    for (pi, prompt) in prompts.iter().enumerate() {
        let prompt_tokens: Vec<u32> = if chat {
            let mut t = Vec::new();
            for piece in ["<|im_start|>", "user", "\n"] {
                t.extend(tokenizer.encode(piece));
            }
            t.extend(tokenizer.encode(prompt));
            for piece in ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"] {
                t.extend(tokenizer.encode(piece));
            }
            t
        } else {
            tokenizer.encode(prompt)
        };
        dn_state.reset(&mut gpu).expect("reset state");
        kv_cache.clear_gpu(&mut gpu).expect("clear kv");
        let t_pre = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &prompt_tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("prefill forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let pre_ms = t_pre.elapsed().as_secs_f64() * 1e3;
        let mut next = llama::argmax(&logits);
        let mut ids: Vec<u32> = vec![next];
        let mut pos = prompt_tokens.len();
        let t_dec = Instant::now();
        while ids.len() < max_tokens && next != config.eos_token && pos < kv_seq {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                next,
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("decode forward failed");
            let logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            ids.push(next);
            pos += 1;
        }
        let dec_s = t_dec.elapsed().as_secs_f64();
        let line = ids
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("PROMPT {pi}: {prompt:?} ({} tokens)", prompt_tokens.len());
        println!("IDS {pi}: {line}");
        println!("TEXT {pi}: {:?}", tokenizer.decode(&ids));
        if timing {
            println!(
                "TIME {pi}: prefill {:.1} ms ({} tok), decode {} tok in {:.2} s = {:.2} tok/s",
                pre_ms,
                prompt_tokens.len(),
                ids.len() - 1,
                dec_s,
                (ids.len() - 1) as f64 / dec_s.max(1e-9)
            );
        }
        if let Some(f) = out.as_mut() {
            writeln!(f, "{line}").expect("write ids");
        }
    }
}
