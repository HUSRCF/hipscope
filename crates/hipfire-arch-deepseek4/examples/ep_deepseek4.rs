// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek-V4-Flash EP (expert-parallel) greedy decode across N GPUs — Ship 6
//! substrate-EP. The 81 GB mq2-lloyd tier does NOT fit one 32 GB card, so this
//! shards the 256 routed experts/layer across `--tp` ranks (shard-aware load:
//! each rank uploads only its owned experts; non-owned → zeroed dummy) and runs
//! the lowered decode through the EP executor: MLA attention replicated, the
//! SHARED expert replicated in ffn_out, only the ROUTED combine all-reduce-EP'd,
//! and `hc_ffn_mix` deferred past the all-reduce.
//!
//! Run (hiptrx, 4× gfx1201):
//!   HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release \
//!       -p hipfire-arch-deepseek4 --example ep_deepseek4 -- \
//!       --model ~/.hipfire/models/deepseek-v4-flash.mq2lloyd --tp 4 --max 48 \
//!       --prompt "The capital of France is"

fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    use hipfire_arch_deepseek4::forward;
    use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::model_source::ModelSource;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::safetensors_source::SafetensorsSource;
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::{DType, GpuTensor};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut overlay: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut prompt_file: Option<PathBuf> = None;
    let mut niah_fixture: Option<PathBuf> = None;
    let mut niah_expected: Vec<String> = Vec::new();
    let mut niah_min_recovered: usize = 0;
    let mut explicit_token_ids: Option<Vec<u32>> = None;
    let mut max: usize = 48;
    let mut warmup: usize = 2;
    let mut tp: usize = 4;
    let mut prefill_tokens: Option<usize> = None;
    let mut batched_prefill = false;
    let mut prefill_batch: usize = 256;
    let mut prefill_logits_out: Option<PathBuf> = None;
    let mut score_out: Option<PathBuf> = None;
    let mut gen_ids_out: Option<PathBuf> = None;
    let mut trace_next: Vec<usize> = Vec::new();
    let mut moe_probe_out: Option<PathBuf> = None;
    let mut no_bos = false;
    let mut chat = false;
    let mut second_prompt: Option<String> = None;
    let mut second_max: usize = 64;
    let mut mtp = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--overlay" => {
                overlay = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--prompt-file" => {
                prompt_file = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--niah-fixture" => {
                niah_fixture = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--token-ids" => {
                let ids = argv[i + 1]
                    .split(',')
                    .map(|value| value.parse().expect("--token-ids"))
                    .collect::<Vec<_>>();
                assert!(!ids.is_empty(), "--token-ids requires at least one ID");
                explicit_token_ids = Some(ids);
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--warmup" => {
                warmup = argv[i + 1].parse().expect("--warmup");
                i += 2;
            }
            "--tp" => {
                tp = argv[i + 1].parse().expect("--tp");
                i += 2;
            }
            "--prefill-tokens" => {
                prefill_tokens = Some(argv[i + 1].parse().expect("--prefill-tokens"));
                i += 2;
            }
            "--batched-prefill" => {
                batched_prefill = true;
                i += 1;
            }
            "--prefill-batch" => {
                prefill_batch = argv[i + 1].parse().expect("--prefill-batch");
                i += 2;
            }
            "--prefill-logits-out" => {
                prefill_logits_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--score-out" => {
                score_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--gen-ids-out" => {
                gen_ids_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--trace-next" => {
                trace_next = argv[i + 1]
                    .split(',')
                    .map(|value| value.parse().expect("--trace-next"))
                    .collect();
                i += 2;
            }
            "--moe-probe-out" => {
                moe_probe_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--no-bos" => {
                no_bos = true;
                i += 1;
            }
            "--chat" => {
                chat = true;
                i += 1;
            }
            "--second-prompt" => {
                second_prompt = Some(argv[i + 1].clone());
                i += 2;
            }
            "--second-max" => {
                second_max = argv[i + 1].parse().expect("--second-max");
                i += 2;
            }
            "--mtp" => {
                mtp = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    if let Some(path) = prompt_file.as_ref() {
        prompt = std::fs::read_to_string(&path).expect("read --prompt-file");
    }
    if let Some(path) = niah_fixture.as_ref() {
        assert!(
            prompt_file.is_none(),
            "do not combine --niah-fixture and --prompt-file"
        );
        let raw = std::fs::read_to_string(path).expect("read --niah-fixture");
        let record = raw.lines().next().expect("empty --niah-fixture");
        let value: serde_json::Value =
            serde_json::from_str(record).expect("parse --niah-fixture JSONL record");
        let filler = value["filler_text"]
            .as_str()
            .expect("--niah-fixture filler_text");
        let question = value["question"].as_str().expect("--niah-fixture question");
        prompt = format!("{filler}\n\n{question}");
        niah_expected = value["expected_answer_substrings"]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .expect("--niah-fixture expected string")
                            .to_string()
                    })
                    .collect()
            })
            .or_else(|| {
                value["expected_answer_substring"]
                    .as_str()
                    .map(|value| vec![value.to_string()])
            })
            .expect("--niah-fixture expected answer");
        niah_min_recovered = value["min_recovered"]
            .as_u64()
            .map(|value| value as usize)
            .unwrap_or(niah_expected.len());
    }
    let model = model.expect("--model required");
    assert!(
        !(chat && no_bos),
        "--chat requires the model BOS; do not combine it with --no-bos"
    );
    assert!(
        !(score_out.is_some() && batched_prefill),
        "--score-out requires sequential prefill; omit --batched-prefill"
    );
    assert!(
        !(moe_probe_out.is_some() && batched_prefill),
        "--moe-probe-out requires sequential prefill; omit --batched-prefill"
    );
    assert!(
        explicit_token_ids.is_none() || (!chat && prompt_file.is_none()),
        "--token-ids bypasses prompt construction; do not combine it with --chat or --prompt-file"
    );
    assert!(
        second_prompt.is_none() || (chat && batched_prefill),
        "--second-prompt requires --chat --batched-prefill"
    );

    // ── config + tokenizer (per-rank loads reopen the file) ─────────────────
    let source_is_dir = model.is_dir();
    let attach_overlay = |hfq: &mut HfqFile| {
        if let Some(path) = overlay.as_deref() {
            let control = HfqFile::open_at_offset(path, 0).expect("open --overlay");
            hfq.attach_overlay(control).expect("attach --overlay");
        }
    };
    let (cfg, tok) = if source_is_dir {
        assert!(
            overlay.is_none(),
            "--overlay is only supported for HFQ input"
        );
        let source = SafetensorsSource::open(&model).expect("open safetensors model");
        let cfg = hipfire_arch_deepseek4::config_from_safetensors(&source)
            .expect("config_from_safetensors");
        let tok_path = source
            .tokenizer_json_path()
            .expect("tokenizer.json in safetensors directory");
        let tok = Tokenizer::from_tokenizer_json(&tok_path)
            .expect("tokenizer parse")
            .expect("tokenizer load");
        (cfg, tok)
    } else {
        let mut hfq0 = HfqFile::open(&model).expect("open model");
        attach_overlay(&mut hfq0);
        let cfg = DeepseekV4::config_from_hfq(&hfq0).expect("config");
        let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
        (cfg, tok)
    };
    let n_exp = cfg.n_routed_experts;
    eprintln!(
        "deepseek4 EP: tp={tp} hidden={} layers={} hash_layers={} experts={}/{} vocab={} route_scale={} swiglu_limit={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        n_exp,
        cfg.num_experts_per_tok,
        cfg.vocab_size,
        cfg.routed_scaling_factor,
        cfg.swiglu_limit,
    );

    // Special tokens (DeepSeek `<｜...｜>` markers live in the tokenizer table).
    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tok.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok = lookup_id("<｜User｜>");
    let asst_tok = lookup_id("<｜Assistant｜>");
    let eos_tok = lookup_id("<｜end▁of▁sentence｜>").unwrap_or(tok.eos_id);
    eprintln!("  bos={bos_tok:?} user={user_tok:?} assistant={asst_tok:?} eos={eos_tok}");
    if chat {
        assert!(
            bos_tok.is_some() && user_tok.is_some() && asst_tok.is_some(),
            "--chat requires BOS/User/Assistant special tokens in model metadata"
        );
    }
    // ── bring up N ranks ────────────────────────────────────────────────────
    let mut gpus = Gpus::init_tp(tp, cfg.num_hidden_layers).expect("init_tp");
    let n = gpus.devices.len();
    assert_eq!(
        n, tp,
        "init_tp gave {n} devices (check HIP_VISIBLE_DEVICES)"
    );
    for (r, d) in gpus.devices.iter().enumerate() {
        eprintln!("  rank {r}: device_id={} arch={}", d.device_id, d.arch);
    }

    // ── shard-aware replicated load (each rank uploads only its owned experts) ─
    let expert_assign = if source_is_dir {
        ExpertAssign::Contiguous
    } else {
        ExpertAssign::Stride
    };
    let shard =
        ShardConfig::new(tp, /*tp_kv_replicate=*/ true, n_exp, expert_assign).expect("ShardConfig");
    let mut weights_per_rank = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        let t = std::time::Instant::now();
        let w = if source_is_dir {
            let source = SafetensorsSource::open(&model).expect("reopen safetensors model");
            DeepseekV4::load_weights_from_safetensors_sharded(
                &source,
                &cfg,
                &mut gpus.devices[r],
                &shard,
                r,
            )
            .expect("safetensors shard-aware load")
        } else {
            let mut hfq = HfqFile::open(&model).expect("reopen model");
            attach_overlay(&mut hfq);
            DeepseekV4::load_weights_sharded(&mut hfq, &cfg, &mut gpus.devices[r], &shard, r)
                .expect("HFQ shard-aware load")
        };
        eprintln!(
            "  [rank {r}] loaded owned shard in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        weights_per_rank.push(w);
    }
    eprintln!("  all ranks loaded (expert assignment: {expert_assign:?})");

    // ── per-rank state + routed partials ([hidden] = ffn_out width) ──────────
    let mut prompt_ids: Vec<u32> = if let Some(ids) = explicit_token_ids.as_ref() {
        ids.clone()
    } else if chat {
        let mut ids = Vec::new();
        ids.push(bos_tok.unwrap());
        ids.push(user_tok.unwrap());
        ids.extend(tok.encode(&prompt));
        ids.push(asst_tok.unwrap());
        // DeepSeek V4 non-thinking template. The model requires this marker
        // immediately after Assistant to skip the reasoning block; omitting it
        // leaves the prompt off-distribution and commonly produces attractors.
        ids.extend(tok.encode("</think>"));
        ids
    } else {
        let mut ids = Vec::new();
        if !no_bos {
            if let Some(b) = bos_tok {
                ids.push(b);
            }
        }
        ids.extend(tok.encode(&prompt));
        ids
    };
    assert!(!prompt_ids.is_empty(), "input token sequence is empty");
    if explicit_token_ids.is_some() {
        eprintln!("explicit token IDs (no BOS/chat/tokenizer additions): {prompt_ids:?}");
    }
    if let Some(target) = prefill_tokens {
        assert!(target > 0, "--prefill-tokens must be greater than zero");
        assert!(
            prompt_ids.len() <= target,
            "prompt already has {} tokens, larger than --prefill-tokens {target}",
            prompt_ids.len()
        );
        let filler = tok.encode(" ");
        assert_eq!(filler.len(), 1, "space must encode to one token");
        prompt_ids.resize(target, filler[0]);
    }

    let mut state_per_rank: Vec<DeepseekV4State> = Vec::with_capacity(n);
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        state_per_rank.push(DeepseekV4State::new(&cfg).expect("state"));
        partials.push(
            gpus.devices[r]
                .zeros(
                    &[
                        if batched_prefill { prefill_batch } else { 1 },
                        cfg.hidden_size,
                    ],
                    DType::F32,
                )
                .expect("partial"),
        );
    }
    let mut pbs_per_rank = Vec::new();
    if batched_prefill {
        assert!(
            prefill_batch > 0,
            "--prefill-batch must be greater than zero"
        );
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind prefill scratch");
            pbs_per_rank.push(
                forward::PrefillBatchScratch::new(&mut gpus.devices[r], &cfg, prefill_batch)
                    .expect("prefill scratch"),
            );
        }
    }
    let peer = gpus.enable_peer_all().expect("enable_peer_all");
    eprintln!("  peer_access_enabled={peer}");
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus).expect("ensure_rank_streams");

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i as u32;
            }
        }
        bi
    };
    let dl_logits = |gpus: &mut Gpus, s: &DeepseekV4State| -> Vec<f32> {
        gpus.devices[0].bind_thread().expect("bind0");
        let l = s.logits.as_ref().expect("logits unset");
        gpus.devices[0].download_f32(l).expect("dl")
    };
    let dump_moe_probe = |gpus: &mut Gpus,
                          states: &[DeepseekV4State],
                          partials: &[GpuTensor],
                          out_dir: &std::path::Path,
                          position: usize| {
        std::fs::create_dir_all(out_dir).expect("create --moe-probe-out");
        let mut manifest = format!(
            "format=hipfire-deepseek4-moe-probe-v1\nlayer={}\nposition={}\n\
             hidden={}\nmoe_intermediate={}\nk_top={}\nranks={}\n",
            cfg.num_hidden_layers - 1,
            position,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
            states.len(),
        );
        for (rank, state) in states.iter().enumerate() {
            gpus.devices[rank].bind_thread().expect("bind probe rank");
            gpus.devices[rank]
                .hip
                .device_synchronize()
                .expect("sync probe rank");

            let tensors = [
                ("ffn_x_plain", state.ffn_x_plain.as_ref()),
                ("ffn_x_rot", state.ffn_x_rot.as_ref()),
                ("router_scores", state.router_scores.as_ref()),
                ("topk_weights", state.moe_topk_weights.as_ref()),
                ("silu_batch", state.moe_gate_batch.as_ref()),
                ("up_batch", state.moe_up_batch.as_ref()),
                ("rot_batch", state.moe_rot_batch.as_ref()),
                ("down_expanded", state.moe_down_expert_outputs.as_ref()),
                ("ffn_out", state.ffn_out.as_ref()),
            ];
            for (name, tensor) in tensors {
                let tensor = tensor.unwrap_or_else(|| panic!("probe tensor {name} unset"));
                let values = gpus.devices[rank]
                    .download_f32(tensor)
                    .unwrap_or_else(|error| panic!("download probe tensor {name}: {error:?}"));
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                let path = out_dir.join(format!("rank{rank}.{name}.f32le"));
                std::fs::write(&path, bytes)
                    .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
                manifest.push_str(&format!("rank{rank}.{name}.count={}\n", values.len()));
            }

            let indices = state
                .moe_topk_indices
                .as_ref()
                .expect("probe topk indices unset");
            let index_bits = gpus.devices[rank]
                .download_f32(indices)
                .expect("download probe topk indices")
                .into_iter()
                .map(|value| value.to_bits() as i32)
                .collect::<Vec<_>>();
            let index_bytes = index_bits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(
                out_dir.join(format!("rank{rank}.topk_indices.i32le")),
                index_bytes,
            )
            .expect("write probe topk indices");
            manifest.push_str(&format!("rank{rank}.topk_indices={index_bits:?}\n"));

            let routed = gpus.devices[rank]
                .download_f32(&partials[rank])
                .expect("download probe routed partial");
            let routed_bytes = routed
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(
                out_dir.join(format!("rank{rank}.routed_partial.f32le")),
                routed_bytes,
            )
            .expect("write probe routed partial");
            manifest.push_str(&format!(
                "rank{rank}.routed_partial.count={}\n",
                routed.len()
            ));
        }
        std::fs::write(out_dir.join("manifest.txt"), manifest).expect("write probe manifest");
        eprintln!(
            "wrote last-layer real-activation MoE probe to {}",
            out_dir.display()
        );
    };
    // Fork-margin diagnostic: top-8 logits at a position, decoded for eyeballing.
    let top8 = |v: &[f32], tok: &Tokenizer| -> String {
        let mut idx: Vec<u32> = (0..v.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            v[b as usize]
                .partial_cmp(&v[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..8]
            .iter()
            .map(|&i| format!("{}={:.4}{:?}", i, v[i as usize], tok.decode(&[i])))
            .collect::<Vec<_>>()
            .join(" ")
    };

    // ── EP prefill (per-token) + greedy decode ──────────────────────────────
    eprintln!(
        "\nprompt {} chars → {} tokens (mode={}, bos-prepended={}, synthetic-fill={})",
        prompt.chars().count(),
        prompt_ids.len(),
        if explicit_token_ids.is_some() {
            "token-ids"
        } else if chat {
            "chat"
        } else {
            "raw"
        },
        explicit_token_ids.is_none() && (chat || !no_bos),
        prefill_tokens.is_some()
    );
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    let mut score_rows: Vec<(u32, f32)> = Vec::new();
    if batched_prefill {
        for (chunk_idx, chunk) in prompt_ids.chunks(prefill_batch).enumerate() {
            let chunk_start = chunk_idx * prefill_batch;
            logits = forward::forward_prefill_batch_chunk_ep(
                &mut gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &pbs_per_rank,
                &partials,
                chunk,
                chunk_start as u32,
            )
            .expect("forward_prefill_batch_chunk_ep");
            let expected_n_tokens = (chunk_start + chunk.len()) as u64;
            let rank_n_tokens = state_per_rank
                .iter()
                .map(|state| state.n_tokens)
                .collect::<Vec<_>>();
            assert!(
                rank_n_tokens
                    .iter()
                    .all(|&n_tokens| n_tokens == expected_n_tokens),
                "batched prefill state cursor mismatch after positions {chunk_start}..={}: expected {expected_n_tokens}, got {rank_n_tokens:?}",
                expected_n_tokens - 1
            );
            eprintln!(
                "  prefill chunk positions {chunk_start}..={} → state.n_tokens={expected_n_tokens} on all {n} ranks",
                expected_n_tokens - 1
            );
        }
    } else {
        for (pos, &t) in prompt_ids.iter().enumerate() {
            forward::forward_ep(
                &mut gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                t,
                pos as u32,
            )
            .expect("forward_ep prefill");
            if score_out.is_some() {
                logits = dl_logits(&mut gpus, &state_per_rank[0]);
                if let Some(&target) = prompt_ids.get(pos + 1) {
                    assert!(
                        logits.iter().all(|value| value.is_finite()),
                        "non-finite teacher-forcing logits at position {pos}"
                    );
                    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = logits
                        .iter()
                        .map(|&value| ((value - max_logit) as f64).exp())
                        .sum::<f64>()
                        .ln()
                        + max_logit as f64;
                    let nll = lse - logits[target as usize] as f64;
                    assert!(nll.is_finite(), "non-finite NLL at position {pos}");
                    score_rows.push((target, nll as f32));
                }
            }
        }
        if score_out.is_none() {
            logits = dl_logits(&mut gpus, &state_per_rank[0]);
        }
    }
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "top8 @pos {} (prefill final): {}",
        prompt_ids.len() - 1,
        top8(&logits, &tok)
    );
    if let Some(path) = prefill_logits_out.as_ref() {
        let bytes = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(path, bytes).expect("write --prefill-logits-out");
        eprintln!(
            "wrote {} F32 prefill logits to {}",
            logits.len(),
            path.display()
        );
    }
    if let Some(path) = score_out.as_ref() {
        assert!(!score_rows.is_empty(), "--score-out produced no positions");
        let mut bytes = Vec::with_capacity(score_rows.len() * 8);
        for &(target, nll) in &score_rows {
            bytes.extend_from_slice(&target.to_le_bytes());
            bytes.extend_from_slice(&nll.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write --score-out");
        let mean_nll =
            score_rows.iter().map(|&(_, nll)| nll as f64).sum::<f64>() / score_rows.len() as f64;
        eprintln!(
            "teacher forcing: {} positions mean_nll={mean_nll:.9} ppl={:.9}; wrote {}",
            score_rows.len(),
            mean_nll.exp(),
            path.display()
        );
    }
    if let Some(path) = moe_probe_out.as_deref() {
        dump_moe_probe(
            &mut gpus,
            &state_per_rank,
            &partials,
            path,
            prompt_ids.len() - 1,
        );
    }

    // ── MTP EP draft (spec-decode drafter under expert parallelism) ─────────
    // After prefill, `logits` predicts t0. Capture h_n (the last-position full
    // HC residual stream) per rank into a DISTINCT buffer (mtp_pre_ffn reads
    // h_n then overwrites residual_streams), then run mtp_forward_ep to draft
    // the token AFTER t0. Compared below to the decode loop's gen[1] (= the
    // true next-next token): a match is a spec-decode "accept" — proof the
    // sharded MTP-layer experts + EP FFN produce the correct draft. (Runs
    // before the decode loop; forward_ep re-inits residual_streams so this
    // doesn't disturb the main path.)
    let mut mtp_draft: Option<u32> = None;
    if mtp {
        let t0 = argmax(&logits);
        let mut h_n_per_rank: Vec<GpuTensor> = Vec::with_capacity(n);
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind");
            let streams = state_per_rank[r]
                .residual_streams
                .as_ref()
                .expect("residual_streams");
            let h = gpus.devices[r]
                .alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], DType::F32)
                .expect("alloc h_n");
            gpus.devices[r]
                .memcpy_dtod_auto(&h.buf, &streams.buf, cfg.hc_mult * cfg.hidden_size * 4)
                .expect("copy h_n");
            h_n_per_rank.push(h);
        }
        let tm = std::time::Instant::now();
        let mtp_logits = forward::mtp_forward_ep(
            &mut gpus,
            &weights_per_rank,
            &cfg,
            &mut state_per_rank,
            &partials,
            &h_n_per_rank,
            t0,
            prompt_ids.len() as u32,
        )
        .expect("mtp_forward_ep");
        let finite = mtp_logits.iter().all(|x| x.is_finite());
        let d = argmax(&mtp_logits);
        mtp_draft = Some(d);
        eprintln!(
            "MTP-EP draft: next_token(t0)={t0} → draft next-next={d} ({:?}) finite={finite} in {:.0}ms",
            tok.decode(&[d]), tm.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut gen = Vec::new();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    let mut steady = 0usize;
    let mut steady_t = std::time::Instant::now();
    let mut ended_on_eos = false;
    for step in 0..max {
        let next = argmax(&logits);
        if next == eos_tok {
            ended_on_eos = true;
            break;
        }
        gen.push(next);
        if step == warmup {
            steady_t = std::time::Instant::now();
            steady = 0;
        }
        forward::forward_ep(
            &mut gpus,
            &weights_per_rank,
            &cfg,
            &mut state_per_rank,
            &partials,
            next,
            pos as u32,
        )
        .expect("forward_ep decode");
        logits = dl_logits(&mut gpus, &state_per_rank[0]);
        if step < 3 || trace_next.contains(&(step + 1)) {
            eprintln!(
                "top8 @pos {} (decode step {}): {}",
                pos,
                step + 1,
                top8(&logits, &tok)
            );
        }
        if step >= warmup {
            steady += 1;
        }
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    let steady_dt = steady_t.elapsed().as_secs_f64();
    let steady_tps = if steady > 0 {
        steady as f64 / steady_dt
    } else {
        f64::NAN
    };
    eprintln!(
        "decoded {} tok in {:.3}s ({:.3} tok/s overall); steady {} tok after {} warmup in {:.3}s ({:.3} tok/s)",
        gen.len(),
        dt,
        gen.len() as f64 / dt,
        steady,
        warmup,
        steady_dt,
        steady_tps,
    );
    let gen_text = tok.decode(&gen);
    println!(
        "=== PROMPT ===\n{prompt}\n=== GENERATION (tp={tp} EP) ===\n{}",
        gen_text
    );
    eprintln!("gen ids: {:?}", &gen[..gen.len().min(40)]);
    eprintln!("gen FNV: 0x{:016x}", fnv1a(&gen));
    eprintln!("generation ended_on_eos={ended_on_eos}");
    if !niah_expected.is_empty() {
        let recovered = niah_expected
            .iter()
            .filter(|expected| gen_text.contains(expected.as_str()))
            .collect::<Vec<_>>();
        eprintln!(
            "NIAH recovered {}/{} (min={}): {:?}",
            recovered.len(),
            niah_expected.len(),
            niah_min_recovered,
            recovered
        );
        assert!(
            recovered.len() >= niah_min_recovered,
            "NIAH FAIL: recovered {} of {}, need at least {}",
            recovered.len(),
            niah_expected.len(),
            niah_min_recovered
        );
    }
    if let Some(path) = gen_ids_out.as_ref() {
        let mut text = gen
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(path, text).expect("write --gen-ids-out");
        eprintln!(
            "wrote {} generated token IDs to {}",
            gen.len(),
            path.display()
        );
    }

    // MTP-EP accept check: the draft predicted the token AFTER t0; the decode
    // loop's gen[1] IS that true token. A match = spec-decode accept.
    if let Some(draft) = mtp_draft {
        let true_next = gen.get(1).copied();
        let accept = Some(draft) == true_next;
        eprintln!(
            "MTP-EP accept check: draft={draft} ({:?}) vs true gen[1]={:?} ({:?}) → {}",
            tok.decode(&[draft]),
            true_next,
            true_next.map(|t| tok.decode(&[t])).unwrap_or_default(),
            if accept {
                "ACCEPT ✓"
            } else {
                "reject (draft≠target; MTP path ran coherently regardless)"
            },
        );
    }

    if let Some(second_prompt) = second_prompt.as_deref() {
        assert!(
            ended_on_eos,
            "first turn reached --max without EOS; refusing to fabricate a second-turn boundary"
        );
        let mut continuation = vec![eos_tok, user_tok.unwrap()];
        continuation.extend(tok.encode(second_prompt));
        continuation.push(asst_tok.unwrap());
        continuation.extend(tok.encode("</think>"));
        eprintln!(
            "\nsecond-turn continuation: {} tokens, starts with EOS at absolute position {pos}",
            continuation.len()
        );
        for (chunk_idx, chunk) in continuation.chunks(prefill_batch).enumerate() {
            let chunk_start = pos + chunk_idx * prefill_batch;
            logits = forward::forward_prefill_batch_chunk_ep(
                &mut gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &pbs_per_rank,
                &partials,
                chunk,
                chunk_start as u32,
            )
            .expect("forward_prefill_batch_chunk_ep second turn");
            let expected_n_tokens = (chunk_start + chunk.len()) as u64;
            let rank_n_tokens = state_per_rank
                .iter()
                .map(|state| state.n_tokens)
                .collect::<Vec<_>>();
            assert!(
                rank_n_tokens
                    .iter()
                    .all(|&n_tokens| n_tokens == expected_n_tokens),
                "second-turn state cursor mismatch: expected {expected_n_tokens}, got {rank_n_tokens:?}"
            );
        }
        pos += continuation.len();

        let mut second_gen = Vec::new();
        let mut second_ended_on_eos = false;
        for _ in 0..second_max {
            let next = argmax(&logits);
            if next == eos_tok {
                second_ended_on_eos = true;
                break;
            }
            second_gen.push(next);
            forward::forward_ep(
                &mut gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                next,
                pos as u32,
            )
            .expect("forward_ep second-turn decode");
            logits = dl_logits(&mut gpus, &state_per_rank[0]);
            pos += 1;
        }
        println!(
            "=== SECOND PROMPT ===\n{second_prompt}\n=== SECOND GENERATION (continued KV) ===\n{}",
            tok.decode(&second_gen)
        );
        eprintln!(
            "second gen ids: {:?}",
            &second_gen[..second_gen.len().min(40)]
        );
        eprintln!("second gen FNV: 0x{:016x}", fnv1a(&second_gen));
        eprintln!(
            "second generation ended_on_eos={second_ended_on_eos}; final state.n_tokens={} on all ranks",
            state_per_rank[0].n_tokens
        );
        assert!(
            state_per_rank
                .iter()
                .all(|state| state.n_tokens == state_per_rank[0].n_tokens),
            "second-turn rank state cursors diverged"
        );
    }
}
