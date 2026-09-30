// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA past-budget parity probe on the real model: GPU vs CPU reference for
//! every QSA layer at 2K..32K context, through the production forward.
//!
//! Per context length `L`: reset, prefill `L - DECODE` tokens of real text
//! (tiled chunks: the batched select and the dense-F16 / grouped attention
//! routes), then teacher-force the last `DECODE` tokens one at a time (the
//! single-row decode routes). A QSA tap reads each layer's scratch and state
//! right after its step and checks, for sampled rows:
//!
//! * pooled keys: the GPU arena against a CPU pool of the GPU's raw index keys
//!   (mean, BF16 RMS norm with the learned weight, HalfSplit RoPE);
//! * selected indices: the GPU row (capacity `budget + compress - 1`, block
//!   order by rank, causal tail, `-1` fill) against a CPU selection over the
//!   GPU's index query and pooled keys, ranked by (score desc, block asc).
//!   Exact equality is required. `fma` scores mirror the kernel's chained
//!   F32 FMA dot (the exact reference); `plain` (separate multiply/add) and
//!   `upstream` (the oracle's `sum(relu(dot)) / sqrt(d)` order) are reported
//!   with their block-set overlap to show how close near-ties sit;
//! * attention output: `ops::qsa_attention` (F32) over the GPU selection, the
//!   GPU's Q/gate row and K/V arenas, against the GPU's gated head output.
//!
//! Then the final-row logits of the decode route are compared against an
//! all-prefill run of the same `L` tokens (no tap), and 24 greedy tokens are
//! decoded from that state for a coherence read.
//!
//! usage: qwen4_qsa_ctx MODEL TEXT OUT.jsonl [CTX=2048,4096,8192,16384,32768]
//!        [STRIDE=256] [DECODE=4]

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward::Qwen4QsaTap;
use hipfire_arch_qwen4::ops;
use hipfire_dispatch::pipeline::IndexedAttentionOp;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

fn read_f32(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<f32>> {
    gpu.download_f32(&tensor.sub_offset(offset, len)).map_err(err)
}

/// `len` bytes of a byte-addressed Raw tensor from byte `offset`.
fn read_raw(gpu: &Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.sub_offset(offset, len).buf)
        .map_err(err)?;
    Ok(bytes)
}

fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    f32::from_bits((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
}

#[derive(Clone, Copy, PartialEq)]
enum Score {
    /// Kernel order: per head a chained F32 FMA dot, `max(dot / sqrt(d), 0)`,
    /// heads summed in order.
    Fma,
    /// Same order with a separate multiply and add.
    Plain,
    /// Upstream oracle order: `sum(max(dot, 0)) / sqrt(d)`.
    Upstream,
}

fn block_scores(query: &[f32], pooled: &[f32], blocks: usize, heads: usize, dim: usize, mode: Score) -> Vec<f32> {
    let scale = (dim as f32).sqrt();
    (0..blocks)
        .map(|block| {
            let key = &pooled[block * dim..(block + 1) * dim];
            let mut score = 0.0f32;
            for head in 0..heads {
                let q = &query[head * dim..(head + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot = match mode {
                        Score::Fma => q[d].mul_add(key[d], dot),
                        _ => dot + q[d] * key[d],
                    };
                }
                score += match mode {
                    Score::Upstream => dot.max(0.0),
                    _ => (dot / scale).max(0.0),
                };
            }
            if mode == Score::Upstream {
                score / scale
            } else {
                score
            }
        })
        .collect()
}

/// Blocks by (score desc, block asc): the documented tie rule.
fn rank(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// The selected row the kernel contract defines: ranked blocks' tokens, then
/// the causal tail, `-1` in every other slot.
fn selection_row(order: &[usize], budget_blocks: usize, compress: usize, visible: usize, capacity: usize) -> Vec<i32> {
    let blocks = visible / compress;
    let chosen = budget_blocks.min(blocks);
    let mut row = vec![-1i32; capacity];
    for (slot, &block) in order.iter().take(chosen).enumerate() {
        for offset in 0..compress {
            let index = slot * compress + offset;
            let token = block * compress + offset;
            if index < capacity && token < visible {
                row[index] = token as i32;
            }
        }
    }
    let mut offset = chosen * compress;
    for token in blocks * compress..visible {
        if offset >= capacity {
            break;
        }
        row[offset] = token as i32;
        offset += 1;
    }
    row
}

fn chosen_set(row: &[i32], compress: usize) -> std::collections::BTreeSet<i32> {
    row.iter()
        .filter(|&&t| t >= 0)
        .map(|&t| t / compress as i32)
        .collect()
}

struct Probe {
    phase: &'static str,
    ctx: usize,
    stride: usize,
    rows: Vec<Value>,
    /// Per slot: the learned index-key norm weight, read once.
    k_norm: Vec<Option<Vec<f32>>>,
    /// Per slot: pooled blocks already checked this session.
    pooled_checked: Vec<usize>,
    wmma_dense: bool,
}

impl Probe {
    fn sampled(&self, rows: usize, start: usize, r: usize) -> bool {
        let visible = start + r + 1;
        rows == 1
            || r + 1 == rows
            || visible % self.stride == 0
            || (visible > 2048 && visible <= 2052)
    }

    fn tap(&mut self, gpu: &mut Gpu, slot: usize, op: &IndexedAttentionOp<'_>) -> Result<()> {
        gpu.hip.device_synchronize().map_err(err)?;
        let n = op.rows;
        let start = op.state.position;
        let end = start + n;
        let compress = op.compress;
        let dim = op.index_dim;
        let budget_blocks = op.budget / compress;
        let capacity = op.state.selected_capacity;
        let index_q_width = op.index_heads * dim;
        let index_width = index_q_width + op.index_kv_heads * dim;
        let q_width = op.heads * op.head_dim;
        let kv_width = op.kv_heads * op.head_dim;
        let blocks_end = end / compress;
        if self.k_norm.len() <= slot {
            self.k_norm.resize(slot + 1, None);
            self.pooled_checked.resize(slot + 1, 0);
        }
        if self.k_norm[slot].is_none() {
            let bytes = read_raw(gpu, op.indexer_k_norm, 0, dim * 2)?;
            self.k_norm[slot] = Some(
                bytes
                    .chunks_exact(2)
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                    .collect(),
            );
        }
        let pooled = read_f32(gpu, op.state.pooled_keys, 0, blocks_end.max(1) * dim)?;

        // Pooled arena vs a CPU pool of the GPU's raw keys (new blocks only).
        let first = self.pooled_checked[slot].min(blocks_end);
        if first < blocks_end {
            let raw = read_f32(
                gpu,
                op.state.raw_index_keys,
                first * compress * dim,
                (blocks_end - first) * compress * dim,
            )?;
            let norm = self.k_norm[slot].as_ref().expect("norm cached");
            let (mut max_abs, mut max_rel, mut exact) = (0.0f32, 0.0f32, 0usize);
            for block in first..blocks_end {
                let base = (block - first) * compress * dim;
                let mut mean = vec![0.0f32; dim];
                for row in 0..compress {
                    for c in 0..dim {
                        mean[c] += raw[base + row * dim + c] / compress as f32;
                    }
                }
                let mean: Vec<f32> = mean.into_iter().map(bf16_round).collect();
                let sum: f32 = mean.iter().map(|&m| bf16_round(m * m)).sum();
                let inv = 1.0 / (sum / dim as f32 + 1.0e-6).sqrt();
                let mut value: Vec<f32> = (0..dim)
                    .map(|c| bf16_round(mean[c] * inv * (1.0 + norm[c])))
                    .collect();
                ops::rope_prefix_halfsplit(&mut value, block * compress, dim.min(64), 10_000_000.0)
                    .map_err(err)?;
                for c in 0..dim {
                    let cpu = bf16_round(value[c]);
                    let gpu_value = pooled[block * dim + c];
                    let diff = (cpu - gpu_value).abs();
                    max_abs = max_abs.max(diff);
                    max_rel = max_rel.max(diff / cpu.abs().max(1.0e-3));
                    exact += usize::from(cpu.to_bits() == gpu_value.to_bits());
                }
            }
            self.rows.push(json!({
                "kind": "pool", "phase": self.phase, "ctx": self.ctx, "slot": slot,
                "blocks": [first, blocks_end], "max_abs": max_abs, "max_rel": max_rel,
                "exact_frac": exact as f64 / ((blocks_end - first) * dim) as f64,
            }));
            self.pooled_checked[slot] = blocks_end;
        }

        let sampled: Vec<usize> = (0..n).filter(|&r| self.sampled(n, start, r)).collect();
        if sampled.is_empty() {
            return Ok(());
        }
        let keys = read_f32(gpu, op.state.full_keys, 0, end * kv_width)?;
        let values = read_f32(gpu, op.state.full_values, 0, end * kv_width)?;
        let route = if n == 1 {
            "per_head"
        } else if self.wmma_dense && n >= 512 && blocks_end <= budget_blocks && end <= capacity {
            "dense_f16"
        } else {
            "hg4"
        };
        for r in sampled {
            let position = start + r;
            let visible = position + 1;
            let blocks = visible / compress;
            let query = read_f32(gpu, op.index_scratch, r * index_width, index_q_width)?;
            let gpu_row: Vec<i32> = read_raw(gpu, op.selected_scratch, r * capacity * 4, capacity * 4)?
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let mut verdict = serde_json::Map::new();
            let mut fma_order = Vec::new();
            let mut fma_scores = Vec::new();
            for (label, mode) in [("fma", Score::Fma), ("plain", Score::Plain), ("upstream", Score::Upstream)] {
                let scores = block_scores(&query, &pooled, blocks, op.index_heads, dim, mode);
                let order = rank(&scores);
                let row = selection_row(&order, budget_blocks, compress, visible, capacity);
                let gpu_set = chosen_set(&gpu_row, compress);
                let cpu_set = chosen_set(&row, compress);
                let overlap = gpu_set.intersection(&cpu_set).count();
                let first_diff = row.iter().zip(&gpu_row).position(|(a, b)| a != b);
                verdict.insert(
                    label.to_string(),
                    json!({
                        "exact": first_diff.is_none(),
                        "first_diff_slot": first_diff,
                        "set_overlap": overlap,
                        "set_size": cpu_set.len(),
                    }),
                );
                if mode == Score::Fma {
                    fma_order = order;
                    fma_scores = scores;
                }
            }
            // Boundary structure of the exact (fma) ranking.
            let zero_blocks = fma_scores.iter().filter(|&&s| s == 0.0).count();
            let (boundary_tie, boundary_gap) = if blocks > budget_blocks {
                let last_in = fma_scores[fma_order[budget_blocks - 1]];
                let first_out = fma_scores[fma_order[budget_blocks]];
                (last_in == first_out, (last_in - first_out) as f64)
            } else {
                (false, f64::NAN)
            };

            // Attention over the GPU selection.
            let qgate = read_f32(gpu, op.qgate_scratch, r * 2 * q_width, 2 * q_width)?;
            let gpu_out = read_f32(gpu, op.qsa_output, r * q_width, q_width)?;
            let selected: Vec<usize> = gpu_row.iter().filter(|&&t| t >= 0).map(|&t| t as usize).collect();
            let mut cpu_out = vec![0.0f32; q_width];
            ops::qsa_attention(
                &qgate,
                &keys[..visible * kv_width],
                &values[..visible * kv_width],
                &selected,
                op.heads,
                op.kv_heads,
                op.head_dim,
                &mut cpu_out,
            )
            .map_err(err)?;
            let (mut max_abs, mut ref_max, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in cpu_out.iter().zip(&gpu_out) {
                let (a, b) = (a as f64, b as f64);
                max_abs = max_abs.max((a - b).abs());
                ref_max = ref_max.max(a.abs());
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            self.rows.push(json!({
                "kind": "row", "phase": self.phase, "ctx": self.ctx, "slot": slot,
                "position": position, "blocks": blocks, "past_budget": blocks > budget_blocks,
                "selected_len": selected.len(), "route": route, "select": Value::Object(verdict),
                "zero_score_blocks": zero_blocks, "boundary_tie": boundary_tie,
                "boundary_gap": boundary_gap,
                "attn_max_abs": max_abs, "attn_ref_max": ref_max,
                "attn_rel": max_abs / ref_max.max(1.0e-30),
                "attn_cos": dot / (na.sqrt() * nb.sqrt()).max(1.0e-30),
            }));
        }
        Ok(())
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
        .0
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = v.iter().map(|&x| (x as f64 - max).exp()).sum();
    v.iter().map(|&x| x as f64 - max - sum.ln()).collect()
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(Ordering::Equal));
    idx.truncate(k);
    idx
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: qwen4_qsa_ctx MODEL TEXT OUT.jsonl [CTX,..] [STRIDE] [DECODE]".into());
    }
    let path = std::path::Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).map_err(err)?;
    let out_path = &args[3];
    let contexts: Vec<usize> = args
        .get(4)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2048, 4096, 8192, 16384, 32768]);
    let stride: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(256);
    let decode: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(4);
    let n_ctx = contexts.iter().copied().max().unwrap_or(2048) + 64;

    let mut hfq = HfqFile::open(path).map_err(err)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(err)?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(err)?;
    let mut gpu = Gpu::init().map_err(err)?;
    eprintln!("gpu {} uma={} n_ctx={n_ctx}", gpu.arch, gpu.is_uma());
    let use_ranges = gpu.is_uma();
    if use_ranges {
        hfq.drop_mmap();
    }
    #[allow(unused_mut)]
    let mut weights = receipt.manifest.weights.clone();
    let mesh = DeviceMesh::single().map_err(err)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            // The carrier's rule: ranges on UMA and for external rows,
            // borrowed mmap bytes for resident tensors on a discrete card.
            if use_ranges || entry.residency.is_external() {
                return source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name));
            }
            let (info, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))?;
            Ok(SourcePayload::Borrowed { info, bytes })
        },
    )
    .map_err(err)?;
    let vocab = receipt.config.vocab_size;
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        transaction,
        &receipt.placements,
        &mut gpu,
        n_ctx,
        receipt.ple,
    )
    .map_err(err)?;
    bundle.attach_forward(&mut gpu, n_ctx).map_err(err)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(err)?;
    let tokens = tokenizer.encode(&text);
    eprintln!("text: {} tokens", tokens.len());

    let probe = Arc::new(Mutex::new(Probe {
        phase: "prefill",
        ctx: 0,
        stride,
        rows: Vec::new(),
        k_norm: Vec::new(),
        pooled_checked: Vec::new(),
        wmma_dense: gpu.arch_caps.has_wmma_w32()
            && std::env::var("HIPFIRE_QWEN4_F16_WMMA").map_or(true, |v| v.trim() != "0"),
    }));
    let mut out = std::fs::File::create(out_path).map_err(err)?;

    for &ctx in &contexts {
        if tokens.len() < ctx {
            return Err(format!("text has {} tokens, need {ctx}", tokens.len()));
        }
        let prompt = &tokens[..ctx];
        // Tapped: prefill then teacher-forced decode.
        bundle.reset(&mut gpu).map_err(err)?;
        {
            let mut p = probe.lock().unwrap();
            p.phase = "prefill";
            p.ctx = ctx;
            p.pooled_checked.iter_mut().for_each(|c| *c = 0);
        }
        let tap_probe = Arc::clone(&probe);
        let tap: Qwen4QsaTap = Box::new(move |gpu, slot, op| tap_probe.lock().unwrap().tap(gpu, slot, op));
        bundle.set_qsa_tap(Some(tap)).map_err(err)?;
        let t = Instant::now();
        bundle
            .forward_chunk_final(&mut gpu, &prompt[..ctx - decode], &logits, None)
            .map_err(err)?;
        let tapped_prefill_s = t.elapsed().as_secs_f64();
        probe.lock().unwrap().phase = "decode";
        for &token in &prompt[ctx - decode..] {
            bundle.forward_token(&mut gpu, token, &logits, None).map_err(err)?;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        let decode_logits = gpu.download_f32(&logits).map_err(err)?;
        bundle.set_qsa_tap(None).map_err(err)?;

        // Untapped: all-prefill of the same tokens.
        bundle.reset(&mut gpu).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let t = Instant::now();
        bundle.forward_chunk_final(&mut gpu, prompt, &logits, None).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let prefill_s = t.elapsed().as_secs_f64();
        let prefill_logits = gpu.download_f32(&logits).map_err(err)?;
        let max_diff = decode_logits
            .iter()
            .zip(&prefill_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (lp, lq) = (log_softmax(&prefill_logits), log_softmax(&decode_logits));
        let kl: f64 = lp.iter().zip(&lq).map(|(&a, &b)| a.exp() * (a - b)).sum();
        let top_p = top_k(&prefill_logits, 5);
        let top_d = top_k(&decode_logits, 5);
        let top5_overlap = top_p.iter().filter(|t| top_d.contains(t)).count();

        // Greedy continuation from the all-prefill state.
        let mut generated = Vec::new();
        let mut next = argmax(&prefill_logits) as u32;
        let t = Instant::now();
        for _ in 0..24 {
            generated.push(next);
            bundle.forward_token(&mut gpu, next, &logits, None).map_err(err)?;
            next = argmax(&gpu.download_f32(&logits).map_err(err)?) as u32;
        }
        let decode_tok_s = 24.0 / t.elapsed().as_secs_f64();
        let summary = json!({
            "kind": "logits", "ctx": ctx, "decode_rows": decode,
            "max_abs_diff": max_diff, "kl_prefill_decode": kl,
            "argmax_prefill": argmax(&prefill_logits), "argmax_decode": argmax(&decode_logits),
            "top5_overlap": top5_overlap,
            "prefill_s": prefill_s, "prefill_tok_s": ctx as f64 / prefill_s,
            "tapped_prefill_s": tapped_prefill_s, "greedy_tok_s": decode_tok_s,
            "greedy_text": tokenizer.decode(&generated),
        });
        eprintln!("{summary}");
        let rows = std::mem::take(&mut probe.lock().unwrap().rows);
        for row in rows.iter().chain(std::iter::once(&summary)) {
            writeln!(out, "{row}").map_err(err)?;
        }
        out.flush().map_err(err)?;
        let checked = rows.iter().filter(|r| r["kind"] == "row").count();
        let fma_exact = rows
            .iter()
            .filter(|r| r["kind"] == "row" && r["select"]["fma"]["exact"] == true)
            .count();
        let worst_rel = rows
            .iter()
            .filter_map(|r| r["attn_rel"].as_f64())
            .fold(0.0f64, f64::max);
        eprintln!("ctx {ctx}: {checked} rows checked, fma-exact {fma_exact}, worst attn rel {worst_rel:.3e}");
    }
    Ok(())
}
