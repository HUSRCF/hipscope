// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE (North-Mini-Code) forward pass (free functions, hot-path static
//! dispatch).
//!
//! The defining structural trait is the **parallel block**: a SINGLE
//! mean-centered `Cohere2LayerNorm` feeds BOTH the attention and the FFN
//! branch, and both add into the residual —
//!   `h = h + o_proj(attn(LN(h))) + ffn(LN(h))`
//! (note: the FFN reads the SAME `LN(h)` as attention, NOT the
//! post-attention residual). Per layer:
//!   normed = layernorm_meancentered(h, input_layernorm)        [gamma only, β=0]
//!   q,k,v  = proj(normed); RoPE only if sliding (full=NoPE); attn; h += o_proj
//!   if dense (first_k_dense_replace prefix): h += down(silu(gate(normed))·up(normed))
//!   if moe:  router = sigmoid(gate·normed); top-8 (NO renorm: norm_topk_prob=false);
//!            h += Σ w_e · expert_e(normed)
//! then logits = lm_head(layernorm_meancentered(h, model.norm)) · logit_scale.
//!
//! Routed experts: the MQ4/MQ6 tiers use the FWHT-pre-rotated indexed-MoE GEMV
//! kernels (exactly the qwen35/lfm2/minimax path). The F16 oracle and Q8 expert
//! tiers have no indexed kernel, so they take a per-expert `weight_gemv` loop
//! (correctness over speed — the KLD/PPL harness is offline).
//!
//! Attention is full causal for sequences ≤ `sliding_window` (4096): at those
//! lengths sliding == full, so only the per-layer NoPE/RoPE split (which
//! matters at ALL lengths) is implemented here. A windowed-mask path for
//! >4096-token context is a follow-up.

use crate::config::{AttnKind, Cohere2MoeConfig};
use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights, Ffn};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu};

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("cohere2moe: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.hidden_size, token_id, &mut state.h)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    let mut logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("cohere2moe: download logits: {e:?}"))?;
    // logit_scale (1.0 for North-Mini-Code → no-op; applied host-side so the
    // device logits stay the raw lm_head output for any downstream re-use).
    if (cfg.logit_scale - 1.0).abs() > f32::EPSILON {
        for v in &mut logits {
            *v *= cfg.logit_scale;
        }
    }
    Ok(logits)
}

/// Seed the residual stream `out` with the embedding row for `token_id`.
/// Dispatches on the (tied) embed dtype: Q8 → dequant kernel; F32 → raw row
/// copy. (The F16 path is unused by the current tiers — embed/lm_head stay Q8
/// across the whole sweep, an engine constraint of the tied-embedding lookup.)
fn embed_lookup(
    gpu: &mut Gpu,
    weights: &Cohere2MoeWeights,
    hidden: usize,
    token_id: u32,
    out: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    match weights.embed_dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup f32: {e:?}")),
        other => Err(format!(
            "cohere2moe: embed dtype {other:?} has no lookup path (use Q8 or F32 tied embeddings)"
        )),
    }
}

/// Per-layer parallel-block stack + final norm + lm_head. Reads `state.h`
/// (seeded by the embedding lookup) and `state.pos_buf` (already staged).
fn decode_step_body(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.layer_norm_eps;
    let seq_len = position as usize + 1;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Parallel block: ONE mean-centered LayerNorm → `normed`, fed to
        //    BOTH the attention and the FFN branch (β = 0, weight only). ──────
        gpu.layernorm_batched(
            &state.h,
            &layer.input_norm,
            &state.ln_beta_zero,
            &state.normed,
            1,
            hidden,
            eps,
        )
        .map_err(|e| format!("cohere2moe L{l}: input layernorm: {e:?}"))?;

        // ── Attention branch (reads `normed`) ──────────────────────────────
        weight_gemv(gpu, &layer.wq, &state.normed, &state.fa_q)
            .map_err(|e| format!("cohere2moe L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.normed, &state.fa_k)
            .map_err(|e| format!("cohere2moe L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.normed, &state.fa_v)
            .map_err(|e| format!("cohere2moe L{l}: v_proj: {e}"))?;

        // NoPE: only `sliding_attention` layers apply RoPE. `full_attention`
        // (global) layers use NO positional embedding (Cohere2 sets
        // sliding_window=None there, gating off rotary).
        //
        // Cohere2 uses the **interleaved (GPT-J)** rotary convention — pairs
        // adjacent dims (2i, 2i+1) — NOT Llama's half-split. The HF
        // `rotate_half` is explicitly commented "different from e.g. Llama":
        // `x1=x[..., ::2]; x2=x[..., 1::2]; rot=stack([-x2,x1]).flatten`. So we
        // MUST use `rope_partial_interleaved_f32` (pairs 2i/2i+1), NOT
        // `rope_f32` (pairs i / i+head_dim/2). Rotary covers the FULL head_dim
        // (no partial_rotary_factor in the config) → n_rot = head_dim.
        if layer.attn_kind == AttnKind::Sliding {
            gpu.rope_interleaved_f32(
                &state.fa_q,
                &state.fa_k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                head_dim, // n_rot = full head_dim (no partial_rotary_factor)
                cfg.rope_theta,
            )
            .map_err(|e| format!("cohere2moe L{l}: rope: {e:?}"))?;
        }

        // KV write (Q8) + GQA attention. Full causal for seq_len ≤ sliding_window
        // (== sliding at those lengths). One KV slot per layer.
        gpu.kv_cache_write_q8_0(&state.kv.k_gpu[l], &state.fa_k, &state.pos_buf, n_kv, head_dim)
            .map_err(|e| format!("cohere2moe L{l}: kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0(&state.kv.v_gpu[l], &state.fa_v, &state.pos_buf, n_kv, head_dim)
            .map_err(|e| format!("cohere2moe L{l}: kv write v: {e:?}"))?;
        gpu.attention_q8_0_kv(
            &state.fa_q,
            &state.kv.k_gpu[l],
            &state.kv.v_gpu[l],
            &state.fa_attn_out,
            &state.pos_buf,
            seq_len,
            n_heads,
            n_kv,
            head_dim,
            state.kv.physical_cap,
        )
        .map_err(|e| format!("cohere2moe L{l}: attention: {e:?}"))?;

        // h += o_proj · attn_out  (attention into the residual).
        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("cohere2moe L{l}: o_proj: {e}"))?;

        // ── FFN branch (reads the SAME `normed`, NOT post-attention `h`) ─────
        match &layer.ffn {
            Ffn::Dense(d) => {
                weight_gemv(gpu, &d.gate, &state.normed, &state.dense_gate)
                    .map_err(|e| format!("cohere2moe L{l}: dense gate_proj: {e}"))?;
                weight_gemv(gpu, &d.up, &state.normed, &state.dense_up)
                    .map_err(|e| format!("cohere2moe L{l}: dense up_proj: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("cohere2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.down, &state.dense_act, &state.h)
                    .map_err(|e| format!("cohere2moe L{l}: dense down_proj: {e}"))?;
            }
            Ffn::Moe(m) => {
                // Router: sigmoid(logits) → top-k. `norm_topk_prob=false` for
                // North-Mini-Code, so the top-8 raw sigmoid scores are the
                // combine weights (NO renormalization). Selection by sigmoid is
                // monotonic in the logits, so it matches HF `expert_selection_fn`.
                weight_gemv(gpu, &m.router, &state.normed, &state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: router: {e}"))?;
                gpu.sigmoid_f32(&state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: sigmoid: {e:?}"))?;
                gpu.moe_topk_renorm_k8(
                    &state.router_logits,
                    &state.topk_indices,
                    &state.topk_weights,
                    n_exp,
                    cfg.norm_topk_prob,
                )
                .map_err(|e| format!("cohere2moe L{l}: topk: {e:?}"))?;

                let edt = m.experts[0].gate_up.gpu_dtype;
                match edt {
                    // FWHT-pre-rotated indexed MoE GEMV (MQ4/MQ6 tiers).
                    DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256 => {
                        let mq6 = matches!(edt, DType::MQ6G256 | DType::HFQ6G256);
                        rotate_x_mq_for(gpu, &m.experts[0].gate_up, &state.normed, &state.ffn_x_rot, hidden)
                            .map_err(|e| format!("cohere2moe L{l}: ffn rotate: {e:?}"))?;
                        if mq6 {
                            gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                                &m.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                                &state.gate_batch, &state.up_batch, 2 * moe_inter, hidden, k_top, 1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: gate_up(mq6): {e:?}"))?;
                        } else {
                            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                                &m.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
                                &state.gate_batch, &state.up_batch, 2 * moe_inter, hidden, k_top, 1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: gate_up(mq4): {e:?}"))?;
                        }
                        fused_silu_mul_rotate_mq_batched_for(
                            gpu, &m.experts[0].down, &state.gate_batch, &state.up_batch,
                            &state.rot_batch, moe_inter, k_top,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: silu_mul_rotate: {e:?}"))?;
                        if mq6 {
                            gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                                &m.expert_down_ptrs, &state.topk_indices, &state.rot_batch,
                                &state.down_expanded, hidden, moe_inter, k_top, 1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: down(mq6): {e:?}"))?;
                        } else {
                            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                                &m.expert_down_ptrs, &state.topk_indices, &state.rot_batch,
                                &state.down_expanded, hidden, moe_inter, k_top, 1,
                            )
                            .map_err(|e| format!("cohere2moe L{l}: down(mq4): {e:?}"))?;
                        }
                        gpu.moe_down_combine_k8_batched(
                            &state.down_expanded, &state.topk_weights, &state.h, hidden, k_top, 1,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: combine: {e:?}"))?;
                    }
                    // Per-expert path for the F16 oracle + Q8 tier (no indexed
                    // kernel for these dtypes). Reads the 8 selected experts off
                    // the device topk buffers and runs a plain GEMV each.
                    DType::Q8_0 | DType::F16 | DType::F32 => {
                        moe_per_expert(gpu, m, state, moe_inter, k_top, l)?;
                    }
                    other => {
                        return Err(format!("cohere2moe L{l}: unsupported expert dtype {other:?}"))
                    }
                }
            }
        }
    }
    state.n_tokens = seq_len;

    // Final mean-centered LayerNorm + lm_head (tied embed).
    gpu.layernorm_batched(
        &state.h,
        &weights.final_norm,
        &state.ln_beta_zero,
        &state.final_norm_buf,
        1,
        hidden,
        eps,
    )
    .map_err(|e| format!("cohere2moe: final layernorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("cohere2moe: lm_head: {e}"))?;
    Ok(())
}

/// Per-expert SwiGLU for non-indexable expert dtypes (F16 oracle / Q8 tier).
/// Recovers the 8 selected expert ids from the device topk buffers (the
/// i32 indices are bit-preserved through `download_f32`), runs a plain
/// `weight_gemv` per selected expert, and accumulates `w_e · down(silu(gate)·up)`
/// into the residual. `normed` is the parallel-block layernorm output.
fn moe_per_expert(
    gpu: &mut Gpu,
    m: &crate::cohere2moe::MoeFfn,
    state: &Cohere2MoeState,
    moe_inter: usize,
    k_top: usize,
    l: usize,
) -> Result<(), String> {
    // i32 expert ids are stored in an F32-typed tensor; download_f32 is a
    // bit-preserving copy, so `.to_bits()` recovers the original index.
    let idx_bits = gpu
        .download_f32(&state.topk_indices)
        .map_err(|e| format!("cohere2moe L{l}: dl topk idx: {e:?}"))?;
    let weights = gpu
        .download_f32(&state.topk_weights)
        .map_err(|e| format!("cohere2moe L{l}: dl topk w: {e:?}"))?;
    for j in 0..k_top {
        let e = (idx_bits[j].to_bits() as usize).min(m.experts.len() - 1);
        let w = weights[j];
        let expert = &m.experts[e];
        // gate_up = [2*moe_inter] (gate ‖ up), then split into halves.
        weight_gemv(gpu, &expert.gate_up, &state.normed, &state.expert_gate_up)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: gate_up gemv: {e2}"))?;
        let gate_view = state.expert_gate_up.sub_offset(0, moe_inter);
        let up_view = state.expert_gate_up.sub_offset(moe_inter, moe_inter);
        gpu.silu_mul_f32(&gate_view, &up_view, &state.expert_act)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: silu_mul: {e2:?}"))?;
        // Fold the router weight into the activation (down is linear:
        // w·down(act) = down(w·act)), then accumulate down(·) into h.
        gpu.scale_f32(&state.expert_act, w)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: scale: {e2:?}"))?;
        weight_gemv_residual(gpu, &expert.down, &state.expert_act, &state.h)
            .map_err(|e2| format!("cohere2moe L{l}E{e}: down gemv: {e2}"))?;
    }
    Ok(())
}
