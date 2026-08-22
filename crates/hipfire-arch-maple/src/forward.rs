// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview forward pass (free functions, hot-path static dispatch).
//!
//! Standard pre-norm block — NOT cohere2moe's parallel block:
//! ```text
//!   h += o_proj(attn(rmsnorm(h, input_layernorm)))
//!   h += moe(rmsnorm(h, post_attention_layernorm))
//!   logits = lm_head(rmsnorm(h, model.norm))
//! ```
//!
//! The four places this arch can go silently wrong, and what pins each:
//!
//! 1. **QK-norm runs BEFORE RoPE** (`modeling_maple.py:300-303`). Both are
//!    in-place on the q/k buffers, so the order here IS the semantics.
//! 2. **Partial rotary, half-split.** `rotary_dim` = 64 of 128; pairs are
//!    `(i, i+n_rot/2)` WITHIN the rotated block and the frequency denominator
//!    is `n_rot`, not `head_dim`. That is `rope_partial_halfsplit_f32` —
//!    reached via `rope_partial_interleaved_f32`, which is a misnomer and
//!    dispatches the half-split kernel by default. The similarly-named
//!    `rope_partial_halved_f32` is Gemma-4's *proportional* convention (pairs
//!    span the full head, denominator `head_dim`) and is NOT interchangeable.
//! 3. **NoPE on the global layers.** RoPE is applied only where
//!    `layer_types[l] == sliding_attention`.
//! 4. **Clamped SwiGLU, asymmetric:** `silu(clamp(gate, max=7)) *
//!    clamp(up, -7, 7)`. `deepseek4_silu_mul_clamp_f32_batched` implements
//!    exactly that (gate capped from above only, up clamped both ways) with the
//!    limit as a parameter — DeepSeek passes 10.0, Maple passes 7.0.
//!
//! **Why this does not call `run_moe_decode`.** The shared MoE executor's
//! gate-side unconditionally runs the shared-expert gate/up GEMVs, and Maple
//! has no shared expert (`num_shared_experts` 0). It also routes the
//! gate→down step through `fused_silu_mul_rotate_mq_batched`, which FWHT-
//! rotates the intermediate — correct for MQ2-Lloyd (qt19) and WRONG for the
//! unrotated MQ2G256LloydU (qt51), and it has no clamp. So this arch drives
//! the same indexed MQ2-Lloyd kernels directly, exactly as `cohere2moe` drives
//! the MQ4/MQ6 indexed kernels for its own no-shared-expert MoE. No new HIP.

use crate::config::{MapleConfig, MapleLayerType};
use crate::maple::{MapleState, MapleWeights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{weight_gemv, weight_gemv_residual};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    state: &mut MapleState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("maple: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.hidden_size, token_id, &state.h)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("maple: download logits: {e:?}"))
}

/// Seed the residual stream with the embedding row for `token_id`.
///
/// The converter carries `model.word_embeddings` as BF16, but there is no BF16
/// embedding-lookup kernel; `MapleWeights::load` therefore widens it to F32 on
/// the host at load time (see the note there), so this is the F32 path. Q8 is
/// accepted for a future requantized export.
fn embed_lookup(
    gpu: &mut Gpu,
    weights: &MapleWeights,
    hidden: usize,
    token_id: u32,
    out: &GpuTensor,
) -> Result<(), String> {
    match weights.embed_dtype {
        DType::F32 => gpu
            .embedding_lookup(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("maple: embed lookup f32: {e:?}")),
        DType::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("maple: embed lookup q8: {e:?}")),
        other => Err(format!(
            "maple: embed dtype {other:?} has no lookup path (expected F32 or Q8)"
        )),
    }
}

/// Per-layer stack + final norm + lm_head. Reads `state.h` (seeded by the
/// embedding lookup) and `state.pos_buf` (already staged).
fn decode_step_body(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    state: &mut MapleState,
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
    let eps = cfg.rms_norm_eps;
    let n_rot = cfg.rotary_dim();
    let seq_len = position as usize + 1;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Attention branch ────────────────────────────────────────────────
        gpu.rmsnorm_batched(&state.h, &layer.input_norm, &state.normed, 1, hidden, eps)
            .map_err(|e| format!("maple L{l}: input rmsnorm: {e:?}"))?;

        weight_gemv(gpu, &layer.wq, &state.normed, &state.fa_q)
            .map_err(|e| format!("maple L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.normed, &state.fa_k)
            .map_err(|e| format!("maple L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.normed, &state.fa_v)
            .map_err(|e| format!("maple L{l}: v_proj: {e}"))?;

        // QK-norm FIRST, per-head over head_dim, in place. Doing this after
        // RoPE still generates text — just worse text.
        gpu.rmsnorm_batched(
            &state.fa_q,
            &layer.q_norm,
            &state.fa_q,
            n_heads,
            head_dim,
            eps,
        )
        .map_err(|e| format!("maple L{l}: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&state.fa_k, &layer.k_norm, &state.fa_k, n_kv, head_dim, eps)
            .map_err(|e| format!("maple L{l}: k_norm: {e:?}"))?;

        // ...THEN RoPE, and only on sliding layers (global layers are NoPE).
        if cfg.applies_rope(l) {
            gpu.rope_partial_interleaved_f32(
                &state.fa_q,
                &state.fa_k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                n_rot,
                cfg.rope_theta,
            )
            .map_err(|e| format!("maple L{l}: rope: {e:?}"))?;
        }

        // KV write (Q8) + windowed flash attention. Sliding layers clip to the
        // last `sliding_window` keys; full layers are full causal (window 0).
        let window = if cfg.layer_type(l) == MapleLayerType::Sliding {
            cfg.sliding_window as i32
        } else {
            0
        };
        let ctx = DispatchCtx::new(gpu);
        let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
            hipfire_dispatch::families::kv_tier::KvTierInputs {
                pos: seq_len - 1,
                q8_windowed: true,
                window,
                ..state.kv.tier_inputs()
            },
        )
        .map_err(|e| format!("maple L{l}: kv tier: {e}"))?;
        let io = hipfire_dispatch::families::attention::AttnParams {
            q: &state.fa_q,
            k: &state.fa_k,
            v: &state.fa_v,
            k_cache: &state.kv.k_gpu[l],
            v_cache: &state.kv.v_gpu[l],
            k_scales: None,
            v_scales: None,
            pos_buf: &state.pos_buf,
            pos: seq_len - 1,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: state.kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&state.flash_partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &state.fa_attn_out,
        };
        hipfire_dispatch::pipeline::execute_steps(
            gpu,
            &ctx,
            &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
        )
        .map_err(|e| format!("maple L{l}: attention: {e:?}"))?;

        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("maple L{l}: o_proj: {e}"))?;

        // ── MoE branch (reads the POST-ATTENTION residual, re-normed) ───────
        gpu.rmsnorm_batched(
            &state.h,
            &layer.post_attn_norm,
            &state.normed,
            1,
            hidden,
            eps,
        )
        .map_err(|e| format!("maple L{l}: post-attn rmsnorm: {e:?}"))?;

        moe_block(cfg, layer, state, gpu, l, hidden, moe_inter, n_exp, k_top)?;
    }

    // ── Head ────────────────────────────────────────────────────────────────
    gpu.rmsnorm_batched(
        &state.h,
        &weights.final_norm,
        &state.final_norm_buf,
        1,
        hidden,
        eps,
    )
    .map_err(|e| format!("maple: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("maple: lm_head: {e}"))?;
    Ok(())
}

/// One MoE block: softmax → top-8 → renormalise, then the indexed MQ2-Lloyd
/// expert GEMVs with the clamped SwiGLU between them.
#[allow(clippy::too_many_arguments)]
fn moe_block(
    cfg: &MapleConfig,
    layer: &crate::maple::MapleLayerWeights,
    state: &MapleState,
    gpu: &mut Gpu,
    l: usize,
    hidden: usize,
    moe_inter: usize,
    n_exp: usize,
    k_top: usize,
) -> Result<(), String> {
    let m = &layer.moe;

    // Router: softmax over ALL experts, take top-k, THEN renormalise the k
    // (`norm_topk_prob` = true). Renormalising before the top-k, or skipping
    // it, both yield plausible-looking but wrong combine weights.
    weight_gemv(gpu, &m.router, &state.normed, &state.router_logits)
        .map_err(|e| format!("maple L{l}: router: {e}"))?;
    gpu.softmax_f32(&state.router_logits)
        .map_err(|e| format!("maple L{l}: router softmax: {e:?}"))?;
    gpu.moe_topk_renorm_k8(
        &state.router_logits,
        &state.topk_indices,
        &state.topk_weights,
        n_exp,
        cfg.norm_topk_prob,
    )
    .map_err(|e| format!("maple L{l}: topk: {e:?}"))?;

    let edt = m.experts[0].gate_up.gpu_dtype;
    if edt != DType::MQ2G256LloydU {
        return Err(format!(
            "maple L{l}: expert dtype {edt:?} unsupported — Maple's experts are \
             natively ternary and must be carried as MQ2G256LloydU (qt=51)"
        ));
    }

    // gate_up: indexed MQ2-Lloyd GEMV over the 8 selected experts. The
    // activation is `normed` in the NATURAL basis — MQ2G256LloydU weights are
    // NOT FWHT-rotated, so feeding a rotated x here is silent garbage.
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &m.expert_gate_up_ptrs,
        &state.topk_indices,
        &state.normed,
        &state.gate_batch,
        &state.up_batch,
        2 * moe_inter,
        hidden,
        k_top,
    )
    .map_err(|e| format!("maple L{l}: gate_up: {e:?}"))?;

    // Clamped SwiGLU, no rotation: silu(min(gate, L)) * clamp(up, ±L).
    gpu.deepseek4_silu_mul_clamp_f32_batched(
        &state.gate_batch,
        &state.up_batch,
        &state.act_batch,
        moe_inter,
        k_top,
        cfg.swiglu_clamp,
    )
    .map_err(|e| format!("maple L{l}: clamped swiglu: {e:?}"))?;

    // down: atomic, weighted, SELF-COMBINING residual GEMV — one launch does
    // down → * topk_weight[krank] → atomicAdd into `h`. There is deliberately
    // NO separate combine after this; adding one double-counts every layer.
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &m.expert_down_ptrs,
        &state.topk_indices,
        &state.topk_weights,
        &state.act_batch,
        &state.h,
        hidden,
        moe_inter,
        k_top,
        false,
    )
    .map_err(|e| format!("maple L{l}: down: {e:?}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MapleConfig {
        let json = r#"{"config":{
            "vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 8,
            "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128,
            "moe_intermediate_size": 512, "num_experts": 256, "num_experts_per_tok": 8,
            "partial_rotary_factor": 0.5, "sliding_window": 512, "rope_theta": 10000,
            "layer_types": [
              "sliding_attention","sliding_attention","sliding_attention","full_attention",
              "sliding_attention","sliding_attention","sliding_attention","full_attention"]
        }}"#;
        MapleConfig::from_metadata_json(json).unwrap()
    }

    /// The reference clamp, transcribed from `modeling_maple.py:110`:
    /// `silu(clamp(gate, max=7)) * clamp(up, min=-7, max=7)`.
    fn reference_swiglu(gate: f32, up: f32, limit: f32) -> f32 {
        let g = gate.min(limit); // capped from ABOVE only
        let u = up.clamp(-limit, limit); // clamped BOTH ways
        (g / (1.0 + (-g).exp())) * u
    }

    #[test]
    fn the_clamp_is_asymmetric_gate_above_only_up_both_ways() {
        let l = 7.0f32;
        // A large positive gate is capped...
        assert_eq!(
            reference_swiglu(100.0, 1.0, l),
            reference_swiglu(7.0, 1.0, l)
        );
        // ...but a large NEGATIVE gate is not. If the gate were clamped from
        // below too, these would be equal.
        assert_ne!(
            reference_swiglu(-100.0, 1.0, l),
            reference_swiglu(-7.0, 1.0, l)
        );
        // `up` is clamped on both sides.
        assert_eq!(
            reference_swiglu(1.0, 100.0, l),
            reference_swiglu(1.0, 7.0, l)
        );
        assert_eq!(
            reference_swiglu(1.0, -100.0, l),
            reference_swiglu(1.0, -7.0, l)
        );
    }

    #[test]
    fn the_clamp_limit_is_seven_not_deepseeks_ten() {
        // The kernel is shared with DeepSeek V4, which passes 10.0. Maple's 7.0
        // comes from the config type, and the two differ on real inputs — so a
        // copied 10.0 would be a silent quality change, not a crash.
        let c = cfg();
        assert_eq!(c.swiglu_clamp, 7.0);
        assert_ne!(
            reference_swiglu(9.0, 1.0, 7.0),
            reference_swiglu(9.0, 1.0, 10.0)
        );
    }

    #[test]
    fn rope_is_applied_on_sliding_layers_only() {
        let c = cfg();
        let roped: Vec<usize> = (0..c.num_hidden_layers)
            .filter(|&l| c.applies_rope(l))
            .collect();
        assert_eq!(roped, vec![0, 1, 2, 4, 5, 6], "layers 3 and 7 are NoPE");
    }

    #[test]
    fn rotary_covers_half_the_head_and_pairs_within_that_block() {
        // n_rot = 64 of head_dim 128. The half-split kernel pairs
        // (i, i + n_rot/2) = (i, i+32) — INSIDE the rotated block. Gemma-4's
        // rope_partial_halved pairs (i, i + head_dim/2) = (i, i+64), which
        // would reach into the pass-through half.
        let c = cfg();
        assert_eq!(c.rotary_dim(), 64);
        assert_eq!(c.rotary_dim() / 2, 32);
        assert_ne!(c.rotary_dim() / 2, c.head_dim / 2);
    }

    #[test]
    fn sliding_window_is_512_and_full_layers_are_unwindowed() {
        let c = cfg();
        assert_eq!(c.attention_span(0, 10_000).len(), 512);
        assert_eq!(c.attention_span(3, 10_000).start, 0);
    }
}
