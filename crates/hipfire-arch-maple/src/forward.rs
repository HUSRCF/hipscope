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

use crate::batch::{dense_qt51_gemm, MAPLE_PREFILL_MAX_B};
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
    // Fail closed on a position past the KV allocation. Without this the KV
    // write runs off the end of the cache and the first symptom is
    // `hipMemcpy D2H: an illegal memory access was encountered` from a
    // downstream kernel — a fault that names the wrong culprit and says
    // nothing about the cause.
    if position as usize >= state.max_seq {
        return Err(format!(
            "maple: position {position} exceeds max_seq {} — allocate the state \
             with a larger max_seq (prompt + generation must fit)",
            state.max_seq
        ));
    }
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

        moe_block_row(cfg, layer, state, gpu, l, &state.normed, &state.h)?;

        // Per-layer residual capture for reference parity. Off unless
        // HIPFIRE_MAPLE_DUMP_HIDDEN is set, so the hot path is unchanged.
        //
        // A cosine cliff at layer n localises the bug to layer n's block —
        // the method that localised the Bonsai double-norm bug to layer 0.
        // The two silent-wrong-answer risks to check first are
        // RoPE-on-a-NoPE-layer and QK-norm ordering.
        if let Some(path) = dump_hidden_path() {
            dump_layer_hidden(gpu, state, l, hidden, &path)?;
        }
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

/// Destination for the per-layer residual dump, or `None` when disabled.
fn dump_hidden_path() -> Option<String> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("HIPFIRE_MAPLE_DUMP_HIDDEN").ok())
        .clone()
}

/// Append `[u32 layer][u32 hidden][hidden f32 LE]` for the current residual.
///
/// Appends rather than truncates so a whole prefill produces one file in
/// (position, layer) order; the reader keys on the layer index.
fn dump_layer_hidden(
    gpu: &mut Gpu,
    state: &MapleState,
    layer: usize,
    hidden: usize,
    path: &str,
) -> Result<(), String> {
    use std::io::Write;
    let h = gpu
        .download_f32(&state.h)
        .map_err(|e| format!("maple: dump hidden L{layer}: {e:?}"))?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("maple: open {path}: {e}"))?;
    let mut buf = Vec::with_capacity(8 + hidden * 4);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for v in h.iter().take(hidden) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| format!("maple: write {path}: {e}"))
}

/// One MoE block for ONE token: softmax → top-8 → renormalise, then the indexed
/// MQ2-Lloyd expert GEMVs with the clamped SwiGLU between them.
///
/// `x` is the post-attention RMSNorm output `[hidden]` and `h` is the residual
/// row `[hidden]` the down-projection atomically accumulates into. Decode passes
/// `&state.normed` / `&state.h`; batched prefill passes row views into
/// `state.b_normed` / `state.b_h`. Everything else (router scratch, top-k
/// buffers, the expert activation batch) is shared per-token scratch, which is
/// why this must stay strictly sequential over rows.
fn moe_block_row(
    cfg: &MapleConfig,
    layer: &crate::maple::MapleLayerWeights,
    state: &MapleState,
    gpu: &mut Gpu,
    l: usize,
    x: &GpuTensor,
    h: &GpuTensor,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let m = &layer.moe;

    // Router: softmax over ALL experts, take top-k, THEN renormalise the k
    // (`norm_topk_prob` = true). Renormalising before the top-k, or skipping
    // it, both yield plausible-looking but wrong combine weights.
    weight_gemv(gpu, &m.router, x, &state.router_logits)
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
        x,
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
        h,
        hidden,
        moe_inter,
        k_top,
        false,
    )
    .map_err(|e| format!("maple L{l}: down: {e:?}"))?;

    Ok(())
}

// ───────────────────────── Batched prefill ─────────────────────────

/// Batched prefill requires uniform qt51 experts (the dense and grouped GEMMs
/// both decode that layout) — anything else falls back to per-token.
pub fn forward_batch_supported(weights: &MapleWeights) -> bool {
    weights.layers.iter().all(|l| {
        l.wq.gpu_dtype == DType::MQ2G256LloydU
            && l.wk.gpu_dtype == DType::MQ2G256LloydU
            && l.wv.gpu_dtype == DType::MQ2G256LloydU
            && l.wo.gpu_dtype == DType::MQ2G256LloydU
            && l.moe.experts[0].gate_up.gpu_dtype == DType::MQ2G256LloydU
            && l.moe.experts[0].down.gpu_dtype == DType::MQ2G256LloydU
    })
}

/// Prefill `tokens` at `[start_pos, start_pos+B)`; returns the LAST token's
/// logits. Errors above `MAPLE_PREFILL_MAX_B` rather than splitting — chunking
/// is the caller's job (`batch::prefill_chunks`).
///
/// The attention half is fully batched: one GEMM per projection per layer over
/// all B rows. The MoE half is deliberately still PER-TOKEN here — correct but
/// slow — so this function is parity-testable before the grouped MoE lands.
pub fn forward_batch(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    state: &mut MapleState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("maple forward_batch: empty token slice".into());
    }
    if b > MAPLE_PREFILL_MAX_B {
        return Err(format!(
            "maple forward_batch: B={b} exceeds scratch cap {MAPLE_PREFILL_MAX_B}"
        ));
    }
    if start_pos + b > state.max_seq {
        return Err(format!(
            "maple forward_batch: positions {start_pos}..{} exceed max_seq {}",
            start_pos + b,
            state.max_seq
        ));
    }
    if !forward_batch_supported(weights) {
        return Err("maple forward_batch: unsupported tier (needs uniform qt51)".into());
    }

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    let n_rot = cfg.rotary_dim();
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();

    // Positions [B] as i32-in-f32, for batched RoPE + the masked windowed attend.
    let pos_bytes: Vec<u8> = (0..b)
        .flat_map(|i| ((start_pos + i) as i32).to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&state.b_positions.buf, &pos_bytes)
        .map_err(|e| format!("maple: htod positions: {e:?}"))?;

    // Re-stamp the dense slot index for THIS B: identity over the live rows and
    // -1 across the BLOCK_M tail, so the grouped kernel skips the pad tile rows
    // instead of computing them from whatever is left in the scratch. The table
    // is allocated for MAPLE_PREFILL_MAX_B and rewritten per chunk (≤ 2 KB, once
    // per chunk — not per layer). `dense_tile_ids` is all-zero for every B, so
    // the load-time upload of that one still stands.
    let slots = crate::batch::dense_slot_index_host(b);
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&state.b_slot_index.buf, &slot_bytes)
        .map_err(|e| format!("maple: htod slot index: {e:?}"))?;

    // Seed the residual rows with embeddings (one row copy per token; the
    // lookup kernels are per-token and this is a memcpy-scale cost).
    for (i, &tok) in tokens.iter().enumerate() {
        let row = row_view(&state.b_h, i * hidden, hidden);
        embed_lookup(gpu, weights, hidden, tok, &row)?;
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        gpu.rmsnorm_batched(
            &state.b_h,
            &layer.input_norm,
            &state.b_normed,
            b,
            hidden,
            eps,
        )
        .map_err(|e| format!("maple L{l}: batch input rmsnorm: {e:?}"))?;

        // Convert to F16 into a buffer WE own and hand THAT to the GEMM. The
        // grouped entry passes an F16 `x_src` straight to the kernel; letting it
        // take the F32 path instead would route through `ensure_fp16_x`, whose
        // conversion is cached on the source POINTER — and `b_normed` is one
        // buffer reused with new contents every layer, so every layer after 0
        // would silently reuse layer 0's activations.
        gpu.deepseek4_convert_f32_to_f16(&state.b_normed, &state.b_normed_f16, (b * hidden) as i64)
            .map_err(|e| format!("maple L{l}: f32->f16: {e:?}"))?;

        for (w_ptrs, m, out) in [
            (&layer.attn_ptr_tables.wq, q_dim, &state.b_q),
            (&layer.attn_ptr_tables.wk, kv_dim, &state.b_k),
            (&layer.attn_ptr_tables.wv, kv_dim, &state.b_v),
        ] {
            dense_qt51_gemm(
                gpu,
                w_ptrs,
                &state.b_tile_ids,
                &state.b_slot_index,
                &state.b_normed_f16,
                out,
                m,
                hidden,
                b,
            )?;
        }

        // QK-norm BEFORE RoPE, per head, across all B tokens. Both are in place,
        // so this order IS the semantics (see the module header).
        gpu.rmsnorm_batched(
            &state.b_q,
            &layer.q_norm,
            &state.b_q,
            n_heads * b,
            head_dim,
            eps,
        )
        .map_err(|e| format!("maple L{l}: batch q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(
            &state.b_k,
            &layer.k_norm,
            &state.b_k,
            n_kv * b,
            head_dim,
            eps,
        )
        .map_err(|e| format!("maple L{l}: batch k_norm: {e:?}"))?;

        // ...THEN RoPE, and only on sliding layers (full layers are NoPE).
        if cfg.applies_rope(l) {
            gpu.rope_partial_interleaved_f32_batched(
                &state.b_q,
                &state.b_k,
                &state.b_positions,
                n_heads,
                n_kv,
                head_dim,
                n_rot,
                cfg.rope_theta,
                b,
                0,
            )
            .map_err(|e| format!("maple L{l}: batch rope: {e:?}"))?;
        }

        batched_attend(cfg, state, gpu, l, b, start_pos)?;

        // o_proj over B rows, then add into the residual. The attention output
        // needs its OWN F16 buffer, distinct from `b_normed_f16` — see the
        // conversion note above; keeping the two apart also keeps the two
        // activations out of any shared conversion scratch.
        gpu.deepseek4_convert_f32_to_f16(
            &state.b_attn_out,
            &state.b_attn_out_f16,
            (b * q_dim) as i64,
        )
        .map_err(|e| format!("maple L{l}: attn_out f32->f16: {e:?}"))?;
        dense_qt51_gemm(
            gpu,
            &layer.attn_ptr_tables.wo,
            &state.b_tile_ids,
            &state.b_slot_index,
            &state.b_attn_out_f16,
            &state.b_proj_out,
            hidden,
            q_dim,
            b,
        )?;
        // `add_inplace_f32` takes NO length — it uses `a.numel()`. So both
        // sides must be views of exactly the live b*hidden elements;
        // `b_proj_out` is [dense_m_total(b) × hidden] and its padding tail must
        // not be folded into the residual.
        let h_live = row_view(&state.b_h, 0, b * hidden);
        let proj_live = row_view(&state.b_proj_out, 0, b * hidden);
        gpu.add_inplace_f32(&h_live, &proj_live)
            .map_err(|e| format!("maple L{l}: o_proj residual add: {e:?}"))?;

        gpu.rmsnorm_batched(
            &state.b_h,
            &layer.post_attn_norm,
            &state.b_normed,
            b,
            hidden,
            eps,
        )
        .map_err(|e| format!("maple L{l}: batch post-attn rmsnorm: {e:?}"))?;

        // Task 3: the MoE stays PER-TOKEN over the batch rows. Correct but slow
        // (one router GEMV + 8 expert GEMVs per token); Task 5 replaces this
        // whole loop with the grouped path.
        for i in 0..b {
            let x_row = row_view(&state.b_normed, i * hidden, hidden);
            let h_row = row_view(&state.b_h, i * hidden, hidden);
            moe_block_row(cfg, layer, state, gpu, l, &x_row, &h_row)?;
        }
    }

    // Head: last row only — the caller wants the next-token distribution, and
    // running lm_head over all B rows would cost B × vocab.
    let last = row_view(&state.b_h, (b - 1) * hidden, hidden);
    gpu.rmsnorm_batched(
        &last,
        &weights.final_norm,
        &state.final_norm_buf,
        1,
        hidden,
        eps,
    )
    .map_err(|e| format!("maple: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("maple: lm_head: {e}"))?;
    state.n_tokens = start_pos + b;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("maple: download logits: {e:?}"))
}

/// Non-owning view of `len` f32 at element `offset` inside `t`.
///
/// Mirrors `slice_moe_f32_view` in `hipfire-dispatch/src/pipeline/mod.rs`
/// (which is private to that crate). The returned tensor MUST NOT be freed —
/// it aliases `t`'s allocation, and `DeviceBuffer::from_raw` has no Drop-time
/// free, so the alias and the owner coexist until the OWNER is freed once.
///
/// Callers guarantee `offset + len <= t.numel()`.
fn row_view(t: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert!(
        offset + len <= t.numel(),
        "row_view out of range: {offset}+{len} > {}",
        t.numel()
    );
    debug_assert_eq!(t.dtype, DType::F32, "row_view is F32-only");
    unsafe {
        let base = t.buf.as_ptr() as *mut u8;
        let ptr = base.add(offset * 4);
        GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len * 4),
            shape: vec![len],
            dtype: DType::F32,
        }
    }
}

/// Batched KV write + windowed masked flash attention for layer `l`.
///
/// `q8_windowed` selects the batched masked windowed Q8 key unconditionally:
/// window > 0 clips sliding layers to the last `sliding_window` keys, window 0
/// is plain causal for the full/NoPE layers. The non-windowed batched key would
/// silently DROP the window on sliding layers past `sliding_window` tokens.
fn batched_attend(
    cfg: &MapleConfig,
    state: &MapleState,
    gpu: &mut Gpu,
    l: usize,
    b: usize,
    start_pos: usize,
) -> Result<(), String> {
    let window = if cfg.layer_type(l) == MapleLayerType::Sliding {
        cfg.sliding_window as i32
    } else {
        0
    };
    let ctx = DispatchCtx::new(gpu);
    let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
        hipfire_dispatch::families::kv_tier::KvTierInputs {
            pos: start_pos + b - 1,
            q8_windowed: true,
            window,
            batch_size: b,
            ..state.kv.tier_inputs()
        },
    )
    .map_err(|e| format!("maple L{l}: kv tier: {e}"))?;
    let io = hipfire_dispatch::families::attention::AttnParams {
        q: &state.b_q,
        k: &state.b_k,
        v: &state.b_v,
        k_cache: &state.kv.k_gpu[l],
        v_cache: &state.kv.v_gpu[l],
        k_scales: None,
        v_scales: None,
        // `pos_buf` / `pos` are the batch_size==1 path; with batch_size > 1 the
        // family reads `positions` instead.
        pos_buf: &state.pos_buf,
        pos: start_pos + b - 1,
        positions: Some(&state.b_positions),
        n_heads: cfg.num_attention_heads,
        n_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: b,
        max_ctx_len: start_pos + b,
        flash_partials: Some(&state.flash_partials),
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &state.b_attn_out,
    };
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        &ctx,
        &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
    )
    .map_err(|e| format!("maple L{l}: batch attention: {e:?}"))
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
