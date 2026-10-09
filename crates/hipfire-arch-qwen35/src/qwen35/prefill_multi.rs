// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-request batched prefill chunk: several independent requests' short
//! row blocks (e.g. MTP verify rows `[seed, drafts..]`) through ONE trunk
//! forward, per request byte-identical to the singleton
//! [`super::forward_prefill_batch`] over that request's rows alone.
//!
//! It runs the singleton chunk body's own stage functions
//! (`forward_batch_chunk_impl`): the row-local stages — embedding, the
//! norm/rotate + projection GEMM stages, the FFN, the final norm — run once
//! over all requests' rows; the stages that read request state — position
//! upload, DeltaNet pre-GDN + recurrence, FullAttention prep + KV write +
//! attend — run per request on that request's rows (a row view of the
//! shared scratch), KV and DeltaNet state. Exactness rests on the shared
//! stages being row-local and selecting the same kernels for the combined
//! row count as for one request's rows: true while every count is in
//! `2..64` (the gfx12 projection GEMMs are row-count independent there —
//! CbSpec probe — and every n-dependent route switch is at >= 64 rows).

use super::*;

/// One request's rows of a multi-request chunk.
pub struct MultiChunkRequest<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
    pub kv_cache: &'a mut llama::KvCache,
    pub dn_state: &'a mut DeltaNetState,
    /// Rollback tape for this request's rows (the singleton verify's
    /// `gdn_tape`, offset 0); `None` = no capture.
    pub gdn_tape: Option<&'a crate::speculative::GdnTape>,
}

/// The largest combined row count: below every n-dependent kernel/route
/// switch of the shared stages (`>= 64` rows).
pub const MULTI_CHUNK_MAX_ROWS: usize = 63;

/// Rows `[r0, r0 + n)` of `pbs` as a scratch of `n` rows. Only row-major
/// fields are re-pointed; they alias `pbs` (never freed through the view).
fn rows_view(pbs: &PrefillBatchScratch, config: &Qwen35Config, r0: usize, n: usize) -> std::mem::ManuallyDrop<PrefillBatchScratch> {
    let dim = config.dim;
    let hidden = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = 2 * k_dim + v_dim;
    let nv = config.linear_num_value_heads;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let v = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
    // SAFETY: a bitwise copy whose every row-major tensor is then replaced
    // by a non-owning view of the same buffer; `ManuallyDrop` keeps the copy
    // from ever releasing anything `pbs` owns.
    unsafe {
        let mut p = std::mem::ManuallyDrop::new(std::ptr::read(pbs));
        let set = |slot: &mut GpuTensor, t: GpuTensor| std::ptr::write(slot, t);
        p.max_batch = n;
        set(&mut p.x_batch, v(&pbs.x_batch, dim));
        set(&mut p.x_rot_batch, v(&pbs.x_rot_batch, dim));
        set(&mut p.x_norm_batch, v(&pbs.x_norm_batch, dim));
        set(&mut p.dn_qkv_batch, v(&pbs.dn_qkv_batch, qkv_dim));
        set(&mut p.dn_z_batch, v(&pbs.dn_z_batch, v_dim));
        set(&mut p.dn_z_fold_batch, v(&pbs.dn_z_fold_batch, v_dim + 256));
        set(&mut p.dn_alpha_batch, v(&pbs.dn_alpha_batch, nv));
        set(&mut p.dn_beta_batch, v(&pbs.dn_beta_batch, nv));
        set(&mut p.dn_q_raw_batch, v(&pbs.dn_q_raw_batch, k_dim));
        set(&mut p.dn_k_raw_batch, v(&pbs.dn_k_raw_batch, k_dim));
        set(&mut p.dn_v_batch, v(&pbs.dn_v_batch, v_dim));
        set(&mut p.dn_q_batch, v(&pbs.dn_q_batch, v_dim));
        set(&mut p.dn_k_batch, v(&pbs.dn_k_batch, v_dim));
        set(&mut p.dn_attn_out_batch, v(&pbs.dn_attn_out_batch, v_dim));
        set(&mut p.dn_normed_batch, v(&pbs.dn_normed_batch, v_dim));
        set(&mut p.gate_ffn_batch, v(&pbs.gate_ffn_batch, hidden));
        set(&mut p.up_batch, v(&pbs.up_batch, hidden));
        set(&mut p.ffn_hidden_batch, v(&pbs.ffn_hidden_batch, hidden));
        set(&mut p.dn_normed_rot_batch, v(&pbs.dn_normed_rot_batch, v_dim));
        set(&mut p.positions, v(&pbs.positions, 1));
        set(&mut p.rope_positions, v(&pbs.rope_positions, 1));
        set(&mut p.pos3, v(&pbs.pos3, 3));
        set(&mut p.ext_emb_index, v(&pbs.ext_emb_index, 1));
        set(&mut p.tokens, v(&pbs.tokens, 1));
        set(&mut p.fa_q_full_batch, v(&pbs.fa_q_full_batch, 2 * q_dim));
        set(&mut p.fa_q_batch, v(&pbs.fa_q_batch, q_dim));
        set(&mut p.fa_gate_batch, v(&pbs.fa_gate_batch, q_dim));
        set(&mut p.fa_k_batch, v(&pbs.fa_k_batch, kv_dim));
        set(&mut p.fa_v_batch, v(&pbs.fa_v_batch, kv_dim));
        set(&mut p.fa_attn_out_batch, v(&pbs.fa_attn_out_batch, q_dim));
        set(&mut p.fa_attn_out_rot_batch, v(&pbs.fa_attn_out_rot_batch, q_dim));
        p
    }
}

/// Run every request's rows through one trunk forward, per request
/// byte-identical to `forward_prefill_batch_with_pbs_opts` over its rows
/// (no tape, no tree, no hidden ring, `DflashFusionCtx::Off`). With
/// `hidden_out`, writes post-output-norm hidden rows `[total x dim]` in
/// request order (`HiddenCapture::Verify`, as the singleton verify).
/// Refuses (before any launch) what it cannot reproduce exactly.
pub fn forward_prefill_batch_multi(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    reqs: &mut [MultiChunkRequest<'_>],
    hidden_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let refuse = |why: &str| Err(HipError::new(0, &format!("forward_prefill_batch_multi: {why}")));
    let total: usize = reqs.iter().map(|r| r.tokens.len()).sum();
    if reqs.is_empty() || total > MULTI_CHUNK_MAX_ROWS || total > pbs.max_batch || pbs.lean {
        return refuse("row count outside 1..=63 / scratch capacity, or lean scratch");
    }
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported");
    }
    let arch = gpu.arch.clone();
    for r in reqs.iter() {
        let n = r.tokens.len();
        // The singleton runs n == 1 through forward_scratch, not this body.
        if n < MIN_BATCH {
            return refuse("every request needs >= 2 rows");
        }
        if !prefill_batch_pbs_eligible(weights, config, r.dn_state, n, &arch, true) {
            return refuse("request not eligible for the batched prefill body");
        }
        let d = &r.dn_state;
        if d.quant != StateQuant::Q8 || d.s_ef_residual.len() != d.s_matrices.len() {
            return refuse("DeltaNet state must be Q8 with error feedback");
        }
        let kv = &r.kv_cache;
        if !(kv.quant_q8 || kv.quant_fp8) || kv.compact_offset != 0 {
            return refuse("KV must be uncompacted Q8 or fp8");
        }
        if r.gdn_tape.is_some_and(|t| t.max_n < n) {
            return refuse("GDN tape smaller than the request's rows");
        }
    }
    if !weights.layers.iter().all(|l| match l {
        LayerWeights::DeltaNet(_) => true,
        LayerWeights::FullAttn(_) => qwen35_layer_batch_admissible(l, config, &arch).is_ok(),
        _ => false,
    }) {
        return refuse("only dense DeltaNet / batched FullAttention layers");
    }
    for r in reqs.iter_mut() {
        let end = checked_kv_end(r.start_pos, r.tokens.len(), "forward_prefill_batch_multi")?;
        release_widened_pbs_for_kv_growth(gpu, r.kv_cache, config, s, end)?;
        r.kv_cache.ensure_mapped_capacity(gpu, end)?;
        r.kv_cache.require_mapped_capacity(end)?;
    }

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let fusion = DflashFusionCtx::Off;
    let sem = BatchSemantics::Sequential;
    // Request row ranges in the shared scratch.
    let mut offs = Vec::with_capacity(reqs.len());
    let mut off = 0usize;
    for r in reqs.iter() {
        offs.push(off);
        off += r.tokens.len();
    }
    let views: Vec<_> = reqs.iter().zip(&offs).map(|(r, &o)| rows_view(pbs, config, o, r.tokens.len())).collect();

    let tokens: Vec<u32> = reqs.iter().flat_map(|r| r.tokens.iter().copied()).collect();
    batch_chunk_embed_tokens(gpu, weights, &tokens, s, pbs, total, dim, dim * 4, true, false, false, None, None)?;
    for (r, view) in reqs.iter().zip(&views) {
        batch_chunk_upload_positions(gpu, view, sem, r.start_pos, r.tokens.len(), None, false)?;
    }
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    let tapes = reqs.iter().any(|r| r.gdn_tape.is_some());
    let ctx = DispatchCtx::new(gpu).with_workload(prefill_dispatch_workload(hidden_out.is_some(), tapes, false));

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // batch_chunk_delta_net_attn, non chunk-scan arm (n < 64).
                batch_chunk_delta_net_input_projection(gpu, layer, config, pbs, total, dim, q8_wmma_arch, fusion, None)?;
                for (r, view) in reqs.iter_mut().zip(&views) {
                    let n = r.tokens.len();
                    let parents = batch_chunk_delta_net_pre_gdn(
                        gpu, layer, config, view, r.dn_state, n, k_dim, v_dim, n_v_heads, hd, sem, None, r.gdn_tape, 0,
                        delta_layer_idx, fusion,
                    )?;
                    if parents.is_some() {
                        return refuse("tree recurrence in a linear verify");
                    }
                    let d = &*r.dn_state;
                    gpu.gated_delta_net_q8_batch_seq(
                        &view.dn_q_batch,
                        &view.dn_k_batch,
                        &view.dn_v_batch,
                        &view.dn_alpha_batch,
                        &view.dn_beta_batch,
                        &d.s_matrices[delta_layer_idx],
                        &d.s_scales[delta_layer_idx],
                        &view.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        d.ef_residual(delta_layer_idx),
                    )?;
                }
                batch_chunk_delta_net_output_projection(
                    gpu,
                    layer,
                    config,
                    pbs,
                    total,
                    n_v_heads,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                    GdnScanOut::F32,
                )?;
                batch_chunk_delta_net_ffn(
                    gpu, layer, config, pbs, total, dim, hidden_dim, q8_wmma_arch, arch_has_wmma, BatchEpilogue::Residual,
                    fusion,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // batch_chunk_full_attn_attn with the per-request flags each
                // request's own singleton chunk computes (all n < 64).
                let gfx12_fa_prep = gpu.arch == "gfx1201"
                    && gpu.flags.gfx12_fa_prep_fused
                    && !gpu.flags.rope_interleaved_legacy
                    && !hipfire_runtime::triattn::tap_enabled()
                    && config.head_dim == 256
                    && (config.n_heads, config.n_kv_heads) == (24, 4)
                    && (config.head_dim as f32 * config.partial_rotary_factor) as usize == 64;
                batch_chunk_full_attn_input_projection(gpu, layer, config, pbs, total, dim, q8_wmma_arch, fusion)?;
                for (r, view) in reqs.iter_mut().zip(&views) {
                    let n = r.tokens.len();
                    let max_ctx_len = r.start_pos + n;
                    let multirow = q8_multirow_attn_admitted(
                        gpu.arch_caps.arch(),
                        r.kv_cache.quant_q8,
                        config.head_dim,
                        n,
                        r.start_pos + n,
                        fa_pertoken_min_ctx(gpu.arch_caps.arch()),
                        false,
                        false,
                        false,
                        false,
                    );
                    batch_chunk_full_attn_prepare(
                        gpu, multirow, layer, config, view, s, r.kv_cache, n, r.start_pos, max_ctx_len, &ctx, sem, None,
                        kv_layer_idx, layer_idx, fusion, gfx12_fa_prep, false, false,
                    )?;
                    batch_chunk_fa_attend(
                        gpu, config, view, s, r.kv_cache, n, r.start_pos, max_ctx_len, &ctx, sem, None, layer_idx, multirow,
                        None, false,
                    )?;
                }
                batch_chunk_full_attn_output_projection(
                    gpu,
                    layer,
                    pbs,
                    total,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                    false,
                    None,
                )?;
                batch_chunk_full_attn_ffn(
                    gpu, layer, config, pbs, total, dim, hidden_dim, q8_wmma_arch, arch_has_wmma, BatchEpilogue::Residual,
                    fusion,
                )?;
                kv_layer_idx += 1;
            }
            _ => return refuse("layer type mismatch"),
        }
    }
    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs,
        total,
        dim,
        dim * 4,
        hidden_out.map(|t| (t, 0, HiddenCapture::Verify)),
        false,
        true,
        &ctx,
    )?;
    Ok(())
}
