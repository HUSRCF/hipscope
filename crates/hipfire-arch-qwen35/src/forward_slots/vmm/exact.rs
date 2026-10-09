// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact VMM batched step: byte-identical, per request, to the default
// singleton route.
//
// Decode rows run the singleton's own lowered layer program
// (`qwen35::forward::{lower_variant, variant_of}` + `Qwen35Bindings`). Only
// the plain projection super-ops (QKVZA / QKV / gate+up) are batched across
// requests: the exact batched RMSNorm+FWHT rotate followed by the x-batched
// scalar V2 GEMV, each row byte-identical to the singleton's norm+GEMV
// (oracle/probe evidence in runs/route-probe). Every other super-op —
// DeltaNet prep, recurrence, gated norm, attention (on the request's own
// VMM `KvCache`), and the residual projections — is the singleton binding
// itself, run once per request on the shared scratch with that request's
// rows copied in and out. Prefill chunks are the singleton prefill
// (`forward_prefill_batch`) on the request's own KV/DN with the singleton
// route's chunk lengths (`exact_prefill_chunk_len`).

use super::{Qwen35RequestState, Qwen35VmmStore, RequestStepKind};
use crate::forward_slots::final_logits_per_slot;
use crate::qwen35::forward::{lower_variant, variant_of, Qwen35Bindings};
use crate::qwen35::{LayerWeights, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{dispatch_super_op, SuperOpKind};
use hipfire_runtime::llama::{fused_rmsnorm_rotate_mq_batched_for, WeightTensor};
use hipfire_runtime::slot_batch::BatchStepPlan;
use rdna_compute::{DType, Gpu, GpuTensor};

/// The exact route reproduces the default singleton route only where its
/// batched projection super-ops are the singleton's norm+GEMV: dense
/// DeltaNet/FullAttention layers, uniform MQ4G256V2 projections, gfx1201,
/// and the lowered decode program (the default; `HIPFIRE_FORWARD_LOWERED=0`
/// selects the hand path, which this route does not mirror).
pub(super) fn supports(gpu: &Gpu, weights: &Qwen35Weights) -> Result<(), String> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err(format!("exact VMM route: arch {} not admitted (gfx1201 only)", gpu.arch));
    }
    if hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() == Some("0") {
        return Err("exact VMM route: singleton is on the hand decode path (HIPFIRE_FORWARD_LOWERED=0)".into());
    }
    let v2 = |w: &WeightTensor| w.gpu_dtype == DType::MQ4G256V2;
    for (i, layer) in weights.layers.iter().enumerate() {
        let ok = match layer {
            LayerWeights::DeltaNet(l) => [&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up].into_iter().all(v2),
            LayerWeights::FullAttn(l) => [&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up].into_iter().all(v2),
            _ => false,
        };
        if !ok {
            return Err(format!("exact VMM route: layer {i} is not a dense uniform-MQ4G256V2 layer"));
        }
    }
    Ok(())
}

/// Singleton route's outer prefill chunk for a request with `remaining`
/// prompt rows (hipfire-generate ar.rs no-eviction path): the ordinary
/// chunk ceiling for this request's own state, then the serve tail rule.
pub(super) fn prefill_chunk_len(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    st: &Qwen35RequestState,
    remaining: usize,
) -> Result<usize, String> {
    let ceiling = crate::qwen35::ordinary_prefill_chunk_limit(gpu, weights, config, &st.dn, &st.kv, None)
        .map_err(|e| format!("exact prefill chunk limit: {e}"))?;
    Ok(crate::qwen35::prefill::ordinary_serve_prefill_chunk_len(remaining, ceiling)
        .unwrap_or(remaining.min(ceiling).max(1)))
}

fn copy_row(gpu: &Gpu, dst: &GpuTensor, dst_row: usize, src: &GpuTensor, src_row: usize, elems: usize) -> HipResult<()> {
    gpu.memcpy_dtod_at_auto(&dst.buf, dst_row * elems * 4, &src.buf, src_row * elems * 4, elems * 4)
}

/// Batched plain projection super-op `ordinal` (0 = attention projection,
/// 1 = gate+up) for every decode row.
fn batched_proj(
    gpu: &mut Gpu,
    st: &Qwen35VmmStore,
    layer: &LayerWeights,
    config: &Qwen35Config,
    ordinal: usize,
    n: usize,
) -> HipResult<()> {
    let p = &st.pbs;
    let (norm, ws): (&GpuTensor, Vec<(&WeightTensor, &GpuTensor)>) = match (layer, ordinal) {
        (LayerWeights::DeltaNet(l), 0) => (
            &l.attn_norm,
            vec![(&l.wqkv, &p.dn_qkv_batch), (&l.wz, &p.dn_z_batch), (&l.w_beta, &p.dn_beta_batch), (&l.w_alpha, &p.dn_alpha_batch)],
        ),
        (LayerWeights::FullAttn(l), 0) => (
            &l.attn_norm,
            vec![(&l.wq, &p.fa_q_full_batch), (&l.wk, &p.fa_k_batch), (&l.wv, &p.fa_v_batch)],
        ),
        (LayerWeights::DeltaNet(l), 1) => (&l.ffn_norm, vec![(&l.w_gate, &p.gate_ffn_batch), (&l.w_up, &p.up_batch)]),
        (LayerWeights::FullAttn(l), 1) => (&l.ffn_norm, vec![(&l.w_gate, &p.gate_ffn_batch), (&l.w_up, &p.up_batch)]),
        _ => return Err(HipError::new(0, "exact VMM step: unsupported projection super-op")),
    };
    fused_rmsnorm_rotate_mq_batched_for(gpu, &p.x_batch, norm, ws[0].0, &p.x_rot_batch, config.dim, config.norm_eps, n)?;
    for (w, y) in ws {
        gpu.gemv_mq4g256v2_xbatch(&w.buf, &p.x_rot_batch, y, w.m, w.k, n)?;
    }
    Ok(())
}

/// Copy decode row `r`'s projection outputs (from super-op `ordinal`) and
/// residual stream into the singleton scratch.
fn copy_in(gpu: &Gpu, st: &Qwen35VmmStore, s: &Qwen35Scratch, config: &Qwen35Config, layer: &LayerWeights, ordinal: Option<usize>, r: usize) -> HipResult<()> {
    let p = &st.pbs;
    copy_row(gpu, &s.x, 0, &p.x_batch, r, config.dim)?;
    match (layer, ordinal) {
        (LayerWeights::DeltaNet(l), Some(0)) => {
            copy_row(gpu, &s.dn_qkv, 0, &p.dn_qkv_batch, r, l.wqkv.m)?;
            copy_row(gpu, &s.dn_z, 0, &p.dn_z_batch, r, l.wz.m)?;
            copy_row(gpu, &s.dn_beta, 0, &p.dn_beta_batch, r, l.w_beta.m)?;
            copy_row(gpu, &s.dn_alpha, 0, &p.dn_alpha_batch, r, l.w_alpha.m)
        }
        (LayerWeights::FullAttn(l), Some(0)) => {
            copy_row(gpu, &s.fa_q_full, 0, &p.fa_q_full_batch, r, l.wq.m)?;
            copy_row(gpu, &s.fa_k, 0, &p.fa_k_batch, r, l.wk.m)?;
            copy_row(gpu, &s.fa_v, 0, &p.fa_v_batch, r, l.wv.m)
        }
        (LayerWeights::DeltaNet(l), Some(1)) => {
            copy_row(gpu, &s.gate_ffn, 0, &p.gate_ffn_batch, r, l.w_gate.m)?;
            copy_row(gpu, &s.up, 0, &p.up_batch, r, l.w_up.m)
        }
        (LayerWeights::FullAttn(l), Some(1)) => {
            copy_row(gpu, &s.gate_ffn, 0, &p.gate_ffn_batch, r, l.w_gate.m)?;
            copy_row(gpu, &s.up, 0, &p.up_batch, r, l.w_up.m)
        }
        (_, None) => Ok(()),
        _ => Err(HipError::new(0, "exact VMM step: unsupported layer kind")),
    }
}

/// Exact decode for the AR rows of `members` (plan request indices). Rows
/// are laid out in `pbs` in member order. Fills `picks`.
pub(super) fn decode(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    members: &[usize],
    picks: &mut [u32],
) -> HipResult<()> {
    let n = members.len();
    if n == 0 {
        return Ok(());
    }
    let dim = config.dim;
    let pb = &plan.batch;
    // Per-row slot / position in member order.
    let rows: Vec<(usize, usize)> = members
        .iter()
        .map(|&i| {
            let r = &plan.requests[i];
            (pb.row_slot[r.rows.begin] as usize, pb.positions[r.rows.begin] as usize)
        })
        .collect();
    let tokens: Vec<i32> = members.iter().map(|&i| pb.tokens[plan.requests[i].rows.begin] as i32).collect();
    let tb: Vec<u8> = tokens.iter().flat_map(|t| t.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&st.pbs.tokens.buf, &tb)?;
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &st.pbs.x_batch, &st.pbs.tokens, n, dim)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let mut delta_layer_idx = 0usize;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let program = lower_variant(variant_of(layer));
        let mut proj_ordinal = 0usize;
        let mut last_proj: Option<usize> = None;
        let mut i = 0usize;
        while i < program.len() {
            if program[i].kind == SuperOpKind::Proj {
                batched_proj(gpu, st, layer, config, proj_ordinal, n)?;
                last_proj = Some(proj_ordinal);
                proj_ordinal += 1;
                i += 1;
                continue;
            }
            // A run of non-batched super-ops: the singleton bindings, per row.
            let end = (i..program.len())
                .find(|&j| program[j].kind == SuperOpKind::Proj)
                .unwrap_or(program.len());
            for (r, &(slot, pos)) in rows.iter().enumerate() {
                copy_in(gpu, st, s, config, layer, last_proj, r)?;
                gpu.hip.memcpy_htod(&s.pos_buf, &(pos as i32).to_ne_bytes())?;
                let req = st.slots[slot].as_mut().expect("provisioned slot");
                let frame = req.dn.s_ef_residual.is_empty().then(|| {
                    let g = rdna_compute::norm::gdn_requant_frame_checkpoint();
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(req.gdn_frame);
                    g
                });
                {
                    let ctx = DispatchCtx::new(gpu);
                    let mut bind = Qwen35Bindings {
                        layer,
                        s,
                        config,
                        kv_cache: &mut req.kv,
                        dn_state: &req.dn,
                        pos,
                        layer_idx,
                        delta_layer_idx,
                        k_dim,
                        v_dim,
                        n_v_heads,
                        hd,
                        precomputed_attn_x_rot: false,
                        fa_output_prerotated: false,
                        defer_routed_combine: false,
                    };
                    for op in &program[i..end] {
                        dispatch_super_op(gpu, &ctx, op, &mut bind)
                            .map_err(|e| HipError::new(0, &e.to_string()))?;
                    }
                }
                if let Some(g) = frame {
                    req.gdn_frame = rdna_compute::norm::gdn_requant_frame_checkpoint();
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(g);
                }
                copy_row(gpu, &st.pbs.x_batch, r, &s.x, 0, dim)?;
            }
            i = end;
        }
        if matches!(layer, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)) {
            delta_layer_idx += 1;
        }
    }

    // Final norm + head: the singleton's rmsnorm_f32 + lm_head GEMV per row.
    let mut b = hipfire_runtime::slot_batch::SlotBatch::default();
    let n_slots = pb.m_per_slot.len();
    b.m_per_slot = vec![0; n_slots];
    for &(slot, pos) in &rows {
        b.m_per_slot[slot] = 1;
        b.tokens.push(0);
        b.positions.push(pos as i32);
        b.row_slot.push(slot as i32);
    }
    // final_logits_per_slot walks slots in order; rows are in slot order.
    st.skip.clear();
    st.skip.resize(n_slots, true);
    for &(slot, _) in &rows {
        st.skip[slot] = false;
    }
    final_logits_per_slot(gpu, weights, config, &b, &st.pbs, s, &st.logits, &st.skip)?;
    for &i in members {
        super::sample_head(gpu, st, config, s, plan, i, picks)?;
    }
    Ok(())
}

/// Exact prefill chunk for plan request `i`: the singleton prefill on the
/// request's own KV/DN; the chunk that completes the prompt picks from the
/// singleton's last-token logits.
pub(super) fn prefill(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    i: usize,
    picks: &mut [u32],
) -> HipResult<()> {
    let r = &plan.requests[i];
    if r.kind != RequestStepKind::Prefill {
        return Err(HipError::new(0, "exact VMM step: not a prefill request"));
    }
    let pb = &plan.batch;
    let slot = pb.row_slot[r.rows.begin] as usize;
    let pos = pb.positions[r.rows.begin] as usize;
    let tokens = &pb.tokens[r.rows.begin..r.rows.begin + r.rows.len];
    let head = st.wants_head(r);
    {
        let req = st.slots[slot].as_mut().expect("provisioned slot");
        let frame = req.dn.s_ef_residual.is_empty().then(|| {
            let g = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(req.gdn_frame);
            g
        });
        crate::qwen35::forward_prefill_batch(
            gpu, weights, config, tokens, pos, &mut req.kv, &mut req.dn, s, None, None, None, None,
        )?;
        if let Some(g) = frame {
            req.gdn_frame = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(g);
        }
    }
    if head {
        let view = st.logits.sub_offset(slot * config.vocab_size, config.vocab_size);
        gpu.memcpy_dtod_at_auto(&view.buf, 0, &s.logits.buf, 0, config.vocab_size * 4)?;
        super::sample_head(gpu, st, config, s, plan, i, picks)?;
    }
    let _ = DType::F32;
    Ok(())
}
