// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact VMM batched step: byte-identical, per request, to the default
// singleton route.
//
// Decode rows run the singleton's own lowered layer program
// (`qwen35::forward::{lower_variant, variant_of}` + `Qwen35Bindings`). The
// projection super-ops are batched across requests, each column
// byte-identical to the singleton's kernel:
// - plain (QKVZA / QKV / gate+up): the exact batched RMSNorm+FWHT rotate,
//   then the PM multi-column V2 GEMV (`Gpu::gemv_mq4g256v2_xbatch_pm`);
// - residual (`wo`, `w_down`): each row's rotated input is staged by the
//   singleton's own prepass (gated-norm rotate / GEMV-family rotate / fused
//   SiLU·up rotate), then the PM multi-column residual GEMV
//   (`Gpu::gemv_mq4g256v2_residual_xbatch`) adds into the residual stream.
// (Oracle/probe evidence in runs/route-probe; kernel oracles in
// kernels/pm-decode/gfx1201.) Every other super-op — DeltaNet prep,
// recurrence, gated norm, attention (on the request's own VMM `KvCache`) —
// is the singleton binding itself, run once per request on the shared
// scratch with that request's rows copied in and out. Prefill chunks are the
// singleton prefill (`forward_prefill_batch`) on the request's own KV/DN with
// the singleton route's chunk lengths (`exact_prefill_chunk_len`).

use super::{Qwen35RequestState, Qwen35VmmStore, RequestStepKind};
use crate::forward_slots::final_logits_per_slot;
use crate::qwen35::forward::{gated_norm_mq_rotate_enabled, lower_variant, variant_of, Qwen35Bindings};
use crate::qwen35::{LayerWeights, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::RotateInputs;
use hipfire_dispatch::pipeline::superop::{dispatch_super_op, SuperOpKind};
use hipfire_runtime::llama::{fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_for, WeightTensor};
use hipfire_runtime::slot_batch::BatchStepPlan;
use rdna_compute::pm_xbatch::PM_XBATCH_MAX;
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
            LayerWeights::DeltaNet(l) => [&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up, &l.wo, &l.w_down].into_iter().all(v2),
            LayerWeights::FullAttn(l) => [&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up, &l.wo, &l.w_down].into_iter().all(v2),
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
        for (b0, nb) in xbatch_chunks(n) {
            gpu.gemv_mq4g256v2_xbatch_pm(
                &w.buf,
                &p.x_rot_batch.sub_offset(b0 * w.k, nb * w.k),
                &y.sub_offset(b0 * w.m, nb * w.m),
                w.m,
                w.k,
                nb,
            )?;
        }
    }
    Ok(())
}

/// `(first row, rows)` launches covering `n` rows, at most
/// [`PM_XBATCH_MAX`] each.
fn xbatch_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).step_by(PM_XBATCH_MAX).map(move |b0| (b0, PM_XBATCH_MAX.min(n - b0)))
}

/// Residual projection of super-op run `ordinal` (0 = attention/DeltaNet
/// `wo`, 1 = `w_down`): its weight and the batch buffer that stages each
/// row's rotated input contiguously as `[n][k]` (a buffer no row scratch
/// view aliases).
fn residual_target<'a>(st: &'a Qwen35VmmStore, layer: &'a LayerWeights, ordinal: usize) -> HipResult<(&'a WeightTensor, &'a GpuTensor)> {
    let p = &st.pbs;
    match (layer, ordinal) {
        (LayerWeights::DeltaNet(l), 0) => Ok((&l.wo, &p.dn_normed_rot_batch)),
        (LayerWeights::FullAttn(l), 0) => Ok((&l.wo, &p.fa_attn_out_rot_batch)),
        (LayerWeights::DeltaNet(l), 1) => Ok((&l.w_down, &p.ffn_hidden_batch)),
        (LayerWeights::FullAttn(l), 1) => Ok((&l.w_down, &p.ffn_hidden_batch)),
        _ => Err(HipError::new(0, "exact VMM step: unsupported residual super-op")),
    }
}

/// One decode row's view of the singleton scratch: the fields its binding
/// run reads or writes as its own row of the step's batch buffers (residual
/// stream, projection outputs, position) alias that row; every other field
/// is the shared scratch. Removes the per-row copy in/out of those rows.
struct RowScratch(std::mem::ManuallyDrop<Qwen35Scratch>);

impl RowScratch {
    fn new(s: &Qwen35Scratch, st: &Qwen35VmmStore, config: &Qwen35Config, r: usize) -> Self {
        let p = &st.pbs;
        let row = |t: &GpuTensor, len: usize| t.sub_offset(r * len, len);
        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let q_full = config.n_heads * config.head_dim * 2;
        let kv_row = config.n_kv_heads * config.head_dim;
        let hidden = s.gate_ffn.numel();
        // SAFETY: a bitwise copy of the shared scratch that is never
        // dropped (`ManuallyDrop`); only the replaced fields below are owned
        // by this value and dropped in `Drop`. Every replacement is a
        // non-owning alias of a live batch buffer row.
        unsafe {
            let mut v = std::mem::ManuallyDrop::new(std::ptr::read(s));
            std::ptr::write(&mut v.x, row(&p.x_batch, config.dim));
            std::ptr::write(&mut v.dn_qkv, row(&p.dn_qkv_batch, qkv_dim));
            std::ptr::write(&mut v.dn_z, row(&p.dn_z_batch, v_dim));
            std::ptr::write(&mut v.dn_beta, row(&p.dn_beta_batch, nv));
            std::ptr::write(&mut v.dn_alpha, row(&p.dn_alpha_batch, nv));
            std::ptr::write(&mut v.fa_q_full, row(&p.fa_q_full_batch, q_full));
            std::ptr::write(&mut v.fa_k, row(&p.fa_k_batch, kv_row));
            std::ptr::write(&mut v.fa_v, row(&p.fa_v_batch, kv_row));
            std::ptr::write(&mut v.gate_ffn, row(&p.gate_ffn_batch, hidden));
            std::ptr::write(&mut v.up, row(&p.up_batch, hidden));
            std::ptr::write(&mut v.pos_buf, p.positions.sub_offset(r, 1).buf);
            Self(v)
        }
    }
}

impl std::ops::Deref for RowScratch {
    type Target = Qwen35Scratch;
    fn deref(&self) -> &Qwen35Scratch {
        &self.0
    }
}

impl Drop for RowScratch {
    fn drop(&mut self) {
        // SAFETY: drop exactly the aliases written in `new`; the remaining
        // fields are bitwise copies owned by the shared scratch.
        unsafe {
            std::ptr::drop_in_place(&mut self.0.x);
            std::ptr::drop_in_place(&mut self.0.dn_qkv);
            std::ptr::drop_in_place(&mut self.0.dn_z);
            std::ptr::drop_in_place(&mut self.0.dn_beta);
            std::ptr::drop_in_place(&mut self.0.dn_alpha);
            std::ptr::drop_in_place(&mut self.0.fa_q_full);
            std::ptr::drop_in_place(&mut self.0.fa_k);
            std::ptr::drop_in_place(&mut self.0.fa_v);
            std::ptr::drop_in_place(&mut self.0.gate_ffn);
            std::ptr::drop_in_place(&mut self.0.up);
            std::ptr::drop_in_place(&mut self.0.pos_buf);
        }
    }
}

/// Stage decode row `r`'s rotated residual-projection input — exactly the
/// vector the singleton's residual super-op feeds `gemv_mq4g256v2_residual`
/// (RESID_WO: the gated-norm rotate output or the GEMV family's rotate of
/// the raw input; RESID_DOWN_SWIGLU: the fused SiLU·up rotate) — into row
/// `r` of the staging buffer.
fn stage_residual_input(
    gpu: &mut Gpu,
    st: &Qwen35VmmStore,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    layer: &LayerWeights,
    ordinal: usize,
    r: usize,
) -> HipResult<()> {
    let (w, stage) = residual_target(st, layer, ordinal)?;
    let k = w.k;
    let dst = stage.sub_offset(r * k, k);
    let rotate_raw = |gpu: &mut Gpu, x: &GpuTensor| -> HipResult<()> {
        let ctx = DispatchCtx::new(gpu);
        let xr = hipfire_runtime::llama::gemv_family()
            .rotate(&ctx, gpu, &w.dispatch_ref(), x, &RotateInputs::default())
            .map_err(|e| HipError::new(0, &e.to_string()))?
            .into_buf();
        copy_row(gpu, &dst, 0, &xr, 0, k)
    };
    match (layer, ordinal) {
        (LayerWeights::DeltaNet(_), 0) => {
            if gated_norm_mq_rotate_enabled(gpu, config, config.linear_num_value_heads, w) {
                copy_row(gpu, &dst, 0, &s.x_rot, 0, k)
            } else {
                rotate_raw(gpu, &s.dn_normed)
            }
        }
        (LayerWeights::FullAttn(_), 0) => rotate_raw(gpu, &s.fa_attn_out),
        (_, 1) => {
            gpu.ensure_mq_signs()?;
            fused_silu_mul_rotate_mq_for(gpu, w, &s.gate_ffn, &s.up, &dst, k)
        }
        _ => Err(HipError::new(0, "exact VMM step: unsupported residual super-op")),
    }
}

/// Batched residual projection `x_batch[r] += W · staged[r]` for every
/// decode row: per column the singleton's `gemv_mq4g256v2_residual`.
fn batched_residual(gpu: &mut Gpu, st: &Qwen35VmmStore, layer: &LayerWeights, ordinal: usize, n: usize) -> HipResult<()> {
    let (w, stage) = residual_target(st, layer, ordinal)?;
    let x_batch = &st.pbs.x_batch;
    for (b0, nb) in xbatch_chunks(n) {
        gpu.gemv_mq4g256v2_residual_xbatch(
            &w.buf,
            &stage.sub_offset(b0 * w.k, nb * w.k),
            &x_batch.sub_offset(b0 * w.m, nb * w.m),
            w.m,
            w.k,
            nb,
        )?;
    }
    Ok(())
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
    // Row positions, uploaded once: each row's scratch view reads its own.
    let pos_bytes: Vec<u8> = rows.iter().flat_map(|&(_, p)| (p as i32).to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&st.pbs.positions.buf, &pos_bytes)?;
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &st.pbs.x_batch, &st.pbs.tokens, n, dim)?;
    let row_scratch: Vec<RowScratch> = (0..n).map(|r| RowScratch::new(s, st, config, r)).collect();

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let mut delta_layer_idx = 0usize;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let program = lower_variant(variant_of(layer));
        let mut proj_ordinal = 0usize;
        let mut last_proj: Option<usize> = None;
        // Whether the run just executed staged the next residual's inputs.
        let mut staged = false;
        let mut i = 0usize;
        while i < program.len() {
            if program[i].kind == SuperOpKind::Proj {
                batched_proj(gpu, st, layer, config, proj_ordinal, n)?;
                last_proj = Some(proj_ordinal);
                proj_ordinal += 1;
                staged = false;
                i += 1;
                continue;
            }
            if program[i].kind == SuperOpKind::ResidualGemv {
                let ordinal = last_proj.ok_or_else(|| HipError::new(0, "exact VMM step: residual before projection"))?;
                if !staged {
                    // Directly after its projection (down after gate+up):
                    // stage from that projection's batched outputs.
                    for (r, rs) in row_scratch.iter().enumerate() {
                        stage_residual_input(gpu, st, rs, config, layer, ordinal, r)?;
                    }
                }
                batched_residual(gpu, st, layer, ordinal, n)?;
                // Later runs of this layer start from the residual stream.
                last_proj = None;
                staged = false;
                i += 1;
                continue;
            }
            // A run of non-batched super-ops: the singleton bindings, per row.
            let end = (i..program.len())
                .find(|&j| matches!(program[j].kind, SuperOpKind::Proj | SuperOpKind::ResidualGemv))
                .unwrap_or(program.len());
            let stage_for = (end < program.len() && program[end].kind == SuperOpKind::ResidualGemv)
                .then_some(last_proj)
                .flatten();
            for (r, &(slot, pos)) in rows.iter().enumerate() {
                let rs = &row_scratch[r];
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
                        s: rs,
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
                if let Some(ordinal) = stage_for {
                    stage_residual_input(gpu, st, rs, config, layer, ordinal, r)?;
                }
            }
            staged = stage_for.is_some();
            i = end;
        }
        if matches!(layer, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)) {
            delta_layer_idx += 1;
        }
    }

    // Final norm + head: the singleton's rmsnorm_f32 + lm_head GEMV per row
    // (`gemv_mq4g256v2_multirow_r2`; the PM multi-column GEMV is not
    // byte-identical to it, so the head is not batched).
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
