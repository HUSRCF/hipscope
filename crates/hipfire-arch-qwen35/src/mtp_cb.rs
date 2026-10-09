// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-request batched MTP speculative decode (continuous batching).
//!
//! One cycle over several requests, each its own singleton MTP window
//! (`spec_step_mtp_compressed_serial_inner`, the `Qwen35MtpDrafter::mtp_step`
//! route) except that the verify trunk and the verify head are shared:
//!
//! 1. draft — per request, unchanged ([`mtp_draft_phase_inner`]);
//! 2. verify trunk — every request's `[seed, drafts..]` rows through ONE
//!    [`forward_prefill_batch_multi`] (per request byte-identical to the
//!    singleton verify forward, rollback tape included);
//! 3. verify head — one row-independent lm_head over all rows, each
//!    request's logits/hidden rows copied into its own `MtpSpecState`;
//! 4. accept/repair — per request, the singleton consumer
//!    ([`mtp_accept_and_rollback`] with the head precomputed).
//!
//! A request whose window cannot take the shared trunk (a single verify row,
//! or a batched-prefill-ineligible row count) runs the singleton verify.

use super::*;
use crate::qwen35::prefill::multi::{
    forward_prefill_batch_multi, multi_chunk_pack_cap, multi_chunk_row_cap, MultiChunkRequest, MultiChunkScratch,
    MULTI_CHUNK_MAX_LANE_ROWS, MULTI_CHUNK_PRODUCT_MAX_ROWS,
};

/// Shared verify buffers for one CB engine: trunk scratch, post-norm hidden
/// rows, head rotation scratch and head logits, all `max_rows` rows (the
/// effective cap: 63, or up to 128 on the wide verify route).
pub struct MtpCbScratch {
    pub max_rows: usize,
    trunk: MultiChunkScratch,
    hidden: GpuTensor,
    rot: GpuTensor,
    logits: GpuTensor,
}

impl MtpCbScratch {
    /// `max_rows` is clamped to `2..=multi_chunk_row_cap(gpu)` (63 unless
    /// `HIPFIRE_CB_VERIFY_CHUNK128` is on, then up to 128); a cycle with more
    /// rows runs in several shared chunks.
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = max_rows.clamp(2, multi_chunk_row_cap(gpu));
        let trunk = MultiChunkScratch::new(gpu, config, max_rows)?;
        let alloc = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], DType::F32);
        let hidden = alloc(gpu, max_rows * config.dim)?;
        let rot = alloc(gpu, max_rows * config.dim)?;
        let logits = alloc(gpu, max_rows * config.vocab_size)?;
        Ok(Self { max_rows, trunk, hidden, rot, logits })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        self.trunk.free_gpu(gpu)?;
        gpu.free_tensor(self.hidden)?;
        gpu.free_tensor(self.rot)?;
        gpu.free_tensor(self.logits)
    }
}

/// One lane's draft request.
pub struct MtpCbDraftLane<'a> {
    pub state: &'a mut MtpSpecState,
    pub cur_pos: usize,
    pub last_committed: u32,
    /// Emitted history ending in `last_committed` (penalty window).
    pub emitted: &'a [u32],
    pub k: usize,
}

/// Does this lane's singleton draft take the greedy full-vocab
/// device-token-chain loop (no proposal graph) whose head the batched
/// drafter shares?
fn draft_batchable(gpu: &Gpu, weights: &Qwen35Weights, head: &Qwen35MtpHead, lane: &MtpCbDraftLane<'_>) -> bool {
    let st = &*lane.state;
    lane.k.min(st.max_n) > 0
        && head.weights.lm_head_draft.is_none()
        && st.sampling.is_greedy()
        && st.p_min <= 0.0
        && mtp_device_token_chain_enabled_from_env()
        && mtp_device_token_chain_eligible_for(weights.embd_format, false, false)
        && weights.output.gpu_dtype == DType::MQ4G256V2
        && gpu.arch_caps.is_gfx1201()
}

/// Draft phase of many lanes, step-synchronous: at each draft step every
/// lane runs its own singleton head block, norm and lm_head input rotate,
/// then ONE multi-column lm_head (the PM twin of the singleton
/// `gemv_mq4g256v2_multirow_r2`, column-identical) over all lanes, then
/// each lane's own argmax into its device token chain. Per lane this is
/// the singleton greedy full-vocab device-chain draft; lanes outside that
/// mode draft alone ([`mtp_cb_draft`]). `cb.rot`/`cb.logits` are free
/// before verify and stage the head columns.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_draft_batched(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbDraftLane<'_>],
) -> HipResult<Vec<MtpDraftOutput>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let dim = config.dim;
    let vocab = config.vocab_size;
    let batched: Vec<bool> = lanes.iter().map(|l| draft_batchable(gpu, weights, head, l)).collect();
    let group: Vec<usize> = (0..lanes.len()).filter(|&i| batched[i]).collect();
    let mut out: Vec<Option<MtpDraftOutput>> = (0..lanes.len()).map(|_| None).collect();
    if group.len() >= 2 && group.len() <= cb.max_rows {
        let n_k: Vec<usize> = group.iter().map(|&i| lanes[i].k.min(lanes[i].state.max_n)).collect();
        for &i in &group {
            let l = &mut lanes[i];
            l.state.penalty.begin_window(l.emitted);
            let seed = l.last_committed as i32;
            gpu.hip.memcpy_htod(&l.state.mtp_token_chain.buf, &seed.to_ne_bytes())?;
        }
        let out_w = &weights.output;
        let k_max = n_k.iter().copied().max().unwrap_or(0);
        for k in 0..k_max {
            let act: Vec<usize> = (0..group.len()).filter(|&g| n_k[g] > k).collect();
            for (col, &g) in act.iter().enumerate() {
                let cur_pos = lanes[group[g]].cur_pos;
                let st = &mut *lanes[group[g]].state;
                let token_slot = st.mtp_token_chain.sub_offset(k, 1);
                embed_device_token_into(gpu, weights, &st.mtp_token_embed, &token_slot, dim)?;
                let prev = if k == 0 { st.prev_hidden.sub_offset(0, dim) } else { st.mtp_t_outs.sub_offset((k - 1) * dim, dim) };
                mtp_head::mtp_head_forward_block_only(
                    gpu,
                    head,
                    &st.mtp_scratch,
                    &mut st.mtp_kv,
                    0,
                    &prev,
                    Some(&st.mtp_token_embed),
                    cur_pos + k,
                    weights,
                )?;
                gpu.rmsnorm_f32(&st.mtp_scratch.t_mtp_out, &head.weights.shared_head_norm, &st.mtp_scratch.tmp, head.config.rms_norm_eps)?;
                // weight_gemv's MQ4 input rotate, into this lane's column.
                let col_x = cb.rot.sub_offset(col * dim, dim);
                llama::rotate_x_mq_for(gpu, out_w, &st.mtp_scratch.tmp, &col_x, out_w.k)?;
            }
            let mut c0 = 0usize;
            while c0 < act.len() {
                let c1 = (c0 + rdna_compute::pm_xbatch::PM_XBATCH_MAX).min(act.len());
                hipfire_dispatch::ops::pm_xbatch::multirow_r2(
                    gpu,
                    &out_w.buf,
                    &cb.rot.sub_offset(c0 * dim, (c1 - c0) * dim),
                    &cb.logits.sub_offset(c0 * vocab, (c1 - c0) * vocab),
                    vocab,
                    dim,
                    c1 - c0,
                )?;
                c0 = c1;
            }
            for (col, &g) in act.iter().enumerate() {
                let st = &mut *lanes[group[g]].state;
                let argmax_view = st.mtp_lm_argmax.sub_offset(0, 1);
                let logits = cb.logits.sub_offset(col * vocab, vocab);
                gpu.argmax_token_chain_f32(&logits, &argmax_view, &st.mtp_token_chain, None, vocab, k + 1)?;
                if k + 1 < n_k[g] {
                    gpu.memcpy_dtod_at_auto(&st.mtp_t_outs.buf, k * dim * 4, &st.mtp_scratch.t_mtp_out.buf, 0, dim * 4)?;
                }
            }
        }
        for (g, &i) in group.iter().enumerate() {
            let l = &mut lanes[i];
            let n = n_k[g];
            let mut host = vec![0i32; n];
            {
                let bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, n * 4) };
                gpu.hip.memcpy_dtoh(bytes, &l.state.mtp_token_chain.sub_offset(1, n).buf)?;
            }
            out[i] = Some(MtpDraftOutput {
                candidates: host.into_iter().map(|t| t as u32).collect(),
                drafts_generated: n,
                chain_truncated: false,
                use_sampling: false,
                sampling: l.state.sampling,
                draft_probs: Vec::new(),
                draft_softmaxes: Vec::new(),
                use_device_token_chain: true,
                cur_pos: l.cur_pos,
                last_committed: l.last_committed,
            });
        }
    }
    for (i, l) in lanes.iter_mut().enumerate() {
        if out[i].is_none() {
            out[i] = Some(mtp_cb_draft(gpu, weights, config, head, l.state, l.cur_pos, l.last_committed, l.emitted, l.k)?);
        }
    }
    Ok(out.into_iter().map(|d| d.expect("every lane drafted")).collect())
}

/// One request's window in a CB cycle: its trunk owners, its MTP drafter
/// state, and the `mtp_step` arguments.
pub struct MtpCbLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut MtpSpecState,
    pub cur_pos: usize,
    pub last_committed: u32,
    /// Emitted history ending in `last_committed` (penalty window).
    pub emitted: &'a [u32],
    pub eos_token_id: u32,
    /// Per-window draft budget (`MtpSpeculator` k, already clamped).
    pub k: usize,
}

/// Synced phase wall times of one cycle (zero unless
/// `HIPFIRE_MTP_PHASE_TIMING=1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MtpCbTiming {
    pub draft_us: f64,
    pub verify_us: f64,
    pub accept_us: f64,
    /// Lanes that took the shared verify / the singleton fallback.
    pub shared_lanes: usize,
    pub singleton_lanes: usize,
}

/// Draft phase of one lane's window: the singleton drafter (penalty window
/// rebuilt from `emitted`, then `mtp_draft_phase_inner`). Touches only the
/// lane's MTP state (head KV rows at and past `cur_pos`, token chain), so a
/// draft that is dropped before verify leaves committed state unchanged.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    emitted: &[u32],
    k: usize,
) -> HipResult<MtpDraftOutput> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    state.penalty.begin_window(emitted);
    mtp_draft_phase_inner(gpu, weights, config, head, state, cur_pos, last_committed, k, false)
}

/// One drafted lane between verify and accept.
pub struct MtpCbVerifyLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut MtpSpecState,
    pub draft: &'a MtpDraftOutput,
    pub eos_token_id: u32,
}

/// A lane after the verify phase.
pub enum MtpCbVerified {
    /// Trunk verified on the shared body, head logits/hidden in the lane's
    /// state: [`mtp_cb_accept`] commits it.
    Pending,
    /// A window the shared trunk cannot take (single verify row) ran the
    /// whole singleton verify/accept/repair; its result.
    Done(MtpSpecResult),
}

/// Can the shared trunk verify this drafted window exactly (the singleton
/// verifies it on the batched-prefill body with a rollback tape)? `pack_cap`
/// is the chunk's packing cap ([`multi_chunk_pack_cap`]); a window never
/// exceeds one request's 63 rows.
fn shared_verify_ok(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config, pack_cap: usize, lane: &MtpCbVerifyLane<'_>) -> bool {
    let n = lane.draft.n_verify();
    n >= 2
        && n <= pack_cap.min(MULTI_CHUNK_MAX_LANE_ROWS)
        && n <= lane.state.trunk_gdn_tape.max_n
        && qwen35::prefill_batch_pbs_eligible(weights, config, lane.dn_state, n, gpu.arch.as_str(), true)
}

/// The verify head of a chunk over 64+ rows (a wide chunk): rotate, then the
/// exact singleton-WMMA MQ4G256V2 lm-head (zero + exact residual), whose rows
/// equal the singleton head's. Chunks below 64 rows keep
/// [`mtp_trunk_verify_lm_head`]. Wide admission guarantees an MQ4G256V2 head;
/// any other dtype is an error, never a product fallback.
#[allow(clippy::too_many_arguments)]
fn mtp_wide_verify_lm_head(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    hidden: &GpuTensor,
    rot: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
) -> HipResult<()> {
    if w_out.gpu_dtype != DType::MQ4G256V2 {
        return Err(hip_bridge::HipError::new(0, "mtp_cb_verify: wide verify head needs an MQ4G256V2 lm_head"));
    }
    let rot = rot.sub_offset(0, rows * w_out.k);
    llama::rotate_x_mq_batched_for(gpu, w_out, hidden, &rot, w_out.k, rows)?;
    gpu.gemm_mq4g256v2_lmhead_verify_exact(&w_out.buf, &rot, logits, w_out.m, w_out.k, rows)
}

/// Verify phase over drafted lanes: DeltaNet snapshot, ONE shared trunk
/// forward (chunks of whole lanes within the packing cap: `cb.max_rows`, held
/// to 63 unless the wide route is admitted for the target) and ONE verify
/// head per chunk; each lane's hidden/logit rows land in its own state.
/// Writes trunk KV/DeltaNet (a failure here leaves the lanes untrusted).
pub fn mtp_cb_verify(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbVerifyLane<'_>],
) -> HipResult<Vec<MtpCbVerified>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let pack_cap = multi_chunk_pack_cap(gpu, weights, config, cb.max_rows);
    let shared: Vec<bool> = lanes.iter().map(|l| shared_verify_ok(gpu, weights, config, pack_cap, l)).collect();
    let mut out: Vec<MtpCbVerified> = (0..lanes.len()).map(|_| MtpCbVerified::Pending).collect();
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut idx: Vec<usize> = (0..lanes.len()).filter(|&i| shared[i]).collect();
    while !idx.is_empty() {
        let mut rows = 0usize;
        let mut take = 0usize;
        while take < idx.len() && rows + lanes[idx[take]].draft.n_verify() <= pack_cap {
            rows += lanes[idx[take]].draft.n_verify();
            take += 1;
        }
        let chunk: Vec<usize> = idx.drain(..take).collect();
        let verify_tokens: Vec<Vec<u32>> = chunk.iter().map(|&i| lanes[i].draft.verify_tokens()).collect();
        for &i in &chunk {
            let lane = &mut lanes[i];
            lane.state.trunk_snap.save_from(lane.dn_state, gpu)?;
        }
        {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(chunk.len());
            let mut rest: &mut [MtpCbVerifyLane<'_>] = lanes;
            let mut base = 0usize;
            for (j, &i) in chunk.iter().enumerate() {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(i - base);
                let (lane, after) = tail.split_first_mut().expect("lane index in range");
                rest = after;
                base = i + 1;
                reqs.push(MultiChunkRequest {
                    tokens: &verify_tokens[j],
                    start_pos: lane.draft.cur_pos,
                    kv_cache: &mut *lane.kv_cache,
                    dn_state: &mut *lane.dn_state,
                    gdn_tape: Some(&lane.state.trunk_gdn_tape),
                    fusion: crate::qwen35::DflashFusionCtx::Off,
                    hidden_rb: None,
                });
            }
            forward_prefill_batch_multi(gpu, weights, config, scratch, &cb.trunk, &mut reqs, Some(&cb.hidden))?;
        }
        let logits = cb.logits.sub_offset(0, rows * vocab);
        if rows > MULTI_CHUNK_PRODUCT_MAX_ROWS {
            mtp_wide_verify_lm_head(gpu, &weights.output, &cb.hidden, &cb.rot, &logits, rows)?;
        } else {
            mtp_trunk_verify_lm_head(gpu, &weights.output, &cb.hidden, &cb.rot, &logits, rows, dim, vocab)?;
        }
        let mut row = 0usize;
        for &i in &chunk {
            let n = lanes[i].draft.n_verify();
            let st = &mut *lanes[i].state;
            gpu.memcpy_dtod_at_auto(&st.verify_hidden.buf, 0, &cb.hidden.buf, row * dim * 4, n * dim * 4)?;
            gpu.memcpy_dtod_at_auto(&st.verify_logits.buf, 0, &cb.logits.buf, row * vocab * 4, n * vocab * 4)?;
            row += n;
        }
    }
    for i in (0..lanes.len()).filter(|&i| !shared[i]) {
        let lane = &mut lanes[i];
        let d = lane.draft;
        out[i] = MtpCbVerified::Done(mtp_shared_verify_accept_rollback_inner(
            gpu,
            weights,
            config,
            lane.kv_cache,
            lane.dn_state,
            scratch,
            lane.state,
            d.cur_pos,
            d.last_committed,
            lane.eos_token_id,
            &d.candidates,
            d.drafts_generated,
            d.chain_truncated,
            false,
            d.use_sampling,
            d.sampling,
            &d.draft_probs,
            &d.draft_softmaxes,
            false,
            d.use_device_token_chain,
        )?);
    }
    Ok(out)
}

/// Accept/repair phase of one [`MtpCbVerified::Pending`] lane: the
/// singleton consumer over its precomputed head logits (greedy/sampled
/// accept, prev_hidden capture, DeltaNet restore + tape replay on a
/// partial accept).
pub fn mtp_cb_accept(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    lane: &mut MtpCbVerifyLane<'_>,
) -> HipResult<MtpSpecResult> {
    let d = lane.draft;
    let verify_tokens = d.verify_tokens();
    mtp_accept_and_rollback(
        gpu,
        weights,
        config,
        lane.kv_cache,
        lane.dn_state,
        scratch,
        lane.state,
        d.n_verify(),
        &verify_tokens,
        &d.candidates,
        d.drafts_generated,
        d.chain_truncated,
        d.use_sampling,
        d.sampling,
        &d.draft_probs,
        &d.draft_softmaxes,
        false,
        d.use_device_token_chain,
        true,
        d.cur_pos,
        lane.eos_token_id,
        true,
    )
}

/// One MTP window per lane, verify shared; results in lane order. Each
/// lane's committed tokens and state equal its singleton window's.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_cycle(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbLane<'_>],
) -> HipResult<(Vec<MtpSpecResult>, MtpCbTiming)> {
    let timed = mtp_phase_timing_enabled();
    let mut timing = MtpCbTiming::default();
    let mark = |gpu: &Gpu, slot: &mut f64, t: &mut Instant| -> HipResult<()> {
        if timed {
            gpu.hip.device_synchronize()?;
            *slot += t.elapsed().as_secs_f64() * 1e6;
            *t = Instant::now();
        }
        Ok(())
    };
    let mut t = Instant::now();
    let drafts = {
        let mut dl: Vec<MtpCbDraftLane<'_>> = lanes
            .iter_mut()
            .map(|l| MtpCbDraftLane {
                state: &mut *l.state,
                cur_pos: l.cur_pos,
                last_committed: l.last_committed,
                emitted: l.emitted,
                k: l.k,
            })
            .collect();
        mtp_cb_draft_batched(gpu, weights, config, head, cb, &mut dl)?
    };
    mark(gpu, &mut timing.draft_us, &mut t)?;
    let mut vlanes: Vec<MtpCbVerifyLane<'_>> = lanes
        .iter_mut()
        .zip(&drafts)
        .map(|(l, d)| MtpCbVerifyLane {
            kv_cache: &mut *l.kv_cache,
            dn_state: &mut *l.dn_state,
            state: &mut *l.state,
            draft: d,
            eos_token_id: l.eos_token_id,
        })
        .collect();
    let verified = mtp_cb_verify(gpu, weights, config, scratch, cb, &mut vlanes)?;
    mark(gpu, &mut timing.verify_us, &mut t)?;
    let mut results = Vec::with_capacity(vlanes.len());
    for (lane, v) in vlanes.iter_mut().zip(verified) {
        results.push(match v {
            MtpCbVerified::Done(r) => {
                timing.singleton_lanes += 1;
                r
            }
            MtpCbVerified::Pending => {
                timing.shared_lanes += 1;
                mtp_cb_accept(gpu, weights, config, scratch, lane)?
            }
        });
    }
    mark(gpu, &mut timing.accept_us, &mut t)?;
    Ok((results, timing))
}
