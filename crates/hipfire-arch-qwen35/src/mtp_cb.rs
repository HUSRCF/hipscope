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
use crate::qwen35::prefill::multi::{forward_prefill_batch_multi, MultiChunkRequest, MultiChunkScratch, MULTI_CHUNK_MAX_ROWS};

/// Shared verify buffers for one CB engine: trunk scratch, post-norm hidden
/// rows, head rotation scratch and head logits, all `max_rows` rows.
pub struct MtpCbScratch {
    pub max_rows: usize,
    trunk: MultiChunkScratch,
    hidden: GpuTensor,
    rot: GpuTensor,
    logits: GpuTensor,
}

impl MtpCbScratch {
    /// `max_rows` is clamped to [`MULTI_CHUNK_MAX_ROWS`]; a cycle with more
    /// rows runs in several shared chunks.
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = max_rows.clamp(2, MULTI_CHUNK_MAX_ROWS);
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
/// verifies it on the batched-prefill body with a rollback tape)?
fn shared_verify_ok(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config, cb: &MtpCbScratch, lane: &MtpCbVerifyLane<'_>) -> bool {
    let n = lane.draft.n_verify();
    n >= 2
        && n <= cb.max_rows
        && n <= lane.state.trunk_gdn_tape.max_n
        && qwen35::prefill_batch_pbs_eligible(weights, config, lane.dn_state, n, gpu.arch.as_str(), true)
}

/// Verify phase over drafted lanes: DeltaNet snapshot, ONE shared trunk
/// forward (chunks of whole lanes within `cb.max_rows`) and ONE verify
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
    let shared: Vec<bool> = lanes.iter().map(|l| shared_verify_ok(gpu, weights, config, cb, l)).collect();
    let mut out: Vec<MtpCbVerified> = (0..lanes.len()).map(|_| MtpCbVerified::Pending).collect();
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut idx: Vec<usize> = (0..lanes.len()).filter(|&i| shared[i]).collect();
    while !idx.is_empty() {
        let mut rows = 0usize;
        let mut take = 0usize;
        while take < idx.len() && rows + lanes[idx[take]].draft.n_verify() <= cb.max_rows {
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
                });
            }
            forward_prefill_batch_multi(gpu, weights, config, scratch, &cb.trunk, &mut reqs, Some(&cb.hidden))?;
        }
        let logits = cb.logits.sub_offset(0, rows * vocab);
        mtp_trunk_verify_lm_head(gpu, &weights.output, &cb.hidden, &cb.rot, &logits, rows, dim, vocab)?;
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
    let mut drafts = Vec::with_capacity(lanes.len());
    for lane in lanes.iter_mut() {
        drafts.push(mtp_cb_draft(
            gpu,
            weights,
            config,
            head,
            lane.state,
            lane.cur_pos,
            lane.last_committed,
            lane.emitted,
            lane.k,
        )?);
    }
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
