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
use crate::qwen35::prefill::multi::{forward_prefill_batch_multi, MultiChunkRequest, MULTI_CHUNK_MAX_ROWS};

/// Shared verify buffers for one CB engine: trunk scratch, post-norm hidden
/// rows, head rotation scratch and head logits, all `max_rows` rows.
pub struct MtpCbScratch {
    pub max_rows: usize,
    pbs: PrefillBatchScratch,
    hidden: GpuTensor,
    rot: GpuTensor,
    logits: GpuTensor,
}

impl MtpCbScratch {
    /// `max_rows` is clamped to [`MULTI_CHUNK_MAX_ROWS`]; a cycle with more
    /// rows runs in several shared chunks.
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = max_rows.clamp(2, MULTI_CHUNK_MAX_ROWS);
        let pbs = PrefillBatchScratch::new(gpu, config, max_rows)?;
        let alloc = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], DType::F32);
        let hidden = alloc(gpu, max_rows * config.dim)?;
        let rot = alloc(gpu, max_rows * config.dim)?;
        let logits = alloc(gpu, max_rows * config.vocab_size)?;
        Ok(Self { max_rows, pbs, hidden, rot, logits })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        self.pbs.free_gpu(gpu)?;
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

/// One MTP window per lane, verify shared; results in lane order. Each
/// lane's committed tokens and state equal its singleton window's.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cb_cycle(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &mut Qwen35Scratch,
    head: &Qwen35MtpHead,
    cb: &MtpCbScratch,
    lanes: &mut [MtpCbLane<'_>],
) -> HipResult<(Vec<MtpSpecResult>, MtpCbTiming)> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
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

    // 1. Draft (per request, the singleton drafter).
    let mut drafts = Vec::with_capacity(lanes.len());
    for lane in lanes.iter_mut() {
        lane.state.penalty.begin_window(lane.emitted);
        drafts.push(mtp_draft_phase_inner(
            gpu,
            weights,
            config,
            head,
            lane.state,
            lane.cur_pos,
            lane.last_committed,
            lane.k,
            false,
        )?);
    }
    mark(gpu, &mut timing.draft_us, &mut t)?;

    let mut results: Vec<Option<MtpSpecResult>> = (0..lanes.len()).map(|_| None).collect();
    // A lane takes the shared trunk iff the singleton would verify it on the
    // batched-prefill body with a rollback tape.
    let shared: Vec<bool> = drafts
        .iter()
        .zip(lanes.iter())
        .map(|(d, lane)| {
            let n = d.n_verify();
            n >= 2
                && n <= cb.max_rows
                && n <= lane.state.trunk_gdn_tape.max_n
                && qwen35::prefill_batch_pbs_eligible(weights, config, lane.dn_state, n, gpu.arch.as_str(), true)
        })
        .collect();

    // 2-4. Shared verify, in chunks of whole lanes within `cb.max_rows`.
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut idx: Vec<usize> = (0..lanes.len()).filter(|&i| shared[i]).collect();
    timing.shared_lanes = idx.len();
    timing.singleton_lanes = lanes.len() - idx.len();
    while !idx.is_empty() {
        let mut rows = 0usize;
        let mut take = 0usize;
        while take < idx.len() && rows + drafts[idx[take]].n_verify() <= cb.max_rows {
            rows += drafts[idx[take]].n_verify();
            take += 1;
        }
        let chunk: Vec<usize> = idx.drain(..take).collect();
        let verify_tokens: Vec<Vec<u32>> = chunk.iter().map(|&i| drafts[i].verify_tokens()).collect();
        for &i in &chunk {
            let lane = &mut lanes[i];
            lane.state.trunk_snap.save_from(lane.dn_state, gpu)?;
        }
        {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(chunk.len());
            let mut rest: &mut [MtpCbLane<'_>] = lanes;
            let mut base = 0usize;
            for (j, &i) in chunk.iter().enumerate() {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(i - base);
                let (lane, after) = tail.split_first_mut().expect("lane index in range");
                rest = after;
                base = i + 1;
                reqs.push(MultiChunkRequest {
                    tokens: &verify_tokens[j],
                    start_pos: lane.cur_pos,
                    kv_cache: &mut *lane.kv_cache,
                    dn_state: &mut *lane.dn_state,
                    gdn_tape: Some(&lane.state.trunk_gdn_tape),
                });
            }
            forward_prefill_batch_multi(gpu, weights, config, scratch, &cb.pbs, &mut reqs, Some(&cb.hidden))?;
        }
        let logits = cb.logits.sub_offset(0, rows * vocab);
        mtp_trunk_verify_lm_head(gpu, &weights.output, &cb.hidden, &cb.rot, &logits, rows, dim, vocab)?;
        let mut row = 0usize;
        for &i in &chunk {
            let n = drafts[i].n_verify();
            let st = &mut *lanes[i].state;
            gpu.memcpy_dtod_at_auto(&st.verify_hidden.buf, 0, &cb.hidden.buf, row * dim * 4, n * dim * 4)?;
            gpu.memcpy_dtod_at_auto(&st.verify_logits.buf, 0, &cb.logits.buf, row * vocab * 4, n * vocab * 4)?;
            row += n;
        }
        mark(gpu, &mut timing.verify_us, &mut t)?;
        for (j, &i) in chunk.iter().enumerate() {
            let d = &drafts[i];
            let lane = &mut lanes[i];
            results[i] = Some(mtp_accept_and_rollback(
                gpu,
                weights,
                config,
                lane.kv_cache,
                lane.dn_state,
                scratch,
                lane.state,
                d.n_verify(),
                &verify_tokens[j],
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
            )?);
        }
        mark(gpu, &mut timing.accept_us, &mut t)?;
    }

    // Singleton fallback lanes.
    for i in (0..lanes.len()).filter(|&i| !shared[i]) {
        let d = &drafts[i];
        let lane = &mut lanes[i];
        results[i] = Some(mtp_shared_verify_accept_rollback_inner(
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
    mark(gpu, &mut timing.accept_us, &mut t)?;
    Ok((results.into_iter().map(|r| r.expect("every lane verified")).collect(), timing))
}
