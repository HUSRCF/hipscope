// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched DFlash lanes of the VMM continuous-batching store.
//!
//! [`DflashVmmLaneState`] is the persistent private draft/verify state of one
//! continuous-batching lane: the singleton [`DflashState`] owners plus the
//! request-local cursors the singleton [`DflashSpeculator`] keeps beside it.
//! It is built by consuming a [`DflashLaneSnapshot`] (`take_vmm_lane`, which
//! transfers every owner and cursor losslessly) or cold, from a lane
//! `DflashState` (`new_dflash_lane_state`) after the lane prompt prefill.
//!
//! The in-crate halves of the shared DFlash verify:
//!
//! * [`dflash_lane_draft`] — the singleton greedy draft call of
//!   `spec_step_dflash` (`draft_dflash_block_rank`) on a lane's own draft
//!   state; returns `[seed, candidates..]`.
//! * [`dflash_cb_head_argmax`] — the singleton DFlash verify head
//!   (`dflash_enqueue_verify_lm_head_argmax` + one packed i32 D2H) over the
//!   first `rows` rows of a shared [`VerifyScratch`].
//! * [`dflash_cb_verify`] — whole lanes packed into `<= 63`-row trunk chunks
//!   (`forward_prefill_batch_multi`, ChainVerify fusion, each lane's own
//!   hidden ring and tape at offset 0), ONE shared head + argmax + D2H per
//!   chunk; each lane keeps its own picks until the greedy accept consumer.

use crate::dflash_spec::{
    DflashCheckpointPolicy, DflashLaneSnapshot, DflashState, DflashWindowMark,
};
use crate::qwen35::prefill::multi::{
    forward_prefill_batch_multi, pack_whole_lanes, MultiChunkRequest, MultiChunkScratch, MULTI_CHUNK_MAX_ROWS,
};
use crate::qwen35::{self, DeltaNetState, DflashFusionCtx};
use crate::qwen35::{Qwen35Config, Qwen35Weights};
use crate::speculative::{
    dflash_download_verify_argmax, dflash_enqueue_verify_lm_head_argmax, draft_dflash_block_rank,
    DeltaNetSnapshot, DflashCbDraft, VerifyScratch,
};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::spec::{request_rng_state, SpecRequestConfig};
use rdna_compute::Gpu;
use std::ops::Range;

/// The singleton's verify block for remaining output budget `max_emit`
/// (`E > 0`): `B = min(block_size, max(E, 2))`. Even `E = 1` verifies two rows
/// (the accept clamp `max_accept = E - 1` then keeps zero drafts).
pub fn dflash_block_for_emit(block_size: usize, max_emit: usize) -> usize {
    block_size.min(max_emit.max(2))
}

/// Private DFlash state of one lane (see the module docs).
pub struct DflashVmmLaneState {
    /// The singleton owners: draft scratch/weights, hidden ring, verify
    /// scratch, target DeltaNet snapshot, GDN tape.
    pub df: DflashState,
    /// Committed target position at capture; the store keeps it equal to the
    /// request's position after every commit.
    pub position: usize,
    pub rng_state: u64,
    pub sample_temp: f32,
    pub sample_top_p: f32,
    pub sample_top_k: usize,
    pub sample_cactus: f32,
    /// Divergent-render checkpoints `(position, snapshot)`.
    pub checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub policy: DflashCheckpointPolicy,
    /// Pre-window mark of the last committed window (terminal-prefix repair).
    pub last_window: Option<DflashWindowMark>,
    /// Target picks of this lane's verified window (`B` ids), kept from the
    /// shared head until the greedy accept consumer runs.
    pub picks: Vec<u32>,
    /// Remaining output budget `E` installed by the driver for the next
    /// window (`0` = not installed; a window needs `E >= 1`).
    pub max_emit: usize,
    /// `picks` hold the verify outcome of the lane's planned window.
    pub verified: bool,
    /// Tokens the last committed window emitted (`committed[1..]`).
    pub last_emit: Vec<u32>,
    /// The request's history length before the last window's emit was
    /// appended (terminal-prefix repair truncates back to it).
    pub last_history_len: usize,
}

// SAFETY: a lane lives inside the store (`Qwen35Bundle::vmm_store`, which must
// be `Send` to ride the model across threads) and, like every GPU owner there
// (`DeviceBuffer` is `Send`), is used only by the thread holding the model.
// The one non-`Send` part of a `DflashState` is its retained-PM4 route, which a
// lane keeps explicitly disabled; the shared draft weights `Arc` is a counted
// reference to the one resident set, read only through GPU calls under that
// same single owner.
unsafe impl Send for DflashVmmLaneState {}

impl DflashVmmLaneState {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        df: DflashState,
        position: usize,
        rng_state: u64,
        sample_temp: f32,
        sample_top_p: f32,
        sample_top_k: usize,
        sample_cactus: f32,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        last_window: Option<DflashWindowMark>,
    ) -> Self {
        let block = df.block_size;
        Self {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
            picks: Vec::with_capacity(block),
            max_emit: 0,
            verified: false,
            last_emit: Vec::with_capacity(block),
            last_history_len: 0,
        }
    }

    /// Consume `snapshot`, moving every owner and cursor into the lane.
    /// Read `snapshot.rows()` before calling.
    pub fn from_snapshot(snapshot: DflashLaneSnapshot) -> Self {
        let DflashLaneSnapshot {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        } = snapshot;
        Self::assemble(
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        )
    }

    /// A lane that just finished its cold prompt prefill (`position` prompt
    /// rows committed): request sampling configured exactly as
    /// `DflashSpeculator::configure_request`, no completed window yet.
    pub fn cold(
        df: DflashState,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        position: usize,
        request: &SpecRequestConfig,
    ) -> Self {
        Self::assemble(
            df,
            position,
            request_rng_state(request.rng_seed),
            request.temp,
            request.top_p,
            request.top_k_cut(),
            request.cactus_delta,
            checkpoints,
            policy,
            None,
        )
    }

    /// Free every owned GPU object exactly once (shared draft weights go
    /// through `DflashState::free_gpu`'s final-owner release).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashVmmLaneState {
            df,
            position: _,
            rng_state: _,
            sample_temp: _,
            sample_top_p: _,
            sample_top_k: _,
            sample_cactus: _,
            checkpoints,
            policy: _,
            last_window: _,
            picks: _,
            max_emit: _,
            verified: _,
            last_emit: _,
            last_history_len: _,
        } = self;
        df.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Greedy chain draft of one lane window of `b` rows at `position` with
/// pending `seed`: exactly the singleton `spec_step_dflash` greedy fast-path
/// call (`ctx_slice=None`, dense target so no draft FFN graph). Mutates the
/// lane's draft scratch (projection watermarks, draft K/V) as the singleton
/// draft does. Returns `[seed, candidates..]` (`b` tokens).
pub fn dflash_lane_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    compact_offset: i32,
    df: &mut DflashState,
    position: usize,
    seed: u32,
    b: usize,
) -> HipResult<Vec<u32>> {
    if config.num_experts > 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash_lane_draft: MoE targets use the draft FFN graph route (not reproduced)",
        ));
    }
    if b < 2 {
        return Err(hip_bridge::HipError::new(0, "dflash_lane_draft: block size must be >= 2"));
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let mut block: Vec<u32> = vec![df.draft_config.mask_token_id; b];
    block[0] = seed;
    draft_dflash_block_rank(
        gpu,
        weights,
        &df.draft_weights,
        &df.draft_config,
        &mut df.draft_scratch,
        &df.verify_scratch,
        &df.target_hidden_host,
        &block,
        position,
        compact_offset,
        None,
        false,
        seed,
        config.vocab_size,
    )
}

/// Shared verify head: the singleton DFlash dispatcher (rotate + batched
/// lm-head + GPU argmax) over `vs.final_hidden[0..rows]`, then one packed i32
/// D2H. Logits land in `vs.logits[0..rows]`, picks in `vs.argmax[0..rows]`.
pub fn dflash_cb_head_argmax(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    vs: &VerifyScratch,
    rows: usize,
) -> HipResult<Vec<u32>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let final_hidden = vs.final_hidden.sub_offset(0, rows * config.dim);
    dflash_enqueue_verify_lm_head_argmax(
        gpu,
        &weights.output,
        &final_hidden,
        vs,
        rows,
        config.vocab_size,
    )?;
    dflash_download_verify_argmax(gpu, vs, rows)
}

/// Shared verify buffers of one DFlash CB engine: the trunk row scratch, ONE
/// head scratch (`final_hidden`/`logits`/`rot`/`argmax`, `max_rows` rows,
/// `hidden_k = dim.next_power_of_two()` like a singleton lane's), and the
/// reusable chunk tables. Allocated once; a cycle allocates no GPU memory.
pub struct DflashCbScratch {
    max_rows: usize,
    trunk: MultiChunkScratch,
    head: VerifyScratch,
    rows: Vec<usize>,
    ranges: Vec<Range<usize>>,
    /// Diagnostic: also copy each lane's head rows (post-norm hidden, logits,
    /// argmax) into its own `df.verify_scratch` (the oracle's proof arrays).
    /// Greedy production leaves this off.
    pub keep_lane_rows: bool,
}

impl DflashCbScratch {
    /// `max_rows` is clamped to `2..=63`: one trunk chunk's rows.
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = max_rows.clamp(2, MULTI_CHUNK_MAX_ROWS);
        let trunk = MultiChunkScratch::new(gpu, config, max_rows)?;
        let head = match VerifyScratch::new(gpu, max_rows, config.dim, config.vocab_size, config.dim.next_power_of_two()) {
            Ok(h) => h,
            Err(e) => {
                let _ = trunk.free_gpu(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            max_rows,
            trunk,
            head,
            rows: Vec::with_capacity(max_rows),
            ranges: Vec::with_capacity(max_rows),
            keep_lane_rows: false,
        })
    }

    /// Rows one trunk chunk can hold (`<= 63`).
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let DflashCbScratch { max_rows: _, trunk, head, rows: _, ranges: _, keep_lane_rows: _ } = self;
        let r = trunk.free_gpu(gpu);
        head.free_gpu(gpu);
        r
    }
}

/// One drafted lane between verify and accept: its trunk owners, its private
/// DFlash state and the window it drafted.
pub struct DflashCbVerifyLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut DflashVmmLaneState,
    pub draft: &'a DflashCbDraft,
}

/// Verify phase over drafted lanes: the pre-verify DeltaNet snapshot of each
/// lane (right after its draft, as the singleton), then whole lanes in order
/// through `forward_prefill_batch_multi` chunks of at most `cb.max_rows()`
/// rows (C1 `[16]`, C3 `[48]`, C4 `[48,16]`, C8 `[48,48,32]`) with each
/// lane's own ChainVerify fusion, hidden ring and tape (offset 0); per chunk
/// ONE head + GPU argmax and ONE D2H of `4 * rows` bytes. Each lane's picks
/// land in `state.picks` (`state.verified = true`) for
/// `dflash_greedy_accept_commit_parts`.
///
/// Everything refusable is checked before the first snapshot is saved or row
/// launched (capture/recording, block sizes, ring/tape capacity, batched-body
/// eligibility). A later failure leaves the lanes of executed chunks
/// advanced: the caller poisons the participants.
pub fn dflash_cb_verify(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbVerifyLane<'_>],
) -> HipResult<()> {
    let refuse = |why: String| Err(HipError::new(0, &format!("dflash_cb_verify: {why}")));
    if lanes.is_empty() {
        return Ok(());
    }
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported".into());
    }
    if config.num_experts > 0 {
        return refuse("MoE targets are not admitted".into());
    }
    cb.rows.clear();
    for (i, lane) in lanes.iter().enumerate() {
        let b = lane.draft.verify_tokens.len();
        if lane.draft.verify_tokens.first() != Some(&lane.draft.seed) {
            return refuse(format!("lane {i}: window does not start with its seed"));
        }
        if b > cb.max_rows {
            return refuse(format!("lane {i}: {b} rows > chunk capacity {}", cb.max_rows));
        }
        let df = &lane.state.df;
        if b > df.gdn_tape.max_n || b > df.hidden_rb.max_batch || b > df.hidden_rb.max_positions {
            return refuse(format!("lane {i}: {b} rows exceed its tape/ring staging"));
        }
        if !qwen35::prefill_batch_pbs_eligible(weights, config, lane.dn_state, b, gpu.arch.as_str(), true) {
            return refuse(format!("lane {i}: {b} rows not eligible for the batched prefill body"));
        }
        cb.rows.push(b);
    }
    pack_whole_lanes(&cb.rows, cb.max_rows, &mut cb.ranges).map_err(|e| HipError::new(0, &format!("dflash_cb_verify: {e}")))?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let (dim, vocab) = (config.dim, config.vocab_size);
    for ci in 0..cb.ranges.len() {
        let range = cb.ranges[ci].clone();
        let rows: usize = cb.rows[range.clone()].iter().sum();
        for lane in lanes[range.clone()].iter_mut() {
            lane.state.df.target_snap.save_from(lane.dn_state, gpu)?;
        }
        {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(range.len());
            for lane in lanes[range.clone()].iter_mut() {
                let DflashCbVerifyLane { kv_cache, dn_state, state, draft } = lane;
                let df = &mut state.df;
                reqs.push(MultiChunkRequest {
                    tokens: &draft.verify_tokens,
                    start_pos: draft.position,
                    kv_cache: &mut **kv_cache,
                    dn_state: &mut **dn_state,
                    gdn_tape: Some(&df.gdn_tape),
                    fusion: DflashFusionCtx::ChainVerify,
                    hidden_rb: Some(&mut df.hidden_rb),
                });
            }
            forward_prefill_batch_multi(gpu, weights, config, scratch, &cb.trunk, &mut reqs, Some(&cb.head.final_hidden))?;
        }
        let picks = dflash_cb_head_argmax(gpu, weights, config, &cb.head, rows)?;
        let mut row = 0usize;
        for lane in lanes[range].iter_mut() {
            let b = lane.draft.verify_tokens.len();
            let st = &mut *lane.state;
            st.picks.clear();
            st.picks.extend_from_slice(&picks[row..row + b]);
            st.verified = true;
            if cb.keep_lane_rows {
                let v = &st.df.verify_scratch;
                gpu.memcpy_dtod_at_auto(&v.final_hidden.buf, 0, &cb.head.final_hidden.buf, row * dim * 4, b * dim * 4)?;
                gpu.memcpy_dtod_at_auto(&v.logits.buf, 0, &cb.head.logits.buf, row * vocab * 4, b * vocab * 4)?;
                gpu.memcpy_dtod_at_auto(&v.argmax.buf, 0, &cb.head.argmax.buf, row * 4, b * 4)?;
            }
            row += b;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::dflash_block_for_emit;

    #[test]
    fn block_follows_the_singleton_budget_rule() {
        // B = min(16, max(E, 2)); E = 1 still verifies two rows.
        let b: Vec<usize> = [1, 2, 3, 15, 16, 17, 1000].iter().map(|&e| dflash_block_for_emit(16, e)).collect();
        assert_eq!(b, vec![2, 2, 3, 15, 16, 16, 16]);
    }
}
