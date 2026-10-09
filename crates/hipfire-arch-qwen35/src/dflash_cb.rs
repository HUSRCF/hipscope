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
//!   first `rows` rows of a shared [`VerifyScratch`]; a head over 64+ rows
//!   (a wide chunk) runs the exact singleton-WMMA head, below 64 the product
//!   head, byte-identical per row.
//! * [`dflash_cb_verify`] — whole lanes packed into trunk chunks of at most
//!   the effective cap (`<= 63` rows; up to 128 with
//!   `HIPFIRE_CB_VERIFY_CHUNK128` on an admitted dense MQ4G256V2 gfx1201
//!   target) (`forward_prefill_batch_multi`, ChainVerify fusion, each lane's
//!   own hidden ring and tape at offset 0), ONE shared head + argmax + D2H
//!   per chunk; each lane keeps its own picks until the greedy accept consumer.

use crate::dflash_spec::{
    DflashCheckpointPolicy, DflashLaneSnapshot, DflashState, DflashWindowMark,
};
use crate::qwen35::prefill::multi::{
    forward_prefill_batch_multi, multi_chunk_pack_cap, multi_chunk_scratch_rows, pack_whole_lanes, MultiChunkRequest,
    MultiChunkScratch, MULTI_CHUNK_MAX_LANE_ROWS, MULTI_CHUNK_PRODUCT_MAX_ROWS,
};
use crate::qwen35::prefill::DenseBatchMath;
use crate::qwen35::{self, DeltaNetState, DflashFusionCtx};
use crate::qwen35::{Qwen35Config, Qwen35Weights};
use crate::speculative::{
    dflash_download_verify_argmax, dflash_draft_batch_eligible, dflash_enqueue_verify_lm_head_argmax,
    draft_dflash_block_rank, draft_dflash_blocks_batched, DeltaNetSnapshot, DflashCbDraft, DflashDraftLane,
    VerifyScratch, DFLASH_DRAFT_BATCH_MAX_ROWS,
};
use hipfire_runtime::dflash::DflashScratch;
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
/// A head over 64+ rows (only a wide chunk reaches it) uses the exact
/// singleton-WMMA lm-head, whose rows equal the singleton head's; below 64
/// rows the product head is the singleton's own.
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
    let math = if rows > MULTI_CHUNK_PRODUCT_MAX_ROWS { DenseBatchMath::SingletonWmma } else { DenseBatchMath::Product };
    dflash_enqueue_verify_lm_head_argmax(
        gpu,
        &weights.output,
        &final_hidden,
        vs,
        rows,
        config.vocab_size,
        math,
    )?;
    dflash_download_verify_argmax(gpu, vs, rows)
}

/// Shared verify buffers of one DFlash CB engine: the trunk row scratch, ONE
/// head scratch (`final_hidden`/`logits`/`rot`/`argmax`, `max_rows` rows,
/// `hidden_k = dim.next_power_of_two()` like a singleton lane's), and the
/// reusable chunk tables. Allocated once; a cycle allocates no GPU memory.
/// `max_rows` is the EFFECTIVE cap (63, or up to 128 on the wide verify route).
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
    /// Row scratch of the batched draft forward, allocated once on the first
    /// batched chunk (the shared draft activations; lane draft state stays in
    /// each lane).
    draft: Option<DflashScratch>,
    /// Fewest lanes a draft chunk needs to take the batched forward (a lone
    /// lane runs the singleton call; the oracle sets 1 to prove the batched
    /// forward at one lane).
    pub draft_min_lanes: usize,
    /// `(lanes, chunks)` the last [`dflash_cb_draft`] call drafted through the
    /// batched forward (the rest ran the singleton call); evidence for the
    /// oracle that the batched path, not its fallback, produced the drafts.
    pub draft_stats: (usize, usize),
}

impl DflashCbScratch {
    /// `max_rows` is the planned rows; the scratch allocates this target's
    /// effective cap ([`multi_chunk_scratch_rows`]: `2..=63`, or up to 128
    /// only when the wide verify route is admitted for `weights`/`config`).
    /// The scratch stores that cap ([`Self::max_rows`]).
    pub fn new(gpu: &mut Gpu, weights: &Qwen35Weights, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let max_rows = multi_chunk_scratch_rows(gpu, weights, config, max_rows);
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
            draft: None,
            draft_min_lanes: 2,
            draft_stats: (0, 0),
        })
    }

    /// Allocate the batched-draft row scratch now (idempotent), so a serving
    /// engine never allocates inside a step; `dflash_cb_draft` otherwise
    /// allocates it on the first batched chunk.
    pub fn reserve_draft(&mut self, gpu: &mut Gpu, draft_config: &hipfire_runtime::dflash::DflashConfig) -> HipResult<()> {
        if self.draft.is_none() {
            self.draft = Some(DflashScratch::new_with_mq(gpu, draft_config, DFLASH_DRAFT_BATCH_MAX_ROWS, 1, true)?);
        }
        Ok(())
    }

    /// Rows one trunk chunk can hold (the effective cap: `<= 63`, or up to
    /// 128 on the wide verify route).
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let DflashCbScratch {
            max_rows: _,
            trunk,
            head,
            rows: _,
            ranges: _,
            keep_lane_rows: _,
            draft,
            draft_min_lanes: _,
            draft_stats: _,
        } = self;
        if let Some(d) = draft {
            d.free_gpu(gpu);
        }
        let r = trunk.free_gpu(gpu);
        head.free_gpu(gpu);
        r
    }
}

/// One lane of [`dflash_cb_draft`]: its private state and the window to draft
/// (`b` rows at `position` with pending `seed`).
pub struct DflashCbDraftLane<'a> {
    pub state: &'a mut DflashVmmLaneState,
    pub position: usize,
    pub seed: u32,
    pub b: usize,
    pub compact_offset: i32,
}

/// Draft phase over lanes. With `batched` (and the exact-gfx1201 MQ4 v2
/// eligibility of [`dflash_draft_batch_eligible`]), full-block lanes are
/// packed in order into chunks of at most
/// `min(DFLASH_DRAFT_BATCH_MAX_ROWS, cb.max_rows())` rows; each chunk of at
/// least `cb.draft_min_lanes` lanes is ONE draft-model forward and ONE shared
/// lm-head ([`draft_dflash_blocks_batched`]) with every lane keeping its own
/// context rings, positions and hidden context. Every other lane (budget
/// tails, lone lanes, ineligible machines/drafts, `batched == false`) runs
/// the singleton call [`dflash_lane_draft`]. Returns each lane's
/// `[seed, candidates..]`, byte-identical to `dflash_lane_draft` per lane.
pub fn dflash_cb_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbDraftLane<'_>],
    batched: bool,
) -> HipResult<Vec<Vec<u32>>> {
    let n = lanes.len();
    let mut out: Vec<Option<Vec<u32>>> = vec![None; n];
    // chunk id per lane (None = singleton draft).
    let mut chunk_of: Vec<Option<usize>> = vec![None; n];
    let mut n_chunks = 0usize;
    if batched && n > 0 && dflash_draft_batch_eligible(gpu, weights, config, &lanes[0].state.df.draft_weights) {
        let cap = DFLASH_DRAFT_BATCH_MAX_ROWS.min(cb.max_rows);
        let mut rows = 0usize;
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, l) in lanes.iter().enumerate() {
            if l.b < 3 || l.b != l.state.df.block_size || l.b > cap {
                continue;
            }
            if members.last().is_none() || rows + l.b > cap {
                members.push(Vec::new());
                rows = 0;
            }
            members.last_mut().unwrap().push(i);
            rows += l.b;
        }
        for m in members {
            if m.len() >= cb.draft_min_lanes.max(1) {
                for &i in &m {
                    chunk_of[i] = Some(n_chunks);
                }
                n_chunks += 1;
            }
        }
    }
    cb.draft_stats = (chunk_of.iter().flatten().count(), n_chunks);
    if n_chunks > 0 && cb.draft.is_none() {
        let cfg = lanes.iter().zip(&chunk_of).find(|(_, c)| c.is_some()).map(|(l, _)| l.state.df.draft_config.clone());
        if let Some(cfg) = cfg {
            cb.draft = Some(DflashScratch::new_with_mq(gpu, &cfg, DFLASH_DRAFT_BATCH_MAX_ROWS, 1, true)?);
        }
    }
    for c in 0..n_chunks {
        let mut dl: Vec<DflashDraftLane<'_>> = Vec::new();
        let mut idx: Vec<usize> = Vec::new();
        for (i, l) in lanes.iter_mut().enumerate() {
            if chunk_of[i] == Some(c) {
                idx.push(i);
                dl.push(DflashDraftLane {
                    df: &mut l.state.df,
                    position: l.position,
                    seed: l.seed,
                    b: l.b,
                    compact_offset: l.compact_offset,
                });
            }
        }
        let DflashCbScratch { draft, head, .. } = &mut *cb;
        let shared = draft.as_mut().ok_or_else(|| HipError::new(0, "dflash_cb_draft: no draft batch scratch"))?;
        let toks = draft_dflash_blocks_batched(gpu, weights, config.vocab_size, shared, head, &mut dl)?;
        for (i, t) in idx.into_iter().zip(toks) {
            out[i] = Some(t);
        }
    }
    for (i, l) in lanes.iter_mut().enumerate() {
        if out[i].is_none() {
            out[i] = Some(dflash_lane_draft(
                gpu,
                weights,
                config,
                l.compact_offset,
                &mut l.state.df,
                l.position,
                l.seed,
                l.b,
            )?);
        }
    }
    Ok(out.into_iter().map(|t| t.expect("every lane drafted")).collect())
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
/// through `forward_prefill_batch_multi` chunks of at most the packing cap
/// ([`multi_chunk_pack_cap`] of `cb.max_rows()`: C1 `[16]`, C3 `[48]`, C4
/// `[48,16]`, C8 `[48,48,32]` at 63 rows; C8 `[128]` with the wide route on a
/// dense MQ4G256V2 gfx1201 target) with each lane's own ChainVerify fusion,
/// hidden ring and tape (offset 0); per chunk ONE head + GPU argmax and ONE
/// D2H of `4 * rows` bytes. Each lane's picks land in `state.picks`
/// (`state.verified = true`) for `dflash_greedy_accept_commit_parts`.
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
    // Effective packing cap: the scratch capacity, held to the product cap
    // unless the wide exact route is admitted for this target.
    let pack_cap = multi_chunk_pack_cap(gpu, weights, config, cb.max_rows);
    cb.rows.clear();
    for (i, lane) in lanes.iter().enumerate() {
        let b = lane.draft.verify_tokens.len();
        if lane.draft.verify_tokens.first() != Some(&lane.draft.seed) {
            return refuse(format!("lane {i}: window does not start with its seed"));
        }
        if b > pack_cap.min(MULTI_CHUNK_MAX_LANE_ROWS) {
            return refuse(format!("lane {i}: {b} rows > chunk capacity {pack_cap}"));
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
    pack_whole_lanes(&cb.rows, pack_cap, &mut cb.ranges).map_err(|e| HipError::new(0, &format!("dflash_cb_verify: {e}")))?;
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
