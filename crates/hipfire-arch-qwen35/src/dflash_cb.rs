// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal DFlash batched-lane seam (Gate 0 replay probe).
//!
//! [`DflashVmmLaneState`] is the persistent private draft/verify state of one
//! continuous-batching lane: the singleton [`DflashState`] owners plus the
//! request-local cursors the singleton [`DflashSpeculator`] keeps beside it.
//! It is built only by consuming a [`DflashLaneSnapshot`] (`take_vmm_lane`),
//! which transfers every owner and cursor losslessly.
//!
//! The two helpers below are the in-crate halves of the shared DFlash verify
//! the Gate 0 probe drives from outside the crate:
//!
//! * [`dflash_lane_draft`] — the singleton greedy draft call of
//!   `spec_step_dflash` (`draft_dflash_block_rank`) on a lane's own draft
//!   state; returns `[seed, candidates..]`.
//! * [`dflash_cb_head_argmax`] — the singleton DFlash verify head
//!   (`dflash_enqueue_verify_lm_head_argmax` + one packed i32 D2H) over the
//!   first `rows` rows of a shared [`VerifyScratch`].

use crate::dflash_spec::{
    DflashCheckpointPolicy, DflashLaneSnapshot, DflashState, DflashWindowMark,
};
use crate::qwen35::{Qwen35Config, Qwen35Weights};
use crate::speculative::{
    dflash_download_verify_argmax, dflash_enqueue_verify_lm_head_argmax, draft_dflash_block_rank,
    DeltaNetSnapshot, VerifyScratch,
};
use hip_bridge::HipResult;
use rdna_compute::Gpu;

/// Private DFlash state of one lane (see the module docs).
pub struct DflashVmmLaneState {
    /// The singleton owners: draft scratch/weights, hidden ring, verify
    /// scratch, target DeltaNet snapshot, GDN tape.
    pub df: DflashState,
    /// Committed target position at capture.
    pub position: usize,
    pub rng_state: u64,
    pub sample_temp: f32,
    pub sample_top_p: f32,
    pub sample_top_k: usize,
    pub sample_cactus: f32,
    /// Divergent-render checkpoints `(position, snapshot)`.
    pub checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub policy: DflashCheckpointPolicy,
    pub last_window: Option<DflashWindowMark>,
}

impl DflashVmmLaneState {
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
        }
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
