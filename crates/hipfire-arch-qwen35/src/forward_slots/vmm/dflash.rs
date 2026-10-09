// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DFlash lanes of the VMM continuous-batching store.
//!
//! A DFlash lane is a request decoded exactly like the singleton DFlash route
//! (`DflashSpeculator`, fixed block, greedy chain): its prompt is filled by
//! the singleton DFlash prompt seed on the request's own KV/DeltaNet, then
//! every decode window is a `RequestStepKind::Verify` request of an executor
//! step — per-request draft (provision), the ChainVerify trunk and head shared
//! with the other DFlash Verify lanes (forward), per-request greedy accept
//! and state commit (commit). Per request the committed ids and the full
//! state (trunk KV, DeltaNet, draft K/V, hidden ring, thlog) equal its
//! isolated singleton DFlash run.
//!
//! Window budget is DFlash's: with remaining output budget `E >= 1` the
//! window is `B = min(block, max(E, 2))` rows and `max_accept = E - 1`
//! ([`Qwen35VmmStore::dflash_set_max_emit`]). A Verify request carries
//! `draft_len = B - 1` (`rows = B`). `RequestAdvance::committed_ids` of a
//! DFlash window is `SpecStepResult::committed[1..]` (accepted drafts then the
//! bonus, which becomes the pending seed); the position advances by
//! `accepted + 1`. Greedy accept never stops at EOS: a consumer that stops
//! inside the window must call [`Qwen35VmmStore::dflash_repair_terminal_prefix`]
//! before retiring the lane.

use super::{Phase, Qwen35RequestState, Qwen35VmmStore};
use crate::dflash_cb::{
    dflash_block_for_emit, dflash_cb_verify, dflash_lane_draft, DflashCbScratch, DflashCbVerifyLane,
    DflashVmmLaneState,
};
use crate::dflash_spec::{
    dflash_prefill_lane_parts, dflash_repair_terminal_prefix_parts, new_dflash_lane_state,
    release_shared_dflash_weights, DflashLaneSnapshot, DflashVmmAssets, DflashWindowMark,
};
use crate::qwen35::prefill::multi::MULTI_CHUNK_MAX_ROWS;
use crate::qwen35::{Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::speculative::{
    dflash_greedy_accept_commit_parts, DeltaNetSnapshot, DflashCbDraft, DflashTargetParts, DflashVerifyOutput,
};
use hip_bridge::HipError;
use hipfire_runtime::slot_batch::{BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind};
use hipfire_runtime::spec::{PrefillOutcome, SpecRequestConfig};
use rdna_compute::Gpu;

/// The store's DFlash engine: the shared draft-weight assets (one resident
/// draft, a counted `Arc` reference) and the shared verify scratch (trunk row
/// scratch + ONE head scratch, `<= 63` rows). Lane state is per request.
pub struct VmmDflashEngine {
    assets: DflashVmmAssets,
    cb: DflashCbScratch,
}

// SAFETY: like every GPU owner in the store (`DeviceBuffer` is `Send`), the
// engine is only used by the thread that currently holds the model. Its asset
// reference is the one resident draft weight set shared (by atomic refcount)
// with the singleton speculator; weights are only read through GPU calls made
// under that same single owner, never concurrently.
unsafe impl Send for VmmDflashEngine {}

/// Loaded-ack evidence of an installed DFlash engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmmDflashReceipt {
    /// Configured (full) verify block, e.g. 16.
    pub block_size: usize,
    /// Rows one shared trunk chunk can hold (`<= 63`, the exactness ceiling).
    pub chunk_row_limit: usize,
    /// Logical context cap of the resolved singleton draft policy.
    pub ctx_capacity: usize,
}

fn refuse_engine<T>(gpu: &mut Gpu, assets: DflashVmmAssets, why: String) -> Result<T, String> {
    release_shared_dflash_weights(gpu, assets.weights);
    Err(format!("VMM DFlash engine: {why}"))
}

impl VmmDflashEngine {
    /// Build the engine over `assets` (`LoadedModel`'s singleton descriptor;
    /// the draft weights are shared, never reloaded). `max_rows` is the
    /// store's planned trunk row capacity; one chunk holds `min(max_rows, 63)`
    /// rows and must hold a full configured block. Consumes `assets`: on
    /// every refusal the asset reference is released (the last owner frees
    /// the raw weights) and nothing is allocated.
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        assets: DflashVmmAssets,
        max_rows: usize,
    ) -> Result<Self, String> {
        let block = assets.block_size;
        if !assets.fixed_chain {
            return refuse_engine(gpu, assets, "assets are not fixed-block chain".into());
        }
        if config.num_experts > 0 {
            return refuse_engine(gpu, assets, "MoE targets are not admitted".into());
        }
        if !(2..=MULTI_CHUNK_MAX_ROWS).contains(&block) {
            return refuse_engine(gpu, assets, format!("block {block} outside 2..={MULTI_CHUNK_MAX_ROWS}"));
        }
        let rows = max_rows.min(MULTI_CHUNK_MAX_ROWS);
        if rows < block {
            return refuse_engine(gpu, assets, format!("{max_rows} planned rows cannot hold one {block}-row block"));
        }
        if assets.staging_rows < block || assets.tape_max_rows < block || assets.ring_positions < block {
            let why = format!(
                "lane ring/tape (staging {}, tape {}, ring {}) cannot hold a {block}-row block",
                assets.staging_rows, assets.tape_max_rows, assets.ring_positions
            );
            return refuse_engine(gpu, assets, why);
        }
        if assets.ctx_capacity < block {
            let why = format!("context capacity {} < block {block}", assets.ctx_capacity);
            return refuse_engine(gpu, assets, why);
        }
        match DflashCbScratch::new(gpu, config, rows) {
            Ok(cb) => Ok(Self { assets, cb }),
            Err(e) => refuse_engine(gpu, assets, format!("verify scratch: {e}")),
        }
    }

    /// Fixed configured block (e.g. 16).
    pub fn block_size(&self) -> usize {
        self.assets.block_size
    }

    /// Resolved singleton logical context cap: a lane's window end
    /// (`position + B`) must stay within it.
    pub fn ctx_capacity(&self) -> usize {
        self.assets.ctx_capacity
    }

    /// Rows one shared trunk chunk holds (`<= 63`).
    pub fn max_rows(&self) -> usize {
        self.cb.max_rows()
    }

    pub fn receipt(&self) -> VmmDflashReceipt {
        VmmDflashReceipt {
            block_size: self.assets.block_size,
            chunk_row_limit: self.cb.max_rows(),
            ctx_capacity: self.assets.ctx_capacity,
        }
    }

    /// Diagnostic (oracle): also keep each lane's head rows in its own
    /// `verify_scratch`. Greedy production leaves it off.
    pub fn set_keep_lane_rows(&mut self, keep: bool) {
        self.cb.keep_lane_rows = keep;
    }

    /// Free the shared scratch and release the asset reference. Every lane
    /// must have been freed first (the last owner of the draft weights frees
    /// them, whichever of lanes / engine / singleton that is).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.hip.device_synchronize();
        let VmmDflashEngine { assets, cb } = self;
        let _ = cb.free_gpu(gpu);
        release_shared_dflash_weights(gpu, assets.weights);
    }
}

impl Qwen35VmmStore {
    /// Install the DFlash engine (staging; once). Refused (engine handed
    /// back, caller frees it) when one is installed or the store's row budget
    /// cannot run even one full block.
    pub fn install_dflash(&mut self, e: VmmDflashEngine) -> Result<(), VmmDflashEngine> {
        if self.dflash_engine.is_some() || self.row_budget < e.block_size() {
            return Err(e);
        }
        self.dflash_engine = Some(e);
        Ok(())
    }

    pub fn dflash_engine(&self) -> Option<&VmmDflashEngine> {
        self.dflash_engine.as_ref()
    }

    /// Make `epoch` a DFlash lane and fill its prompt by the singleton DFlash
    /// prompt seed (cold: fresh owners, positions from 0) on the request's own
    /// KV/DeltaNet. The lane's snapshot/DeltaNet-sized buffers come from the
    /// REQUEST's own `DeltaNetState`. Returns `Ready { first_token }` (greedy
    /// host argmax of the last prompt row's logits) and leaves it as the
    /// lane's pending seed (also pushed to `history`) at position
    /// `prompt.len()`; `Aborted` leaves the lane fresh (position 0, no DFlash
    /// state — the caller retires it). A failure poisons the lane; the lane
    /// state built so far is freed on every path.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_prefill(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
        prompt: &[u32],
        request: SpecRequestConfig,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        if !matches!(self.phase, Phase::Idle) {
            return Err("dflash prefill during an uncommitted step".into());
        }
        let engine = self.dflash_engine.as_ref().ok_or("dflash prefill: no DFlash engine staged")?;
        if prompt.is_empty() || prompt.len() >= engine.ctx_capacity() {
            return Err(format!("dflash prefill: prompt {} outside 1..{}", prompt.len(), engine.ctx_capacity()));
        }
        self.spec_provision(gpu, epoch, prompt.len())?;
        let Self { slots, dflash_engine, .. } = self;
        let engine = dflash_engine.as_ref().expect("checked above");
        let s = slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("dflash prefill: unknown epoch {epoch:?}"))?;
        if s.position != 0 || s.pending_seed.is_some() || s.mtp.is_some() || s.dflash.is_some() {
            return Err("dflash prefill: lane is not fresh".into());
        }
        let r = (|| -> Result<PrefillOutcome, String> {
            let mut df = new_dflash_lane_state(gpu, &engine.assets, config, &s.dn)
                .map_err(|e| format!("dflash prefill: lane state: {e}"))?;
            let mut checkpoints: Vec<(usize, DeltaNetSnapshot)> = Vec::new();
            let outcome = {
                let mut target = DflashTargetParts {
                    weights,
                    config,
                    kv_cache: &mut s.kv,
                    dn_state: &mut s.dn,
                    scratch,
                };
                dflash_prefill_lane_parts(gpu, &mut target, &engine.assets, &mut df, &mut checkpoints, prompt, abort)
            };
            match outcome {
                Ok(PrefillOutcome::Ready { first_token }) => {
                    s.dflash = Some(DflashVmmLaneState::cold(
                        df,
                        checkpoints,
                        engine.assets.checkpoint,
                        prompt.len(),
                        &request,
                    ));
                    s.position = prompt.len();
                    s.pending_seed = Some(first_token);
                    s.history.push(first_token);
                    Ok(PrefillOutcome::Ready { first_token })
                }
                Ok(PrefillOutcome::Aborted) => {
                    free_lane_parts(gpu, df, checkpoints);
                    Ok(PrefillOutcome::Aborted)
                }
                Err(e) => {
                    free_lane_parts(gpu, df, checkpoints);
                    Err(format!("dflash prefill: {e}"))
                }
            }
        })();
        if r.is_err() {
            s.poisoned = true;
        }
        r
    }

    /// Continue a promoted singleton DFlash request as a lane: its admitted
    /// owners (trunk KV/DeltaNet, `position`, `pending_seed`) came from the
    /// singleton, `snapshot` carries its draft/hidden-ring/checkpoint state
    /// (`DflashSpeculator::take_vmm_lane`). Validates asset identity
    /// (`Arc` pointer + resolved policy), position and idle/fresh exclusivity
    /// with MTP; installs the moved owners without resetting a cursor, except
    /// the pre-window DeltaNet snapshot, re-sized from the REQUEST's own
    /// `DeltaNetState` (a promoted request's pooled DN tensors can be larger
    /// than the resident one's; `save_from` copies by tensor size) and the
    /// last-window mark (the previous window's snapshot was in the old
    /// sizing). Consumes `snapshot` on success and on every error.
    pub fn dflash_adopt(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        snapshot: DflashLaneSnapshot,
    ) -> Result<(), String> {
        let mut lane = Some(DflashVmmLaneState::from_snapshot(snapshot));
        let r = self.dflash_adopt_inner(gpu, epoch, &mut lane);
        if let Some(l) = lane {
            l.free_gpu(gpu);
        }
        r
    }

    fn dflash_adopt_inner(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        slot_lane: &mut Option<DflashVmmLaneState>,
    ) -> Result<(), String> {
        if !matches!(self.phase, Phase::Idle) {
            return Err("dflash adopt during an uncommitted step".into());
        }
        let Self { slots, dflash_engine, .. } = self;
        let engine = dflash_engine.as_ref().ok_or("dflash adopt: no DFlash engine staged")?;
        let s = slots
            .iter_mut()
            .flatten()
            .find(|s| s.epoch == *epoch)
            .ok_or_else(|| format!("dflash adopt: unknown epoch {epoch:?}"))?;
        let lane = slot_lane.as_mut().expect("lane present");
        if s.mtp.is_some() || s.dflash.is_some() || s.pending_seed.is_none() || s.position != lane.position {
            return Err(format!(
                "dflash adopt: lane position {} / snapshot position {} / seed {:?} / other drafter {}",
                s.position,
                lane.position,
                s.pending_seed,
                s.mtp.is_some() || s.dflash.is_some()
            ));
        }
        if s.poisoned {
            return Err("dflash adopt: lane is poisoned".into());
        }
        if !engine.assets.matches_state(&lane.df) {
            return Err("dflash adopt: snapshot does not match the engine's shared assets".into());
        }
        if s.position + 2 > engine.ctx_capacity() {
            return Err(format!(
                "dflash adopt: position {} leaves no window within capacity {}",
                s.position,
                engine.ctx_capacity()
            ));
        }
        let fresh = DeltaNetSnapshot::new_for(gpu, &s.dn).map_err(|e| format!("dflash adopt: DeltaNet snapshot: {e}"))?;
        std::mem::replace(&mut lane.df.target_snap, fresh).free_gpu(gpu);
        lane.last_window = None;
        lane.last_emit.clear();
        lane.verified = false;
        lane.picks.clear();
        s.dflash = slot_lane.take();
        Ok(())
    }

    /// Install the remaining output budget `max_emit = E >= 1` of the lane's
    /// next window (the driver calls it between windows, before planning).
    /// Returns the singleton's block `B = min(block, max(E, 2))` the planned
    /// Verify request must carry (`draft_len = B - 1`, rows `B`); the accept
    /// clamp is `E - 1`.
    pub fn dflash_set_max_emit(&mut self, epoch: &RequestEpoch, max_emit: usize) -> Result<usize, String> {
        if max_emit == 0 {
            return Err("dflash_set_max_emit: no remaining output budget".into());
        }
        let block = self.dflash_engine.as_ref().ok_or("dflash_set_max_emit: no DFlash engine staged")?.block_size();
        let s = self
            .request_state_mut(epoch)
            .ok_or_else(|| format!("dflash_set_max_emit: unknown epoch {epoch:?}"))?;
        let Qwen35RequestState { dflash, dflash_draft, .. } = s;
        let lane = dflash.as_mut().ok_or("dflash_set_max_emit: not a DFlash lane")?;
        if dflash_draft.is_some() || lane.verified {
            return Err("dflash_set_max_emit: a window is in flight".into());
        }
        lane.max_emit = max_emit;
        Ok(dflash_block_for_emit(block, max_emit))
    }

    /// Terminal-prefix repair of a lane that stopped inside its last window:
    /// the window started at `position` with pending `seed`; the producer
    /// consumed `consumed` — a (possibly empty) prefix of that window's
    /// emitted tail, the last consumed token being the new pending seed.
    /// Restores the pre-window DeltaNet snapshot and thlog mark, re-seeds
    /// `[seed, consumed[..n-1]]` through the singleton suffix forward, then
    /// corrects the lane: position `= position + consumed.len()`, pending
    /// seed `= consumed.last()` (the window seed for an empty prefix), history
    /// truncated to its pre-window length plus `consumed`. A fully consumed
    /// window needs no repair (no-op). Call before retiring/evidencing the
    /// lane; the store must be idle. A failure poisons the lane.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_repair_terminal_prefix(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
        position: usize,
        seed: u32,
        consumed: &[u32],
    ) -> Result<(), String> {
        if !matches!(self.phase, Phase::Idle) {
            return Err("dflash terminal repair during an uncommitted step".into());
        }
        let s = self
            .request_state_mut(epoch)
            .ok_or_else(|| format!("dflash terminal repair: unknown epoch {epoch:?}"))?;
        if s.poisoned {
            return Err("dflash terminal repair: lane is poisoned".into());
        }
        let Qwen35RequestState {
            kv,
            dn,
            dflash,
            dflash_draft,
            position: lane_position,
            pending_seed,
            history,
            poisoned,
            ..
        } = s;
        let lane = dflash.as_mut().ok_or("dflash terminal repair: not a DFlash lane")?;
        if dflash_draft.is_some() || lane.verified {
            return Err("dflash terminal repair: a window is in flight".into());
        }
        let mark = lane.last_window.ok_or("dflash terminal repair: no completed window available")?;
        if mark.position != position || mark.seed != seed {
            return Err(format!(
                "dflash terminal repair: window mismatch (saved pos={} seed={}, requested pos={position} seed={seed})",
                mark.position, mark.seed
            ));
        }
        let emitted = lane.last_emit.len();
        if *lane_position != position + emitted {
            return Err(format!(
                "dflash terminal repair: lane position {} != window start {position} + emit {emitted}",
                *lane_position
            ));
        }
        if consumed.len() > emitted || consumed != &lane.last_emit[..consumed.len()] {
            return Err("dflash terminal repair: consumed tokens are not a prefix of the window's emit".into());
        }
        if consumed.len() == emitted {
            return Ok(());
        }
        let policy = lane.policy;
        let repaired = {
            let mut target = DflashTargetParts {
                weights,
                config,
                kv_cache: &mut *kv,
                dn_state: &mut *dn,
                scratch,
            };
            dflash_repair_terminal_prefix_parts(
                gpu,
                &mut target,
                &mut lane.df,
                &mut lane.checkpoints,
                &policy,
                mark.target_hidden,
                position,
                seed,
                consumed,
            )
        };
        if let Err(e) = repaired {
            *poisoned = true;
            return Err(format!("dflash terminal repair: {e}"));
        }
        lane.last_window = None;
        lane.last_emit.clear();
        *lane_position = position + consumed.len();
        lane.position = *lane_position;
        *pending_seed = Some(consumed.last().copied().unwrap_or(seed));
        history.truncate(lane.last_history_len);
        history.extend_from_slice(consumed);
        Ok(())
    }

    // ── Verify rows: provision gate / draft / forward / commit ──

    /// Provision gate of one planned DFlash Verify request: an idle lane with
    /// an installed budget, its seed in the first row, `rows == draft_len + 1
    /// == B(E)`, and the window within the lane's draft capacity.
    pub(super) fn dflash_check_verify(
        &self,
        s: &Qwen35RequestState,
        r: &RequestRows,
        draft_len: usize,
        seed_row_token: u32,
    ) -> Result<(), String> {
        let engine = self
            .dflash_engine
            .as_ref()
            .ok_or("provision_step: DFlash Verify rows need the staged DFlash engine")?;
        let lane = s.dflash.as_ref().expect("dispatched on the DFlash tag");
        if s.dflash_draft.is_some() || lane.verified {
            return Err(format!("provision_step: {:?} is not an idle DFlash lane", r.epoch));
        }
        if lane.max_emit == 0 {
            return Err(format!("provision_step: {:?} has no installed DFlash output budget", r.epoch));
        }
        let b = dflash_block_for_emit(engine.block_size(), lane.max_emit);
        if draft_len + 1 != b || r.rows.len != b {
            return Err(format!(
                "provision_step: DFlash Verify draft_len {draft_len} (rows {}) != window B={b} for budget {}",
                r.rows.len, lane.max_emit
            ));
        }
        if s.pending_seed != Some(seed_row_token) {
            return Err(format!("provision_step: Verify seed row {seed_row_token} != pending seed {:?}", s.pending_seed));
        }
        if s.position + b > engine.ctx_capacity() {
            return Err(format!(
                "provision_step: DFlash window end {} > draft context capacity {}",
                s.position + b,
                engine.ctx_capacity()
            ));
        }
        Ok(())
    }

    /// Provision: the singleton greedy draft of every planned DFlash Verify
    /// lane on its own draft state (epoch-tagged: the draft and its thlog mark
    /// are held until commit/abort). Touches only uncommitted drafter state
    /// (projection watermarks, draft K/V of the window), never target state.
    pub(super) fn dflash_draft_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        plan: &BatchStepPlan,
    ) -> Result<(), String> {
        let Self { slots, .. } = self;
        for r in &plan.requests {
            let RequestStepKind::Verify { draft_len } = r.kind else {
                continue;
            };
            let s = slots
                .iter_mut()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .ok_or_else(|| format!("provision_step: epoch {:?} vanished", r.epoch))?;
            let Qwen35RequestState { kv, dflash, dflash_draft, position, pending_seed, .. } = s;
            let Some(lane) = dflash.as_mut() else {
                continue;
            };
            let seed = pending_seed.ok_or("provision_step: DFlash lane without a pending seed")?;
            let mark = lane.df.draft_scratch.thlog.mark();
            let compact = kv.compact_offset as i32;
            let tokens = dflash_lane_draft(gpu, weights, config, compact, &mut lane.df, *position, seed, draft_len + 1)
                .map_err(|e| format!("provision_step: DFlash draft: {e}"))?;
            *dflash_draft = Some(DflashCbDraft {
                position: *position,
                seed,
                verify_tokens: tokens,
                max_accept: lane.max_emit.saturating_sub(1),
                thlog_mark: mark,
            });
        }
        Ok(())
    }

    /// Forward: every planned DFlash Verify lane through the shared ChainVerify
    /// trunk (whole lanes in `<= 63`-row chunks) and ONE head + argmax + D2H
    /// per chunk ([`dflash_cb_verify`]). Picks wait in each lane for commit.
    pub(super) fn dflash_verify_planned(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        plan: &BatchStepPlan,
    ) -> hip_bridge::HipResult<()> {
        let epochs: Vec<RequestEpoch> = plan
            .requests
            .iter()
            .filter(|r| {
                matches!(r.kind, RequestStepKind::Verify { .. })
                    && self.request_state(&r.epoch).is_some_and(|s| s.dflash.is_some())
            })
            .map(|r| r.epoch)
            .collect();
        if epochs.is_empty() {
            return Ok(());
        }
        let Self { slots, dflash_engine, .. } = self;
        let engine = dflash_engine
            .as_mut()
            .ok_or_else(|| HipError::new(0, "verify: DFlash lanes without a DFlash engine"))?;
        let mut owners: Vec<Option<&mut Qwen35RequestState>> = epochs.iter().map(|_| None).collect();
        for s in slots.iter_mut().flatten() {
            if let Some(i) = epochs.iter().position(|e| *e == s.epoch) {
                owners[i] = Some(s);
            }
        }
        let mut lanes: Vec<DflashCbVerifyLane<'_>> = Vec::with_capacity(epochs.len());
        for s in owners {
            let s = s.ok_or_else(|| HipError::new(0, "verify: planned owner vanished"))?;
            let Qwen35RequestState { kv, dn, dflash, dflash_draft, .. } = s;
            let draft = dflash_draft.as_ref().ok_or_else(|| HipError::new(0, "verify: no DFlash draft"))?;
            lanes.push(DflashCbVerifyLane {
                kv_cache: kv,
                dn_state: dn,
                state: dflash.as_mut().expect("checked above"),
                draft,
            });
        }
        dflash_cb_verify(gpu, weights, config, scratch, &mut engine.cb, &mut lanes)
    }

    /// Commit one verified DFlash window: the singleton greedy accept
    /// consumer (`dflash_greedy_accept_commit_parts`: `eos=None` accept,
    /// pre-commit `max_accept` clamp, hidden scatter/thlog append, snapshot
    /// replay) over the lane's picks, then publish its window — position
    /// `+= accepted + 1`, pending seed = bonus, history `+= committed[1..]` —
    /// and keep the window mark for terminal-prefix repair.
    pub(super) fn dflash_commit_verify(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        epoch: &RequestEpoch,
    ) -> Result<RequestAdvance, String> {
        let ctx_cap = self
            .dflash_engine
            .as_ref()
            .ok_or("commit_step: no DFlash engine staged")?
            .ctx_capacity();
        let s = self
            .request_state_mut(epoch)
            .ok_or_else(|| format!("commit_step: epoch {epoch:?} no longer owns its slot"))?;
        let Qwen35RequestState { kv, dn, dflash, dflash_draft, position, pending_seed, history, stop_ids, .. } = s;
        let draft = dflash_draft.take().ok_or("commit_step: DFlash verify without a draft")?;
        let lane = dflash.as_mut().ok_or("commit_step: lane lost its DFlash state")?;
        if !std::mem::replace(&mut lane.verified, false) {
            return Err("commit_step: DFlash verify without an outcome".into());
        }
        if draft.position != *position {
            return Err(format!("commit_step: DFlash window at {} but lane at {}", draft.position, *position));
        }
        let verified = DflashVerifyOutput {
            argmax_per_pos: std::mem::take(&mut lane.picks),
            logits_per_pos: Vec::new(),
        };
        let result = {
            let mut target = DflashTargetParts {
                weights,
                config,
                kv_cache: &mut *kv,
                dn_state: &mut *dn,
                scratch,
            };
            dflash_greedy_accept_commit_parts(gpu, &mut target, &mut lane.df, &draft, &verified)
                .map_err(|e| format!("commit_step: DFlash accept: {e}"))?
        };
        let mut picks = verified.argmax_per_pos;
        picks.clear();
        lane.picks = picks;

        let emit = &result.committed[1..];
        debug_assert_eq!(emit.len(), result.accepted + 1);
        lane.last_history_len = history.len();
        history.extend_from_slice(emit);
        lane.last_emit.clear();
        lane.last_emit.extend_from_slice(emit);
        lane.last_window = Some(DflashWindowMark {
            position: draft.position,
            seed: draft.seed,
            target_hidden: draft.thlog_mark,
        });
        *position += emit.len();
        lane.position = *position;
        // The budget belongs to one window: the driver installs the next
        // window's `E` (`dflash_set_max_emit`) before planning it.
        lane.max_emit = 0;
        *pending_seed = Some(result.bonus_token);
        let finish = if emit.iter().any(|t| stop_ids.contains(t)) {
            Some("stop".to_string())
        } else if *position >= kv.vmm_logical_bound() || *position + 2 > ctx_cap {
            Some("length".to_string())
        } else {
            None
        };
        Ok(RequestAdvance {
            epoch: *epoch,
            committed_ids: emit.to_vec(),
            committed_position: *position,
            accepted_drafts: result.accepted,
            verified_rows: draft.verify_tokens.len(),
            finish,
        })
    }
}

/// Free a lane's owners that never became a [`DflashVmmLaneState`].
fn free_lane_parts(gpu: &mut Gpu, df: crate::dflash_spec::DflashState, checkpoints: Vec<(usize, DeltaNetSnapshot)>) {
    df.free_gpu(gpu);
    for (_, snap) in checkpoints {
        snap.free_gpu(gpu);
    }
}
