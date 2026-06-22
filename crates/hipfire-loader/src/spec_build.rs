// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Speculative-decode build/glue that lives at the top of the DAG, where both
//! `LoadedModel`/`ModelState` and the arch crates are in scope.
//!
//! Stage 0: the [`Qwen35SlotGuard`] only — the RAII scope that the daemon's
//! DFlash loop will use to borrow the target bundle. `DflashSpeculator` and
//! `build_speculator` land here at Stages 1-2.

use crate::{DflashState, ModelState};
use hipfire_arch_qwen35::speculative::{
    spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, ModelSlot,
    ModelSlotConfig, Phase2Snapshots, SpecStepResult,
};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{SpecGrammar, SpecStep, SpecTarget, Speculator};
use rdna_compute::Gpu;
use std::path::Path;

/// RAII scope that moves the live `Qwen35Bundle` out of `m.state`, lends it to
/// the spec-decode loop as a [`ModelSlot`], and — on `Drop`, via *every* exit
/// path including `?`, early return, and panic-unwind — restores it into
/// `m.state`.
///
/// This is the single chokepoint that replaces the eight hand-written
/// `m.state = Some(ModelState::Qwen35(..))` reconstructions in the daemon's
/// DFlash loop, structurally eliminating the "forgot to restore on early
/// return" cross-request state-bleed class (#462): there is no longer a code
/// path on which the bundle fails to return to `m.state`.
///
/// The `HfqFile` (an mmap handle that `ModelSlot` carries but the spec kernels
/// never read) is opened **lazily**, on the first [`slot`](Self::slot) call.
/// Two payoffs: (1) an autoregressive caller that only needs the bundle fields
/// never pays the mmap, and (2) an open failure leaves the bundle parked for
/// `Drop` to restore — so a reopen error can surface as `Err` without ever
/// leaving `m.state == None`.
pub struct Qwen35SlotGuard<'m> {
    state_back: &'m mut Option<ModelState>,
    model_path: String,
    // `Option` only so `Drop` can move the contents out; it is `Some` for the
    // guard's entire observable lifetime.
    parked: Option<Parked>,
}

// Both variants hold the same ~5.6 KB of live model state by value — that is
// the point: the guard *moves* the bundle, it does not copy it. The two differ
// by only the lazily-opened `HfqFile` handle + name + slot_config (~240 B), so
// boxing to flatten the delta would mean boxing BOTH variants (two ~5.8 KB
// heap alloc/free per generation) for no real saving on a short-lived
// stack-local guard. Keep it inline.
#[allow(clippy::large_enum_variant)]
enum Parked {
    /// The bundle as taken — fields untouched, no `HfqFile` opened yet.
    Bundle(Qwen35Bundle),
    /// The bundle assembled into a `ModelSlot` (HfqFile opened) for the spec
    /// helpers. `Drop` rebuilds the bundle from these fields.
    Slot(ModelSlot),
}

impl<'m> Qwen35SlotGuard<'m> {
    /// Take the `Qwen35Bundle` out of `state`. Returns `Err` (leaving `state`
    /// untouched) if the model is not a loaded Qwen3.5 bundle — note the
    /// `matches!` guard *before* `take()` so a non-Qwen35 model is never moved
    /// out and dropped.
    pub fn take(state: &'m mut Option<ModelState>, model_path: &str) -> Result<Self, String> {
        if !matches!(state, Some(ModelState::Qwen35(_))) {
            return Err("Qwen35SlotGuard: model state is not a loaded Qwen3.5 bundle".into());
        }
        let Some(ModelState::Qwen35(bundle)) = state.take() else {
            unreachable!("guarded by the matches! above")
        };
        Ok(Self {
            state_back: state,
            model_path: model_path.to_string(),
            parked: Some(Parked::Bundle(bundle)),
        })
    }

    /// Borrow the target as a [`ModelSlot`], opening the `HfqFile` on first use.
    /// On reopen failure the bundle stays parked (so `Drop` still restores it)
    /// and the error is returned.
    pub fn slot(&mut self) -> Result<&mut ModelSlot, String> {
        if let Some(Parked::Bundle(_)) = self.parked {
            let Some(Parked::Bundle(bundle)) = self.parked.take() else {
                unreachable!("guarded by the if-let above")
            };
            let hfq = match HfqFile::open(Path::new(&self.model_path)) {
                Ok(h) => h,
                Err(e) => {
                    // Park the bundle back so `Drop` restores it — no leak.
                    self.parked = Some(Parked::Bundle(bundle));
                    return Err(format!("reopen model: {e}"));
                }
            };
            let Qwen35Bundle {
                config,
                weights,
                scratch,
                kv_cache,
                dn_state,
            } = bundle;
            self.parked = Some(Parked::Slot(ModelSlot {
                name: String::from("target"),
                hfq,
                config,
                weights,
                kv_cache,
                dn_state,
                scratch,
                slot_config: ModelSlotConfig::default(),
            }));
        }
        match self.parked.as_mut() {
            Some(Parked::Slot(slot)) => Ok(slot),
            _ => unreachable!("slot() leaves `parked` as Slot on success"),
        }
    }
}

impl Drop for Qwen35SlotGuard<'_> {
    fn drop(&mut self) {
        let bundle = match self.parked.take() {
            Some(Parked::Bundle(b)) => b,
            Some(Parked::Slot(slot)) => {
                // slot.hfq (mmap), slot.name, slot.slot_config drop here; the
                // five live pieces go back into the bundle.
                Qwen35Bundle {
                    config: slot.config,
                    weights: slot.weights,
                    scratch: slot.scratch,
                    kv_cache: slot.kv_cache,
                    dn_state: slot.dn_state,
                }
            }
            None => return, // only reachable if `Drop` ran twice — it cannot.
        };
        *self.state_back = Some(ModelState::Qwen35(bundle));
    }
}

// ─── DflashSpeculator ───────────────────────────────────────────────────

/// Lower a qwen35 `SpecStepResult` onto the arch-generic `SpecStep`.
///
/// The daemon-called `spec_step_*` build `committed = [seed, drafts.., bonus]`,
/// so `committed[1..]` is exactly the daemon's `committed_tail` (the tokens
/// emitted this window) and its length is `accepted + 1` — which is why the
/// unified loop advances `position` by `emit.len()`.
fn lower_qwen35(r: SpecStepResult) -> SpecStep {
    SpecStep::new(
        r.committed[1..].iter().copied(),
        r.bonus_token,
        r.drafted.len(),
        r.accepted,
    )
}

/// DFlash / DDTree speculator: wraps the qwen35 `spec_step_*` chain/tree
/// kernels behind the arch-generic [`Speculator`] trait. Chain-vs-tree and
/// path_c are internal detail resolved at build (`ddtree` presence comes from
/// the loaded `DflashState`; `path_c_mode` from `HIPFIRE_DDTREE_PATH_C`).
///
/// Owns the `DflashState` moved out of `LoadedModel.dflash`. The divergent-
/// render checkpoint ring (`dflash_checkpoints`) stays daemon-managed until
/// Stage 2 reconciles the prompt-cache, so `checkpoint`/`rewind_to` are no-ops
/// here.
pub struct DflashSpeculator {
    df: DflashState,
    path_c_mode: Option<&'static str>,
    rng_state: u64,
}

impl DflashSpeculator {
    /// `path_c_mode` is the validated `HIPFIRE_DDTREE_PATH_C` value
    /// (`Some("phase1"|"phase2")` or `None`), resolved once at build.
    pub fn new(df: DflashState, path_c_mode: Option<&'static str>) -> Self {
        Self {
            df,
            path_c_mode,
            // Same fixed seed the daemon's DFlash loop used (greedy decode does
            // not consume it, but the signature requires an RNG state cell).
            rng_state: 0x13579BDF,
        }
    }
}

impl Speculator for DflashSpeculator {
    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<SpecStep, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Verbatim 3-way dispatch from the daemon's old generate_dflash loop:
        // DDTree path_c → DDTree batched → chain-mode DFlash. The grammar arg is
        // ignored — qwen35 enforces tool-call grammar post-hoc in the daemon.
        let result = if let Some(dd) = self.df.ddtree.as_mut() {
            if self.path_c_mode == Some("phase1") || self.path_c_mode == Some("phase2") {
                let phase2_snaps = if self.path_c_mode == Some("phase2") {
                    Some(Phase2Snapshots {
                        parent_pre_snap: &mut dd.path_c_parent_pre_snap,
                        main_end_snap: &mut dd.path_c_main_end_snap,
                    })
                } else {
                    None
                };
                spec_step_ddtree_path_c(
                    gpu,
                    slot,
                    &self.df.draft_weights,
                    &self.df.draft_config,
                    &mut self.df.draft_scratch,
                    &mut self.df.hidden_rb,
                    &mut self.df.target_hidden_host,
                    &mut self.df.target_snap,
                    &mut self.df.gdn_tape,
                    &self.df.verify_scratch,
                    position,
                    seed,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                    phase2_snaps,
                )
            } else {
                spec_step_ddtree_batched(
                    gpu,
                    slot,
                    &self.df.draft_weights,
                    &self.df.draft_config,
                    &mut self.df.draft_scratch,
                    &mut self.df.hidden_rb,
                    &mut self.df.target_hidden_host,
                    &mut self.df.target_snap,
                    &mut dd.post_seed_snap,
                    &mut self.df.gdn_tape,
                    &dd.scratch,
                    &self.df.verify_scratch,
                    position,
                    seed,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                )
            }
        } else {
            spec_step_dflash(
                gpu,
                slot,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                &mut self.df.target_snap,
                &self.df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                Some(&mut self.df.gdn_tape),
                0.0_f32, // temperature (greedy)
                &mut self.rng_state,
                None, // block_size override
                None, // ngram_cache
                emitted,
                0.0_f32, // cactus_delta
                None,    // pld_spine
                1.0_f32, // repeat_penalty (off)
                0,       // repeat_window
            )
        };

        result.map(lower_qwen35).map_err(|e| e.to_string())
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Drafter-local reset: invalidate cached suffix projections so the next
        // conversation re-projects from scratch. Target KV/recurrent reset is
        // the daemon's job (it owns the bundle).
        self.df.draft_scratch.reset_upload_tracking();
    }

    fn checkpoint(&mut self, _gpu: &mut Gpu, _position: usize) -> Result<(), String> {
        // Divergent-render checkpoints stay on `LoadedModel.dflash_checkpoints`
        // (daemon-managed) until Stage 2 reconciles the prompt-cache.
        Ok(())
    }

    fn rewind_to(&mut self, _gpu: &mut Gpu, position: usize) -> Result<usize, String> {
        Ok(position)
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors the current `unload_model` dflash teardown exactly.
        let DflashSpeculator { df, .. } = *self;
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
    }
}
