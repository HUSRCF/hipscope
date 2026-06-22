// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Speculative-decode build/glue that lives at the top of the DAG, where both
//! `LoadedModel`/`ModelState` and the arch crates are in scope.
//!
//! Contents: the [`Qwen35SlotGuard`] RAII target borrow, the [`DflashSpeculator`]
//! impl (which owns `DflashState` + the divergent-render checkpoint ring),
//! [`build_dflash_speculator`] (its load-time constructor), and the generic
//! [`build_speculator`] registry that dispatches on draft kind: a loaded DFlash
//! draft → [`DflashSpeculator`], else (opt-in) the model-free n-gram drafter
//! (`ChainSpeculator<NgramDrafter>` from `spec_ngram`). The registry is what lets
//! the loader pick a drafter at load time without the daemon learning which ran.

use crate::{DflashState, ModelState};
use hipfire_arch_qwen35::speculative::{
    apply_eviction_retain_to_draft, scatter_hidden_block_to_interleaved,
    seed_target_hidden_from_prompt_abortable, seed_target_hidden_suffix_abortable,
    spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, DeltaNetSnapshot,
    ModelSlot, ModelSlotConfig, Phase2Snapshots, SpecStepResult,
};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{
    EvictRetain, PrefillOutcome, SpecGrammar, SpecStep, SpecTarget, Speculator,
};
use hipfire_runtime::spec_ngram::{ChainSpeculator, NgramDrafter};
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
/// Owns the `DflashState` moved out of `LoadedModel.dflash`, plus the divergent-
/// render DeltaNet checkpoint ring folded in from `LoadedModel.dflash_checkpoints`.
pub struct DflashSpeculator {
    df: DflashState,
    path_c_mode: Option<&'static str>,
    rng_state: u64,
    /// Divergent-render checkpoint ring. Populated by `prefill`'s seed when
    /// `resume_enabled`; freed on `reset`/`free`.
    checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    resume_enabled: bool,
    ck_interval: usize,
    ck_cap: usize,
}

impl DflashSpeculator {
    /// `path_c_mode` is the validated `HIPFIRE_DDTREE_PATH_C` value
    /// (`Some("phase1"|"phase2")` or `None`); `resume_enabled`/`ck_interval`/
    /// `ck_cap` mirror the daemon's `ckpt_resume_enabled()`/`ckpt_interval()`/
    /// `ckpt_max()` — passed in by `build_dflash_speculator` so `new` itself is
    /// env-free (and unit-testable).
    pub fn new(
        df: DflashState,
        path_c_mode: Option<&'static str>,
        resume_enabled: bool,
        ck_interval: usize,
        ck_cap: usize,
    ) -> Self {
        Self {
            df,
            path_c_mode,
            // Same fixed seed the daemon's DFlash loop used (greedy decode does
            // not consume it, but the signature requires an RNG state cell).
            rng_state: 0x13579BDF,
            checkpoints: Vec::new(),
            resume_enabled,
            ck_interval,
            ck_cap,
        }
    }
}

impl Speculator for DflashSpeculator {
    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Mirror the daemon's pre-seed drafter setup (generate_dflash 4064-4072):
        // always clear the host hidden buffer; on a full prefill drop the draft's
        // upload/projection tracking. On a cache HIT it is PRESERVED so the draft
        // reuses the cached [0..start_pos] projections and only projects the suffix.
        self.df.target_hidden_host.clear();
        if !cache_hit {
            self.df.draft_scratch.reset_upload_tracking();
        }

        // Seed the target's hidden state into the drafter ring (chunked prefill
        // with hidden extraction). Cache hit → seed only the suffix from
        // `prefill_start`, reusing the prior turn's KV + recurrent state; miss →
        // seed the full prompt (the seed fn resets target state itself).
        let (ck_interval, ck_cap) = (self.ck_interval, self.ck_cap);
        let ckpt_sink = if self.resume_enabled {
            Some(&mut self.checkpoints)
        } else {
            None
        };
        let aborted = if cache_hit {
            seed_target_hidden_suffix_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                prefill_tokens,
                prefill_start,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        } else {
            seed_target_hidden_from_prompt_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                prefill_tokens,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        }
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller resets conversation state + emits aborted/done; the slot
            // guard restores the target bundle on the way out.
            return Ok(PrefillOutcome::Aborted);
        }

        // Prime/extend the draft's GPU target_hidden buffer. On a hit, scatter
        // only the suffix rows at `prefill_start` (the prefix is preserved);
        // on a miss, scatter all prompt rows from 0.
        let (scatter_off, scatter_len) = if cache_hit {
            (prefill_start, prefill_tokens.len())
        } else {
            (0, prompt_tokens.len())
        };
        if let Err(e) = scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            scatter_off,
            scatter_len,
            scatter_len,
        ) {
            eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
        }
        self.df.draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
        self.df.draft_scratch.target_hidden_abs_positions =
            (0..prompt_tokens.len() as i32).collect();
        if let Some(ckpt) = resume_from {
            // Divergent rows [ckpt..len) were just overwritten; drop the draft's
            // projection cursor so the first spec step re-projects from `ckpt`.
            self.df.draft_scratch.draft_ctx_cached_rows = ckpt;
        }

        // First emit = target argmax at the final prompt position (seed already
        // ran the per-token forward; scratch.logits holds the post-prompt logits).
        let first_logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        let first_token = first_logits
            .iter()
            .enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv {
                    (i as u32, v)
                } else {
                    (best, bv)
                }
            })
            .0;
        Ok(PrefillOutcome::Ready { first_token })
    }

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

    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        // Compact the drafter's cached target-hidden rows to match the target KV
        // after the FlashCASK eviction the daemon already applied to the target.
        let ne = self.df.draft_config.num_extract();
        let h = self.df.draft_config.hidden;
        apply_eviction_retain_to_draft(
            gpu,
            &mut self.df.draft_scratch,
            &retain.retain_mask,
            ne,
            h,
            retain.pre_phys,
        )
        .map_err(|e| e.to_string())
    }

    fn reset(&mut self, gpu: &mut Gpu) {
        // Drafter-local reset: invalidate cached suffix projections and free the
        // divergent-render checkpoint ring (the target KV/recurrent reset is the
        // daemon's job — it owns the bundle).
        self.df.draft_scratch.reset_upload_tracking();
        for (_, snap) in self.checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
    }

    fn block_size(&self) -> usize {
        self.df.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.df.ctx_capacity
    }

    fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints.iter().map(|(p, _)| *p).collect()
    }

    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        // Restore the target's DeltaNet recurrent state to the checkpoint at
        // `position` and drop the now-stale tail of the ring (mirrors the old
        // divergent-render resume at generate_dflash 4021-4036). Caller rewinds
        // seq_pos / conversation_tokens to match.
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        if let Some(idx) = self.checkpoints.iter().rposition(|(p, _)| *p == position) {
            let _ = self.checkpoints[idx].1.restore_to(&mut slot.dn_state, gpu);
            for (_, snap) in self.checkpoints.drain(idx + 1..) {
                snap.free_gpu(gpu);
            }
        }
        Ok(position)
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors the `unload_model` dflash teardown + the checkpoint-ring free.
        let DflashSpeculator {
            df, checkpoints, ..
        } = *self;
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Construct the DFlash speculator from a freshly-loaded `DflashState`, resolving
/// the env config the daemon's old `generate_dflash` read inline: `path_c_mode`
/// (`HIPFIRE_DDTREE_PATH_C`), checkpoint resume (`HIPFIRE_DFLASH_CKPT_RESUME` +
/// no-eviction), and interval/cap (`HIPFIRE_CACHE_CKPT_INTERVAL`/`_MAX`, matching
/// the daemon's `ckpt_interval()`/`ckpt_max()` defaults). Called once at load.
pub fn build_dflash_speculator(df: DflashState, eviction_is_none: bool) -> Box<dyn Speculator> {
    let path_c_mode: Option<&'static str> =
        match std::env::var("HIPFIRE_DDTREE_PATH_C").ok().as_deref() {
            Some("phase1") => Some("phase1"),
            Some("phase2") => Some("phase2"),
            _ => None,
        };
    let resume_enabled = std::env::var("HIPFIRE_DFLASH_CKPT_RESUME").ok().as_deref() != Some("0")
        && eviction_is_none;
    let ck_interval = std::env::var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize)
        .max(256);
    let ck_cap = std::env::var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    Box::new(DflashSpeculator::new(
        df,
        path_c_mode,
        resume_enabled,
        ck_interval,
        ck_cap,
    ))
}

/// Pick the speculative-decode drafter for a freshly-loaded model. This is the
/// single load-time registry the daemon's `generate_dflash` routes through —
/// it never learns which arm was chosen.
///
/// Dispatch:
/// 1. A loaded DFlash draft (`dflash = Some`) → [`DflashSpeculator`].
/// 2. Else, when `HIPFIRE_NGRAM_DRAFT=1` and the arch has a `SpecTarget` impl
///    (qwen35 5/6, llama 0/1), the model-free `ChainSpeculator<NgramDrafter>` —
///    spec-decode with no draft model. Opt-in until validated.
/// 3. Otherwise `None` (AR-only).
///
/// The n-gram arm is arch-typeless: it builds its target-side verify scratch
/// lazily on first `prefill` via `SpecTarget::new_spec_scratch`, so this fn needs
/// only `arch_id`, the drafter env, and the target's `ctx_capacity`. `arch_id`
/// gates which arches the model-free arm is enabled for (qwen35 5/6 today; llama
/// added with its `SpecTarget` impl).
pub fn build_speculator(
    arch_id: u32,
    dflash: Option<DflashState>,
    eviction_is_none: bool,
    ctx_capacity: usize,
) -> Option<Box<dyn Speculator>> {
    if let Some(df) = dflash {
        return Some(build_dflash_speculator(df, eviction_is_none));
    }
    let ngram_enabled = std::env::var("HIPFIRE_NGRAM_DRAFT").ok().as_deref() == Some("1");
    // Spec-capable arches with a `SpecTarget` impl: qwen35 DeltaNet (5/6) and
    // the dense LLaMA family (0 = LLaMA/Mistral, 1 = plain Qwen3/Qwen2).
    if ngram_enabled && matches!(arch_id, 0 | 1 | 5 | 6) {
        let block_size = std::env::var("HIPFIRE_NGRAM_DRAFT_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize)
            .max(2);
        let min_count = std::env::var("HIPFIRE_NGRAM_MIN_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2u32);
        eprintln!(
            "  n-gram speculator enabled (model-free, K={}, min_count={})",
            block_size, min_count
        );
        return Some(Box::new(ChainSpeculator::new(
            NgramDrafter::new(min_count, block_size),
            block_size,
            ctx_capacity,
        )));
    }
    None
}
