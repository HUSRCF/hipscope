// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Speculative-decode build/glue that lives at the top of the DAG, where both
//! `LoadedModel`/`ModelState` and the arch crates are in scope.
//!
//! Stage 0: the [`Qwen35SlotGuard`] only — the RAII scope that the daemon's
//! DFlash loop will use to borrow the target bundle. `DflashSpeculator` and
//! `build_speculator` land here at Stages 1-2.

use crate::ModelState;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_runtime::hfq::HfqFile;
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
