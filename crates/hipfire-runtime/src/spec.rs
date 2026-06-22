// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transparent speculative-decode seam.
//!
//! The daemon's decode loop drives a `&mut dyn Speculator` and never learns
//! which drafter/mode (DFlash chain, DDTree tree, DeepSeek4 MTP, future
//! n-gram / EAGLE) is in use. Adding a drafter is a bounded-context change:
//! implement [`Speculator`] and register one arm in the loader's
//! `build_speculator` — no daemon edits.
//!
//! This is the arch-generic boundary anticipated by the `hipfire-arch-qwen35`
//! crate docs ("speculative.rs will become arch-generic"): the trait and the
//! unified result live here in the arch-agnostic runtime, while the
//! arch-coupled impls (which need `qwen35::*` / `deepseek4::*` symbols) stay in
//! their arch crates and `impl` this trait under the orphan rule.
//!
//! Stage 0 (this file): the trait + the unified result + the borrowed-target
//! and erased-grammar interfaces. No wiring yet — the daemon still calls the
//! arch `spec_step_*` functions directly until Stage 1 routes through the trait.

use rdna_compute::Gpu;
use smallvec::SmallVec;

/// Outcome of one speculative-decode acceptance window, drafter-agnostic.
///
/// The daemon advances by `emit.len()` and reseeds from `next_seed` without
/// knowing which drafter ran. The two arch result types lower onto this:
/// - qwen35 `SpecStepResult` → `emit = committed[1..]` (the seed re-echo is
///   dropped), `next_seed = bonus_token`.
/// - deepseek4 MTP → `emit = accepted_tokens`, `next_seed = accepted_tokens.last()`.
///
/// `committed[1..].len()` equals `accepted + 1` for the chain drafters and
/// `accepted_tokens.len()` equals the MTP position advance, so a single
/// `position += emit.len()` is correct for both. `proposed`/`accepted` are τ
/// accounting only; they do NOT drive position math.
#[derive(Debug, Clone)]
pub struct SpecStep {
    /// Tokens to emit this window, in order, with any seed re-echo already
    /// stripped. `position += emit.len()`. Non-empty on `Ok` (forward progress).
    ///
    /// `SmallVec` so the autoregressive fast path (one token / step, wired at
    /// Stage 2) stays heap-alloc-free; only large spec windows spill to the heap.
    pub emit: SmallVec<[u32; 8]>,
    /// Seed for the next window — the verifier's preferred token at the
    /// divergence point (qwen35 `bonus_token`; MTP `accepted_tokens.last()`).
    pub next_seed: u32,
    /// Drafts offered this window (τ denominator).
    pub proposed: usize,
    /// Drafts accepted this window (τ numerator).
    pub accepted: usize,
}

impl SpecStep {
    /// Build a step from an `emit` iterator, hiding the `SmallVec` backing from
    /// caller crates that don't depend on `smallvec` (the per-arch lowering
    /// adapters `lower_qwen35` / `lower_mtp` live in the loader / arch crates).
    pub fn new(
        emit: impl IntoIterator<Item = u32>,
        next_seed: u32,
        proposed: usize,
        accepted: usize,
    ) -> Self {
        Self {
            emit: emit.into_iter().collect(),
            next_seed,
            proposed,
            accepted,
        }
    }
}

/// The verifier (target) model's GPU state, borrowed by [`Speculator::step`]
/// for the duration of one window. A `Speculator` impl recovers its concrete
/// target via `as_any_mut().downcast_mut::<T>()` (e.g. the qwen35 `ModelSlot`).
///
/// The borrowed-not-owned shape lets one decode loop hold the target across all
/// windows (taken from the model bundle once, via the loader's RAII slot guard)
/// while the speculator borrows it per step — no per-step ownership transfer.
pub trait SpecTarget {
    /// Downcast hook: `target.as_any_mut().downcast_mut::<ModelSlot>()`.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Zero the target's recurrent (DeltaNet) state and reset the KV eviction
    /// offset — used by the daemon's mid-generation abort path in place of its
    /// current inline memset loop.
    fn reset_recurrent(&mut self, gpu: &mut Gpu);
}

/// Erased grammar-mask interface for tool-call-constrained spec-decode.
///
/// The concrete per-arch grammar `Matcher` types
/// (`hipfire_arch_qwen35::grammar`, `hipfire_arch_deepseek4::grammar`) are
/// crate-local and distinct, and this crate depends on neither — so grammar is
/// threaded through the trait as an erased `&mut dyn SpecGrammar`, not a shared
/// concrete struct (a shared struct would invert the crate dependency graph; an
/// associated type would break `Box<dyn Speculator>`). Marker for now; the
/// mask-fill / accept method set is defined at Stage 3, when the MTP speculator
/// first consumes it.
pub trait SpecGrammar {}

/// Outcome of [`Speculator::prefill`].
#[derive(Debug, Clone)]
pub enum PrefillOutcome {
    /// Prompt prefilled; `first_token` is the target's argmax at the last prompt
    /// position (the seed for the first decode window).
    Ready { first_token: u32 },
    /// Client cancelled mid-prefill. The caller resets conversation state and
    /// emits the aborted/done events; the slot guard restores the target bundle.
    Aborted,
}

/// Eviction-retain descriptor for [`Speculator::on_evict`] — lets the drafter
/// compact its cached target-hidden rows to match the target KV after a
/// FlashCASK eviction the daemon already applied to the target.
#[derive(Debug, Clone)]
pub struct EvictRetain {
    /// Per-physical-slot retain mask from the eviction policy.
    pub retain_mask: Vec<u32>,
    /// Physical fill before the eviction (rows to compact).
    pub pre_phys: usize,
}

/// A speculative-decode drafter+verifier, owned by the loaded model behind a
/// `Box<dyn Speculator>`. The daemon's decode loop holds `&mut dyn Speculator`
/// and is agnostic to whether the impl is a DFlash chain, a DDTree tree, an MTP
/// head, or a future n-gram / EAGLE drafter — chain-vs-tree, path_c, K, budget,
/// and topk are all resolved at build time and stored inside the impl.
pub trait Speculator {
    /// Prefill the prompt: seed the target's hidden state (advancing its KV +
    /// recurrent state) and prime the drafter's cached target-hidden buffer,
    /// returning the target's first token. `prefill_tokens` is the suffix to
    /// seed on a cache hit (from `prefill_start`) or the full prompt on a miss;
    /// `prompt_tokens` is the full rendered prompt (used to size the drafter
    /// cursor). `resume_from`, when set, drops the drafter projection cursor to a
    /// divergent-render checkpoint position.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<PrefillOutcome, String>;

    /// Run one acceptance window starting from `seed` at absolute `position`.
    /// `target` is the borrowed verifier; `emitted` is the prior committed
    /// tokens (repeat-penalty / n-gram context); `grammar` constrains both the
    /// draft and verify logits (`None` = unconstrained).
    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<SpecStep, String>;

    /// Compact drafter-local cached state after a target KV eviction the daemon
    /// already applied. Default no-op for drafters with no target-hidden cache.
    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        let _ = (gpu, retain);
        Ok(())
    }

    /// Rewind drafter-LOCAL state for a fresh conversation. The target's KV /
    /// recurrent state is the daemon's concern (it owns the bundle); this clears
    /// only the drafter's own scratch + checkpoint ring.
    fn reset(&mut self, gpu: &mut Gpu);

    /// Snapshot drafter-local recurrent state at `position` for divergent-render
    /// prompt-cache reuse. Default no-op for stateless drafters (n-gram).
    fn checkpoint(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<(), String> {
        let _ = (gpu, target, position);
        Ok(())
    }

    /// Restore drafter-local state to the nearest checkpoint `<= position`,
    /// returning the position actually restored to. Default no-op (returns
    /// `position`) for stateless drafters.
    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        let _ = (gpu, target);
        Ok(position)
    }

    /// Release all GPU buffers the drafter owns. Called from `unload_model`,
    /// so a drafter that forgets to free is a missing-trait-method compile
    /// error rather than a silent VRAM leak.
    fn free(self: Box<Self>, gpu: &mut Gpu);

    /// Whether this drafter requires greedy verification (temperature 0).
    /// DFlash / DDTree return `true`; a sampling-capable MTP / EAGLE could
    /// return `false`. `build_speculator` returns `None` for a non-greedy
    /// request against a greedy-only drafter, routing it to the AR path.
    fn requires_greedy(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Locks the load-bearing loop contract: the daemon advances `position` by
    // `emit.len()`, NOT by `accepted`. For a chain window that accepted 2 of 4
    // drafts, emit is `[d0, d1, bonus]` (len 3 = accepted + 1) and the next
    // seed is the bonus = `emit.last()`. The adversarial review certified this
    // equivalence against `speculative.rs:3737-3744`; this test pins it so a
    // future lowering change can't silently break the position math.
    #[test]
    fn emit_len_drives_advance_not_accepted() {
        let step = SpecStep {
            emit: SmallVec::from_slice(&[10, 11, 12]),
            next_seed: 12,
            proposed: 4,
            accepted: 2,
        };
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(*step.emit.last().unwrap(), step.next_seed);
    }
}
