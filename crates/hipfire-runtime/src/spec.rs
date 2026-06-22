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
//! Status: the trait, the unified [`SpecStep`] result, and the borrowed-target /
//! erased-grammar interfaces are live. The daemon's DFlash decode loop drives a
//! `&mut dyn Speculator` (`examples/daemon.rs::generate_dflash`), with the
//! loader's `DflashSpeculator` as the sole impl. Still future work: a generic
//! `build_speculator` registry (dispatch on arch/draft kind) and additional
//! drafters (n-gram, MTP, EAGLE) — the AR one-token path still runs through
//! `generate()`, not this trait.

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
    /// `SmallVec` so a future one-token-per-step drafter (e.g. n-gram) stays
    /// heap-alloc-free; only large spec windows spill to the heap.
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

    // ── Arch-generic speculation primitives ─────────────────────────────────
    //
    // These let a *model-free* speculator (n-gram / PLD) drive any arch's target
    // without knowing its internals: the target owns ALL verify mechanics (the
    // batched forward, the per-position lm_head, the recurrent snapshot/rewind,
    // and the arch-specific scratch), while the speculator owns only policy
    // (drafting + acceptance). The arch-specific verify scratch is created by the
    // target via [`new_spec_scratch`](Self::new_spec_scratch) and handed back on
    // every call as an erased `&mut dyn SpecScratch`, so no arch type leaks into
    // the speculator and the speculator owns the scratch's lifetime.

    /// Allocate arch-specific verify scratch sized to `block_size` (the max
    /// speculation window). The speculator owns the returned box for its lifetime
    /// and frees it via [`SpecScratch::free`].
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String>;

    /// Advance the target over `tokens` from absolute `start_pos` (chunked,
    /// abortable), returning the greedy argmax at the LAST position. `reset`
    /// zeroes recurrent + KV state first (cache-miss prefill); `false` continues
    /// from the current state (cache-hit suffix, or the partial-accept replay).
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String>;

    /// Run the target over `block` at absolute `position`, returning the greedy
    /// argmax at each of the `block.len()` positions (`argmax[i]` is the target's
    /// next-token prediction after consuming `block[0..=i]`). Leaves target state
    /// advanced by `block.len()`.
    ///
    /// CONTRACT: this MUST first snapshot whatever recurrent state
    /// [`commit_prefix`](Self::commit_prefix) needs to rewind (e.g. the DeltaNet
    /// S/conv state AND the Q8 error-feedback residual) INTO `scratch`, *before*
    /// running the forward that advances it. Stateless (pure-attention) arches
    /// snapshot nothing.
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<Vec<u32>, String>;

    /// Fix target state to reflect exactly the committed prefix
    /// `block[..accept_len + 1]` (after [`verify_block`](Self::verify_block)
    /// over-advanced it by `block.len()`). Cases:
    /// - full accept (`accept_len == block.len() - 1`): no-op — verify already
    ///   left state at the right position;
    /// - recurrent + partial: restore the snapshot saved in `scratch` (incl. the
    ///   s_ef residual) and replay `block[..accept_len + 1]` with the SAME batched
    ///   forward `verify_block` used (numerics must match the accepted argmax);
    /// - stateless + partial: no-op — the accepted-prefix KV the verify wrote is
    ///   already correct, and the rejected tail is overwritten by the next verify.
    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String>;

    /// The target's EOS token id (for the daemon's decode-loop terminator check).
    fn eos_token(&self) -> u32;

    /// The target's usable context capacity (decode-loop overflow guard).
    fn ctx_capacity(&self) -> usize;

    /// The target's KV cache, for the daemon's FlashCASK eviction (which operates
    /// on the shared `llama::KvCache` for all spec-capable arches). Eviction is
    /// `None` by default for pure-attention arches, so this is exercised rarely.
    fn kv_cache_mut(&mut self) -> &mut crate::llama::KvCache;
}

/// Erased, arch-specific verify scratch owned by a model-free speculator.
///
/// The concrete scratch (qwen35: `VerifyScratch` + `DeltaNetSnapshot` + s_ef
/// backup + hidden ring; llama: lm_head/argmax buffers) is crate-local to each
/// arch and this crate depends on none of them — so it is threaded through the
/// trait erased, mirroring [`SpecGrammar`]. The arch's [`SpecTarget`] impl
/// recovers it via `scratch.as_any_mut().downcast_mut::<T>()`.
pub trait SpecScratch {
    /// Downcast hook for the owning [`SpecTarget`] impl.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Release all GPU buffers the scratch owns. Explicit because `GpuTensor` /
    /// `DeviceBuffer` have no `Drop` in this codebase — a bare `drop` of the box
    /// would orphan device memory. Called from the speculator's `free`.
    fn free(self: Box<Self>, gpu: &mut Gpu);
}

/// Outcome of [`SpecTarget::spec_advance`].
#[derive(Debug, Clone)]
pub enum SpecAdvance {
    /// Advanced to the end; `last_argmax` is the greedy token at the final
    /// position (the first decode seed on prefill; ignored on replay).
    Ready { last_argmax: u32 },
    /// Client cancelled mid-advance; the target reset its own state.
    Aborted,
}

/// Erased grammar-mask interface for tool-call-constrained spec-decode.
///
/// The concrete per-arch grammar `Matcher` types
/// (`hipfire_arch_qwen35::grammar`, `hipfire_arch_deepseek4::grammar`) are
/// crate-local and distinct, and this crate depends on neither — so grammar is
/// threaded through the trait as an erased `&mut dyn SpecGrammar`, not a shared
/// concrete struct (a shared struct would invert the crate dependency graph; an
/// associated type would break `Box<dyn Speculator>`). Marker for now; the
/// mask-fill / accept method set will be defined when a grammar-consuming
/// drafter (MTP / EAGLE) first needs it. `DflashSpeculator` ignores grammar —
/// qwen35 enforces tool-call grammar post-hoc in the daemon.
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

    /// The drafter's speculation window (DFlash block size / MTP K). The daemon
    /// uses it for capacity checks and the decode-loop overflow guard.
    fn block_size(&self) -> usize;

    /// The target's usable context capacity (for the loop overflow guard).
    fn ctx_capacity(&self) -> usize;

    /// Divergent-render checkpoint positions (ascending), for prompt-cache
    /// resume planning. Default empty for drafters with no checkpoint ring.
    fn checkpoint_positions(&self) -> Vec<usize> {
        Vec::new()
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

// ─── Model-free drafting sources (arch-agnostic, pure CPU) ──────────────────
//
// Moved here from `hipfire-arch-qwen35::speculative` so the arch-generic
// `NgramSpeculator` can use them without an arch-crate dependency. qwen35's
// `spec_step_dflash` still uses them via a `pub use` re-export in that crate.

/// Rolling bigram n-gram cache. Keyed by the last two committed tokens
/// `(a, b)`; value is a small map from possible next-token to count.
///
/// Populated incrementally from the committed output stream. Used as a
/// "free" second opinion on top of the DFlash draft: if the cache has
/// seen a (a, b) → c transition with high enough count, and the DFlash
/// draft proposed something else at that position, the n-gram's `c`
/// often turns out to match the target's argmax.
///
/// Scales: the cache size is bounded by the number of distinct bigrams
/// in the committed output — typically a few hundred per session, so
/// no eviction policy needed.
pub struct NgramCache {
    /// `(a, b) → { next: count, ... }` with the next-token histogram.
    pub bigram: std::collections::HashMap<(u32, u32), std::collections::HashMap<u32, u32>>,
    /// Minimum count before we trust the prediction. Smaller = more
    /// aggressive (more overrides), larger = more conservative. 3 is a
    /// reasonable default on hot-loop code / repetitive text.
    pub min_count: u32,
}

impl NgramCache {
    pub fn new(min_count: u32) -> Self {
        Self {
            bigram: std::collections::HashMap::new(),
            min_count,
        }
    }

    /// Record the triple `(a, b) → c` in the cache.
    #[inline]
    pub fn observe(&mut self, a: u32, b: u32, c: u32) {
        *self.bigram.entry((a, b)).or_default().entry(c).or_insert(0) += 1;
    }

    /// Predict `c` from last-two `(a, b)` if the max-count next-token
    /// reaches `min_count`. Returns (token, count).
    #[inline]
    pub fn predict(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let map = self.bigram.get(&(a, b))?;
        let (&tok, &cnt) = map.iter().max_by_key(|(_, &c)| c)?;
        if cnt >= self.min_count {
            Some((tok, cnt))
        } else {
            None
        }
    }

    /// Record every consecutive triple in a slice of committed tokens.
    /// Caller supplies the full token stream; this walks it in-place.
    pub fn observe_many(&mut self, tokens: &[u32]) {
        if tokens.len() >= 3 {
            for w in tokens.windows(3) {
                self.observe(w[0], w[1], w[2]);
            }
        }
    }
}

/// Prompt Lookup Decoding (Saxena 2023): training-free deterministic draft
/// built from context suffix self-match. If the last N tokens of context
/// appeared earlier in context, the tokens that followed that earlier
/// occurrence are a high-quality continuation guess.
///
/// Used as the draft source in Goose bypass mode (Jin et al. 2026,
/// arXiv:2604.02047 §4.3): PLD-matched tokens have 2–18× higher acceptance
/// than bigram (TR) tokens (median 6× across 5 models × 5 benchmarks).
/// When PLD confidence is high, the spine — a deep linear chain of
/// PLD-matched tokens — is verified in one target forward pass without
/// tree construction. That's exactly what we need on Qwen3.5 hybrid
/// (24 DeltaNet + 8 FullAttention): linear verify sidesteps the
/// state-forking problem that tree verify imposes on recurrent LA layers.
pub struct PldMatcher {
    /// n-gram suffix lengths to try, longest first. Paper uses {5,4,3}.
    /// Longer matches are more selective; if the longest fails we fall
    /// back to shorter. Order matters: we return the first (longest) hit.
    pub ngram_lens: Vec<usize>,
    /// Hard cap on spine length. Paper uses 8 — sufficient for typical
    /// block sizes and avoids running off the end of a match into drift.
    pub max_extract: usize,
    /// Minimum extracted length to count as a usable spine. Very short
    /// spines aren't worth the PLD path (bigram covers 1-token lookahead
    /// at lower risk); require at least this many continuation tokens.
    pub min_extract: usize,
}

impl Default for PldMatcher {
    fn default() -> Self {
        Self {
            ngram_lens: vec![5, 4, 3],
            max_extract: 8,
            min_extract: 3,
        }
    }
}

/// Result of a successful PLD lookup.
#[derive(Debug, Clone)]
pub struct PldMatch {
    /// The extracted spine (continuation tokens after the matched suffix).
    pub tokens: Vec<u32>,
    /// The suffix length that produced this match (the longest that hit).
    pub n: usize,
    /// Number of tried n-gram lengths that agreed on `tokens[0]`. Paper
    /// §4.3 uses this as part of the bypass-mode confidence signal;
    /// higher consensus = more reliable spine. Ranges 1..=ngram_lens.len().
    pub consensus: usize,
}

impl PldMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a spine continuation for `context`. Returns `None` if no tried
    /// n-gram length produces a match of length ≥ `self.min_extract`.
    ///
    /// For each n in `self.ngram_lens`: take the last-n tokens as the
    /// suffix, search for its last occurrence earlier in context, and
    /// extract the `max_extract` tokens that followed it (stopping before
    /// the suffix itself so we don't include tokens that would be about
    /// to be re-predicted). Returns the longest-n match with a usable
    /// spine; consensus counts how many alternate n's produced the same
    /// first continuation token.
    pub fn lookup(&self, context: &[u32]) -> Option<PldMatch> {
        if self.ngram_lens.is_empty() {
            return None;
        }
        // Per-n continuation, collected to compute consensus across lengths.
        let mut firsts: Vec<u32> = Vec::with_capacity(self.ngram_lens.len());
        let mut best: Option<(usize, Vec<u32>)> = None; // (n, spine)
        for &n in &self.ngram_lens {
            if context.len() <= n {
                continue;
            }
            let suffix_start = context.len() - n;
            let suffix = &context[suffix_start..];
            let haystack = &context[..suffix_start];
            if haystack.len() < n {
                continue;
            }
            // Last occurrence (freshest) of `suffix` in `haystack`.
            let mut found: Option<usize> = None;
            for i in (0..=haystack.len() - n).rev() {
                if &haystack[i..i + n] == suffix {
                    found = Some(i);
                    break;
                }
            }
            let start = match found {
                Some(s) => s,
                None => continue,
            };
            let cont_start = start + n;
            let cont_end = (cont_start + self.max_extract).min(suffix_start);
            if cont_end <= cont_start {
                continue;
            }
            let spine: Vec<u32> = context[cont_start..cont_end].to_vec();
            if spine.len() < self.min_extract {
                continue;
            }
            firsts.push(spine[0]);
            if best.is_none() {
                best = Some((n, spine));
            }
        }

        let (n, tokens) = best?;
        let consensus = firsts.iter().filter(|&&t| t == tokens[0]).count();
        Some(PldMatch {
            tokens,
            n,
            consensus,
        })
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
