// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Model-free n-gram / PLD speculator — the arch-agnostic [`Speculator`] arm.
//!
//! Carries **no draft model**. It proposes a continuation block from two
//! training-free sources over the committed token history (prompt + emitted):
//! (1) Prompt Lookup Decoding (Saxena 2023) — context-suffix self-match, and
//! (2) a rolling bigram chain ([`NgramCache`]) as the fallback. It then verifies
//! with the **target only**, accepting the longest prefix the target's greedy
//! argmax agrees with. Verification is exact, so an over-eager draft only costs
//! τ, never coherence.
//!
//! This type is **100% arch-agnostic**: it never names a `ModelSlot`, a forward
//! function, or a verify-scratch type. All target mechanics — the batched verify
//! forward, the per-position lm_head/argmax, the recurrent snapshot/rewind, and
//! the arch-specific GPU scratch — live behind [`SpecTarget`]
//! (`verify_block` / `commit_prefix` / `spec_advance` / `new_spec_scratch`). Any
//! arch that implements `SpecTarget` gets this drafter for free. The drafter owns
//! only *policy*: drafting + acceptance.
//!
//! **Perf is situational on recurrent (DeltaNet) targets** — opt-in for that
//! reason (`HIPFIRE_NGRAM_DRAFT=1`). On those, the verify forward runs the
//! recurrence sequentially over the `b`-token block, so it only wins when PLD
//! acceptance is high (high prompt-copy content). On pure-attention targets the
//! verify is block-parallel, so model-free spec wins broadly.

use crate::spec::{
    accept_greedy_prefix, NgramCache, PldMatcher, PrefillOutcome, SpecAdvance, SpecGrammar,
    SpecScratch, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// Model-free n-gram / PLD drafter. See module docs for the contract.
pub struct NgramSpeculator {
    /// Max block size including the seed: a window verifies `[seed, draft..]`
    /// with `draft.len() <= block_size - 1`, so `b <= block_size`.
    block_size: usize,
    ctx_capacity: usize,
    /// Bigram fallback predictor; seeded from the prompt at prefill and grown
    /// from committed tokens each step.
    ngram: NgramCache,
    /// PLD self-match matcher (primary draft source).
    pld: PldMatcher,
    /// The rendered prompt, kept so PLD/bigram see the full context (prompt +
    /// emitted), not just the decode tail — PLD's biggest wins are copies from
    /// the prompt. Refreshed each `prefill`.
    prompt: Vec<u32>,
    /// Arch-specific target verify scratch, created lazily on first `prefill`
    /// via [`SpecTarget::new_spec_scratch`] (the target isn't available at
    /// construction). Reused across requests; freed in [`Speculator::free`].
    scratch: Option<Box<dyn SpecScratch>>,
}

impl NgramSpeculator {
    /// `block_size` is the n-gram draft window (incl. seed); `min_count` is the
    /// bigram trust threshold. No GPU / arch types — scratch is lazy.
    pub fn new(block_size: usize, ctx_capacity: usize, min_count: u32) -> Self {
        let block_size = block_size.max(2);
        // PLD extracts at most block_size-1 continuation tokens (cap b at
        // block_size). min_extract = 1 keeps even short self-matches usable —
        // exact verify gates them anyway.
        let pld = PldMatcher {
            ngram_lens: vec![5, 4, 3],
            max_extract: block_size - 1,
            min_extract: 1,
        };
        Self {
            block_size,
            ctx_capacity,
            ngram: NgramCache::new(min_count),
            pld,
            prompt: Vec::new(),
            scratch: None,
        }
    }

    /// Propose up to `block_size - 1` continuation tokens after `ctx`'s last
    /// token (the seed). PLD first (longest self-match), bigram chain as
    /// fallback. Returns an empty vec when neither source fires (→ pure AR step).
    fn build_draft(&self, ctx: &[u32]) -> Vec<u32> {
        let max_draft = self.block_size - 1;
        if let Some(m) = self.pld.lookup(ctx) {
            let mut d = m.tokens;
            d.truncate(max_draft);
            if !d.is_empty() {
                return d;
            }
        }
        if ctx.len() >= 2 {
            let mut d = Vec::with_capacity(max_draft);
            let mut a = ctx[ctx.len() - 2];
            let mut b = ctx[ctx.len() - 1];
            while d.len() < max_draft {
                match self.ngram.predict(a, b) {
                    Some((c, _)) => {
                        d.push(c);
                        a = b;
                        b = c;
                    }
                    None => break,
                }
            }
            return d;
        }
        Vec::new()
    }
}

impl Speculator for NgramSpeculator {
    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        _resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        // Lazily build the arch-specific verify scratch (target available now).
        if self.scratch.is_none() {
            self.scratch = Some(target.new_spec_scratch(gpu, self.block_size)?);
        }

        // Advance the target over the prompt (miss → reset + full) or just the
        // new suffix (hit → no reset, from prefill_start). The target owns the
        // chunked/abortable forward; we only need its KV + recurrent state moved.
        let start = if cache_hit { prefill_start } else { 0 };
        let adv = target.spec_advance(gpu, prefill_tokens, start, !cache_hit, abort)?;
        let first_token = match adv {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready { last_argmax } => last_argmax,
        };

        // Refresh drafter context: keep the full prompt for PLD self-match and
        // seed the bigram cache from it.
        self.prompt.clear();
        self.prompt.extend_from_slice(prompt_tokens);
        self.ngram.observe_many(prompt_tokens);

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
        // Full context = prompt ++ emitted; its last token is `seed`. Build the
        // draft (pure CPU) BEFORE borrowing scratch so the `&self` draft borrow
        // doesn't collide with the `&mut self.scratch` field borrow below.
        let mut ctx = Vec::with_capacity(self.prompt.len() + emitted.len());
        ctx.extend_from_slice(&self.prompt);
        ctx.extend_from_slice(emitted);
        let draft = self.build_draft(&ctx);

        // block = [seed, draft..] ; b = block.len() in 1..=block_size.
        let mut block = Vec::with_capacity(draft.len() + 1);
        block.push(seed);
        block.extend_from_slice(&draft);

        let scratch = self
            .scratch
            .as_deref_mut()
            .ok_or("NgramSpeculator: step before prefill")?;

        // Verify: target snapshots its pre-state into `scratch`, runs the block,
        // returns per-position greedy argmax, leaves state advanced by b.
        let argmax = target.verify_block(gpu, &block, position, scratch)?;

        // Shared greedy accept-prefix (eos=None: never early-stops, always emits
        // a bonus — EOS is handled downstream by the daemon decode loop).
        // `emit` = accepted drafts ++ bonus = the committed tail.
        let acc = accept_greedy_prefix(&draft, &argmax, None);
        let accept_len = acc.accepted;
        let bonus = *acc
            .committed
            .last()
            .expect("eos=None always yields a bonus");

        // Fix target state to the committed prefix block[..accept_len+1] (the
        // target decides full-accept-skip vs rewind+replay vs no-op internally).
        target.commit_prefix(gpu, &block, accept_len, position, scratch)?;

        // Grow the bigram cache with the new tokens, including the triples
        // spanning the previous-context boundary.
        let pre = ctx.len().saturating_sub(2);
        let mut window: Vec<u32> = ctx[pre..].to_vec();
        window.extend_from_slice(&acc.committed);
        self.ngram.observe_many(&window);

        Ok(SpecStep::new(
            acc.committed.iter().copied(),
            bonus,
            draft.len(),
            accept_len,
        ))
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Drafter-local reset for a fresh conversation: clear CPU draft state.
        // The verify scratch is reusable GPU state — kept across conversations.
        self.ngram = NgramCache::new(self.ngram.min_count);
        self.prompt.clear();
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn free(mut self: Box<Self>, gpu: &mut Gpu) {
        if let Some(scratch) = self.scratch.take() {
            scratch.free(gpu);
        }
    }
}
