// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Model-free n-gram / PLD speculator — the second [`Speculator`] arm, and the
//! one that justifies the `build_speculator` registry.
//!
//! Unlike [`crate::spec_build::DflashSpeculator`], this drafter carries **no
//! draft model**. It proposes a continuation block from two training-free
//! sources over the committed token history (prompt + emitted):
//!   1. Prompt Lookup Decoding (Saxena 2023) — context-suffix self-match,
//!   2. a rolling bigram chain ([`NgramCache`]) as the fallback,
//! and then **verifies with the target only**, accepting the longest prefix the
//! target's greedy argmax agrees with. Verification is exact, so an over-eager
//! draft only costs τ, never coherence.
//!
//! **Perf is situational on DeltaNet (Qwen3.5) targets** — opt-in for that
//! reason (`HIPFIRE_NGRAM_DRAFT=1`). The verify forward over `b` tokens runs the
//! GatedDeltaNet recurrence *sequentially* (it does not parallelize across the
//! block the way attention does), so a window costs ~`b`× the DeltaNet part for
//! `accept+1` tokens. It only wins when PLD acceptance is high — high
//! prompt-copy content (edit / refactor / verbatim), where measured +15% decode
//! on a 9B copy task. On low-copy "write" prompts the verify cost dominates and
//! it loses to plain AR. (A draft *model* like DFlash wins broadly because its
//! verify uses block-parallel kernels + a cheap GDN-tape rollback; the
//! model-free arm has neither.)
//!
//! State machinery follows `speculative.rs::spec_step_dflash`'s no-tape path:
//! snapshot the target DeltaNet state, run the verify block (over-advances the
//! target by `b`), then:
//!   - **Full accept** (`accept_len == draft.len()`): the verify already left
//!     the target at exactly the next window's position — keep it, no rewind.
//!     This is the high-acceptance path, so it must be the cheap one.
//!   - **Partial accept**: rewind the recurrent state and replay the
//!     `accept_len + 1` committed tokens with the same batched forward the verify
//!     used (consistent GDN numerics), leaving the target at
//!     `position + accept_len + 1`.
//! The bonus token becomes the next window's seed (`block[0]`); the FullAttention
//! KV written past the commit point is stale but harmless — the next verify
//! overwrites it before it can be read as context.
//!
//! The only GPU state it owns is target-side verify scratch: a [`VerifyScratch`]
//! (per-position lm_head + argmax buffers), a [`DeltaNetSnapshot`] (+ a local
//! `s_ef_residual` backup the snapshot type omits) for the rewind, and a
//! [`HiddenStateRingBuffer`] built with `num_extract = 0` so it allocates
//! **zero** hidden buffers — it exists only to satisfy `verify_dflash_block`'s
//! required `&mut` arg; the forward writes nothing to it and nothing is read back.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Config};
use hipfire_arch_qwen35::speculative::{
    verify_dflash_block, DeltaNetSnapshot, HiddenStateRingBuffer, ModelSlot, NgramCache,
    PldMatcher, VerifyScratch,
};
use hipfire_runtime::spec::{PrefillOutcome, SpecGrammar, SpecStep, SpecTarget, Speculator};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Single-pass argmax over a logit row.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0
}

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
    // ── target-side verify scratch (GPU) ────────────────────────────────
    verify_scratch: VerifyScratch,
    hidden_rb: HiddenStateRingBuffer,
    target_snap: DeltaNetSnapshot,
    /// Pre-verify backup of the DeltaNet Q8 error-feedback residual
    /// (`DeltaNetState::s_ef_residual`). [`DeltaNetSnapshot`] does NOT cover this
    /// field, so without restoring it the batched verify's `b`-step advance of
    /// s_ef would never be undone — it accumulates ~b extra steps of residual per
    /// window and corrupts the recurrent state into an attractor. Empty when
    /// error-feedback is disabled (`HIPFIRE_DN_STATE_EF=0` / non-Q8 state).
    s_ef_snap: Vec<GpuTensor>,
}

impl NgramSpeculator {
    /// Build the per-target verify scratch from the target config + a snapshot
    /// matching the target's DeltaNet shapes. `block_size` is the n-gram draft
    /// window (incl. seed); `min_count` is the bigram trust threshold.
    pub fn new(
        gpu: &mut Gpu,
        target_config: &Qwen35Config,
        target_dn: &DeltaNetState,
        ctx_capacity: usize,
        block_size: usize,
        min_count: u32,
    ) -> Result<Self, String> {
        let block_size = block_size.max(2);
        let dim = target_config.dim;
        let vocab = target_config.vocab_size;
        let hidden_k = dim.next_power_of_two();
        // max_n = block_size covers the largest verify block (b <= block_size).
        let verify_scratch = VerifyScratch::new(gpu, block_size, dim, vocab, hidden_k)
            .map_err(|e| format!("NgramSpeculator VerifyScratch: {e}"))?;
        // num_extract = 0 ⇒ no hidden buffers allocated; the forward's hidden
        // extraction is a no-op and the ring is never read.
        let hidden_rb = HiddenStateRingBuffer::new(
            gpu,
            target_config.n_layers,
            0,
            dim,
            ctx_capacity,
            block_size,
        )
        .map_err(|e| format!("NgramSpeculator HiddenStateRingBuffer: {e}"))?;
        let target_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
            .map_err(|e| format!("NgramSpeculator DeltaNetSnapshot: {e}"))?;
        // Backup buffers for s_ef_residual (F16), matching the live shapes. Empty
        // vec when error-feedback is off (then there is nothing to restore).
        let mut s_ef_snap = Vec::with_capacity(target_dn.s_ef_residual.len());
        for t in &target_dn.s_ef_residual {
            s_ef_snap.push(
                gpu.alloc_tensor(&t.shape, DType::F16)
                    .map_err(|e| format!("NgramSpeculator s_ef snapshot: {e}"))?,
            );
        }
        // PLD extracts at most block_size-1 continuation tokens (cap b at
        // block_size). min_extract = 1 keeps even short self-matches usable —
        // exact verify gates them anyway.
        let pld = PldMatcher {
            ngram_lens: vec![5, 4, 3],
            max_extract: block_size - 1,
            min_extract: 1,
        };
        Ok(Self {
            block_size,
            ctx_capacity,
            ngram: NgramCache::new(min_count),
            pld,
            prompt: Vec::new(),
            verify_scratch,
            hidden_rb,
            target_snap,
            s_ef_snap,
        })
    }

    /// Copy the live s_ef_residual into the backup (pre-verify).
    fn save_s_ef(&mut self, gpu: &mut Gpu, dn: &DeltaNetState) -> Result<(), String> {
        for (dst, src) in self.s_ef_snap.iter().zip(dn.s_ef_residual.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Restore s_ef_residual from the backup (post-verify, pre-replay).
    fn restore_s_ef(&self, gpu: &mut Gpu, dn: &DeltaNetState) -> Result<(), String> {
        for (src, dst) in self.s_ef_snap.iter().zip(dn.s_ef_residual.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
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
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("NgramSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Plain target advance over the prompt (miss) or just the new suffix
        // (hit), chunked at PREFILL_MAX_BATCH with abort checks between chunks.
        // No hidden extraction — the n-gram drafter needs only the target's KV
        // + recurrent state advanced, not the DFlash hidden ring.
        if !cache_hit {
            slot.reset_state(gpu);
        }
        let chunk_max = qwen35::PREFILL_MAX_BATCH;
        let mut off = 0usize;
        let mut pos = if cache_hit { prefill_start } else { 0 };
        while off < prefill_tokens.len() {
            if abort() {
                slot.reset_state(gpu);
                return Ok(PrefillOutcome::Aborted);
            }
            let end = (off + chunk_max).min(prefill_tokens.len());
            qwen35::forward_prefill_batch(
                gpu,
                &slot.weights,
                &slot.config,
                &prefill_tokens[off..end],
                pos,
                &mut slot.kv_cache,
                &mut slot.dn_state,
                &slot.scratch,
                None,
                None,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
            pos += end - off;
            off = end;
        }

        // Refresh drafter context: keep the full prompt for PLD self-match and
        // seed the bigram cache from it.
        self.prompt.clear();
        self.prompt.extend_from_slice(prompt_tokens);
        self.ngram.observe_many(prompt_tokens);

        // First token = target argmax at the last prompt position (the per-token
        // forward left last-token logits in scratch.logits).
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        Ok(PrefillOutcome::Ready {
            first_token: argmax(&logits),
        })
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
            .ok_or("NgramSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Full context = prompt ++ emitted; its last token is `seed`.
        let mut ctx = Vec::with_capacity(self.prompt.len() + emitted.len());
        ctx.extend_from_slice(&self.prompt);
        ctx.extend_from_slice(emitted);

        // block = [seed, draft..] ; b = block.len() in 1..=block_size.
        let draft = self.build_draft(&ctx);
        let mut block = Vec::with_capacity(draft.len() + 1);
        block.push(seed);
        block.extend_from_slice(&draft);

        // Snapshot DeltaNet pre-verify (incl. the s_ef residual the snapshot
        // type omits), then verify (advances target by b).
        self.target_snap
            .save_from(&slot.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        self.save_s_ef(gpu, &slot.dn_state)?;
        let out = verify_dflash_block(
            gpu,
            slot,
            &block,
            position,
            &mut self.hidden_rb,
            None,  // gdn_tape: rewind by replay below, no tape
            false, // greedy: GPU argmax, no full-logit D2H
            &self.verify_scratch,
        )
        .map_err(|e| e.to_string())?;

        // Greedy acceptance: longest prefix where argmax[i] == block[i+1].
        let mut accept_len = 0usize;
        while accept_len < draft.len() && out.argmax_per_pos[accept_len] == block[accept_len + 1] {
            accept_len += 1;
        }
        let bonus = out.argmax_per_pos[accept_len];

        // committed = [seed, accepted drafts.., bonus] (len = accept_len + 2).
        let mut committed = Vec::with_capacity(accept_len + 2);
        committed.push(seed);
        committed.extend_from_slice(&draft[..accept_len]);
        committed.push(bonus);

        // State fixup after verify, which advanced the target by `b` tokens.
        //
        // FULL ACCEPT (accept_len == draft.len()): the verify already left the
        // target at exactly `position + b` = the next window's position, and the
        // bonus is the next seed (predicted, not yet fed). Nothing to undo — skip
        // the rewind entirely. This is the high-acceptance case where spec-decode
        // actually wins, so it must be the cheap path.
        //
        // PARTIAL ACCEPT: the verify over-advanced by `b - (accept_len + 1)`
        // rejected tokens. Rewind the recurrent state (incl. the s_ef residual the
        // snapshot omits) to pre-verify and replay the committed prefix with the
        // SAME batched `forward_prefill_batch` the verify ran — its GatedDeltaNet
        // numerics match the verify's, so the recurrent state the next window sees
        // is consistent with the argmax that was just accepted (a per-token replay
        // would advance with different GDN numerics → drift off the verified
        // trajectory). The bonus is NOT replayed — it is next window's block[0].
        // The stale FullAttention KV at [position+accept+1 .. position+b) is
        // overwritten by the next verify before it can be read as context.
        if accept_len < draft.len() {
            self.target_snap
                .restore_to(&mut slot.dn_state, gpu)
                .map_err(|e| e.to_string())?;
            self.restore_s_ef(gpu, &slot.dn_state)?;
            qwen35::forward_prefill_batch(
                gpu,
                &slot.weights,
                &slot.config,
                &committed[..accept_len + 1],
                position,
                &mut slot.kv_cache,
                &mut slot.dn_state,
                &slot.scratch,
                None,
                None,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
        }

        // Grow the bigram cache with the newly committed tokens, including the
        // triples spanning the previous-context boundary.
        let pre = ctx.len().saturating_sub(2);
        let mut window: Vec<u32> = ctx[pre..].to_vec();
        window.extend_from_slice(&committed[1..]);
        self.ngram.observe_many(&window);

        // emit = committed[1..] (seed re-echo stripped); next_seed = bonus.
        Ok(SpecStep::new(
            committed[1..].iter().copied(),
            bonus,
            draft.len(),
            accept_len,
        ))
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Drafter-local reset for a fresh conversation: clear CPU draft state.
        // The verify scratch / snapshot are reusable GPU buffers — kept.
        self.ngram = NgramCache::new(self.ngram.min_count);
        self.prompt.clear();
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let NgramSpeculator {
            verify_scratch,
            hidden_rb,
            target_snap,
            s_ef_snap,
            ..
        } = *self;
        for t in s_ef_snap {
            let _ = gpu.free_tensor(t);
        }
        verify_scratch.free_gpu(gpu);
        // `HiddenStateRingBuffer` has no `free_gpu`; free its buffers directly.
        // Both vecs are empty here (num_extract = 0) — this is a no-op that
        // stays correct if the extract count ever changes.
        for t in hidden_rb.layer_bufs {
            let _ = gpu.free_tensor(t);
        }
        for t in hidden_rb.staging_bufs {
            let _ = gpu.free_tensor(t);
        }
        target_snap.free_gpu(gpu);
    }
}
