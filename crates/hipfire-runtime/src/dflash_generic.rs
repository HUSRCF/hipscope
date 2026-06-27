// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Target-generic chain-mode DFlash speculator.
//!
//! This is the arch-free twin of `hipfire_arch_qwen35::speculative::spec_step_dflash`
//! / `dflash_spec::DflashSpeculator`. It drives the SAME block-diffusion drafter
//! forward ([`crate::dflash`]) but verifies through the arch-generic
//! [`SpecTarget`] trait instead of a concrete qwen35 `ModelSlot`.
//!
//! Because the target is reached only through `SpecTarget`, the whole
//! DeltaNet-specific apparatus the qwen35 path carries — recurrent snapshot
//! ([`DeltaNetSnapshot`]), the GDN innovation tape ([`GdnTape`]), the hidden-state
//! ring buffer, and the post-verify rewind/replay — DISAPPEARS. For a stateless
//! dense-attention target (LLaMA / plain Qwen3) verify is one block-parallel
//! forward whose accepted-prefix KV is already correct; nothing to rewind.
//!
//! Mechanically the generic skeleton extracted from `spec_step_dflash` is:
//!   1. build the masked block `[seed, MASK, …, MASK]`,
//!   2. draft it in ONE [`draft_forward`] (broadcast the target's mask-token
//!      embedding as the noise input; positions per the qwen35 path),
//!   3. derive `drafts` by applying the TARGET lm_head to the draft hidden rows
//!      (`draft_scratch.x` rows `1..b`) and argmax-ing,
//!   4. verify `[seed, drafts…]` through [`SpecTarget::verify_block`], which
//!      returns the per-position target argmax AND the per-position hidden rows,
//!   5. greedy-accept the longest matching prefix + bonus, append ONLY the
//!      committed-prefix hidden to `target_hidden_host`, and advance.
//!
//! Chain (linear) verify only — greedy, temp-0. The temp>0 SWOR path is a later
//! milestone (M5), so [`requires_greedy`](Speculator::requires_greedy) is `true`
//! and [`supports_temp_verify`](Speculator::supports_temp_verify) is `false`.

use crate::dflash::{draft_forward, DflashConfig, DflashScratch, DflashWeights};
use crate::hfq::HfqFile;
use crate::llama;
use crate::spec::{
    accept_greedy_prefix, PrefillOutcome, SpecAdvance, SpecGrammar, SpecScratch, SpecStep,
    SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::path::Path;

/// Target-generic chain-mode DFlash speculator.
///
/// Owns the loaded draft weights/scratch/config, the cumulative target-hidden
/// host buffer (`[committed_rows × num_extract × hidden]` f32, row-major,
/// extract-layers ascending), and the arch-specific verify scratch the target
/// minted via [`SpecTarget::new_spec_scratch`]. The target itself is borrowed
/// per call — never owned — exactly like the qwen35 `DflashSpeculator`.
pub struct GenericDflashSpeculator {
    weights: DflashWeights,
    scratch: DflashScratch,
    config: DflashConfig,
    /// Cumulative committed target-hidden rows. Authoritative CPU shadow handed
    /// to `draft_forward` each cycle (the generic path always uses the host
    /// buffer — there is no D2D scatter fast path here, unlike qwen35's GPU-side
    /// hidden_rb). Grows by exactly `accept+1` rows per accepted cycle.
    target_hidden_host: Vec<f32>,
    /// Per-target verify scratch, minted by `SpecTarget::new_spec_scratch`.
    /// `Option` so `free` can move it out for explicit GPU release.
    verify_scratch: Option<Box<dyn SpecScratch>>,
    block_size: usize,
    ctx_capacity: usize,
}

impl GenericDflashSpeculator {
    fn num_extract(&self) -> usize {
        self.config.num_extract()
    }
}

/// Pure (testable) truncation math for the partial-accept host-buffer update.
///
/// After a cycle that committed `committed_len` tokens (= `accepted + 2`:
/// `[seed, accepted drafts…, bonus]`), `draft_forward`'s incremental contract
/// grows the cached prefix by exactly `accepted + 1` rows. So the host buffer
/// must hold `(position + accepted + 1) × row_stride` floats, where
/// `row_stride = num_extract × hidden`. The verify produced `block_hidden` for
/// all `b+1 = drafts+1` positions; we keep the first `accepted + 1` rows and
/// discard the rejected tail.
///
/// Returns the number of f32 elements the host buffer should have after the
/// append. Factored out so the truncation invariant is unit-testable without a
/// GPU.
fn committed_host_len(
    position: usize,
    accepted: usize,
    num_extract: usize,
    hidden: usize,
) -> usize {
    (position + accepted + 1) * num_extract * hidden
}

/// Number of leading f32 elements of a verify `block_hidden` buffer that belong
/// to the committed prefix (the first `accepted + 1` rows). The rejected tail
/// (rows `accepted+1 ..= drafts`) is discarded.
fn committed_block_hidden_elems(accepted: usize, num_extract: usize, hidden: usize) -> usize {
    (accepted + 1) * num_extract * hidden
}

impl Speculator for GenericDflashSpeculator {
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
        // Mirror the qwen35 prefill cache-hit/miss split, minus the DeltaNet /
        // hidden-ring machinery: on a miss we re-seed the whole prompt and reset
        // the draft's incremental-upload cursor; on a hit we advance only the
        // suffix from `prefill_start` and keep the cached prefix projections.
        let (fill_tokens, start_pos): (&[u32], usize) = if cache_hit {
            (prefill_tokens, prefill_start)
        } else {
            (prompt_tokens, 0)
        };

        if !cache_hit {
            // Cold start: drop both the cumulative host shadow and the draft's
            // upload/projection tracking so the first step re-uploads from row 0.
            self.target_hidden_host.clear();
            self.scratch.reset_upload_tracking();
        }

        // Advance the target AND capture its residual hidden into the cumulative
        // host buffer in one pass. `reset = !cache_hit` zeroes the target's KV
        // (and recurrent, for arches that have it) on a miss. Capture only fires
        // when the target's `dflash_extract_layers()` is `Some` (set at build).
        let adv = target.spec_advance(
            gpu,
            fill_tokens,
            start_pos,
            !cache_hit,
            abort,
            Some(&mut self.target_hidden_host),
        )?;
        let first_token = match adv {
            SpecAdvance::Ready { last_argmax } => last_argmax,
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        _temp: f32, // chain verify is greedy-only; temp is an M5 milestone
    ) -> Result<SpecStep, String> {
        let b = self.config.block_size;
        assert!(b >= 2, "dflash block size must be >= 2");
        let h = self.config.hidden;
        let ne = self.num_extract();

        // ── 1. Build the masked block: [seed, MASK, …, MASK] ────────────────
        let mut block: Vec<u32> = vec![self.config.mask_token_id; b];
        block[0] = seed;

        // ── 2. Build the block-diffusion draft INPUT (review finding H1) ────
        // noise_embedding: the target's mask-token embedding row, broadcast across
        // all `b` masked block positions (b×hidden f32). The qwen35 path writes the
        // per-slot embedding directly into draft_scratch.x via D2D; we go through
        // the host (one embed_row lookup) and let draft_forward upload it.
        let mask_row = target.embed_row(gpu, self.config.mask_token_id)?;
        debug_assert_eq!(mask_row.len(), h, "embed_row length != hidden");
        let mut noise_embedding: Vec<f32> = Vec::with_capacity(b * h);
        for _ in 0..b {
            noise_embedding.extend_from_slice(&mask_row);
        }

        // positions_q: absolute positions of the block slots [position .. position+b).
        // positions_k: context positions [0 .. position) then block [position .. position+b).
        // (The generic path has no FlashCASK eviction, so positions are contiguous —
        // matching the qwen35 pre-eviction layout byte-for-byte.)
        let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
        let positions_k: Vec<i32> = (0i32..(position + b) as i32).collect();

        // ── 3. Draft forward over the cumulative target-hidden prefix ───────
        // The host buffer is authoritative: hand draft_forward rows [0..position).
        // Its incremental-upload fast path keys off scratch.uploaded_target_hidden_rows.
        let ctx_elems = position * ne * h;
        assert_eq!(
            self.target_hidden_host.len(),
            ctx_elems,
            "target_hidden_host len {} != position {} * ne {} * h {}",
            self.target_hidden_host.len(),
            position,
            ne,
            h
        );
        draft_forward(
            gpu,
            &self.weights,
            &self.config,
            Some(&noise_embedding),
            Some(&self.target_hidden_host[..ctx_elems]),
            &positions_q,
            &positions_k,
            b,
            position,
            &mut self.scratch,
        )
        .map_err(|e| format!("draft_forward: {e}"))?;

        // ── 3b. Draft → tokens: apply the TARGET lm_head to draft hidden rows ─
        // The draft's final-hidden rows live in draft_scratch.x; row 0 is the seed
        // slot, rows 1..b are the drafted positions (mirrors qwen35's
        // draft_scratch.x.sub_offset(h, batch*h)). We argmax the target lm_head
        // over those b-1 rows to get the drafted tokens.
        let batch = b - 1;
        let draft_hidden = self.scratch.x.sub_offset(h, batch * h);
        let draft_logits = target.lm_head_logits(gpu, &draft_hidden, batch)?;
        debug_assert_eq!(draft_logits.len(), batch * self.config.vocab_size);
        let vocab = self.config.vocab_size;
        let mut drafts: Vec<u32> = Vec::with_capacity(batch);
        for i in 0..batch {
            drafts.push(llama::argmax(&draft_logits[i * vocab..(i + 1) * vocab]));
        }
        for (i, &d) in drafts.iter().enumerate() {
            block[i + 1] = d;
        }

        // ── 4. Verify + accept + truncation (review finding H2) ─────────────
        // Verify [seed, drafts…] through the target. Returns the per-position
        // greedy argmax (length b) AND fills block_hidden with the per-position
        // residual hidden (b rows × ne × h). The target leaves its state advanced
        // by `b`; commit_prefix fixes it to the committed prefix afterward.
        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
        let mut block_hidden: Vec<f32> = Vec::with_capacity(b * ne * h);
        let target_pick =
            target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut block_hidden))?;
        debug_assert_eq!(target_pick.len(), b, "verify_block returned != b argmax");
        debug_assert_eq!(
            block_hidden.len(),
            b * ne * h,
            "verify_block hidden != b*ne*h"
        );

        // Greedy accept: drafts = block[1..b], target_pick is the verifier's argmax
        // after each of the b positions. eos=None — DFlash never early-stops on
        // EOS here (the daemon handles EOS downstream). committed = accepted prefix
        // + bonus (= target_pick[accepted]).
        let acc = accept_greedy_prefix(&drafts, &target_pick, None);
        let accepted = acc.accepted;
        let bonus = *acc.committed.last().expect("eos=None yields a bonus");

        // Fix the target state to the committed prefix [seed, accepted…, bonus].
        // For a stateless target this is a no-op; for a recurrent one it restores
        // the snapshot verify_block stashed in `vs`. accept_len passed is the
        // number of accepted DRAFTS (block[1..=accept_len] accepted).
        target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;

        // Append ONLY the committed-prefix hidden — the first accepted+1 rows of
        // block_hidden (positions [position .. position+accepted], i.e. seed +
        // accepted drafts). The bonus's hidden is NOT appended: its proper hidden
        // materializes on the NEXT cycle's verify when it is forwarded as block[0].
        // This grows the prefix by accept+1, matching draft_forward's contract.
        let keep_elems = committed_block_hidden_elems(accepted, ne, h);
        self.target_hidden_host
            .extend_from_slice(&block_hidden[..keep_elems]);
        // draft_forward owns the uploaded_target_hidden_rows cursor (it delta-uploads the appended host rows next step); the generic path never scatters to GPU itself, so we must NOT set it here.
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, accepted, ne, h),
            "host buffer length mismatch after commit"
        );

        // ── 5. Lower to SpecStep: emit = committed[1..] (accepted drafts + bonus,
        // seed dropped), next_seed = bonus. emit.len() == accepted + 1 drives the
        // daemon's position += emit.len().
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        Ok(SpecStep::new(emit, bonus, drafts.len(), accepted))
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Drafter-local reset: clear the cumulative host shadow + the draft's
        // upload/projection tracking. The target's KV/recurrent reset is the
        // daemon's job (it owns the bundle).
        self.target_hidden_host.clear();
        self.scratch.reset_upload_tracking();
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn supports_temp_verify(&self) -> bool {
        false
    }

    fn requires_greedy(&self) -> bool {
        true
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let GenericDflashSpeculator {
            weights,
            scratch,
            verify_scratch,
            ..
        } = *self;
        weights.free_gpu(gpu);
        scratch.free_gpu(gpu);
        if let Some(vs) = verify_scratch {
            vs.free(gpu);
        }
    }
}

/// Build a target-generic chain DFlash speculator from a converted draft HFQ.
///
/// The draft must be the F16 product of `dflash_convert` (`has_mq = false`), so
/// the scratch is built with [`DflashScratch::new`] (the `new_with_mq` path is
/// only for an MQ-quantized draft — review finding L3; qwen35 regressed exactly
/// this once at `dflash_spec.rs:91`).
///
/// The verify scratch is minted by the TARGET (`new_spec_scratch`) so no arch
/// type leaks here, and the target is told the draft's extract layers via
/// [`SpecTarget::set_dflash_extract_layers`] so hidden capture matches the
/// drafter's `fc` input layout.
pub fn build_generic_dflash_speculator(
    gpu: &mut Gpu,
    draft_hfq_path: &str,
    target: &mut dyn SpecTarget,
    ctx_capacity: usize,
) -> Result<Box<dyn Speculator>, String> {
    let draft_hfq = HfqFile::open(Path::new(draft_hfq_path)).map_err(|e| format!("{e}"))?;
    let config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let weights = DflashWeights::load(gpu, &draft_hfq, &config).map_err(|e| format!("{e}"))?;
    let block_size = config.block_size;
    // L3: F16 drafts (dflash_convert) → has_mq=false → DflashScratch::new.
    // new_with_mq only for an MQ-quantized draft.
    let scratch = if weights.has_mq {
        DflashScratch::new_with_mq(gpu, &config, block_size, ctx_capacity, true)
            .map_err(|e| format!("{e}"))?
    } else {
        DflashScratch::new(gpu, &config, block_size, ctx_capacity).map_err(|e| format!("{e}"))?
    };
    let _ = draft_hfq;

    // Tell the target which residual-hidden layers to capture (the drafter's
    // target_layer_ids), and mint the per-target verify scratch.
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    let verify_scratch = target.new_spec_scratch(gpu, block_size)?;

    Ok(Box::new(GenericDflashSpeculator {
        weights,
        scratch,
        config,
        target_hidden_host: Vec::new(),
        verify_scratch: Some(verify_scratch),
        block_size,
        ctx_capacity,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::SpecStep;
    use smallvec::SmallVec;

    // The SpecStep lowering the chain step produces: emit = accepted drafts +
    // bonus (seed dropped), next_seed = bonus = emit.last(). Mirrors
    // `spec.rs::emit_len_drives_advance_not_accepted` — pins the load-bearing
    // loop contract that the daemon advances `position` by emit.len(), not by
    // `accepted`.
    fn lower_chain(drafts: &[u32], accepted: usize, bonus: u32) -> SpecStep {
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        SpecStep::new(emit, bonus, drafts.len(), accepted)
    }

    #[test]
    fn chain_lowering_emit_len_is_accepted_plus_one() {
        // 4 drafts, accepted 2, bonus 99 → emit = [d0, d1, 99] (len 3).
        let drafts = [10u32, 11, 12, 13];
        let step = lower_chain(&drafts, 2, 99);
        assert_eq!(step.emit.as_slice(), &[10, 11, 99]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(*step.emit.last().unwrap(), step.next_seed);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 4);
        assert_eq!(step.accepted, 2);
    }

    #[test]
    fn chain_lowering_full_accept() {
        // all 3 drafts accepted, bonus 77 → emit = [d0,d1,d2,77] (len 4).
        let drafts = [10u32, 11, 12];
        let step = lower_chain(&drafts, 3, 77);
        assert_eq!(step.emit.as_slice(), &[10, 11, 12, 77]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(step.next_seed, 77);
    }

    #[test]
    fn chain_lowering_zero_accept() {
        // 0 accepted → emit = [bonus] only (still non-empty, forward progress).
        let drafts = [10u32, 11];
        let step = lower_chain(&drafts, 0, 42);
        assert_eq!(step.emit.as_slice(), &[42]);
        assert_eq!(step.emit.len(), 1);
        assert_eq!(step.next_seed, 42);
        assert_eq!(step.accepted, 0);
    }

    // The lowering is byte-equivalent to building a SmallVec by hand (guards the
    // SpecStep::new IntoIterator path against an accidental reorder).
    #[test]
    fn chain_lowering_matches_manual_smallvec() {
        let drafts = [5u32, 6, 7, 8];
        let step = lower_chain(&drafts, 1, 99);
        let manual: SmallVec<[u32; 8]> = SmallVec::from_slice(&[5, 99]);
        assert_eq!(step.emit, manual);
    }

    // Partial-accept truncation math (review finding H2): after a step that
    // accepted `accepted` drafts at absolute `position`, the cumulative host
    // buffer must hold (position + accepted + 1) committed rows × ne × h, and we
    // keep exactly (accepted + 1) rows of the verify's block_hidden.
    #[test]
    fn truncation_keeps_accept_plus_one_rows() {
        let ne = 5usize;
        let h = 4096usize;
        // Simulate appending the committed-prefix hidden onto a host buffer that
        // already holds `position` rows.
        for &(position, drafts, accepted) in
            &[(0usize, 16usize, 0usize), (10, 16, 7), (100, 16, 16)]
        {
            let row_stride = ne * h;
            // verify produced b = drafts+1 rows of hidden.
            let b = drafts + 1;
            let block_hidden = vec![1.0f32; b * row_stride];
            let keep = committed_block_hidden_elems(accepted, ne, h);
            assert_eq!(keep, (accepted + 1) * row_stride);
            // Host buffer pre-step holds `position` rows.
            let mut host = vec![0.0f32; position * row_stride];
            host.extend_from_slice(&block_hidden[..keep]);
            assert_eq!(host.len(), committed_host_len(position, accepted, ne, h));
            assert_eq!(host.len(), (position + accepted + 1) * row_stride);
        }
    }

    // Full-accept and zero-accept boundaries of the truncation math.
    #[test]
    fn truncation_boundaries() {
        let ne = 2usize;
        let h = 8usize;
        // zero accept → keep exactly 1 row (the seed position's hidden).
        assert_eq!(committed_block_hidden_elems(0, ne, h), ne * h);
        // full accept of 15 drafts → keep 16 rows.
        assert_eq!(committed_block_hidden_elems(15, ne, h), 16 * ne * h);
        // host length advances by accept+1 rows from position.
        assert_eq!(committed_host_len(50, 3, ne, h), (50 + 4) * ne * h);
    }
}
