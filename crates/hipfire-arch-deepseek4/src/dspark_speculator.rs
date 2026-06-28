// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 **DSpark** `MtpDrafter` impl — the DSpark draft module wired
//! into the unified MTP spec-decode core ([`hipfire_runtime::spec::MtpDrafter`]
//! + [`MtpSpeculator`]), mirroring [`crate::mtp_speculator::Deepseek4MtpDrafter`].
//!
//! The ONLY difference from the MTP drafter is the DRAFT SOURCE: instead of K
//! iterations of `mtp_forward`, one [`crate::forward::dspark_forward`] call
//! produces all `block_size` draft tokens in a single block-batched pass. The
//! VERIFY + ACCEPT machinery is the shared trunk-forward + `accept_greedy_prefix`
//! used by [`crate::spec_decode`].
//!
//! ## main_hidden bookkeeping (the crux)
//!
//! `dspark_forward(main_hidden@P, prev_token=token@P, position=P)` drafts the
//! tokens at positions `P+1 ..= P+block`. So before drafting at the seed
//! position `P` we need the trunk's captured `[40,41,42]` main_hidden FOR the
//! seed token at `P`. The seed token is freshly committed (it has never been
//! forwarded through the trunk), so we materialize its main_hidden with a single
//! 1-token capture-armed trunk forward, caching its position in
//! `self.main_hidden_pos`. The window's K+1 verify forward also captures, but it
//! captures the *seed + drafts* positions — NOT the next seed (the bonus), which
//! is a brand-new token. Hence the bootstrap forward fires once per window.
//! (Warming the DSpark stage KV rings during prefill — a τ optimisation — is a
//! TODO; see `mtp_prefill`.)

use crate::forward::{self, dspark_assemble_main_hidden, dspark_forward, PrefillBatchScratch};
use crate::mtp_speculator::Deepseek4SpecGrammar;
use crate::spec_decode::logits_argmax;
use crate::spec_impl::Deepseek4Bundle;
use hipfire_runtime::spec::{
    accept_greedy_prefix, MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// DeepSeek V4 DSpark drafter. Holds its own trunk-sized `PrefillBatchScratch`
/// (the verify + bootstrap forwards run through it) allocated lazily on the
/// first `mtp_prefill`. `main_hidden_pos` tracks which absolute position the
/// seed's main_hidden currently in `state.dspark_main_hidden` belongs to, so a
/// window can skip the bootstrap forward when it's already in sync (it never is
/// today — each window's next seed is a fresh token — but the guard keeps the
/// contract explicit and makes a future fold cheap).
pub struct Deepseek4DsparkDrafter {
    pbs: Option<PrefillBatchScratch>,
    /// Absolute position of the seed token whose main_hidden lives in
    /// `state.dspark_main_hidden`. `None` ⇒ must bootstrap.
    main_hidden_pos: Option<usize>,
    block: usize,
    ctx_capacity: usize,
}

impl Deepseek4DsparkDrafter {
    pub fn new(block: usize, ctx_capacity: usize) -> Self {
        Self {
            pbs: None,
            main_hidden_pos: None,
            block: block.clamp(1, 8),
            ctx_capacity,
        }
    }

    fn bundle(target: &mut dyn SpecTarget) -> Result<&mut Deepseek4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Deepseek4Bundle>()
            .ok_or_else(|| "Deepseek4DsparkDrafter: target is not a Deepseek4Bundle".to_string())
    }

    fn pbs_max_batch() -> usize {
        std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024)
    }
}

impl MtpDrafter for Deepseek4DsparkDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        if !cache_hit {
            target.reset_recurrent(gpu);
            self.main_hidden_pos = None;
        }

        if self.pbs.is_none() {
            let bundle = Self::bundle(target)?;
            self.pbs = Some(
                PrefillBatchScratch::new(gpu, &bundle.config, Self::pbs_max_batch())
                    .map_err(|e| format!("Deepseek4DsparkDrafter: alloc PBS: {e}"))?,
            );
        }

        let bundle = Self::bundle(target)?;
        let Deepseek4Bundle {
            config,
            weights,
            state,
            ..
        } = bundle;

        if weights.dspark.is_none() {
            return Err("Deepseek4DsparkDrafter: weights.dspark is None".into());
        }

        // Arm the [40,41,42] target-hidden capture for the prefill forward.
        state.dspark_target_layers = weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .target_layer_ids
            .clone();
        state.dspark_capture_active = true;

        let pbs = self.pbs.as_ref().expect("just built");
        // Strict batched-only trunk prefill with capture armed (same path the
        // validated dspark_forward_smoke uses). Returns the LAST position's
        // trunk logits; their argmax is the AR seed.
        let last_logits = forward::forward_prefill_batch_chunked(
            config,
            weights,
            state,
            gpu,
            fill_tokens,
            start_pos as u32,
            pbs,
        )
        .map_err(|e| format!("dspark prefill: {e}"))?;

        // Assemble main_hidden for the LAST prefilled position (the seed's
        // PREDECESSOR — the trunk position whose logits produced the seed). The
        // seed itself sits one position later and is materialised on the first
        // mtp_step via the bootstrap forward, so we leave main_hidden_pos = None.
        //
        // TODO(perf): warm each DSpark stage's main_kv SWA ring over the prompt
        // here (the reference primes the stage caches during prefill). Skipped
        // for now — single-step drafting is correct without it; multi-step τ may
        // be lower than the warmed ceiling until this lands.
        self.main_hidden_pos = None;

        Ok(logits_argmax(&last_logits) as u32)
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // DSpark drafts the whole block at once; the in-step grammar (tool-call)
        // path is not yet wired for DSpark — downcast to surface a wrong pairing
        // loudly rather than silently dropping the mask, but otherwise ignore it
        // (the daemon's post-hoc emission-layer grammar still applies).
        if let Some(g) = grammar {
            let _ = g
                .as_any_mut()
                .downcast_mut::<Deepseek4SpecGrammar>()
                .ok_or("Deepseek4DsparkDrafter: grammar handle is not a Deepseek4SpecGrammar")?;
        }

        let bundle = Self::bundle(target)?;
        // Detach config (small, Clone) so it doesn't pin `&bundle`. The remaining
        // accesses go through disjoint field paths (`bundle.weights.*` immutable,
        // `bundle.state` mutable) which the borrow checker allows.
        let config = bundle.config.clone();

        if bundle.weights.dspark.is_none() {
            return Err("Deepseek4DsparkDrafter: weights.dspark is None".into());
        }
        let target_layers = bundle
            .weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .target_layer_ids
            .clone();
        let block = bundle
            .weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .block_size
            .min(k)
            .max(1);

        // Trunk tensors (embedding / lm_head / output norm). shallow_clone()
        // detaches them from the `weights` borrow so they can coexist with the
        // `&mut state` the forwards take.
        let token_embd = bundle
            .weights
            .token_embd
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.token_embd is None")?
            .shallow_clone();
        let head = bundle
            .weights
            .head
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.head is None")?
            .shallow_clone();
        let output_norm = bundle
            .weights
            .output_norm
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.output_norm is None")?
            .shallow_clone();

        // Read the in-sync guard before borrowing pbs (which pins `self`); all
        // writes to `self.main_hidden_pos` happen after pbs's last use (step 5).
        let need_bootstrap = self.main_hidden_pos != Some(position);
        let pbs = self
            .pbs
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: mtp_step before mtp_prefill")?;

        // ── 1. Ensure main_hidden@position for the seed ─────────────────────
        // The seed is a fresh token; materialise its captured [40,41,42] hidden
        // with a single 1-token capture-armed trunk forward. (Guard lets a
        // future verify-fold skip this when already in sync.)
        if need_bootstrap {
            bundle.state.dspark_target_layers = target_layers.clone();
            bundle.state.dspark_capture_active = true;
            forward::forward_prefill_batch_chunk(
                &config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                pbs,
                &[seed],
                position as u32,
            )
            .map_err(|e| format!("dspark bootstrap forward: {e}"))?;
            dspark_assemble_main_hidden(&mut bundle.state, gpu, &config, 0)
                .map_err(|e| format!("dspark assemble bootstrap main_hidden: {e}"))?;
        }

        // ── 2. Draft the block with DSpark ──────────────────────────────────
        let main_hidden = bundle
            .state
            .dspark_main_hidden
            .as_ref()
            .ok_or("dspark: main_hidden missing after bootstrap")?
            .shallow_clone();
        let draft = dspark_forward(
            &config,
            bundle.weights.dspark.as_ref().unwrap(),
            &mut bundle.state,
            gpu,
            &main_hidden,
            &token_embd,
            &head,
            &output_norm,
            seed,
            position as u32,
        )
        .map_err(|e| format!("dspark draft: {e}"))?;
        let drafts: Vec<u32> = draft.tokens.into_iter().take(block).collect();
        let n_proposed = drafts.len();

        // ── 3. Verify: trunk forward [seed, draft0..draft_{n-1}] ────────────
        // Placed at their TRUE trunk positions (seed@position, drafts at
        // position+1..). Capture armed so the verify pass also refreshes the
        // captures, though the next seed (bonus) is a fresh token captured by the
        // next window's bootstrap forward.
        let verify_tokens: Vec<u32> = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect();
        if pbs.max_batch < verify_tokens.len() {
            return Err(format!(
                "dspark verify: PBS max_batch ({}) < verify len ({})",
                pbs.max_batch,
                verify_tokens.len()
            ));
        }
        bundle.state.dspark_target_layers = target_layers.clone();
        bundle.state.dspark_capture_active = true;
        forward::forward_prefill_batch_chunk(
            &config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            pbs,
            &verify_tokens,
            position as u32,
        )
        .map_err(|e| format!("dspark verify forward: {e}"))?;

        let all_logits = forward::final_norm_and_head_all_batched(
            &config,
            &bundle.weights,
            &mut bundle.state,
            pbs,
            gpu,
            verify_tokens.len(),
        )
        .map_err(|e| format!("dspark verify head: {e}"))?;

        // ── 4. Greedy accept (shared core). target_pick[i] = argmax at verify
        //    slot i = the trunk's prediction for position+i+1. EOS-aware so an
        //    accepted EOS draft stops the window without a stale bonus. ─────────
        let target_pick: Vec<u32> = all_logits.iter().map(|l| logits_argmax(l) as u32).collect();
        let acc = accept_greedy_prefix(&drafts, &target_pick, Some(eos));
        let committed = acc.committed;
        let n_accepted = acc.accepted;

        // ── 5. Advance trunk position + invalidate the next seed's main_hidden.
        // The verify forward wrote ring slots position..position+n_proposed using
        // (possibly rejected) drafts; only the first committed.len() are real.
        // The next window's bootstrap forward overwrites the next-seed slot.
        bundle.state.n_tokens = (position + committed.len()) as u64;
        // The next seed (committed.last()) is a fresh token — its main_hidden is
        // NOT in the capture buffer, so force a bootstrap next window.
        self.main_hidden_pos = None;

        Ok(MtpWindow {
            committed,
            accepted: n_accepted,
            drafts_generated: n_proposed,
        })
    }

    fn mtp_reset(&mut self, _gpu: &mut Gpu) {
        // No drafter-local conversation state beyond `pbs` (scratch). The target
        // bundle's recurrent reset is the daemon's job. Invalidate the cached
        // main_hidden position so the next prefill re-bootstraps cleanly.
        self.main_hidden_pos = None;
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(pbs) = self.pbs {
            pbs.free_gpu(gpu);
        }
    }

    fn k(&self) -> usize {
        self.block
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }
}

/// Build the deepseek4 DSpark speculator (the boxed `dyn Speculator` the loader
/// returns when a `-dspark` sidecar is present). The trunk-sized
/// `PrefillBatchScratch` is allocated lazily on the first `mtp_prefill`.
pub fn build_deepseek4_dspark_speculator(block: usize, ctx_capacity: usize) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Deepseek4DsparkDrafter::new(
        block,
        ctx_capacity,
    )))
}
