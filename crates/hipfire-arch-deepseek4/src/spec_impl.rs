// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 impl of the arch-generic `hipfire_runtime::spec::SpecTarget`.
//!
//! [`Deepseek4Bundle`] owns the model pieces the daemon + MTP drafter need
//! (config + weights + recurrent state + eos) so deepseek4 can be borrowed as a
//! `&mut dyn SpecTarget` exactly like the qwen35 `ModelSlot` — the prerequisite
//! for routing it through the unified spec loop. The MTP draft+verify itself is
//! the [`crate::spec_decode`] fused step, reached by downcasting this bundle in
//! the deepseek4 `MtpDrafter` impl; deepseek4 never pairs with the model-free
//! n-gram drafter, so the n-gram-verify primitives are intentional error stubs.
//!
//! The four DSpark-specific `SpecTarget` hooks (`new_spec_scratch`,
//! `verify_block`, `commit_prefix`, `capture_seed_main_hidden`) ARE
//! implemented here so the generic `DsparkDrafter` in `dspark_core` can
//! route verify + bootstrap through the trait without downcasting — the
//! byte-identical gate depends on these hitting the same kernel paths as
//! the old inline `Deepseek4DsparkDrafter`.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{
    self, dspark_assemble_main_hidden, final_norm_and_argmax_all_batched,
    forward_prefill_batch_chunk, PrefillBatchScratch,
};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// Owned deepseek4 model state — the future `ModelState::Deepseek4` payload and
/// the spec-decode target. Bundles config + weights + recurrent state + eos so
/// the daemon can borrow it as `&mut dyn SpecTarget`.
pub struct Deepseek4Bundle {
    pub config: DeepseekV4Config,
    pub weights: DeepseekV4Weights,
    pub state: DeepseekV4State,
    pub eos_tok: u32,
}

/// Thin verify scratch for the DSpark `DsparkDrafter` path. DeepSeek V4's SWA
/// attention is stateless (no recurrent rewind needed between verify and
/// commit_prefix), so the scratch carries no GPU buffers — the PBS lives in
/// `state.dspark_verify_pbs` and is reused across windows.
pub struct Deepseek4DsparkScratch;

impl SpecScratch for Deepseek4DsparkScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // No GPU buffers owned by this scratch.
    }
}

/// Max batch for the trunk-side verify PBS (bootstrap 1-token + verify up to
/// block+1 tokens). Mirror of `Deepseek4DsparkDrafter::pbs_max_batch`.
fn dspark_verify_pbs_max_batch() -> usize {
    std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}

impl SpecTarget for Deepseek4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // n_tokens → 0 + mtp_last_hidden cleared; the position-indexed KV / SWA /
        // compressed-KV rings are overwritten by the next prefill, never read
        // beyond n_tokens (see `DeepseekV4State::reset`). No GPU work needed.
        self.state.reset();
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.config.max_position_embeddings
    }

    // ── n-gram-verify primitives (intentionally unsupported) ────────────────
    // deepseek4's MTP drafter downcasts this bundle and runs `spec_decode` —
    // those paths never hit these hooks. The DSpark drafter DOES use
    // `new_spec_scratch` / `verify_block` / `commit_prefix`; see below.

    /// Advance the trunk over `tokens` from `start_pos`, returning the greedy
    /// argmax at the last position. Used by `DsparkDrafter::mtp_prefill` to
    /// run the prompt through the trunk in a single pass.
    ///
    /// `reset` is always `false` here — the caller (`DsparkDrafter::mtp_prefill`)
    /// calls `reset_recurrent` separately on cache miss. `abort` and `hidden_out`
    /// are ignored for this arch (deepseek4 is not abort-capable in this path
    /// and does not expose hidden states via `spec_advance`).
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        _reset: bool,
        _abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Lazily allocate the trunk-sized PBS.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle::spec_advance: alloc PBS: {e}"))?,
            );
        }
        // Take the PBS out of state to avoid a simultaneous immutable + mutable
        // borrow of self.state (forward_prefill_batch_chunked takes &mut state).
        // Restore it afterward (it is always Some after the lazy alloc above).
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let result = forward::forward_prefill_batch_chunked(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            tokens,
            start_pos as u32,
            &pbs,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        let last_logits =
            result.map_err(|e| format!("Deepseek4Bundle::spec_advance prefill: {e}"))?;
        let last_argmax = crate::spec_decode::logits_argmax(&last_logits) as u32;
        Ok(SpecAdvance::Ready { last_argmax })
    }

    // ── DSpark verify primitives ──────────────────────────────────────────
    //
    // The generic `DsparkDrafter` in `dspark_core` calls these three methods
    // to verify draft tokens against the trunk. They route to the IDENTICAL
    // kernel paths the old inline `Deepseek4DsparkDrafter` used —
    // `forward_prefill_batch_chunk` + `final_norm_and_argmax_all_batched` —
    // so the byte-identical gate passes without any numeric change.

    /// Allocate the thin DSpark verify scratch. The PBS lives in
    /// `state.dspark_verify_pbs` (lazily allocated here on first call);
    /// `Deepseek4DsparkScratch` itself carries no GPU buffers.
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle: alloc dspark_verify_pbs: {e}"))?,
            );
        }
        Ok(Box::new(Deepseek4DsparkScratch))
    }

    /// Run the trunk forward over `block` at absolute `position`, returning
    /// per-slot target argmaxes. Mirrors `Deepseek4DsparkDrafter::mtp_step`
    /// steps 3–4 exactly: capture armed, `forward_prefill_batch_chunk` then
    /// `final_norm_and_argmax_all_batched`. `hidden_out` is ignored (DSpark
    /// bootstrap uses `capture_seed_main_hidden`, not `hidden_out`).
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        {
            let pbs_ref = self
                .state
                .dspark_verify_pbs
                .as_ref()
                .ok_or("Deepseek4Bundle::verify_block: dspark_verify_pbs not allocated (call new_spec_scratch first)")?;
            if pbs_ref.max_batch < block.len() {
                return Err(format!(
                    "Deepseek4Bundle::verify_block: PBS max_batch ({}) < block len ({})",
                    pbs_ref.max_batch,
                    block.len()
                ));
            }
        }
        // Arm capture so the verify pass populates `state.dspark_caps` —
        // the next window's `capture_seed_main_hidden` will re-capture
        // anyway (bonus is always a fresh token), but keeping capture armed
        // is the exact same behaviour as the old inline drafter.
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable + mutable borrow collision.
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let fwd_result = forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
            block,
            position as u32,
        );
        // Restore the PBS before propagating any error so the state is
        // always consistent on exit.
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result.map_err(|e| format!("Deepseek4Bundle::verify_block forward: {e}"))?;

        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let argmax_result = final_norm_and_argmax_all_batched(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            block.len(),
        );
        self.state.dspark_verify_pbs = Some(pbs);
        argmax_result.map_err(|e| format!("Deepseek4Bundle::verify_block head+argmax: {e}"))
    }

    /// Advance `state.n_tokens` to reflect the committed prefix. DeepSeek
    /// V4's SWA attention is stateless so no recurrent rewind is needed;
    /// the next verify forward simply overwrites the rejected tail slots.
    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        accept_len: usize,
        position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Mirrors the old inline drafter:
        // `bundle.state.n_tokens = (position + committed.len()) as u64`
        // where `committed.len() = accept_len + 1` (accepted drafts + bonus).
        self.state.n_tokens = (position + accept_len + 1) as u64;
        Ok(())
    }

    // ── DSpark bootstrap primitive ─────────────────────────────────────────

    /// Run a 1-token trunk forward with capture armed at `layers`, assemble
    /// the concatenated `[layers.len() * hidden]` main-hidden vector, and
    /// return it as a host-side `Vec<f32>`. Mirrors the bootstrap step of
    /// the old `Deepseek4DsparkDrafter::mtp_step` (steps 1a–1c) exactly.
    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| {
                        format!("Deepseek4Bundle: alloc dspark_verify_pbs (bootstrap): {e}")
                    })?,
            );
        }

        self.state.dspark_target_layers = layers.to_vec();
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable+mutable borrow conflict.
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let fwd_result = forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
            &[seed],
            position as u32,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden forward: {e}"))?;

        dspark_assemble_main_hidden(&mut self.state, gpu, &self.config, 0)
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden assemble: {e}"))?;

        let n = layers.len() * self.config.hidden_size;
        let mut host = vec![0.0f32; n];
        {
            let main_hidden = self
                .state
                .dspark_main_hidden
                .as_ref()
                .ok_or("Deepseek4Bundle::capture_seed_main_hidden: dspark_main_hidden is None after assemble")?;
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &main_hidden.buf)
                .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden d2h: {e:?}"))?;
        }
        Ok(host)
    }
}
