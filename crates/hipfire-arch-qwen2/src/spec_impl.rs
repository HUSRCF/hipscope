// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen2-family implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for Qwen2Bundle` lets the model-free `NgramSpeculator` drive
//! a Qwen2 target (e.g. VibeThinker-3B, arch_id=7) with no arch knowledge. Pure
//! GQA attention, no recurrent state, so `commit_prefix` is a no-op (the accepted-
//! prefix KV the verify wrote is already correct; the rejected tail is overwritten
//! by the next verify). Unlike llama/qwen35, Qwen2 keeps its KV in its own
//! `Qwen2State` (not the shared `llama::KvCache`), so [`SpecTarget::kv_cache_mut`]
//! stays at its `None` default — arch_id=7 has no FlashCASK eviction.
//!
//! VERIFY IS CURRENTLY SEQUENTIAL (one `forward_step` per block token), so it is
//! CORRECT — each step's split-K flash-decode attention reads the FULL KV history
//! — but it does NOT yet get the block-parallel speedup llama enjoys. The reason
//! is a kernel gap: qwen2 keeps F32 KV, and the batched-with-history attention
//! kernels in `rdna-compute` are all quantized-KV (q8/asym/fwht); qwen2's only
//! batched attention (`attention_causal_batched`, used by
//! `forward_prefill_batch_embeds`) is INTRA-batch and cannot see prior KV, so it
//! is unusable for a mid-sequence verify. A real block-parallel verify needs an
//! F32-KV batched-decode-with-history attention kernel (tracked follow-up). Until
//! then n-gram on qwen2 is functional/coherent but not a throughput win — keep it
//! opt-in (`HIPFIRE_NGRAM_DRAFT=1`).

use crate::carrier::Qwen2Bundle;
use crate::qwen2;
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// Qwen2 verify scratch: nothing persistent. The sequential verify reuses the
/// bundle's own `Qwen2State` scratch (dense attention → no recurrent snapshot to
/// carry between windows).
pub struct Qwen2SpecScratch;

impl SpecScratch for Qwen2SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for Qwen2Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // Pure attention: no recurrent state to zero. Rewind the KV position
        // cursor so the next prefill writes from slot 0 (O(1); KV is overwritten
        // in place). Mirrors the daemon's arch_id=7 reset handler.
        self.state.reset();
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(Qwen2SpecScratch))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        // Pure attention: "reset" rewinds the position cursor; the per-token
        // prefill then overwrites KV at the absolute positions it writes.
        if reset {
            self.state.reset();
        }
        self.state.next_pos = start_pos;
        for &tok in tokens {
            if abort() {
                self.state.reset();
                return Ok(SpecAdvance::Aborted);
            }
            qwen2::forward_step(gpu, &self.weights, &self.config, &mut self.state, tok)
                .map_err(|e| format!("{e:?}"))?;
        }
        // forward_step leaves the last position's logits in state.logits.
        let last_argmax = gpu
            .argmax_f32(&self.state.logits, self.config.vocab_size)
            .map_err(|e| format!("{e:?}"))?;
        Ok(SpecAdvance::Ready { last_argmax })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<Vec<u32>, String> {
        // Sequential verify: `forward_step(block[i])` predicts the token AFTER
        // block[i] (with block[0..i] already in the KV cache), which is exactly
        // `argmax[i]` — the verifier's pick at slot i. Each step's flash-decode
        // attention reads the FULL KV history (prompt + accepted prefix), so this
        // is correct where a naive intra-batch forward would be blind to the
        // prompt. Position the cursor at `position` first so the writes land at
        // the right absolute slots (overwriting any rejected-tail KV from a prior
        // window). See the module header for why this isn't block-parallel yet.
        self.state.next_pos = position;
        let mut out = Vec::with_capacity(block.len());
        for &tok in block {
            qwen2::forward_step(gpu, &self.weights, &self.config, &mut self.state, tok)
                .map_err(|e| format!("{e:?}"))?;
            out.push(
                gpu.argmax_f32(&self.state.logits, self.config.vocab_size)
                    .map_err(|e| format!("{e:?}"))?,
            );
        }
        Ok(out)
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // rejected tail is overwritten by the next verify. Nothing to rewind.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }

    // kv_cache_mut: defaulted to `None` — Qwen2State is not a `llama::KvCache`,
    // and arch_id=7 has no eviction (the daemon's eviction sites are
    // `if let Some(ev)`-gated, so this is never reached).
}
