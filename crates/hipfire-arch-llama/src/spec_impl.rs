// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for LlamaBundle` lets the model-free `NgramSpeculator` drive
//! a dense-attention target with no arch knowledge. Pure attention makes this the
//! *cheap* spec case: `verify_block` runs ONE block-parallel batched forward
//! (`llama::verify_block_argmax`), there is no recurrent state to snapshot, and
//! `commit_prefix` is a no-op — the accepted-prefix KV the verify wrote is
//! already correct and the rejected tail is overwritten by the next verify. So
//! the verify of a `b`-token block costs ~one token's forward latency, and
//! model-free spec wins broadly here (unlike the qwen35 DeltaNet target, where
//! the recurrence serializes the verify).

use crate::LlamaBundle;
use hipfire_runtime::llama::{self, KvCache, PrefillBatchScratch};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// LLaMA target-verify scratch: just the per-block batched-forward scratch
/// (`PrefillBatchScratch`). No recurrent snapshot — pure attention.
pub struct LlamaSpecScratch {
    pbs: PrefillBatchScratch,
}

impl SpecScratch for LlamaSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        self.pbs.free_gpu(gpu);
    }
}

impl SpecTarget for LlamaBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // Pure attention: no recurrent state to zero. Drop the KV eviction offset
        // so the next conversation rotates from absolute 0.
        self.kv.compact_offset = 0;
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(2);
        let pbs = PrefillBatchScratch::new(gpu, &self.config, block_size, self.kv.physical_cap)
            .map_err(|e| format!("LlamaSpecScratch PrefillBatchScratch: {e:?}"))?;
        Ok(Box::new(LlamaSpecScratch { pbs }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Pure attention: "reset" just rewinds the eviction offset; the prefill
        // forward overwrites KV at the absolute positions it writes.
        if reset {
            self.kv.compact_offset = 0;
        }
        // DFlash hidden capture: only when the drafter configured extract layers
        // AND the caller passed a sink. The two together form the gate; either
        // missing → no capture (the `_hidden_out` ignored default of Task 1).
        let extract = self.dflash_extract_layers.clone();
        let want_capture = !extract.is_empty() && hidden_out.is_some();
        let chunk_max = llama::PREFILL_MAX_BATCH;
        let mut off = 0usize;
        let mut pos = start_pos;
        while off < tokens.len() {
            if abort() {
                self.kv.compact_offset = 0;
                return Ok(SpecAdvance::Aborted);
            }
            let end = (off + chunk_max).min(tokens.len());
            let mut sink = if want_capture {
                Some(llama::HiddenCaptureSink {
                    extract_layers: &extract,
                    hidden: hidden_out.as_deref_mut().unwrap(),
                })
            } else {
                None
            };
            llama::forward_prefill_batch_capture(
                gpu,
                &self.weights,
                &self.config,
                &tokens[off..end],
                pos,
                &mut self.kv,
                &self.scratch,
                None,
                sink.as_mut(),
            )
            .map_err(|e| format!("{e:?}"))?;
            pos += end - off;
            off = end;
        }
        // forward_prefill_batch leaves last-row logits in scratch.logits.
        let logits = gpu
            .download_f32(&self.scratch.logits)
            .map_err(|e| format!("{e:?}"))?;
        Ok(SpecAdvance::Ready {
            last_argmax: llama::argmax(&logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_block: scratch is not LlamaSpecScratch")?;
        let extract = &self.dflash_extract_layers;
        let mut sink = match hidden_out {
            Some(h) if !extract.is_empty() => Some(llama::HiddenCaptureSink {
                extract_layers: extract,
                hidden: h,
            }),
            _ => None,
        };
        hipfire_runtime::llama_spec::verify_block_argmax(
            gpu,
            &self.weights,
            &self.config,
            block,
            position,
            &mut self.kv,
            &self.scratch,
            &s.pbs,
            sink.as_mut(),
        )
        .map_err(|e| format!("{e:?}"))
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
        self.config.eos_token
    }

    fn ctx_capacity(&self) -> usize {
        self.kv.physical_cap
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.kv)
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        if self.dflash_extract_layers.is_empty() {
            None
        } else {
            Some(&self.dflash_extract_layers)
        }
    }
}
