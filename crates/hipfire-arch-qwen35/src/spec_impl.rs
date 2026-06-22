// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 implementations of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! Stage 0: `impl SpecTarget for ModelSlot` only — the borrowed-verifier hook
//! the daemon's spec loop will hand to a `Speculator`. The `DflashSpeculator`
//! itself (which owns `DflashState`) lives in `hipfire-loader` because that is
//! where `DflashState` lives; it lands here in spirit but is wired at Stage 1.

use crate::speculative::ModelSlot;
use hipfire_runtime::spec::SpecTarget;
use rdna_compute::Gpu;

impl SpecTarget for ModelSlot {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) {
        // Reuse the canonical DeltaNet reset (zeroes s_matrices / s_scales /
        // conv_states / s_ef_residual, stream-aware) rather than re-inlining the
        // memset loop the daemon abort path currently hand-writes, then drop the
        // KV eviction offset so the next conversation rotates from absolute 0.
        self.dn_state.reset(gpu);
        self.kv_cache.compact_offset = 0;
    }
}
