// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 implementations of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! Provides `impl SpecTarget for ModelSlot` — the borrowed-verifier hook the
//! daemon's spec loop hands to a `Speculator`. The `DflashSpeculator` impl
//! itself lives in `hipfire-loader` (alongside `DflashState`, which it owns),
//! not here: the orphan rule plus where `DflashState` is defined put it there.

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
