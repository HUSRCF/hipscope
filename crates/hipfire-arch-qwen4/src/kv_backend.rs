// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 QSA context storage backend: stable-VA VMM owners with demand-mapped
//! pages, or legacy full-capacity allocations. Load admission resolves the
//! backend; state owners only receive it.

use rdna_compute::Gpu;

/// Storage of Qwen4 QSA context arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4KvBackend {
    /// Full admitted-capacity physical allocations, zeroed at construction.
    Legacy,
    /// Registered VMM owners reserving the admitted capacity; pages are
    /// mapped on demand.
    Vmm,
}

/// The only architecture whose Qwen4 QSA allocator smoke is proven.
const VMM_ARCH: &str = "gfx1151";

/// Capability check only: whether Qwen4 QSA context arenas on `gpu` may be
/// VMM owners. Exactly gfx1151, not Windows (whose driver needs the full-map
/// workaround), the platform VMM KV refusal
/// ([`hipfire_config::devices::vmm_kv_platform_refusal`], the one load
/// admission applies) is clear, and the HIP VMM runtime reports an allocation
/// granularity (admission's `vmm_runtime_available`).
pub(crate) fn qwen4_vmm_supported(gpu: &Gpu) -> bool {
    !cfg!(windows)
        && gpu.arch == VMM_ARCH
        && hipfire_config::devices::vmm_kv_platform_refusal().is_none()
        && gpu.vmm_recommended_granularity().is_ok()
}
