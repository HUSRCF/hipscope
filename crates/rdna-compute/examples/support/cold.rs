// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Cold-cache timing shared by `hip_loop` (lane nodes) and `pm_loop` (`iu4_gemm`, `qkvzaset`):
//! the flush buffers and eviction, the rotating weight-pool sizing, and the record fields.
//!
//! Production runs a verify GEMM against a cold L2 / MALL / TLB / instruction cache: the other ~25
//! kernels of a decode step evict it between two launches of the same GEMM, and each layer reads its
//! OWN weights. A hot loop (the same weights, back to back) lets a <= 64 MiB weight matrix stay
//! resident in the 64 MB last-level cache and under-reads DRAM-bound kernels (`pm_loop`, N=64
//! `full_add_b1s`: 100.6 -> 62.6 us for a change that read 122.3 -> 98.4 us in the model).
//!
//! * **Flush**: a 2 x [`FLUSH_BYTES`] device-to-device copy after every timed dispatch, enqueued on
//!   the default stream OUTSIDE the per-dispatch hipEvent interval ([`Flush::evict`]).
//! * **Pool**: distinct seeded weight sets whose total exceeds [`POOL_BYTES`]; dispatch `i` uses set
//!   `i % sets` ([`pool_sets`], odd so ABBA arms do not alias set parity).
//!
//! Included with `#[path = "support/cold.rs"] mod cold;`.
#![allow(dead_code)]

use hip_bridge::{DeviceBuffer, HipRuntime};
use serde_json::{json, Value};

/// A timing pool's weight sets together exceed this many bytes (production runs against a cold
/// last-level cache: the other ~25 kernels between two launches of a decode step evict it).
pub const POOL_BYTES: usize = 192 << 20;
/// Bytes of each of the two device-to-device flush buffers.
pub const FLUSH_BYTES: usize = 96 << 20;
/// Most weight sets a pool holds.
pub const MAX_SETS: usize = 161;

/// How many distinct weight sets a timing pool of `set_bytes` each needs: their total exceeds
/// [`POOL_BYTES`], the count is odd (so ABBA arms do not alias set parity), clamped to 3..=161.
pub fn pool_sets(set_bytes: usize) -> usize {
    let mut n = POOL_BYTES / set_bytes.max(1) + 1;
    if n % 2 == 0 {
        n += 1;
    }
    n.clamp(3, MAX_SETS)
}

/// The record's `timing_pool` object (`hip_loop` and `pm_loop` share the key names).
pub fn pool_json(sets: usize, bytes_per_set: usize) -> Value {
    json!({"sets": sets, "bytes_per_set": bytes_per_set,
        "total_bytes": sets * bytes_per_set, "dispatch_i_uses_set": "i % sets"})
}

/// `timing.flush_method`.
pub fn flush_method() -> String {
    format!(
        "2 x {} MiB device-to-device copy after every timed dispatch, outside the per-dispatch hipEvent interval",
        FLUSH_BYTES >> 20
    )
}

/// The two flush buffers: [`Flush::evict`] copies `a` into `b`, streaming `2 x FLUSH_BYTES` through
/// L2, the last-level cache and the TLBs, and running the copy kernel through the instruction cache.
pub struct Flush {
    a: DeviceBuffer,
    b: DeviceBuffer,
}

impl Flush {
    pub fn alloc(hip: &HipRuntime) -> Result<Self, String> {
        let a = hip.malloc(FLUSH_BYTES).map_err(|e| e.to_string())?;
        match hip.malloc(FLUSH_BYTES) {
            Ok(b) => Ok(Self { a, b }),
            Err(e) => {
                let _ = hip.free(a);
                Err(e.to_string())
            }
        }
    }

    pub fn evict(&self, hip: &HipRuntime) -> Result<(), String> {
        hip.memcpy_dtod(&self.b, &self.a, FLUSH_BYTES)
            .map_err(|e| e.to_string())
    }

    pub fn free(self, hip: &HipRuntime) {
        let _ = hip.free(self.a);
        let _ = hip.free(self.b);
    }
}

/// Evict through `flush` when present (the timing loops' `evict` step).
pub fn evict(hip: &HipRuntime, flush: Option<&Flush>) -> Result<(), String> {
    match flush {
        Some(f) => f.evict(hip),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_exceeds_the_last_level_cache_with_an_odd_count() {
        for set_bytes in [
            1usize << 20,
            20 << 20,
            45 << 20,
            47_349_760,
            94_699_520,
            300 << 20,
        ] {
            let n = pool_sets(set_bytes);
            assert_eq!(n % 2, 1, "{set_bytes}");
            assert!((3..=MAX_SETS).contains(&n));
            assert!(n * set_bytes > POOL_BYTES || n == MAX_SETS, "{set_bytes}");
        }
        assert_eq!(pool_sets(0), MAX_SETS);
        assert_eq!(pool_sets(usize::MAX), 3);
    }

    #[test]
    fn record_fields_name_the_pool_and_the_flush() {
        let p = pool_json(5, 47_349_760);
        assert_eq!(p["sets"], 5);
        assert_eq!(p["total_bytes"], 5u64 * 47_349_760);
        assert_eq!(p["dispatch_i_uses_set"], "i % sets");
        assert!(flush_method().starts_with("2 x 96 MiB"));
        assert_eq!(FLUSH_BYTES, 96 << 20);
        assert_eq!(POOL_BYTES, 192 << 20);
    }
}
