// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! THROWAWAY smoke (not permanent wiring): gfx1201 grid.y/grid.z launch guard.
//!
//! Real-GPU proof of the `hip-bridge` fail-closed guard through the public
//! `Gpu::launch_kernel_blob` entry:
//!   * grid `[1,65535,1]` and `[1,1,65535]` launch and write every expected
//!     word (a canary word past the last row stays untouched);
//!   * grid `[1,65536,1]` and `[1,1,65536]` return `HipError` (code 1, message
//!     names the axis and value) BEFORE the device sees them: the output
//!     buffer stays all-zero after a device sync.
//!
//! The record/capture rejection (dispatch `launch_maybe_blob_bound` /
//! `launch_blob_recorded`, `scratch::launch_maybe_blob`) is not reachable from
//! an example: `ReplayLaunchBindings` lives in the private `dispatch` module
//! and `launch_maybe_blob` is `pub(crate)`. Those funnels call the same
//! `HipRuntime::validate_launch_grid` exercised here.
//!
//! Skips (exit 0) on any arch other than gfx1201: only gfx1201 is guarded.
//!
//! Usage: cargo run --release -p rdna-compute --example tmp_grid_yz_guard_smoke

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

const NAME: &str = "tmp_grid_yz_guard_smoke";
const SRC: &str = r#"
extern "C" __global__ void tmp_grid_yz_guard_smoke(unsigned int* out, int rows) {
    unsigned int idx = blockIdx.z * gridDim.y + blockIdx.y;
    if ((int)idx < rows) {
        out[idx] = 0xA5000000u ^ idx;
    }
}
"#;

/// 65535 rows written; word 65535 is the canary that must stay zero.
const ROWS: usize = 65_535;
const WORDS: usize = ROWS + 1;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("SKIP: arch={} (guard applies to gfx1201 only)", gpu.arch);
        return;
    }
    gpu.ensure_kernel_public(NAME, SRC, NAME).expect("jit");

    let run = |gpu: &mut Gpu, grid: [u32; 3]| -> (Result<(), hip_bridge::HipError>, Vec<u32>) {
        let out = gpu.zeros(&[WORDS], DType::F32).expect("zeros");
        let mut kb = KernargBlob::new();
        kb.push_ptr(out.buf.as_ptr() as *const c_void);
        kb.push_i32(ROWS as i32);
        kb.pad_to(16);
        let r = gpu.launch_kernel_blob(NAME, grid, [1, 1, 1], 0, kb.as_mut_slice());
        gpu.hip.device_synchronize().expect("sync");
        let bytes = gpu.download_raw_bytes(&out).expect("download");
        (r, words(&bytes))
    };

    let mut failures = 0usize;
    let mut check = |ok: bool, what: &str| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    // Boundary acceptance: y = 65535, z = 1 and y = 1, z = 65535.
    for grid in [[1u32, 65_535, 1], [1, 1, 65_535]] {
        let (r, w) = run(&mut gpu, grid);
        check(r.is_ok(), &format!("{grid:?} launch accepted ({r:?})"));
        let rows_ok = (0..ROWS).all(|i| w[i] == (0xA500_0000u32 ^ i as u32));
        check(rows_ok, &format!("{grid:?} wrote all {ROWS} expected words"));
        check(w[ROWS] == 0, &format!("{grid:?} canary word untouched"));
    }

    // Oversized rejection: error before the device, output never touched.
    for (grid, axis) in [([1u32, 65_536, 1], "grid.y=65536"), ([1, 1, 65_536], "grid.z=65536")] {
        let (r, w) = run(&mut gpu, grid);
        match r {
            Ok(()) => check(false, &format!("{grid:?} must be rejected but launched")),
            Err(e) => {
                check(e.code == 1, &format!("{grid:?} HipError code 1 (got {})", e.code));
                check(
                    e.message.contains(axis),
                    &format!("{grid:?} message names {axis}: {}", e.message),
                );
            }
        }
        check(
            w.iter().all(|&v| v == 0),
            &format!("{grid:?} output buffer untouched (all zero)"),
        );
    }

    // x is unbounded: a large grid.x with y = z = 1 must still launch.
    let (r, _) = run(&mut gpu, [1 << 20, 1, 1]);
    check(r.is_ok(), &format!("[1<<20,1,1] grid.x accepted ({r:?})"));

    if failures != 0 {
        eprintln!("{failures} check(s) FAILED");
        std::process::exit(1);
    }
    println!("tmp_grid_yz_guard_smoke: all checks passed");
}
