// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed launch-grid guard.
//!
//! `hipModuleLaunchKernel` on gfx1201 (RDNA4) rejects any launch whose
//! `gridDim.y` or `gridDim.z` is >= 65536 (measured on raw HIP: <= 65535
//! launches, >= 65536 fails). `gridDim.x` is not bound. The failure only shows
//! up at submission, which is too late for the graph-capture / replay-record
//! tapes: a rejected geometry must be refused *before* it is recorded or
//! captured so the tapes can never hold a launch HIP will not run.
//!
//! The limit is applied only to architectures with measured evidence
//! ([`grid_yz_limit_for_arch`]); every other architecture keeps its historic
//! behaviour. A geometry over the limit is an error, never clamped or
//! truncated: silently shrinking a grid would drop rows.
//!
//! One policy table and one checker live here; every ingress (raw HIP launch,
//! dispatch recording funnels, PM4/AQL replay preparation) calls them.

use crate::error::{HipError, HipResult};

/// Largest `gridDim.y` / `gridDim.z` raw HIP accepts on gfx1201.
pub const GFX1201_MAX_GRID_YZ: u32 = 65_535;

/// `hipErrorInvalidValue`.
const HIP_ERROR_INVALID_VALUE: u32 = 1;

/// Per-architecture `gridDim.y`/`gridDim.z` ceiling, or `None` when the
/// architecture has no recorded ceiling and launches are left to HIP.
pub fn grid_yz_limit_for_arch(arch: &str) -> Option<u32> {
    arch.eq_ignore_ascii_case("gfx1201")
        .then_some(GFX1201_MAX_GRID_YZ)
}

/// Reject a launch geometry whose `y` or `z` axis exceeds `limit`.
///
/// `limit == None` accepts everything (no guard for this architecture).
pub fn check_launch_grid(grid: [u32; 3], limit: Option<u32>) -> HipResult<()> {
    let Some(limit) = limit else {
        return Ok(());
    };
    for (axis, name) in [(1usize, "y"), (2, "z")] {
        if grid[axis] > limit {
            return Err(HipError::new(
                HIP_ERROR_INVALID_VALUE,
                &format!(
                    "launch grid {grid:?}: grid.{name}={} exceeds the device grid.y/grid.z limit \
                     {limit}; refusing to record or launch (never truncated) — chunk the launch \
                     or fold the axis into grid.x in the caller",
                    grid[axis]
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1201_accepts_boundary_and_rejects_next() {
        let limit = grid_yz_limit_for_arch("gfx1201");
        assert_eq!(limit, Some(65_535));
        assert!(check_launch_grid([1, 65_535, 1], limit).is_ok());
        assert!(check_launch_grid([1, 1, 65_535], limit).is_ok());
        assert!(check_launch_grid([u32::MAX, 65_535, 65_535], limit).is_ok());
        let y = check_launch_grid([8, 65_536, 1], limit).unwrap_err();
        assert!(y.message.contains("grid.y=65536"), "{}", y.message);
        let z = check_launch_grid([8, 1, 65_536], limit).unwrap_err();
        assert!(z.message.contains("grid.z=65536"), "{}", z.message);
    }

    #[test]
    fn grid_x_is_not_bound() {
        let limit = grid_yz_limit_for_arch("gfx1201");
        assert!(check_launch_grid([1 << 20, 1, 1], limit).is_ok());
    }

    #[test]
    fn other_architectures_are_unguarded() {
        for arch in ["gfx1100", "gfx1200", "gfx1151", "gfx942", "gfx1010"] {
            assert_eq!(grid_yz_limit_for_arch(arch), None, "{arch}");
        }
        assert!(check_launch_grid([1, 1 << 20, 1 << 20], None).is_ok());
    }
}
