// SPDX-License-Identifier: Apache-2.0
//! Standalone gfx1100 MQ4-v2 execution-format probes.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

impl Gpu {
    /// Exact affine-MQ4 probe using a load-time lane-major execution copy.
    ///
    /// This entry is intentionally not reachable from serving dispatch. The
    /// weight tensor uses the same byte count and numerical contract as
    /// HFQ4-G256, but is reordered by 16-row tiles for coalesced packed-LDS
    /// staging and conflict-light register decode.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4v2_lane_major_packed_lds_prequant(
        &mut self,
        a_execution: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() || m % 64 != 0 || k % 256 != 0 || batch_size % 256 != 0 {
            return Err(HipError::new(
                0,
                "MQ4-v2 lane-major probe requires gfx1100 and M%64=K%256=N%256=0",
            ));
        }
        let expected = m
            .checked_mul(k / 256)
            .and_then(|groups| groups.checked_mul(136))
            .ok_or_else(|| HipError::new(0, "MQ4-v2 execution weight size overflow"))?;
        if a_execution.buf.size() != expected {
            return Err(HipError::new(
                0,
                &format!(
                    "MQ4-v2 lane-major probe expects {expected} weight bytes, got {}",
                    a_execution.buf.size()
                ),
            ));
        }

        const MODULE: &str = "gemm_mq4v2_lane_major_packed_lds_gfx1100";
        let kernel = if add {
            "gemm_mq4v2_lane_major_packed_lds_add"
        } else {
            "gemm_mq4v2_lane_major_packed_lds_set"
        };
        self.ensure_kernel(
            MODULE,
            kernels::GEMM_MQ4V2_LANE_MAJOR_PACKED_LDS_GFX1100_SRC,
            kernel,
        )?;

        let mut a_ptr = a_execution.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        // Q8 activation tile + 64-row packed weight tile + per-column sums.
        const SHARED_BYTES: u32 = ((256 * 36 + 64 * 32 + 64 + 256) * 4) as u32;
        self.launch_maybe_blob(
            kernel,
            [(m / 64) as u32, (batch_size / 256) as u32, 1],
            [32, 8, 1],
            SHARED_BYTES,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        )
    }
}
