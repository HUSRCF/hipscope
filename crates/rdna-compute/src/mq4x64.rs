// SPDX-License-Identifier: Apache-2.0
//! Standalone gfx1100 X64/Y64 packed-MQ4 occupancy probe.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

impl Gpu {
    /// Exact group128 Q8 x affine-MQ4 probe with a 64-token output tile.
    ///
    /// This is deliberately unreachable from serving. Compared with the
    /// production X256/Y64 kernel it quarters the accumulator footprint and
    /// reduces dynamic LDS below 32 KiB, at the cost of loading each weight
    /// tile for four times as many workgroups.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_mmq_prequant_x64y64_group128(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        add: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() || m % 64 != 0 || k % 256 != 0 || batch_size % 64 != 0 {
            return Err(HipError::new(
                0,
                "X64/Y64 group128 probe requires gfx1100 and M%64=K%256=N%64=0",
            ));
        }

        const MODULE: &str = "gemm_hfq4g256_mmq_x64y64_group128";
        let kernel = if add {
            "gemm_hfq4g256_mmq_x64y64_group128_full_add"
        } else {
            "gemm_hfq4g256_mmq_x64y64_group128_full_set"
        };
        static SOURCE: OnceLock<String> = OnceLock::new();
        let source = SOURCE.get_or_init(|| {
            let body = kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC
                .replace("gemm_hfq4g256_residual_mmq", MODULE);
            format!(
                "#define MMQ_X 64\n\
                 #define MMQ_Y 64\n\
                 #define MMQ_NWARPS 8\n\
                 #define MMQ_MIN_BLOCKS_PER_CU 2\n\
                 #define MMQ_ROW_FRAGS 2\n\
                 #define MMQ_COL_GROUPS 4\n\
                 #define MMQ_PERM_NIBBLE 1\n\
                 #define MMQ_Q8_GROUP128 1\n\
                 #define MMQ_WEIGHT_QUAD_ROW_U32X2 1\n\
                 {body}"
            )
        });
        self.ensure_kernel(MODULE, source, kernel)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = i32::from(add);
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // 64-column Q8 tile + 64-row expanded weight tile + per-column sums.
        const SHARED_BYTES: u32 = ((64 * 36 + 64 * 76 + 64) * 4) as u32;
        self.launch_maybe_blob(
            kernel,
            [(m / 64) as u32, (batch_size / 64) as u32, 1],
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
                b.push_i32(add_val);
                b
            },
        )
    }
}
