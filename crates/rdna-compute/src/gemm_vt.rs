// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12 MQ4G256V2 "verify-tile" dense WMMA GEMMs for 16 < N < 64 rows.
//!
//! The multi-request speculative-verify trunk packs several requests'
//! 4-row (MTP) / 16-row (DFlash) blocks into one forward of N <= 63 rows. The
//! one-tile gfx12 kernels (`gemm_{qkvza,qkv,gate_up}_mq4g256v2_wmma_gfx12`,
//! `gemm_mq4g256v2_residual_wmma_gfx12`) launch one wave per 16x16 tile, so
//! every block re-streams the whole activation panel (N x K fp16) through the
//! caches for just 16 weight rows: the GEMM is activation-bandwidth bound and
//! its cost scales ~linearly with N (batch-tiling inside one wave, the BT4
//! kernels, shares the weight fragment but not that traffic). The
//! `_vt{2,3,4}w{4,8}` kernels
//! (`kernels/src/gemm_mq4g256v2_wmma_gfx12_vt_core.hip`) run 4 or 8 waves per
//! block, each owning 16 weight rows, over the whole batch panel (BT =
//! ceil(N/16) 16-row tiles, one grid column): each 128-K slab of the panel is
//! staged in LDS once per block and feeds BT independent accumulators per wave.
//! Per (weight row, batch row) output the arithmetic is the one-tile kernel's
//! exactly (same WMMA chain, K order, headers, dequant, epilogue), so rows are
//! byte-identical to the one-tile launch at any N.
//!
//! Selection ([`Gpu::gfx12_verify_bt_active`]): exact gfx1201, `16 < N < 64`,
//! K % 256 == 0, outside capture/replay recording (the historical launch
//! contract), and `HIPFIRE_WMMA_BATCH_TILES` not `0`. N <= 16 never takes it.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Wide (`w8`) blocks share each staged activation slab across more weight
/// rows and win when there are enough 16-row tiles to keep every CU busy;
/// narrow (`w4`) blocks keep small-M projections (residual) spread over the
/// GPU. Cut measured on gfx1201 (R9700) at N = 32..63.
const VT_WIDE_MIN_ROW_TILES: usize = 1024;

/// Symbol table `[BT - 2][wide]` for a projection prefix.
macro_rules! vt_symbols {
    ($p:literal) => {
        [
            [concat!($p, "_vt2w4"), concat!($p, "_vt2w8")],
            [concat!($p, "_vt3w4"), concat!($p, "_vt3w8")],
            [concat!($p, "_vt4w4"), concat!($p, "_vt4w8")],
        ]
    };
}

impl Gpu {
    /// Whether the batch-tiled verify route applies to a dense MQ4V2 gfx12
    /// GEMM of `batch_size` rows with reduction length `k`.
    pub(crate) fn gfx12_verify_bt_active(&self, batch_size: usize, k: usize) -> bool {
        self.flags.wmma_batch_tiles
            && self.arch == "gfx1201"
            && (17..64).contains(&batch_size)
            && k % 256 == 0
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
    }

    /// Shared launch: `ptrs` then `ints` are the kernel arguments in ABI order
    /// (the `_vt` kernels keep the one-tile kernels' ABI). Grid x covers
    /// `total_m` weight rows at `16 * waves` per block; grid y is the batch
    /// panel in 16 * BT-row blocks.
    #[allow(clippy::too_many_arguments)]
    fn launch_gemm_vt(
        &mut self,
        module: &'static str,
        source: &'static str,
        symbols: [[&'static str; 2]; 3],
        total_m: usize,
        batch_size: usize,
        mut ptrs: Vec<*mut c_void>,
        mut ints: Vec<i32>,
        bytes: usize,
    ) -> HipResult<()> {
        let bt = (batch_size + 15) / 16;
        debug_assert!((2..=4).contains(&bt));
        let wide = (total_m + 15) / 16 >= VT_WIDE_MIN_ROW_TILES;
        let waves = if wide { 8 } else { 4 };
        let func_name = symbols[bt - 2][wide as usize];
        self.ensure_kernel(module, source, func_name)?;
        let mut params: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| p as *mut _ as *mut c_void)
            .chain(ints.iter_mut().map(|v| v as *mut _ as *mut c_void))
            .collect();
        let grid_x = (total_m + 16 * waves - 1) / (16 * waves);
        let grid_y = (batch_size + 16 * bt - 1) / (16 * bt);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x as u32, grid_y as u32, 1],
            [(32 * waves) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in &ptrs {
                    b.push_ptr(*p);
                }
                for v in &ints {
                    b.push_i32(*v);
                }
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// `gemm_qkvza_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_qkvza_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_qkvza_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_qkv.buf.as_ptr(),
                a_z.buf.as_ptr(),
                a_beta.buf.as_ptr(),
                a_alpha.buf.as_ptr(),
                x_f16,
                y_qkv.buf.as_ptr(),
                y_z.buf.as_ptr(),
                y_beta.buf.as_ptr(),
                y_alpha.buf.as_ptr(),
            ],
            vec![
                qkv_m as i32,
                z_m as i32,
                beta_m as i32,
                alpha_m as i32,
                k as i32,
                batch_size as i32,
            ],
            bytes,
        )
    }

    /// `gemm_qkv_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_qkv_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = q_m + k_m + v_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_qkv_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_q.buf.as_ptr(),
                a_k.buf.as_ptr(),
                a_v.buf.as_ptr(),
                x_f16,
                y_q.buf.as_ptr(),
                y_k.buf.as_ptr(),
                y_v.buf.as_ptr(),
            ],
            vec![q_m as i32, k_m as i32, v_m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }

    /// `gemm_gate_up_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_gate_up_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = gate_m + up_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_gate_up_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_gate.buf.as_ptr(),
                a_up.buf.as_ptr(),
                x_f16,
                y_gate.buf.as_ptr(),
                y_up.buf.as_ptr(),
            ],
            vec![gate_m as i32, up_m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }

    /// `gemm_mq4g256v2_residual_wmma_gfx12` (`Y += W @ X`, also the batched
    /// lm_head) for 16 < N < 64 (see module docs).
    pub(crate) fn gemm_mq4g256v2_residual_wmma_gfx12_vt(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_mq4g256v2_residual_wmma_gfx12_vt",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_mq4g256v2_residual_wmma_gfx12"),
            m,
            batch_size,
            vec![a_raw.buf.as_ptr(), x_f16, y.buf.as_ptr()],
            vec![m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }
}
