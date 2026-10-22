// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder (PeaceMaker) multi-column row kernels of the Flash-Next lane step,
//! exact gfx1151 (`peacemaker-isa` module `qwen4_lane_rows_pm`).
//!
//! Each export computes up to [`LANE_ROWS_PM_MAX`] activation columns per
//! launch. Every column keeps the arithmetic DAG of its 1-row HIP incumbent,
//! so column `c` of `y` is byte-identical to the 1-row kernel launched on
//! `x[c*K..]` and to the incumbent rows kernel's column `c`; what changes is
//! that the activation loads are shared across several weight rows (the
//! incumbents re-read the activations once per weight row).
//!
//! | launcher | column `c` equals | grid x block |
//! |---|---|---|
//! | [`Gpu::gemv_mq6g256v2_xbatch_pm`] | `gemv_mq6g256v2` on `x[c]` | `ceil(M/4)` x 32 |
//! | [`Gpu::gemv_mq6g256v2_x4_xbatch_pm`] | `gemv_mq6g256v2_x4` on `x[c]` | `sum ceil(Mi/4)` x 32 |
//! | [`Gpu::gemv_q8_0_k8_xbatch_pm`] | `gemv_q8_0_k8` on `x[c]` | `ceil(M/4)` x 256 |
//! | [`Gpu::gemv_q8_0_k320_xbatch_pm`] | `gemv_q8_0` (K = 320, `gemv_q8_0_k320_staged`) on `x[c]` | `ceil(M/16)` x 64 |
//!
//! `x` is F32 `[B][K]` row-major and `y` F32 `[B][M]` (column `c` at
//! `y + c*M*4`), `B` in `1..=8`. The embedded image is
//! [`kernels::QWEN4_LANE_ROWS_PM_GFX1151`], pinned by
//! [`kernels::QWEN4_LANE_ROWS_PM_GFX1151_SHA256`] and verified once before the
//! first load (fails closed).
//!
//! The launchers are routed from the four incumbent rows launchers
//! (`gemm_mq6g256v2_f32_rows`, `gemm_mq6g256v2_f32_rows_x4`,
//! `gemv_q8_0_k8_rows`, `gemv_q8_0_staged_rows` at K = 320) when
//! [`Gpu::lane_rows_pm_enabled`] holds; they themselves do not consult the
//! flag (the byte oracle `lane_rows_pm_oracle` calls them directly).

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::sync::LazyLock;

/// Largest column count of one launch.
pub const LANE_ROWS_PM_MAX: usize = 8;
/// Largest column count of one S2 wide launch (the `_wide_r{16,32}_pm` exports
/// serve 9..=this; `<= LANE_ROWS_PM_MAX` keeps the narrow exports).
pub const LANE_ROWS_PM_WIDE_MAX: usize = 32;

const MODULE: &str = "qwen4_lane_rows_pm";
const MQ6: &str = "gemv_mq6g256v2_xbatch_pm";
const MQ6_X4: &str = "gemv_mq6g256v2_x4_xbatch_pm";
const Q8_K8: &str = "gemv_q8_0_k8_xbatch_pm";
const Q8_K320: &str = "gemv_q8_0_k320_xbatch_pm";

// S2 exact-wide exports of the same module and image (`[r16, r32]`): block 256 with
// eight weight rows per workgroup (HC k8: block 512, two rows). The activations
// are staged in bounded row tiles, every column keeps the 1-row DAG.
pub(crate) const MQ6_WIDE: [&str; 2] = [
    "gemv_mq6g256v2_xbatch_wide_r16_pm",
    "gemv_mq6g256v2_xbatch_wide_r32_pm",
];
pub(crate) const MQ6_X4_WIDE: [&str; 2] = [
    "gemv_mq6g256v2_x4_xbatch_wide_r16_pm",
    "gemv_mq6g256v2_x4_xbatch_wide_r32_pm",
];
pub(crate) const Q8_K8_WIDE: [&str; 2] = [
    "gemv_q8_0_k8_xbatch_wide_r16_pm",
    "gemv_q8_0_k8_xbatch_wide_r32_pm",
];
pub(crate) const Q8_K320_WIDE: [&str; 2] = [
    "gemv_q8_0_k320_xbatch_wide_r16_pm",
    "gemv_q8_0_k320_xbatch_wide_r32_pm",
];
pub(crate) const Q8_K2560_WIDE: [&str; 2] = [
    "gemv_q8_0_k2560_xbatch_wide_r16_pm",
    "gemv_q8_0_k2560_xbatch_wide_r32_pm",
];
pub(crate) const Q8_SMALL_WIDE: [&str; 2] = [
    "gemv_q8_0_wide_xbatch_wide_r16_pm",
    "gemv_q8_0_wide_xbatch_wide_r32_pm",
];
pub(crate) const Q8_K2560_X2_WIDE: [&str; 2] = [
    "gemv_q8_0_k2560_x2_xbatch_wide_r16_pm",
    "gemv_q8_0_k2560_x2_xbatch_wide_r32_pm",
];
/// The MQ6 symbols of the module's wide exports, for route closure.
pub(crate) const MQ6_WIDE_SYMBOLS: [&str; 4] =
    [MQ6_WIDE[0], MQ6_WIDE[1], MQ6_X4_WIDE[0], MQ6_X4_WIDE[1]];
/// The Q8 symbols of the module's wide exports, for route closure.
pub(crate) const Q8_WIDE_SYMBOLS: [&str; 10] = [
    Q8_K8_WIDE[0],
    Q8_K8_WIDE[1],
    Q8_K320_WIDE[0],
    Q8_K320_WIDE[1],
    Q8_K2560_WIDE[0],
    Q8_K2560_WIDE[1],
    Q8_SMALL_WIDE[0],
    Q8_SMALL_WIDE[1],
    Q8_K2560_X2_WIDE[0],
    Q8_K2560_X2_WIDE[1],
];

/// MQ6G256V2 bytes per 256-weight group.
const MQ6_GROUP_BYTES: usize = 200;
/// Q8_0 bytes per 32-weight block.
const Q8_BLOCK_BYTES: usize = 34;
/// K of the Q8_0 K-split kernel's slices: eight waves of 256.
const K8_K_MULTIPLE: usize = 2048;
/// K of the HC read-up Q8_0 kernel.
const K320: usize = 320;

static VERIFIED: LazyLock<Result<(), String>> = LazyLock::new(|| {
    let want = kernels::QWEN4_LANE_ROWS_PM_GFX1151_SHA256;
    if want.is_empty() {
        return Err(format!(
            "{MODULE}: QWEN4_LANE_ROWS_PM_GFX1151_SHA256 is not pinned; set it to the sha256sum of kernels/qwen4_lane_rows_pm_gfx1151.hxaco"
        ));
    }
    let got: String = Sha256::digest(kernels::QWEN4_LANE_ROWS_PM_GFX1151)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if got != want {
        return Err(format!(
            "{MODULE}: embedded image sha256 {got} is not the pinned {want}"
        ));
    }
    Ok(())
});

/// The kernels address activations, weights and outputs with 32-bit byte
/// offsets: every operand must fit them.
fn fits_u32(bytes: Option<usize>) -> bool {
    bytes.is_some_and(|b| b <= u32::MAX as usize)
}

fn fits_i32(n: usize) -> bool {
    n <= i32::MAX as usize
}

/// `batch` is a narrow-launch column count (`1..=`[`LANE_ROWS_PM_MAX`]).
fn narrow(batch: usize) -> bool {
    (1..=LANE_ROWS_PM_MAX).contains(&batch)
}

/// `batch` is a wide-launch column count (`LANE_ROWS_PM_MAX + 1..=`[`LANE_ROWS_PM_WIDE_MAX`]).
fn wide(batch: usize) -> bool {
    (LANE_ROWS_PM_MAX + 1..=LANE_ROWS_PM_WIDE_MAX).contains(&batch)
}

/// Column-count-independent MQ6 shape rule.
fn mq6_shape(m: usize, k: usize, batch: usize) -> bool {
    m > 0
        && k > 0
        && k.is_multiple_of(256)
        && fits_i32(m)
        && fits_i32(k)
        && fits_u32(m.checked_mul(k / 256 * MQ6_GROUP_BYTES))
        && fits_u32(batch.checked_mul(k).and_then(|n| n.checked_mul(4)))
        && fits_u32(batch.checked_mul(m).and_then(|n| n.checked_mul(4)))
}

/// Column-count-independent Q8_0 shape rule (m, k, 32-bit byte offsets).
fn q8_shape(m: usize, k: usize, batch: usize) -> bool {
    m > 0
        && k > 0
        && fits_i32(m)
        && fits_i32(k)
        && fits_u32(m.checked_mul(k / 32 * Q8_BLOCK_BYTES))
        && fits_u32(batch.checked_mul(k).and_then(|n| n.checked_mul(4)))
        && fits_u32(batch.checked_mul(m).and_then(|n| n.checked_mul(4)))
}

/// `ms` (unused slots are 0) are admitted MQ6 heights for `mq6_x4`'s shape rule.
fn mq6_x4_shape(ms: [usize; 4], k: usize, batch: usize) -> bool {
    ms.iter().any(|&m| m > 0)
        && ms.iter().all(|&m| m == 0 || mq6_shape(m, k, batch))
        && ms
            .iter()
            .try_fold(0usize, |acc, &m| acc.checked_add(m.div_ceil(4)))
            .is_some_and(|g| g <= u32::MAX as usize)
}

/// `gemv_mq6g256v2_xbatch_pm` admits `(m, k, batch)`.
pub(crate) fn mq6_admits(m: usize, k: usize, batch: usize) -> bool {
    narrow(batch) && mq6_shape(m, k, batch)
}

/// `gemv_mq6g256v2_xbatch_wide_r{16,32}_pm` admits `(m, k, rows)`.
pub(crate) fn mq6_wide_admits(m: usize, k: usize, rows: usize) -> bool {
    wide(rows) && mq6_shape(m, k, rows)
}

/// `gemv_mq6g256v2_x4_xbatch_pm` admits four matrices of heights `ms` (unused
/// slots are 0) reading one `[batch, k]` x. At least one matrix is non-empty.
pub(crate) fn mq6_x4_admits(ms: [usize; 4], k: usize, batch: usize) -> bool {
    narrow(batch) && mq6_x4_shape(ms, k, batch)
}

/// `gemv_mq6g256v2_x4_xbatch_wide_r{16,32}_pm` admits `(ms, k, rows)`; the
/// grid is `sum ceil(Mi / 8)`, so the 4-row bound of the narrow rule is
/// only a sufficient 32-bit grid check.
pub(crate) fn mq6_x4_wide_admits(ms: [usize; 4], k: usize, rows: usize) -> bool {
    wide(rows) && mq6_x4_shape(ms, k, rows)
}

/// `gemv_q8_0_k8_xbatch_pm` admits `(m, k, batch)`.
pub(crate) fn q8_k8_admits(m: usize, k: usize, batch: usize) -> bool {
    narrow(batch) && k.is_multiple_of(K8_K_MULTIPLE) && q8_shape(m, k, batch)
}

/// `gemv_q8_0_k8_xbatch_wide_r{16,32}_pm` admits `(m, k, rows)`: `K % 256 == 0`
/// (short final partition tiles included, as the HIP `gemv_q8_0_k8` singleton);
/// the narrow export keeps `K % 2048`.
pub(crate) fn q8_k8_wide_admits(m: usize, k: usize, rows: usize) -> bool {
    wide(rows) && k.is_multiple_of(256) && q8_shape(m, k, rows)
}

/// `gemv_q8_0_k320_xbatch_pm` admits `(m, k, batch)`.
pub(crate) fn q8_k320_admits(m: usize, k: usize, batch: usize) -> bool {
    narrow(batch) && k == K320 && q8_shape(m, K320, batch)
}

/// `gemv_q8_0_k320_xbatch_wide_r{16,32}_pm` admits `(m, k, rows)`.
pub(crate) fn q8_k320_wide_admits(m: usize, k: usize, rows: usize) -> bool {
    wide(rows) && k == K320 && q8_shape(m, K320, rows)
}

/// `gemv_q8_0_k2560_xbatch_wide_r{16,32}_pm` and the x2 pair admit `(m, k, rows)`.
pub(crate) fn q8_k2560_wide_admits(m: usize, k: usize, rows: usize) -> bool {
    wide(rows) && k == 2560 && q8_shape(m, 2560, rows)
}

/// `gemv_q8_0_wide_xbatch_wide_r{16,32}_pm` (four-chain small K: every staged
/// K <= 1536, K % 32 == 0 other than K = 320) admits `(m, k, rows)`.
pub(crate) fn q8_small_wide_admits(m: usize, k: usize, rows: usize) -> bool {
    wide(rows) && k != K320 && k <= 1536 && k.is_multiple_of(32) && q8_shape(m, k, rows)
}

fn need(name: &str, what: &str, tensor: &GpuTensor, bytes: usize) -> HipResult<()> {
    if tensor.buf.size() < bytes {
        return Err(HipError::new(
            1,
            &format!(
                "{name}: {what} view holds {} bytes, {bytes} required",
                tensor.buf.size()
            ),
        ));
    }
    Ok(())
}

fn refuse(name: &str, detail: &str) -> HipError {
    HipError::new(1, &format!("{name}: {detail}"))
}

impl Gpu {
    /// Whether the incumbent rows launchers route to the lane-rows PM kernels:
    /// the `fn_lane_pm_rows` flag (`HIPFIRE_FN_LANE_PM_ROWS`, gfx1151 default on) on
    /// exact gfx1151, inside the lane step or a Qwen4 speculative verify
    /// ([`Gpu::lane_rows_scope`]), with no recorder or graph capture active.
    pub fn lane_rows_pm_enabled(&self) -> bool {
        self.flags.fn_lane_pm_rows_enabled()
            && self.arch_caps.is_gfx1151()
            && self.lane_rows_scope
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
    }

    /// Verify the pinned `qwen4_lane_rows_pm` image and load `symbol` (narrow or
    /// S2 `_wide_r{16,32}_pm` export; exact gfx1151 only), so byte oracles can
    /// launch any export by name with raw blobs.
    pub fn ensure_lane_rows_pm_entry(&mut self, symbol: &str) -> HipResult<()> {
        if !self.arch_caps.is_gfx1151() {
            return Err(refuse(
                symbol,
                "the lane-rows PM kernels are exact gfx1151 only",
            ));
        }
        VERIFIED
            .clone()
            .map_err(|reason| HipError::new(0, &reason))?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, kernels::QWEN4_LANE_ROWS_PM_GFX1151, symbol)
    }

    /// Verify the pinned image and load `symbol` of the module.
    pub(crate) fn ensure_lane_rows_pm(
        &mut self,
        name: &str,
        symbol: &'static str,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1151() {
            return Err(refuse(
                name,
                "the lane-rows PM kernels are exact gfx1151 only",
            ));
        }
        VERIFIED
            .clone()
            .map_err(|reason| HipError::new(0, &reason))?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, kernels::QWEN4_LANE_ROWS_PM_GFX1151, symbol)
    }

    /// `y[c] = W · x[c]` for `c < batch` (<= [`LANE_ROWS_PM_MAX`]) over an
    /// MQ6G256V2 weight, each column byte-identical to `gemv_mq6g256v2` on
    /// `x[c]`. `k % 256 == 0`; x `[batch, k]`, y `[batch, m]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq6g256v2_xbatch_pm(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        const NAME: &str = "gemv_mq6g256v2_xbatch_pm";
        if !mq6_admits(m, k, batch) {
            return Err(refuse(
                NAME,
                &format!("batch {batch} in 1..=8, m {m} > 0, k {k} % 256 == 0 and 32-bit byte offsets required"),
            ));
        }
        need(NAME, "weight", a_raw, m * (k / 256) * MQ6_GROUP_BYTES)?;
        need(NAME, "x", x, batch * k * 4)?;
        need(NAME, "y", y, batch * m * 4)?;
        self.ensure_lane_rows_pm(NAME, MQ6)?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let (m_val, k_val, b_val) = (m as i32, k as i32, batch as i32);
        let mut params: [*mut c_void; 6] = [
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            MQ6,
            [m.div_ceil(4) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(b_val);
                b
            },
        )
    }

    /// [`Gpu::gemv_mq6g256v2_xbatch_pm`] of up to four `(weight, y, m)`
    /// matrices reading the same pre-rotated `[batch, k]` x in one launch
    /// (unused slots carry `m = 0`); each column of each output is byte-identical
    /// to the one-matrix `gemv_mq6g256v2` on `x[c]`.
    pub fn gemv_mq6g256v2_x4_xbatch_pm(
        &mut self,
        parts: [(&GpuTensor, &GpuTensor, usize); 4],
        x: &GpuTensor,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        const NAME: &str = "gemv_mq6g256v2_x4_xbatch_pm";
        let ms = parts.map(|(_, _, m)| m);
        if !mq6_x4_admits(ms, k, batch) {
            return Err(refuse(
                NAME,
                &format!("batch {batch} in 1..=8, heights {ms:?} (not all 0), k {k} % 256 == 0 and 32-bit byte offsets required"),
            ));
        }
        need(NAME, "x", x, batch * k * 4)?;
        for (w, y, m) in parts {
            if m > 0 {
                need(NAME, "weight", w, m * (k / 256) * MQ6_GROUP_BYTES)?;
                need(NAME, "y", y, batch * m * 4)?;
            }
        }
        self.ensure_lane_rows_pm(NAME, MQ6_X4)?;
        let a = parts.map(|(w, _, _)| w.buf.as_ptr());
        let y = parts.map(|(_, y, _)| y.buf.as_ptr());
        let m = ms.map(|m| m as i32);
        let x_ptr = x.buf.as_ptr();
        let (k_val, b_val) = (k as i32, batch as i32);
        let grid: usize = ms.iter().map(|m| m.div_ceil(4)).sum();
        let mut params: Vec<*mut c_void> = Vec::with_capacity(15);
        for p in &a {
            params.push(p as *const _ as *mut c_void);
        }
        params.push(&x_ptr as *const _ as *mut c_void);
        for p in &y {
            params.push(p as *const _ as *mut c_void);
        }
        for v in &m {
            params.push(v as *const _ as *mut c_void);
        }
        params.push(&k_val as *const _ as *mut c_void);
        params.push(&b_val as *const _ as *mut c_void);
        self.launch_maybe_blob(
            MQ6_X4,
            [grid as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in a {
                    b.push_ptr(p);
                }
                b.push_ptr(x_ptr);
                for p in y {
                    b.push_ptr(p);
                }
                for v in m {
                    b.push_i32(v);
                }
                b.push_i32(k_val);
                b.push_i32(b_val);
                b
            },
        )
    }

    /// `y[c] = epilogue(W · x[c])` for `c < batch` (<= [`LANE_ROWS_PM_MAX`])
    /// over a Q8_0 weight, each column byte-identical to `gemv_q8_0_k8` on
    /// `x[c]` (K split over eight waves; `hc_act_scale = Some(s)` runs the HC
    /// activation epilogue with scale `s`). `k % 2048 == 0`; x `[batch, k]`,
    /// y `[batch, m]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_0_k8_xbatch_pm(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        hc_act_scale: Option<f32>,
        batch: usize,
    ) -> HipResult<()> {
        const NAME: &str = "gemv_q8_0_k8_xbatch_pm";
        if !q8_k8_admits(m, k, batch) {
            return Err(refuse(
                NAME,
                &format!("batch {batch} in 1..=8, m {m} > 0, k {k} % 2048 == 0 and 32-bit byte offsets required"),
            ));
        }
        need(NAME, "weight", weight, m * (k / 32) * Q8_BLOCK_BYTES)?;
        need(NAME, "x", x, batch * k * 4)?;
        need(NAME, "y", y, batch * m * 4)?;
        self.ensure_lane_rows_pm(NAME, Q8_K8)?;
        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let (m_val, k_val, b_val) = (m as i32, k as i32, batch as i32);
        let scale = hc_act_scale.unwrap_or(1.0);
        let hc_act = i32::from(hc_act_scale.is_some());
        let mut params: [*mut c_void; 8] = [
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &hc_act as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            Q8_K8,
            [m.div_ceil(4) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(b_val);
                b.push_f32(scale);
                b.push_i32(hc_act);
                b
            },
        )
    }

    /// `y[c] = W · x[c]` for `c < batch` (<= [`LANE_ROWS_PM_MAX`]) over a
    /// K = 320 Q8_0 weight (the HC read-up projection), each column
    /// byte-identical to the gfx1151 `gemv_q8_0_k320_staged` on `x[c]`; x
    /// `[batch, 320]`, y `[batch, m]`.
    pub fn gemv_q8_0_k320_xbatch_pm(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        batch: usize,
    ) -> HipResult<()> {
        const NAME: &str = "gemv_q8_0_k320_xbatch_pm";
        if !q8_k320_admits(m, K320, batch) {
            return Err(refuse(
                NAME,
                &format!("batch {batch} in 1..=8, m {m} > 0 and 32-bit byte offsets required"),
            ));
        }
        need(NAME, "weight", a_raw, m * (K320 / 32) * Q8_BLOCK_BYTES)?;
        need(NAME, "x", x, batch * K320 * 4)?;
        need(NAME, "y", y, batch * m * 4)?;
        self.ensure_lane_rows_pm(NAME, Q8_K320)?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let (m_val, b_val) = (m as i32, batch as i32);
        let mut params: [*mut c_void; 5] = [
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            Q8_K320,
            [m.div_ceil(16) as u32, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(b_val);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_image_matches_its_pinned_digest() {
        VERIFIED.clone().unwrap();
    }

    #[test]
    fn mq6_admission_is_k256_columns_and_32_bit_offsets() {
        assert!(mq6_admits(1, 256, 1));
        assert!(mq6_admits(248_320, 2560, 8));
        assert!(!mq6_admits(0, 256, 2));
        assert!(!mq6_admits(8, 0, 2));
        assert!(!mq6_admits(8, 2559, 2));
        assert!(!mq6_admits(8, 2560, 0));
        assert!(!mq6_admits(8, 2560, 9));
        // 21.5M rows of K = 2560 exceed 4 GiB of weights.
        assert!(!mq6_admits(21_500_000, 2560, 2));
    }

    #[test]
    fn mq6_x4_admission_needs_one_matrix_and_admitted_members() {
        assert!(mq6_x4_admits([12288, 640, 512, 512], 2560, 8));
        assert!(mq6_x4_admits([3, 0, 5, 0], 2560, 2));
        assert!(!mq6_x4_admits([0, 0, 0, 0], 2560, 2));
        assert!(!mq6_x4_admits([3, 0, 5, 0], 2559, 2));
        assert!(!mq6_x4_admits([3, 0, 5, 0], 2560, 9));
    }

    #[test]
    fn q8_admission_follows_the_incumbent_shapes() {
        assert!(q8_k8_admits(320, 10240, 8));
        assert!(q8_k8_admits(7, 2048, 2));
        assert!(!q8_k8_admits(7, 2304, 2));
        assert!(!q8_k8_admits(0, 2048, 2));
        assert!(q8_k320_admits(10240, 320, 8));
        assert!(q8_k320_admits(1, 320, 2));
        assert!(!q8_k320_admits(17, 640, 2));
        assert!(!q8_k320_admits(17, 320, 9));
    }
}
