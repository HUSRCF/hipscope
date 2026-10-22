// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-INT8 prefill GEMM for MQ6G256V2 weights on exact gfx1201 and gfx1151.
//!
//! The 6-bit codes enter `v_wmma_i32_16x16x16_iu8` raw and unsigned
//! (`kernels/src/gemm_mq6g256v2_mmq_i8.gfx12.hip` / `.gfx11.hip`); activations
//! are the existing A8 `block_i8_128` blocks (136 B per token per 128 K, layout
//! `[K/128][N]`) the MQ4 A8 route already produces. Per (row, token, 128-K
//! block) the accumulator is seeded `I8_MAGIC - 32*s` so it ends at
//! `I8_MAGIC + sum (q - 32) * qx` (`|C| <= 32*127*128 < 2^22`, exact in f32),
//! then folded in K order with `fma(RN(s_h*d), C, acc)` and
//! `fma(RN(z'_h*d), s, acc)`, `z'_h = z_h + 32*s_h`. Output F32 or BF16 bits
//! (RNE, the incumbent's formula), overwrite (SET).
//!
//! Producer ([`Gpu::mq6_i8_prepare_x`]): one pass, [`Gpu::rotate_x_mq_i8_batched`]
//! (`mq_rotate_x_i8_gfx12` / `_gfx11`): the same FWHT as the F16 route emitting the
//! A8 blocks in registers, the 128-K blocks coinciding with the MQ6 128-halves.
//! The blocks are byte-identical to the former two-pass chain
//! (`rotate_x_mq_batched` into an F32 scratch, then `ensure_int8_mmq_x`).
//!
//! Admission: default on for exact gfx1201 inside the Qwen4 forward at
//! `route_rows >= QWEN4_F16_WMMA_MIN_TOKENS` (the F16 PM route it replaces);
//! measured on Flash-Next pp16384 +1.1% with KLD inside the bar
//! (`release-0.4.3/fp8-lean/RESULTS.md`). gfx1151 is opt-in only: there the
//! iu8 rate equals F16 and the route measured 0.79-0.95x of the incumbent GEMM.
//! Env (developer): `HIPFIRE_MQ6_I8` (`1` on for any scope, `0` off, unset =
//! the default above); `HIPFIRE_MQ6_I8_PM` (the GEMM itself runs the certified
//! builder (PeaceMaker) twin `kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201` on exact
//! gfx1201 whenever [`mq6_i8_pm_admits`] holds, else the hipcc kernel; `0` = hipcc
//! always, `1` and unset = the twin where admitted). The twin's output is
//! bit-identical to the hipcc kernel's.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// MQ6G256V2 group bytes (8 B header + 192 B 6-bit payload).
const MQ6_GROUP_BYTES: usize = 200;
/// A8 `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_I8_128_BYTES: usize = 136;

/// Archs where the unset-flag default runs the INT8 route in qwen4 scope.
pub const MQ6_I8_DEFAULT_ARCHS: &[&str] = &["gfx1201"];

/// Dynamic LDS of the gfx12 entries (header constant `MQ6_I8_GFX12_LDS_BYTES`
/// of `gemm_mq6g256v2_mmq_i8.gfx12.hip`: two K64 slots of 8192 B A + 8192 B
/// unpacked W, plus 2 x 1024 B DS and 2 x 1024 B SZ planes).
const MQ6_I8_GFX12_LDS_BYTES: u32 = 36864;
/// Dynamic LDS of the gfx11 128-token build (512 threads).
const MQ6_I8_GFX11_LDS_BYTES: u32 = 32768;
/// Dynamic LDS of the gfx11 256-token build (1024 threads, `_t256` entries).
const MQ6_I8_GFX11_T256_LDS_BYTES: u32 = 65536;

/// Embedded module of the PeaceMaker twin (gfx1201) and its entries.
const MQ6_I8_PM_MODULE: &str = "gemm_mq6g256v2_mmq_i8_pm_gfx1201";
const MQ6_I8_PM_ENTRY_F32: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201";
const MQ6_I8_PM_ENTRY_BF16: &str = "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201";
/// The twin's CTA tile (128 rows x 128 tokens, 256 threads) and dynamic LDS
/// (the same figures as the hipcc kernel).
const MQ6_I8_PM_TILE: usize = 128;

/// Whether the PeaceMaker twin runs one launch (`flag` = `HIPFIRE_MQ6_I8_PM`): not
/// switched off (`0`; unset and `1` both admit), exact gfx1201, `k % 256 == 0`, `m % 4 == 0` (its row quads are stored
/// whole), a 4-byte aligned `y`, and the 32-bit address contract (`m`, `k`, `n`
/// and the output and weight extents under 2^31 bytes). Anything else keeps the
/// hipcc kernel.
fn mq6_i8_pm_admits(
    flag: Option<bool>,
    arch: &str,
    m: usize,
    k: usize,
    n: usize,
    out_elem_bytes: usize,
    y_ptr: usize,
) -> bool {
    flag != Some(false)
        && arch == "gfx1201"
        && k > 0
        && k % 256 == 0
        && m > 0
        && m % 4 == 0
        && n > 0
        && y_ptr % 4 == 0
        && m.max(k).max(n) <= i32::MAX as usize
        && crate::gemm::pm_output_extent_bytes(n, m, out_elem_bytes).is_some()
        && m.checked_mul(k / 256)
            .and_then(|g| g.checked_mul(MQ6_GROUP_BYTES))
            .is_some_and(|w| w <= crate::gemm::PM_OUTPUT_EXTENT_LIMIT_BYTES)
}

/// Scope decision of the MQ6 INT8 route past the device/shape preconditions:
/// `Some(false)` off, `Some(true)` on for any scope, unset on in qwen4 scope
/// iff `arch` is in [`MQ6_I8_DEFAULT_ARCHS`] and `route_rows` reaches
/// [`crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS`].
fn mq6_i8_scope_admits(
    flag: Option<bool>,
    arch: &str,
    qwen4_scope: bool,
    route_rows: usize,
) -> bool {
    match flag {
        Some(on) => on,
        None => {
            qwen4_scope
                && MQ6_I8_DEFAULT_ARCHS.contains(&arch)
                && route_rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        }
    }
}

/// `(module, source, block threads, dynamic LDS bytes, CTA tile)` of one
/// launch of `rows` weight rows x `n` tokens. gfx1201 runs the 128 x 128 tile;
/// gfx1151 takes the 256 x 256 build (one 1024-thread CTA per WGP) whenever it
/// still yields at least `cu_count` CTAs (the A8 MQ4 rule, `a8_launch`), else
/// the 128 x 128 build. `None` off those archs.
fn mq6_i8_launch(
    arch: &str,
    rows: usize,
    n: usize,
    cu_count: usize,
) -> Option<(&'static str, &'static str, u32, u32, usize)> {
    match arch {
        "gfx1201" => Some((
            "gemm_mq6g256v2_mmq_i8_gfx12",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC,
            256,
            MQ6_I8_GFX12_LDS_BYTES,
            128,
        )),
        "gfx1151" if rows.div_ceil(256) * n.div_ceil(256) >= cu_count => Some((
            "gemm_mq6g256v2_mmq_i8_gfx11_t256",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_T256_SRC,
            1024,
            MQ6_I8_GFX11_T256_LDS_BYTES,
            256,
        )),
        "gfx1151" => Some((
            "gemm_mq6g256v2_mmq_i8_gfx11",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_SRC,
            512,
            MQ6_I8_GFX11_LDS_BYTES,
            128,
        )),
        _ => None,
    }
}

/// Entry symbol of the SET kernel in the module [`mq6_i8_launch`] picked.
fn mq6_i8_entry(arch: &str, tile: usize, bf16: bool) -> &'static str {
    match (arch == "gfx1201", tile, bf16) {
        (true, _, false) => "gemm_mq6g256v2_mmq_i8_set_gfx12",
        (true, _, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12",
        (false, 256, false) => "gemm_mq6g256v2_mmq_i8_set_gfx11_t256",
        (false, 256, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256",
        (false, _, false) => "gemm_mq6g256v2_mmq_i8_set_gfx11",
        (false, _, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11",
    }
}

impl Gpu {
    /// Whether the MQ6 INT8 route admits one launch of `m` weight rows, `k`
    /// and `batch_size` tokens: exact gfx1201 or gfx1151, eager (no replay
    /// recording / graph capture), `k % 256 == 0`, non-empty dims, and the
    /// scope decision (`HIPFIRE_MQ6_I8`: `1` any scope, `0` off, unset = qwen4
    /// scope on [`MQ6_I8_DEFAULT_ARCHS`] at `route_rows >= 512`).
    pub fn gemm_mq6g256v2_i8_applies(&self, m: usize, k: usize, batch_size: usize) -> bool {
        matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && k > 0
            && k % 256 == 0
            && m > 0
            && batch_size > 0
            && mq6_i8_scope_admits(
                self.flags.mq6_i8,
                self.arch.as_str(),
                self.qwen4_scope,
                self.route_rows(batch_size),
            )
    }

    /// Name of the kernel [`Gpu::gemm_mq6g256v2_i8_prequant`] launches for these
    /// arguments (the PeaceMaker twin entry or the hipcc one); `None` off gfx1201 /
    /// gfx1151.
    pub fn gemm_mq6g256v2_i8_kernel_name(
        &self,
        m: usize,
        k: usize,
        batch_size: usize,
        y: &GpuTensor,
    ) -> Option<&'static str> {
        let bf16 = y.dtype == DType::BF16;
        if mq6_i8_pm_admits(
            self.flags.mq6_i8_pm,
            &self.arch,
            m,
            k,
            batch_size,
            if bf16 { 2 } else { 4 },
            y.buf.as_ptr() as usize,
        ) {
            return Some(if bf16 {
                MQ6_I8_PM_ENTRY_BF16
            } else {
                MQ6_I8_PM_ENTRY_F32
            });
        }
        let (_, _, _, _, tile) =
            mq6_i8_launch(&self.arch, m, batch_size, self.cu_count_or_default())?;
        Some(mq6_i8_entry(&self.arch, tile, bf16))
    }

    /// Rotate F32 `x` (`[batch_size x k]`) with the MQ FWHT (same signs as
    /// [`Gpu::rotate_x_mq_batched_f16`]) and quantize it into A8 `block_i8_128`
    /// blocks (`[k/128][batch_size]`) in one fused launch
    /// ([`Gpu::rotate_x_mq_i8_batched`]).
    ///
    /// The returned pointer belongs to this `Gpu` and stays valid until the
    /// next INT8 activation producer call.
    pub fn mq6_i8_prepare_x(
        &mut self,
        x: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<*mut c_void> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "mq6_i8_prepare_x: eager-only"));
        }
        if x.dtype != DType::F32 {
            return Err(HipError::new(0, "mq6_i8_prepare_x: x must be F32"));
        }
        if !matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
            || k == 0
            || k % 256 != 0
            || batch_size == 0
        {
            return Err(HipError::new(
                0,
                "mq6_i8_prepare_x: exact gfx1201/gfx1151, K%256==0, N>0 required",
            ));
        }
        let elems = batch_size
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "mq6_i8_prepare_x: batch*k overflow"))?;
        if x.numel() < elems {
            return Err(HipError::new(0, "mq6_i8_prepare_x: undersized x"));
        }
        // `x` may be a residual owed an add (the deferred fold): read it flushed.
        self.flush_residual_fold()?;
        // One pass: FWHT rotate straight into the A8 blocks (no F32 round trip).
        let reservation = self.reserve_int8_mmq(k, batch_size)?;
        let prepared = self.rotate_x_mq_i8_batched(x, None, reservation, k, batch_size)?;
        self.int8_mmq_prepared_ptr(&prepared, k, batch_size)
    }

    /// Overwrite `y` (`[batch_size x m]`, F32, or BF16 bits when
    /// `y.dtype == DType::BF16`) with `W . X` for MQ6G256V2 weights `a`
    /// (`m` rows x `k/256` groups x 200 B) and the A8 blocks `xq` of
    /// [`Gpu::mq6_i8_prepare_x`]. `m` and `batch_size` need not be tile
    /// multiples. gfx1151 takes the 256-token build when
    /// `ceil(m/256) * ceil(n/256) >= cu_count`, else the 128-token one.
    pub fn gemm_mq6g256v2_i8_prequant(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "gemm_mq6g256v2_i8: eager-only"));
        }
        if m == 0 || batch_size == 0 || k == 0 || k % 256 != 0 || xq.is_null() {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: need M>0, N>0, K>0, K%256==0 and prepared blocks",
            ));
        }
        let bf16 = match y.dtype {
            DType::F32 => false,
            DType::BF16 => true,
            _ => return Err(HipError::new(0, "gemm_mq6g256v2_i8: y must be F32 or BF16")),
        };
        let weight_bytes = m
            .checked_mul(k / 256)
            .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_i8: weight bytes overflow"))?;
        let y_elems = batch_size
            .checked_mul(m)
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_i8: batch*m overflow"))?;
        let out_bytes = y_elems * if bf16 { 2 } else { 4 };
        if a.buf.size() < weight_bytes || y.buf.size() < out_bytes {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: undersized weight or output tensor",
            ));
        }
        let Some((module, source, threads, lds, tile)) =
            mq6_i8_launch(&self.arch, m, batch_size, self.cu_count_or_default())
        else {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: exact gfx1201 or gfx1151 required",
            ));
        };
        let pm = mq6_i8_pm_admits(
            self.flags.mq6_i8_pm,
            &self.arch,
            m,
            k,
            batch_size,
            if bf16 { 2 } else { 4 },
            y.buf.as_ptr() as usize,
        );
        let kernel = if pm {
            if bf16 {
                MQ6_I8_PM_ENTRY_BF16
            } else {
                MQ6_I8_PM_ENTRY_F32
            }
        } else {
            mq6_i8_entry(&self.arch, tile, bf16)
        };
        let tile = if pm { MQ6_I8_PM_TILE } else { tile };
        if pm {
            self.ensure_embedded_kernel(
                MQ6_I8_PM_MODULE,
                kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201,
                kernel,
            )?;
        } else {
            self.ensure_kernel(module, source, kernel)?;
        }
        let mut ap = a.buf.as_ptr();
        let mut xp = xq;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut nv = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        // The PM image keeps K1's FullSet kernarg layout: a seventh explicit
        // by-value word (`unused`, offset 36) precedes the hidden block counts.
        let mut unused = 0i32;
        if pm {
            params.push(&mut unused as *mut _ as *mut c_void);
        }
        let bytes = weight_bytes + batch_size * (k / 128) * BLOCK_I8_128_BYTES + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [m.div_ceil(tile) as u32, batch_size.div_ceil(tile) as u32, 1],
            [threads, 1, 1],
            lds,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                if pm {
                    b.push_i32(unused);
                }
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // `y` was overwritten: caches keyed on its pointer are stale.
        self.invalidate_x_caches_for(y.buf.as_ptr());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_is_gfx1201_qwen4_prefill_only() {
        assert_eq!(MQ6_I8_DEFAULT_ARCHS, &["gfx1201"]);
        assert!(mq6_i8_scope_admits(None, "gfx1201", true, 512));
        assert!(mq6_i8_scope_admits(None, "gfx1201", true, 8192));
        assert!(!mq6_i8_scope_admits(None, "gfx1201", true, 511));
        assert!(!mq6_i8_scope_admits(None, "gfx1201", false, 8192));
        for arch in ["gfx1151", "gfx1100"] {
            assert!(!mq6_i8_scope_admits(None, arch, true, 8192));
        }
    }

    #[test]
    fn flag_forces_on_any_scope_and_off_wins() {
        for scope in [false, true] {
            for arch in ["gfx1201", "gfx1151"] {
                assert!(mq6_i8_scope_admits(Some(true), arch, scope, 1));
                assert!(!mq6_i8_scope_admits(Some(false), arch, scope, 8192));
            }
        }
    }

    #[test]
    fn launch_selection_per_arch() {
        // gfx1201: one 128x128 build, 256 threads.
        for (rows, n) in [(48, 1), (10240, 4096), (248_320, 8192)] {
            let (module, _, threads, lds, tile) = mq6_i8_launch("gfx1201", rows, n, 48).unwrap();
            assert_eq!(
                (module, threads, lds, tile),
                (
                    "gemm_mq6g256v2_mmq_i8_gfx12",
                    256,
                    MQ6_I8_GFX12_LDS_BYTES,
                    128
                )
            );
        }
        // gfx1151 (cu_count 20): t256 once ceil(m/256)*ceil(n/256) >= 20.
        let t128 = mq6_i8_launch("gfx1151", 2560, 512, 20).unwrap(); // 10 * 2 = 20 -> t256
        assert_eq!(t128.0, "gemm_mq6g256v2_mmq_i8_gfx11_t256");
        assert_eq!(
            (t128.2, t128.3, t128.4),
            (1024, MQ6_I8_GFX11_T256_LDS_BYTES, 256)
        );
        let small = mq6_i8_launch("gfx1151", 2560, 256, 20).unwrap(); // 10 * 1 = 10 -> t128
        assert_eq!(small.0, "gemm_mq6g256v2_mmq_i8_gfx11");
        assert_eq!(
            (small.2, small.3, small.4),
            (512, MQ6_I8_GFX11_LDS_BYTES, 128)
        );
        assert!(mq6_i8_launch("gfx1100", 2560, 512, 20).is_none());
        assert!(mq6_i8_launch("gfx942", 2560, 512, 20).is_none());
        // Budgets stay within the 64 KiB LDS of one CU.
        for lds in [
            MQ6_I8_GFX12_LDS_BYTES,
            MQ6_I8_GFX11_LDS_BYTES,
            MQ6_I8_GFX11_T256_LDS_BYTES,
        ] {
            assert!(lds <= 65536);
        }
    }

    #[test]
    fn pm_twin_is_the_gfx1201_default_with_the_32_bit_address_contract() {
        let ok = |flag, arch, m, k, n, eb, y| mq6_i8_pm_admits(flag, arch, m, k, n, eb, y);
        assert!(ok(Some(true), "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(ok(Some(true), "gfx1201", 48, 256, 1, 2, 4));
        // FN's largest M at a vocabulary-sized token count stays inside 2^31 bytes.
        assert!(ok(Some(true), "gfx1201", 248_320, 5120, 2162, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 248_320, 5120, 4000, 4, 256));
        // Unset is the default: PM wherever admitted, hipcc where not; `0` keeps hipcc.
        assert!(ok(None, "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(!ok(None, "gfx1201", 248_320, 5120, 4000, 4, 256));
        assert!(!ok(Some(false), "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(!ok(Some(false), "gfx1201", 48, 256, 1, 2, 4));
        // Other archs, K % 256, M % 4, an unaligned Y and empty dims fall back.
        assert!(!ok(Some(true), "gfx1151", 2560, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5000, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2562, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5120, 4000, 4, 258));
        assert!(!ok(Some(true), "gfx1201", 0, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5120, 0, 4, 256));
        // The embedded image carries both entries.
        for entry in [MQ6_I8_PM_ENTRY_F32, MQ6_I8_PM_ENTRY_BF16] {
            assert!(
                kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201
                    .windows(entry.len())
                    .any(|w| w == entry.as_bytes()),
                "image lacks {entry}"
            );
        }
        // The twin's tile and LDS are the hipcc gfx12 build's.
        let (_, _, threads, lds, tile) = mq6_i8_launch("gfx1201", 2560, 512, 48).unwrap();
        assert_eq!(
            (threads, lds, tile),
            (256, MQ6_I8_GFX12_LDS_BYTES, MQ6_I8_PM_TILE)
        );
    }

    #[test]
    fn entries_follow_arch_tile_and_output() {
        assert_eq!(
            mq6_i8_entry("gfx1201", 128, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx12"
        );
        assert_eq!(
            mq6_i8_entry("gfx1201", 128, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 128, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx11"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 128, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 256, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx11_t256"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 256, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256"
        );
    }

    /// Every entry the launcher names exists in the module source it compiles
    /// from, and the gfx12 LDS contract matches the kernel header.
    #[test]
    fn kernel_sources_carry_the_launcher_symbols_and_lds_contract() {
        let sets = [
            (
                kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC,
                "gfx1201",
                128usize,
            ),
            (kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_SRC, "gfx1151", 128),
            (
                kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_T256_SRC,
                "gfx1151",
                256,
            ),
        ];
        for (src, arch, tile) in sets {
            for bf16 in [false, true] {
                let entry = mq6_i8_entry(arch, tile, bf16);
                // The gfx11 source names its entries through `I8_ENTRY(base)`,
                // which appends `_t256` under `I8_TILE 256`.
                let (base, t256) = match entry.strip_suffix("_t256") {
                    Some(base) => (base, true),
                    None => (entry, false),
                };
                let named = if arch == "gfx1201" {
                    src.contains(base)
                } else {
                    src.contains(&format!("I8_ENTRY({base})"))
                };
                assert!(named, "source lacks entry `{base}`");
                assert_eq!(
                    src.contains("#define I8_TILE 256"),
                    t256,
                    "tile define of `{entry}`"
                );
            }
            assert!(
                src.contains("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1"),
                "module must not re-export the standalone A8 quantizer"
            );
        }
        let gfx12 = format!("#define MQ6_I8_GFX12_LDS_BYTES {MQ6_I8_GFX12_LDS_BYTES}");
        assert!(
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC.contains(&gfx12),
            "gfx12 LDS contract drifted from the kernel header"
        );
    }

    /// The one-pass producer is the plain A8 rotate recipe: quantizer prelude without its
    /// standalone kernel, the emit aliases, the entry define, then the unmodified rotate body
    /// with neither the sigmoid gate nor AWQ.
    #[test]
    fn one_pass_producer_sources_are_the_plain_a8_rotate_recipe() {
        for (src, entry) in [
            (kernels::MQ_ROTATE_X_I8_GFX12_SRC, "mq_rotate_x_i8_gfx12"),
            (kernels::MQ_ROTATE_X_I8_GFX11_SRC, "mq_rotate_x_i8_gfx11"),
        ] {
            assert!(src.starts_with("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n"));
            assert!(src.contains(&format!("#define HIPFIRE_ROTATE_KERNEL {entry}\n")));
            assert!(src.contains(
                "#define emit_iu4_sidecar_from_producer8 emit_i8_sidecar_from_producer8"
            ));
            for gate in [
                "#define HIPFIRE_ROTATE_SIGMOID_GATE",
                "#define HIPFIRE_ROTATE_AWQ",
                "#define HIPFIRE_IU4_RUNTIME_CAND",
            ] {
                assert!(!src.contains(gate), "{entry}: unexpected {gate}");
            }
            let emit = src.find("emit_i8_sidecar_from_producer8(").unwrap();
            let body = src.find("HIPFIRE_ROTATE_KERNEL(").unwrap();
            assert!(emit < body, "{entry}: emit must precede the rotate body");
        }
    }

    #[test]
    fn ensure_kernel_module_names_are_stable() {
        // The lazy-kernel allowlist rows name these three modules.
        let names: Vec<&str> = [
            ("gfx1201", 48, 1),
            ("gfx1151", 2560, 512),
            ("gfx1151", 48, 1),
        ]
        .iter()
        .map(|&(a, r, n)| mq6_i8_launch(a, r, n, 20).unwrap().0)
        .collect();
        assert_eq!(
            names,
            [
                "gemm_mq6g256v2_mmq_i8_gfx12",
                "gemm_mq6g256v2_mmq_i8_gfx11_t256",
                "gemm_mq6g256v2_mmq_i8_gfx11"
            ]
        );
    }
}
