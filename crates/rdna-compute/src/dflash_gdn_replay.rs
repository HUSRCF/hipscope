// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6c: `Gpu` launchers for the multi-layer DFlash/MTP GDN
//! replay (gfx1201).
//!
//! `GdnTape::replay_gdn` advances every LinearAttention layer's conv ring and
//! recurrent state by `n_steps` tokens from the verify tape. Layers are
//! independent, so the per-layer launch sequence (conv1d + QK norm +
//! interleave + GDN: 4 launches x 48 layers on gfx1201) collapses to two
//! launches over a device pointer table with one row per layer:
//!
//! 1. [`Gpu::dflash_gdn_replay_pre_ml`] — the `dflash_gdn_pre_replay_gfx1100`
//!    preamble body per (block, layer), grid.y = layer.
//! 2. [`Gpu::gated_delta_net_q8_fast_ml`] — `gated_delta_net_q8_fast` built
//!    with the shipping defines plus `HIPFIRE_GDN_LAYER_TABLE`, grid.z =
//!    layer, frames reserved exactly as the per-layer launches would.
//!
//! Both kernels read pointers from device memory, so a Redline tape could
//! not attribute their effects; the launchers therefore decline (return
//! `Ok(false)`) while a Redline recording is open. HipGraph capture is fine
//! (`launch_maybe_blob`). The pointer tables are owned by the caller.

use crate::dispatch::Gpu;
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

/// Preamble kernel source.
pub const DFLASH_GDN_REPLAY_PRE_ML_SRC: &str =
    include_str!("../../../kernels/src/dflash_gdn_replay_pre_ml.hip");
/// Compiled-module key and device symbol for the preamble kernel.
pub const DFLASH_GDN_REPLAY_PRE_ML_SYMBOL: &str = "dflash_gdn_replay_pre_ml";
/// Recurrence kernel source: the shipping `gated_delta_net_q8_fast` defines
/// (`GATED_DELTA_NET_Q8_FAST_SRC`) plus the layer-table entry.
pub const GATED_DELTA_NET_Q8_FAST_ML_SRC: &str = concat!(
    "#define HIPFIRE_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_PREFETCH 1\n",
    "#define HIPFIRE_GDN_LAYER_TABLE 1\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_fast_ml\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
/// Compiled-module key and device symbol for the recurrence kernel.
pub const GATED_DELTA_NET_Q8_FAST_ML_SYMBOL: &str = "gated_delta_net_q8_fast_ml";

/// Preamble threads per block (matches `GDN_PRE_BLOCK`).
pub const DFLASH_GDN_REPLAY_PRE_BLOCK: u32 = 256;
/// Only head_dim == 128 is supported (matches `GDN_PRE_HD` / the GDN `HD`).
pub const DFLASH_GDN_REPLAY_HEAD_DIM: usize = 128;
/// Replay step ceiling (matches `GDN_PRE_MAXN` LDS staging).
pub const DFLASH_GDN_REPLAY_MAX_STEPS: usize = 16;
/// GDN tile rows per block (the shipping `HIPFIRE_GDN_TILE_ROWS`).
const GDN_TILE_ROWS: u32 = 4;

/// One preamble row per LinearAttention layer. `#[repr(C)]` layout matches
/// `DflashReplayPreLayer` in `kernels/src/dflash_gdn_replay_pre_ml.hip`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashReplayPreLayer {
    pub qkv_tape: u64,
    pub conv_w: u64,
    pub conv_state: u64,
    pub v_out: u64,
    pub q_dst: u64,
    pub k_dst: u64,
}

/// One recurrence row per LinearAttention layer. `#[repr(C)]` layout matches
/// `GdnLayerTable` in `kernels/src/gated_delta_net_q8_fast.hip`. `ef` is 0
/// when the layer has no error-feedback residual.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLayerTable {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub gate: u64,
    pub beta: u64,
    pub s_q8: u64,
    pub s_scales: u64,
    pub output: u64,
    pub ef: u64,
}

/// Byte view of a `#[repr(C)]` table of plain `u64`s for one `memcpy_htod`.
pub fn table_bytes<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: callers pass the two repr(C) all-u64 row types above.
    unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, std::mem::size_of_val(rows)) }
}

impl Gpu {
    /// Host-side eligibility shared by both launchers: exact gfx1201, the
    /// `HIPFIRE_GDN_REPLAY_ML_OFF` opt-out, no open Redline recording, the
    /// fast (single-end requant) GDN kernel, head_dim 128, GQA divisibility,
    /// and 1 <= n_steps <= 16.
    pub fn gdn_replay_ml_eligible(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        n_steps: usize,
    ) -> bool {
        self.arch_caps.is_gfx1201()
            && !self.flags.gdn_replay_ml_off
            && !self.replay.is_recording()
            && !crate::norm::dn_requant_per_token()
            && key_head_dim == DFLASH_GDN_REPLAY_HEAD_DIM
            && value_head_dim == DFLASH_GDN_REPLAY_HEAD_DIM
            && n_key_heads > 0
            && n_v_heads % n_key_heads == 0
            && (1..=DFLASH_GDN_REPLAY_MAX_STEPS).contains(&n_steps)
    }

    /// JIT both kernels (idempotent). Called when the tape arms its tables,
    /// before any capture.
    pub fn ensure_dflash_gdn_replay_ml(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
            DFLASH_GDN_REPLAY_PRE_ML_SRC,
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
        )?;
        self.ensure_kernel(
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
            GATED_DELTA_NET_Q8_FAST_ML_SRC,
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
        )
    }

    /// Conv1d + QK norm + interleave for `n_layers` table rows in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_gdn_replay_pre_ml(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_v_heads: usize,
        n_key_heads: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_steps: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml()?;
        let tp = table as *mut c_void;
        let nvh = n_v_heads as i32;
        let nkh = n_key_heads as i32;
        let ratio = (n_v_heads / n_key_heads) as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let qd = qkv_dim as i32;
        let ns = n_steps as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nvh as *const _ as *mut c_void,
            &nkh as *const _ as *mut c_void,
            &ratio as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &ns as *const _ as *mut c_void,
            &q_scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let v_blocks = v_dim.div_ceil(DFLASH_GDN_REPLAY_PRE_BLOCK as usize);
        self.launch_maybe_blob(
            DFLASH_GDN_REPLAY_PRE_ML_SYMBOL,
            [(n_key_heads + v_blocks) as u32, n_layers as u32, 1],
            [DFLASH_GDN_REPLAY_PRE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nvh);
                b.push_i32(nkh);
                b.push_i32(ratio);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(qd);
                b.push_i32(ns);
                b.push_f32(q_scale);
                b.push_f32(eps);
                b
            },
        )
    }

    /// GDN Q8 recurrence (fast, EF-capable) for `n_layers` table rows in one
    /// launch. Reserves `n_layers * n_steps` stochastic-rounding frames —
    /// the same total, in the same per-layer order, as `n_layers` sequential
    /// `gated_delta_net_q8_batch_seq` calls.
    pub fn gated_delta_net_q8_fast_ml(
        &mut self,
        table: *const c_void,
        n_layers: usize,
        n_steps: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_dflash_gdn_replay_ml()?;
        let tp = table as *mut c_void;
        let nt = n_steps as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = crate::norm::reserve_gdn_requant_frames((n_layers * n_steps) as u32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            GATED_DELTA_NET_Q8_FAST_ML_SYMBOL,
            [
                n_heads as u32,
                DFLASH_GDN_REPLAY_HEAD_DIM as u32 / GDN_TILE_ROWS,
                n_layers as u32,
            ],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_u32(fr);
                b
            },
        )
    }
}
