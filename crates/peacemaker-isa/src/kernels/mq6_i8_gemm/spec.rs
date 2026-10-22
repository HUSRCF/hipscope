// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Variant axes and fixed geometry of the MQ6G256V2 x `block_i8_128` GEMM.
use crate::{kernels::iu4_k1::Variant, Arch, KernargLayout};

/// Epilogue: F32 overwrite, or BF16 bits (`isfinite(v) ? RNE : u >> 16`, `v = 0 + y`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi {
    Set,
    SetBf16,
}
/// CTA raster. `Banded` is the product (hipcc `IU4_G12_RASTER`, band 8). `Linear`
/// (`rs = x * 128`, `bs = y * 128`) exists only so the CPU interpreter, which has no
/// SALU float-reciprocal semantics, can run the data path of the same kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Raster {
    Banded,
    Linear,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub epi: Epi,
    pub raster: Raster,
    pub arch: Arch,
}

/// 128 weight rows x 128 tokens x 8 waves (32 rows x 64 tokens per wave).
pub const TILE_ROWS: u32 = 128;
pub const TILE_TOKENS: u32 = 128;
pub const THREADS: u32 = 256;
pub const VGPR_CEILING: u16 = 192;
pub const RASTER_BAND: u32 = 8;
/// `block_i8_128`: `{f32 d, i32 s, int8 qs[128]}`.
pub const XBLK_BYTES: u32 = 136;
/// MQ6G256V2 group: 8 header bytes + 192 payload bytes.
pub const GROUP_BYTES: u32 = 200;

/// LDS byte layout (every plane a separate builder slot).
///   slot s: A_s = 8192 (4 token-group pairs x 4 K16 steps x 512 B),
///           W_s = 8192 (4 row-group pairs x 4 K16 steps x 512 B, decoded codes)
///   DS_p = 1024 per K128 parity: planar token `d` [128 f32] then `float(s)` [128 f32]
///   SZ_p = 1024 per K128 parity: planar row `sc` [128 f32] then `z'` [128 f32]
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub a: [u32; 2],
    pub w: [u32; 2],
    pub ds: [u32; 2],
    pub sz: [u32; 2],
    pub a_bytes: u32,
    pub w_bytes: u32,
    pub meta_bytes: u32,
    pub launch: u32,
}
pub const LAYOUT: Layout = Layout {
    a: [0, 16384],
    w: [8192, 24576],
    ds: [32768, 33792],
    sz: [34816, 35840],
    a_bytes: 8192,
    w_bytes: 8192,
    meta_bytes: 1024,
    launch: 36864,
};
/// The dynamic LDS bytes the launch must pass (the same figure as the hipcc twin's
/// `MQ6_I8_GFX12_LDS_BYTES`).
pub const LAUNCH_LDS_BYTES: u32 = 36864;

impl Epi {
    pub fn name(self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::SetBf16 => "set_bf16",
        }
    }
}
impl Spec {
    pub fn product(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Banded,
            arch: Arch::Gfx1201,
        }
    }
    /// The interpreter twin: identical data path, linear raster.
    pub fn emulation(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Linear,
            arch: Arch::Gfx1201,
        }
    }
    pub fn module() -> &'static str {
        "gemm_mq6g256v2_mmq_i8_pm_gfx1201"
    }
    pub fn symbol(self) -> String {
        let base = format!("gemm_mq6g256v2_mmq_i8_pm_{}_gfx1201", self.epi.name());
        match self.raster {
            Raster::Banded => base,
            Raster::Linear => format!("{base}_linear"),
        }
    }
    pub fn variant_name(self) -> String {
        format!(
            "128x128x8-{}{}",
            self.epi.name(),
            match self.raster {
                Raster::Banded => "",
                Raster::Linear => "-linear",
            }
        )
    }
    /// The hipcc twin's ABI `(A, Xq, Y, M, K, N)` followed by K1's hidden block.
    pub fn kernargs(self) -> KernargLayout {
        Variant::FullSet.kernargs()
    }
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 {
            return Err("mq6_i8_gemm targets gfx1201 only".into());
        }
        Ok(())
    }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "set" => Ok(Self::Set),
            "set_bf16" => Ok(Self::SetBf16),
            _ => Err(format!("unknown mq6_i8_gemm epilogue {s}")),
        }
    }
}
