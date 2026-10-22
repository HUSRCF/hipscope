// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Grouped-position down activation plane for the gfx1201 A3B MoE (module `qwen4_moe_down_gp_pm_gfx1201`, bundle
//! `kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco`). Three byte-identical twins of the flat-slot route, in this order:
//!
//! * `moe_rotate_i4_gp_pm_gfx1201`: [`super::moe_rotate_i4_pm::emit_gp`], the down A4 producer writing record
//!   `g * P + row` (72 bytes: `d` F32, `s` I32, 64 `qs`) for the grouped row `row` instead of `g * L + sorted[row]`.
//! * `qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4` / `..._nt8`: [`super::qwen4_moe_sym::emit_down_gp`], the NT4 / NT8 down
//!   entries reading record `g * m_total + 16 * tile + lane` instead of gathering `sorted`.
//!
//! The record bytes, every fold, every rounding and every store are those of the incumbent entries (the originals stay in
//! `qwen4_moe_iu4_sym_pm_gfx1201` / `moe_rotate_i4_pm_gfx1201` as the oracle anchors); only where a record lives
//! changes, so a tile's 16 slots read 1152 adjacent bytes per epoch instead of 16 scattered records.
use super::{moe_rotate_i4_pm, qwen4_moe_sym};
use crate::{Arch, Emitted};

/// Module key (also the bundle stem).
pub fn module_name(arch: Arch) -> String {
    format!("qwen4_moe_down_gp_pm_{}", arch.name())
}

/// The three exports in emission order.
pub fn symbols(arch: Arch) -> [String; 3] {
    let down = |nt| {
        qwen4_moe_sym::Spec {
            arch,
            kind: qwen4_moe_sym::Kind::Down,
            nt,
            rr: 1,
            codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
        }
        .symbol()
        .replacen("_sym_pm_", "_sym_gp_pm_", 1)
    };
    [moe_rotate_i4_pm::symbol_gp(arch), down(4), down(8)]
}

pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    if arch != Arch::Gfx1201 {
        return Err("qwen4_moe_down_gp: exact gfx1201 only".into());
    }
    let down = |nt| {
        qwen4_moe_sym::emit_down_gp(qwen4_moe_sym::Spec {
            arch,
            kind: qwen4_moe_sym::Kind::Down,
            nt,
            rr: 1,
            codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
        })
    };
    let emitted = vec![
        moe_rotate_i4_pm::emit_gp(moe_rotate_i4_pm::Spec { arch })?,
        down(4)?,
        down(8)?,
    ];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module_name(arch))?;
    Ok((emitted, text, proof))
}
