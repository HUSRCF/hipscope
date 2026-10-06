// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{Arch, Emitted};

pub fn symbol(arch: Arch) -> String { format!("indexed_attention_select_from_scores_pm_{}", arch.name()) }

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 { return Err("qsa_topk: built for gfx1151 only".into()) }
    Err(format!("qsa_topk: {} is not authored yet", symbol(arch)))
}
