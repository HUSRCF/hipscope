// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Arch-agnostic loader contract. Concrete `ModelState`/`LoadedModel`
//! and the registry live top-of-DAG in `hipfire-loader`; this module
//! holds only what the arch crates need to implement a carrier.

use std::path::PathBuf;
use crate::hfq::HfqFile;
use rdna_compute::Gpu;

/// A model on disk, before we know its arch. Carries either a parsed
/// HFQ header or a directory (safetensors/ParoQuant — probed later).
pub enum ModelSource {
    Hfq(HfqFile),
    Dir(PathBuf),
}

impl ModelSource {
    /// Convenience: the HFQ arch_id if this is an HFQ source.
    pub fn arch_id(&self) -> Option<u32> {
        match self {
            ModelSource::Hfq(h) => Some(h.arch_id),
            ModelSource::Dir(_) => None,
        }
    }
}

/// Everything a carrier's `load` needs beyond the source itself.
pub struct LoadCtx<'a> {
    pub path: &'a str,
    pub max_seq: usize,
    pub draft_path: Option<&'a str>,
    pub kv_mode_override: Option<&'a str>,
    pub kv_adaptive_override: Option<&'a str>,
    pub state_quant_override: Option<&'a str>,
    pub cask: &'a CaskConfig,
    pub pp: usize,
    pub gpu: &'a mut Gpu,
}

/// One arch's load contract. Object-safe — usable as `&dyn Carrier`.
/// Implementations live in `hipfire-loader::carriers`.
pub trait Carrier {
    fn name(&self) -> &'static str;
    fn probe(&self, src: &ModelSource) -> bool;
}

/// CASK/TriAttention params forwarded by the CLI at load time.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    pub cask_m_folding: bool,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}


