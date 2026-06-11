// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch Carrier structs for the Step-A probe/name registry.
//! Step B moves `Carrier` impls into the individual arch crates
//! and adds associated `Bundle` types.

use hipfire_runtime::loader_api::{Carrier, ModelSource};

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str { "qwen2" }
    fn probe(&self, src: &ModelSource) -> bool { src.arch_id() == Some(7) }
}

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str { "qwen35" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(5) | Some(6))
    }
}

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str { "llama" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(id) if id < 5)
    }
}
