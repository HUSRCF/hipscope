// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Gemma 4 E-series text architecture support.
//!
//! Runtime types are implemented here before loader registration. Until the
//! Carrier and quantizer land together, no existing serving route selects this
//! crate and no unloadable Gemma 4 HFQ artifact is emitted.

pub mod arch;
pub mod config;
pub mod forward;
pub mod gemma4;

pub use arch::{Gemma4, ARCH_ID};
pub use config::{Gemma4Config, Gemma4ESeriesVariant, LayerType, RopeType};
pub use forward::forward_batch;
pub use gemma4::{FullLayerWeights, Gemma4State, Gemma4Weights, LayerWeights, SlidingLayerWeights};
