// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Gemma 4 text architecture support.
//!
//! The first integration stage intentionally exposes only the configuration
//! contract. It recognizes the E2B and E4B text towers without registering a
//! loader or changing serving dispatch.

pub mod config;

pub use config::{Gemma4Config, Gemma4ESeriesVariant, LayerType, RopeType};
