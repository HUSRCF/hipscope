//! Kernel Atlas: typed schema + JSONL writer for hipfire bench corpus.
//!
//! This crate is the **Rust collection layer** of the three-layer Atlas
//! architecture (see `docs/methodology/kernel-atlas-architecture.md`):
//!
//! 1. **Collection (this crate)** — bench tools emit typed `AtlasRow`
//!    values directly. No stdout-scraping, no regex.
//! 2. **Analysis** — `scripts/kernel_atlas.py` (on the HIPa branch)
//!    handles ranking, render, suggest, task-bundle generation.
//! 3. **Advisor** — future autotuner / advisor model consumes the corpus.
//!
//! The on-disk JSONL shape matches the Python harness so both layers
//! share a corpus.

pub mod schema;

pub use schema::{
    load_row, load_rows, truncate_jsonl, value_object, AtlasRow, ATLAS_SCHEMA,
};
