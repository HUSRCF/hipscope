// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The contract between `pm_loop` (driver: certify, load, ABBA, JSON) and a
//! production-generator kernel module (`iu4.rs`, `moe.rs`): knob parsing, shape
//! grammar, emit/link of the candidate, the shipped baseline, deterministic
//! synthetic inputs, kernarg packing, and the CPU reference.
//!
//! Included with `#[path = "pm_loop/workload.rs"] mod workload;`.
#![allow(dead_code)]

use rdna_compute::Gpu;
use serde_json::Value;
use std::collections::BTreeMap;

pub type Result<T, E = String> = std::result::Result<T, E>;

/// Which kernel a launch/output belongs to. `Baseline` is timing arm A,
/// `Candidate` arm B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Baseline,
    Candidate,
}

/// One kernel dispatch exactly as the production launcher issues it.
#[derive(Clone, Debug)]
pub struct Launch {
    /// Complete kernarg blob (explicit args; the driver passes it to
    /// `hip.launch_kernel_blob`, which fills the hidden block).
    pub kernarg: Vec<u8>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic LDS bytes.
    pub lds: u32,
}

/// A loadable code object (ELF `.co` or clang offload bundle `.hxaco`) and
/// the symbol the workload launches.
#[derive(Clone, Debug)]
pub struct Object {
    pub bytes: Vec<u8>,
    /// Kernel symbol launched by timing/correctness.
    pub launch_symbol: String,
    /// Every export of `bytes` (all certified for the candidate; resources
    /// are reported for each).
    pub exports: Vec<String>,
    /// JSON description: `{knobs:{..}}` for an emitted object, or
    /// `{artifact:"<path>"}` for a committed object.
    pub desc: Value,
    /// JSON provenance of the selection: `{source:"file:line", ..}`.
    pub source: Value,
}

/// Candidate-vs-CPU-reference result.
#[derive(Clone, Debug)]
pub struct RefReport {
    /// Human name of the reference (e.g. `cpu_iu4_fold_rn`).
    pub reference: String,
    pub max_abs_err: f64,
    pub tolerance: f64,
    /// `max_abs_err <= tolerance` and no NaN/Inf mismatch.
    pub pass: bool,
}

/// Raw `--key value` flags the driver did not consume. A module `take`s the
/// flags it understands and the driver errors on whatever is left.
#[derive(Clone, Debug, Default)]
pub struct Knobs(pub BTreeMap<String, String>);

impl Knobs {
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
    /// `take`, or `default`.
    pub fn take_or(&mut self, key: &str, default: &str) -> String {
        self.take(key).unwrap_or_else(|| default.to_owned())
    }
    /// Error naming every flag nobody consumed.
    pub fn finish(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unknown option(s) for this kernel: {}",
                self.0
                    .keys()
                    .map(|k| format!("--{k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }
}

/// One production-generator kernel under test. Constructed GPU-free (knob
/// parsing + shape grammar + candidate/baseline selection); `setup` then
/// allocates and fills the device inputs.
pub trait Workload {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    /// Target arch (always `gfx1201` today).
    fn arch(&self) -> &'static str;
    /// Parsed shape as JSON (every dimension the launch derives from).
    fn shape_json(&self) -> Value;
    /// Candidate knobs as JSON (the resolved Spec axes).
    fn candidate_knobs(&self) -> Value;

    /// Emit + link the candidate with the native writer (the path of
    /// `peacemaker-isa emit --co`). Errors (e.g. a closed fold axis) are
    /// reported verbatim by the driver.
    fn candidate(&self) -> Result<Object>;
    /// The shipped baseline (what production selects for gfx1201), or the
    /// `--base-*` knob variant. `source` names file:line / artifact + sha256.
    fn baseline(&self) -> Result<Object>;

    /// Allocate device buffers and upload deterministic synthetic inputs.
    fn setup(&mut self, gpu: &Gpu) -> Result<()>;
    /// Launch parameters for `arm` (kernarg points at that arm's buffers).
    fn launch(&self, arm: Arm) -> Launch;
    /// Restore `arm`'s accumulating state (e.g. `Y` for epi add; zero `Y`
    /// otherwise) to its pre-launch bytes. Called outside event intervals.
    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()>;
    /// Download `arm`'s output (the payload the correctness check covers).
    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>>;
    /// Compare a downloaded output to the CPU reference.
    fn check(&self, out: &[u8]) -> Result<RefReport>;
    /// Max absolute element difference between two outputs (interpreting the
    /// bytes in the kernel's output dtype); `inf` if any element differs in
    /// NaN-ness. 0.0 iff element-wise equal.
    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64>;

    /// `--emu`: run the candidate on the peacemaker emulator at a small
    /// shape if every instruction is qualified; otherwise
    /// `Value::String("refused (<reason>)")`. Never fabricate a result.
    fn emu(&self) -> Value;
}
