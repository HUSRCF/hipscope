// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4/R5/R6 developer-only autotuner: bounded legal-space search over existing
//! PM generator knobs (never edited source), an evidence-keyed result store with
//! a deterministic exported table, and the promotion gate pipeline.
//!
//! * [`domain`] (R4): `enumerate(&WorkloadCaseV1, node, tokens)` -> the ordered
//!   candidate domain of one corpus node; no GPU probing.
//! * [`run`] (R4): the phase driver (offline legality screen -> A/A noise floor ->
//!   discovery -> held-out confirmation in fresh processes) over `pm_loop`
//!   subprocesses, with a resumable ledger and work/time budgets.
//! * [`store`] (R5): append-only JSONL store of [`TuneRecordV1`], deterministic
//!   [`TableV1`] export and exact-key `lookup`. Only promoted rows are exported
//!   as selectable; nothing in the product reads this today.
//! * [`promote`] (R6): `can_promote(&TuneEvidenceV1, &ClaimPolicyV1)`; the
//!   KLD/state stage is an explicit unwired hook that always refuses.
//!
//! The types below are the frozen contract between those modules.

pub mod domain;
pub mod promote;
pub mod run;
pub mod store;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// sha256 (hex) of [`crate::noise::canonical`] JSON (sorted keys, no whitespace).
pub fn canonical_sha256(v: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(crate::noise::canonical(v).as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub const DOMAIN_SCHEMA: &str = "hipfire.tune_domain.v1";
pub const RECORD_SCHEMA: &str = "hipfire.tune_record.v1";
pub const TABLE_SCHEMA: &str = "hipfire.tune_table.v1";

/// One point of the legal knob space, exactly the `pm_loop` arguments that
/// select it (`--kernel K --arch A --shape S --<knob> V ...`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CandidateSpecV1 {
    pub kernel: String,
    pub arch: String,
    pub shape: String,
    pub knobs: BTreeMap<String, String>,
}

impl CandidateSpecV1 {
    /// First 16 hex of sha256 over the canonical JSON of the spec.
    pub fn id(&self) -> String {
        let v = serde_json::to_value(self).expect("spec serializes");
        canonical_sha256(&v)[..16].to_owned()
    }
    /// `pm_loop` arguments selecting this candidate (driver flags excluded).
    pub fn pm_loop_args(&self) -> Vec<String> {
        let mut a = vec![
            "--kernel".into(),
            self.kernel.clone(),
            "--arch".into(),
            self.arch.clone(),
            "--shape".into(),
            self.shape.clone(),
        ];
        for (k, v) in &self.knobs {
            a.push(format!("--{k}"));
            a.push(v.clone());
        }
        a
    }
}

/// The enumerated domain of one corpus node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainV1 {
    pub schema: String,
    pub case_id: String,
    pub node_index: usize,
    pub node_symbol: String,
    pub kernel: String,
    pub arch: String,
    pub shape: String,
    /// Knob value set of the shipped incumbent selection (the corpus node's symbol).
    pub incumbent: CandidateSpecV1,
    /// Deterministic order; includes the incumbent's generator twin.
    pub candidates: Vec<CandidateSpecV1>,
    /// How the shape was derived from the node (params checked, presets used).
    pub derivation: Vec<String>,
    /// sha256 of the canonical JSON of `(kernel, arch, shape, incumbent, candidates)`.
    pub domain_hash: String,
}

/// The fields of one `pm_loop` record the gates read (parsed, not trusted blindly:
/// the raw record path is kept so the evidence can be re-read).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptV1 {
    pub candidate_id: String,
    /// `offline`, `aa`, `discovery` or `confirm`.
    pub phase: String,
    pub replicate: u32,
    pub record_path: String,
    pub exit_code: i32,
    /// `error.stage` of the record, if any.
    pub error_stage: Option<String>,
    pub error_message: Option<String>,
    pub m7_status: Option<String>,
    pub correctness_pass: Option<bool>,
    pub identical_to_baseline: Option<bool>,
    pub baseline_launch_symbol: Option<String>,
    pub candidate_launch_symbol: Option<String>,
    pub baseline_co_sha256: Option<String>,
    pub candidate_co_sha256: Option<String>,
    /// `resources[].code_sha256` of the launch symbol of each arm.
    pub baseline_code_sha256: Option<String>,
    pub candidate_code_sha256: Option<String>,
    pub separable: Option<bool>,
    pub direction: Option<String>,
    pub sign_p: Option<f64>,
    pub block_ratio: Option<f64>,
    pub material: Option<bool>,
    pub rounds_run: Option<u64>,
    /// `timing.vs_noise_floor.floor` / `.exceeds` (None when no matching A/A group).
    pub noise_floor: Option<f64>,
    pub exceeds_noise_floor: Option<bool>,
    pub device_uuid: Option<String>,
    pub fabric_job: Option<String>,
}

/// Everything `can_promote` judges for one candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneEvidenceV1 {
    pub candidate: CandidateSpecV1,
    pub discovery: AttemptV1,
    pub confirms: Vec<AttemptV1>,
    /// A/A floor (q95 |block_ratio - 1|) of the incumbent at this shape.
    pub noise_floor: Option<f64>,
    pub aa_runs: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimPolicyV1 {
    pub sign_p_max: f64,
    pub min_effect: f64,
    /// Fresh-process confirmations required after discovery.
    pub confirm_processes: usize,
    /// A/A processes required for the noise floor.
    pub aa_processes: usize,
}

impl Default for ClaimPolicyV1 {
    fn default() -> Self {
        Self {
            sign_p_max: 1e-3,
            min_effect: 0.005,
            confirm_processes: 1,
            aa_processes: 3,
        }
    }
}

/// A refused promotion: the first failing stage, its reason, and every stage
/// that passed before it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    pub stage: String,
    pub reason: String,
    pub passed: Vec<String>,
}

/// A granted promotion (unreachable while the KLD/state hook is unwired).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionV1 {
    pub passed: Vec<String>,
    pub speedup: f64,
}

/// Store key: any change of any field is a different key (a miss).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TuneKeyV1 {
    pub arch: String,
    pub kernel: String,
    pub shape: String,
    /// `exact` for code whose output is bit-identical to the incumbent.
    pub numeric_contract: String,
    /// `eager` today.
    pub mode: String,
    pub baseline_code_sha256: String,
    pub candidate_co_sha256: String,
    pub candidate_code_sha256: String,
    /// hipfire commit of the generator/checker that produced the candidate.
    pub commit: String,
    /// sha256 of the `pm_loop` binary that measured it (toolchain identity).
    pub pm_loop_sha256: String,
}

/// One search outcome for one key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneRecordV1 {
    pub schema: String,
    pub key: TuneKeyV1,
    pub domain_hash: String,
    pub candidate: CandidateSpecV1,
    /// `winner` (separable, confirmed) or `not_winner`.
    pub verdict: String,
    pub evidence: TuneEvidenceV1,
    pub promotion: Result<PromotionV1, Refusal>,
}

/// A row the runtime would be allowed to select (only promoted records).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotedVariantV1 {
    pub key: TuneKeyV1,
    pub candidate: CandidateSpecV1,
    pub speedup: f64,
}

/// Deterministic export of a store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableV1 {
    pub schema: String,
    /// Sorted by key; promoted rows only.
    pub rows: Vec<PromotedVariantV1>,
    /// Sorted `(key, refusal)` of winner records that were refused promotion.
    pub refused: Vec<(TuneKeyV1, Refusal)>,
    /// Store lines that did not parse as a `TuneRecordV1` (`line N: error`); never rows.
    pub malformed: Vec<String>,
    /// sha256 of the canonical JSON of `rows`.
    pub rows_digest: String,
}
