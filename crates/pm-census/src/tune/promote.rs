// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R6 promotion pipeline skeleton: `can_promote` evaluates ordered stages and
//! refuses at the first failure. The final `kld_state` stage is an explicit hook
//! that ALWAYS refuses today (no model-level KLD / state-equality evidence
//! producer exists), so nothing is promotable into product yet.

use super::{AttemptV1, ClaimPolicyV1, PromotionV1, Refusal, TuneEvidenceV1};

pub const STAGES: [&str; 6] = [
    "m7",
    "correctness",
    "abba",
    "noise_floor",
    "confirm",
    "kld_state",
];

fn refuse(idx: usize, reason: String) -> Refusal {
    Refusal {
        stage: STAGES[idx].to_owned(),
        reason,
        passed: STAGES[..idx].iter().map(|s| (*s).to_owned()).collect(),
    }
}

fn attempts(ev: &TuneEvidenceV1) -> impl Iterator<Item = (String, &AttemptV1)> {
    std::iter::once(("discovery".to_owned(), &ev.discovery)).chain(
        ev.confirms
            .iter()
            .enumerate()
            .map(|(i, a)| (format!("confirm[{i}]"), a)),
    )
}

fn stage_m7(ev: &TuneEvidenceV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.m7_status.as_deref() != Some("pass") {
            return Err(format!(
                "{n}: m7_status is {:?}, need \"pass\"",
                a.m7_status
            ));
        }
    }
    Ok(())
}

fn stage_correctness(ev: &TuneEvidenceV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.correctness_pass != Some(true) {
            return Err(format!(
                "{n}: correctness_pass is {:?}, need true",
                a.correctness_pass
            ));
        }
        if a.identical_to_baseline != Some(true) {
            return Err(format!(
                "{n}: identical_to_baseline is {:?}; NUMERIC candidates are not admitted by this policy (numeric contract is exact only)",
                a.identical_to_baseline
            ));
        }
    }
    Ok(())
}

/// ABBA criteria for one attempt (stage 3).
fn abba_ok(n: &str, a: &AttemptV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if a.separable != Some(true) {
        return Err(format!("{n}: separable is {:?}, need true", a.separable));
    }
    if a.direction.as_deref() != Some("faster") {
        return Err(format!(
            "{n}: direction is {:?}, need \"faster\"",
            a.direction
        ));
    }
    match a.sign_p {
        Some(p) if p <= policy.sign_p_max => {}
        other => {
            return Err(format!(
                "{n}: sign_p {other:?} not <= {}",
                policy.sign_p_max
            ))
        }
    }
    if a.material != Some(true) {
        return Err(format!("{n}: material is {:?}, need true", a.material));
    }
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() >= policy.min_effect => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect below min_effect {}",
                policy.min_effect
            ))
        }
    }
    Ok(())
}

/// Noise-floor criteria for one attempt against the evidence floor (stage 4).
fn noise_ok(n: &str, a: &AttemptV1, floor: f64) -> Result<(), String> {
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() > floor => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect does not exceed noise floor {floor}"
            ))
        }
    }
    if a.exceeds_noise_floor != Some(true) {
        return Err(format!(
            "{n}: exceeds_noise_floor is {:?}, need true",
            a.exceeds_noise_floor
        ));
    }
    Ok(())
}

fn stage_abba(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    abba_ok("discovery", &ev.discovery, policy)
}

fn stage_noise(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.aa_runs < policy.aa_processes {
        return Err(format!(
            "aa_runs {} < required {}",
            ev.aa_runs, policy.aa_processes
        ));
    }
    let Some(floor) = ev.noise_floor else {
        return Err("no noise floor measured; a missing floor is never a pass".to_owned());
    };
    noise_ok("discovery", &ev.discovery, floor)
}

fn stage_confirm(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.confirms.len() < policy.confirm_processes {
        return Err(format!(
            "{} confirmation(s) < required {}",
            ev.confirms.len(),
            policy.confirm_processes
        ));
    }
    let floor = ev
        .noise_floor
        .ok_or_else(|| "no noise floor for confirmation".to_owned())?;
    for (i, c) in ev.confirms.iter().enumerate() {
        let n = format!("confirm[{i}]");
        abba_ok(&n, c, policy)?;
        noise_ok(&n, c, floor)?;
        if c.candidate_code_sha256 != ev.discovery.candidate_code_sha256
            || c.candidate_code_sha256.is_none()
        {
            return Err(format!("{n}: candidate_code_sha256 differs from discovery"));
        }
        if c.baseline_code_sha256 != ev.discovery.baseline_code_sha256
            || c.baseline_code_sha256.is_none()
        {
            return Err(format!("{n}: baseline_code_sha256 differs from discovery"));
        }
    }
    Ok(())
}

/// Unwired model-level KLD / state-equality hook: always refuses.
pub fn kld_state_hook(_ev: &TuneEvidenceV1) -> Result<(), String> {
    Err("KLD/state hook not wired: no model-level KLD / state-equality evidence producer exists for this route; promotion into product is refused until R6 wires it".to_owned())
}

fn run_stages_1_to_5(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), Refusal> {
    stage_m7(ev).map_err(|r| refuse(0, r))?;
    stage_correctness(ev).map_err(|r| refuse(1, r))?;
    stage_abba(ev, policy).map_err(|r| refuse(2, r))?;
    stage_noise(ev, policy).map_err(|r| refuse(3, r))?;
    stage_confirm(ev, policy).map_err(|r| refuse(4, r))?;
    Ok(())
}

/// Stages 1-5 all pass (KLD/state not considered).
pub fn is_winner(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> bool {
    run_stages_1_to_5(ev, policy).is_ok()
}

pub fn can_promote(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<PromotionV1, Refusal> {
    run_stages_1_to_5(ev, policy)?;
    kld_state_hook(ev).map_err(|r| refuse(5, r))?;
    // Unreachable while the hook refuses; kept so a wired hook yields a grant.
    let ratios: Vec<f64> = attempts(ev).filter_map(|(_, a)| a.block_ratio).collect();
    let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    Ok(PromotionV1 {
        passed: STAGES.iter().map(|s| (*s).to_owned()).collect(),
        speedup: 1.0 / g,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::CandidateSpecV1;
    use std::collections::BTreeMap;

    fn attempt(phase: &str) -> AttemptV1 {
        AttemptV1 {
            candidate_id: "c".into(),
            phase: phase.into(),
            replicate: 0,
            record_path: "r.json".into(),
            exit_code: 0,
            error_stage: None,
            error_message: None,
            m7_status: Some("pass".into()),
            correctness_pass: Some(true),
            identical_to_baseline: Some(true),
            baseline_launch_symbol: None,
            candidate_launch_symbol: None,
            baseline_co_sha256: None,
            candidate_co_sha256: None,
            baseline_code_sha256: Some("b".into()),
            candidate_code_sha256: Some("c".into()),
            separable: Some(true),
            direction: Some("faster".into()),
            sign_p: Some(1e-5),
            block_ratio: Some(0.95),
            material: Some(true),
            rounds_run: Some(10),
            noise_floor: Some(0.01),
            exceeds_noise_floor: Some(true),
            device_uuid: None,
            fabric_job: None,
        }
    }

    fn ev() -> TuneEvidenceV1 {
        TuneEvidenceV1 {
            candidate: CandidateSpecV1 {
                kernel: "k".into(),
                arch: "gfx1100".into(),
                shape: "s".into(),
                knobs: BTreeMap::new(),
            },
            discovery: attempt("discovery"),
            confirms: vec![attempt("confirm")],
            noise_floor: Some(0.01),
            aa_runs: 3,
        }
    }

    fn stage_of(ev: &TuneEvidenceV1) -> String {
        can_promote(ev, &ClaimPolicyV1::default())
            .unwrap_err()
            .stage
    }

    #[test]
    fn full_pass_is_winner_but_kld_refuses() {
        let e = ev();
        let p = ClaimPolicyV1::default();
        assert!(is_winner(&e, &p));
        let r = can_promote(&e, &p).unwrap_err();
        assert_eq!(r.stage, "kld_state");
        assert_eq!(r.passed, STAGES[..5].to_vec());
        assert!(r.reason.contains("KLD/state hook not wired"));
    }

    #[test]
    fn each_stage_refuses_at_itself() {
        let p = ClaimPolicyV1::default();
        let mut e = ev();
        e.discovery.m7_status = Some("fail".into());
        assert_eq!(stage_of(&e), "m7");
        assert!(!is_winner(&e, &p));

        let mut e = ev();
        e.discovery.identical_to_baseline = Some(false);
        let r = can_promote(&e, &p).unwrap_err();
        assert_eq!(r.stage, "correctness");
        assert!(r.reason.contains("NUMERIC candidates are not admitted"));
        let mut e = ev();
        e.discovery.correctness_pass = Some(false);
        assert_eq!(stage_of(&e), "correctness");

        let mut e = ev();
        e.discovery.separable = Some(false);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.direction = Some("slower".into());
        e.discovery.block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.block_ratio = Some(0.998);
        assert_eq!(stage_of(&e), "abba");

        let mut e = ev();
        e.noise_floor = None;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.aa_runs = 2;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.noise_floor = Some(0.2);
        assert_eq!(stage_of(&e), "noise_floor");

        let mut e = ev();
        e.confirms.clear();
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].direction = Some("slower".into());
        e.confirms[0].block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].candidate_code_sha256 = Some("other".into());
        assert_eq!(stage_of(&e), "confirm");
    }
}
