// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R5 evidence-keyed result store: append-only JSONL of [`TuneRecordV1`], a
//! deterministic [`TableV1`] export, and exact-key `lookup`.
//!
//! * No runtime reads this today; there is no startup benchmarking.
//! * An unrepresented key is a miss and the caller keeps the existing incumbent
//!   selector. Lookup is exact key equality, never nearest/prefix.
//! * Malformed lines are refusals (reported in `malformed`), never hits.
//! * If two promoted records share a key, the one appearing LAST in the file wins
//!   (later evidence supersedes earlier).

use super::{
    canonical_sha256, PromotedVariantV1, Refusal, TableV1, TuneKeyV1, TuneRecordV1, RECORD_SCHEMA,
    TABLE_SCHEMA,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// Append one canonical JSON line (file created if missing).
pub fn append(path: &Path, rec: &TuneRecordV1) -> Result<(), String> {
    let v = serde_json::to_value(rec).map_err(|e| format!("serialize record: {e}"))?;
    let mut line = crate::noise::canonical(&v);
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    f.write_all(line.as_bytes())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate(rec: &TuneRecordV1) -> Result<(), String> {
    if rec.schema != RECORD_SCHEMA {
        return Err(format!("schema {:?}, expected {RECORD_SCHEMA}", rec.schema));
    }
    let k = &rec.key;
    for (n, v) in [
        ("arch", &k.arch),
        ("kernel", &k.kernel),
        ("shape", &k.shape),
        ("numeric_contract", &k.numeric_contract),
        ("mode", &k.mode),
    ] {
        if v.is_empty() {
            return Err(format!("key.{n} is empty"));
        }
    }
    for (n, v) in [
        ("baseline_code_sha256", &k.baseline_code_sha256),
        ("candidate_co_sha256", &k.candidate_co_sha256),
        ("candidate_code_sha256", &k.candidate_code_sha256),
        ("pm_loop_sha256", &k.pm_loop_sha256),
    ] {
        if !is_hex(v, 64) {
            return Err(format!("key.{n} is not 64 lowercase hex"));
        }
    }
    if !is_hex(&k.commit, 40) {
        return Err("key.commit is not 40 lowercase hex".to_owned());
    }
    Ok(())
}

/// Parse a store: `(records, malformed "line N: err")`.
pub fn import(text: &str) -> (Vec<TuneRecordV1>, Vec<String>) {
    let mut recs = Vec::new();
    let mut bad = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let n = i + 1;
        match serde_json::from_str::<TuneRecordV1>(line) {
            Err(e) => bad.push(format!("line {n}: {e}")),
            Ok(r) => match validate(&r) {
                Ok(()) => recs.push(r),
                Err(e) => bad.push(format!("line {n}: {e}")),
            },
        }
    }
    (recs, bad)
}

/// Deterministic table: promoted rows (last record per key wins) sorted by key,
/// refused winners sorted by key, malformed lines, and the rows digest.
pub fn export(text: &str) -> TableV1 {
    let (recs, malformed) = import(text);
    let mut rows: BTreeMap<TuneKeyV1, PromotedVariantV1> = BTreeMap::new();
    let mut refused: Vec<(TuneKeyV1, Refusal)> = Vec::new();
    for r in recs {
        match r.promotion {
            Ok(p) => {
                rows.insert(
                    r.key.clone(),
                    PromotedVariantV1 {
                        key: r.key,
                        candidate: r.candidate,
                        speedup: p.speedup,
                    },
                );
            }
            Err(refusal) => {
                if r.verdict == "winner" {
                    refused.push((r.key, refusal));
                }
            }
        }
    }
    refused.sort_by(|a, b| a.0.cmp(&b.0));
    let rows: Vec<PromotedVariantV1> = rows.into_values().collect();
    let rows_digest = canonical_sha256(&serde_json::to_value(&rows).expect("rows serialize"));
    TableV1 {
        schema: TABLE_SCHEMA.to_owned(),
        rows,
        refused,
        malformed,
        rows_digest,
    }
}

/// Exact key equality only.
pub fn lookup<'a>(t: &'a TableV1, key: &TuneKeyV1) -> Option<&'a PromotedVariantV1> {
    t.rows
        .binary_search_by(|r| r.key.cmp(key))
        .ok()
        .map(|i| &t.rows[i])
}

/// `pm-census tune export --store FILE [--out FILE]`.
pub fn cli(args: &[String]) -> Result<(), String> {
    let mut store: Option<&str> = None;
    let mut out: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--store" => store = Some(val),
            "--out" => out = Some(val),
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    let store = store.ok_or("usage: pm-census tune export --store FILE [--out FILE]")?;
    let text = std::fs::read_to_string(store).map_err(|e| format!("read {store}: {e}"))?;
    let table = export(&text);
    let v = serde_json::to_value(&table).map_err(|e| format!("serialize table: {e}"))?;
    println!("{}", crate::noise::canonical(&v));
    if let Some(o) = out {
        let pretty = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
        std::fs::write(o, pretty + "\n").map_err(|e| format!("write {o}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{AttemptV1, CandidateSpecV1, ClaimPolicyV1, PromotionV1, TuneEvidenceV1};

    fn h(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn key(tag: char) -> TuneKeyV1 {
        TuneKeyV1 {
            arch: "gfx1100".into(),
            kernel: "gemv".into(),
            shape: "4096x4096".into(),
            numeric_contract: "exact".into(),
            mode: "eager".into(),
            baseline_code_sha256: h('a'),
            candidate_co_sha256: h('b'),
            candidate_code_sha256: h(tag),
            commit: "c".repeat(40),
            pm_loop_sha256: h('d'),
        }
    }

    fn spec() -> CandidateSpecV1 {
        CandidateSpecV1 {
            kernel: "gemv".into(),
            arch: "gfx1100".into(),
            shape: "4096x4096".into(),
            knobs: BTreeMap::new(),
        }
    }

    fn attempt() -> AttemptV1 {
        AttemptV1 {
            candidate_id: "c".into(),
            phase: "discovery".into(),
            replicate: 0,
            record_path: "r".into(),
            exit_code: 0,
            error_stage: None,
            error_message: None,
            m7_status: None,
            correctness_pass: None,
            identical_to_baseline: None,
            baseline_launch_symbol: None,
            candidate_launch_symbol: None,
            baseline_co_sha256: None,
            candidate_co_sha256: None,
            baseline_code_sha256: None,
            candidate_code_sha256: None,
            separable: None,
            direction: None,
            sign_p: None,
            block_ratio: None,
            material: None,
            rounds_run: None,
            noise_floor: None,
            exceeds_noise_floor: None,
            device_uuid: None,
            fabric_job: None,
        }
    }

    fn rec(tag: char, verdict: &str, promotion: Result<PromotionV1, Refusal>) -> TuneRecordV1 {
        TuneRecordV1 {
            schema: RECORD_SCHEMA.into(),
            key: key(tag),
            domain_hash: h('e'),
            candidate: spec(),
            verdict: verdict.into(),
            evidence: TuneEvidenceV1 {
                candidate: spec(),
                discovery: attempt(),
                confirms: vec![],
                noise_floor: None,
                aa_runs: 0,
            },
            promotion,
        }
    }

    fn promoted(tag: char, speedup: f64) -> TuneRecordV1 {
        rec(
            tag,
            "winner",
            Ok(PromotionV1 {
                passed: vec!["m7".into()],
                speedup,
            }),
        )
    }

    fn refusal() -> Refusal {
        Refusal {
            stage: "kld_state".into(),
            reason: "unwired".into(),
            passed: vec![],
        }
    }

    fn text(recs: &[TuneRecordV1]) -> String {
        recs.iter()
            .map(|r| crate::noise::canonical(&serde_json::to_value(r).unwrap()) + "\n")
            .collect()
    }

    #[test]
    fn hit_and_mutation_matrix() {
        let t = export(&text(&[promoted('1', 1.1)]));
        let k = key('1');
        assert_eq!(lookup(&t, &k).unwrap().speedup, 1.1);
        let muts: Vec<Box<dyn Fn(&mut TuneKeyV1)>> = vec![
            Box::new(|k| k.arch.push('x')),
            Box::new(|k| k.kernel.push('x')),
            Box::new(|k| k.shape.push('x')),
            Box::new(|k| k.numeric_contract = "numeric".into()),
            Box::new(|k| k.mode = "graph".into()),
            Box::new(|k| k.baseline_code_sha256 = h('9')),
            Box::new(|k| k.candidate_co_sha256 = h('9')),
            Box::new(|k| k.candidate_code_sha256 = h('9')),
            Box::new(|k| k.commit = "9".repeat(40)),
            Box::new(|k| k.pm_loop_sha256 = h('9')),
        ];
        assert_eq!(muts.len(), 10);
        for (i, m) in muts.iter().enumerate() {
            let mut q = k.clone();
            m(&mut q);
            assert_ne!(q, k, "mutation {i} changed nothing");
            assert!(lookup(&t, &q).is_none(), "mutation {i} hit");
        }
    }

    #[test]
    fn refused_and_not_winner_routing() {
        let t = export(&text(&[
            promoted('1', 1.1),
            rec('2', "winner", Err(refusal())),
            rec('3', "not_winner", Err(refusal())),
        ]));
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.refused.len(), 1);
        assert_eq!(t.refused[0].0, key('2'));
        assert!(lookup(&t, &key('2')).is_none());
        assert!(lookup(&t, &key('3')).is_none());
        assert!(t.malformed.is_empty());
    }

    #[test]
    fn malformed_lines_do_not_affect_rows() {
        let good = text(&[promoted('1', 1.1)]);
        let mut wrong_schema = promoted('2', 1.2);
        wrong_schema.schema = "nope".into();
        let mut bad_commit = promoted('3', 1.3);
        bad_commit.key.commit = "abc".into();
        let mut all = good.clone();
        all.push_str("{\"schema\": trunc\n");
        all.push_str("garbage\n\n");
        all.push_str(&text(&[wrong_schema, bad_commit]));
        let t = export(&all);
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.malformed.len(), 4);
        assert!(t.malformed[0].starts_with("line 2:"));
        assert_eq!(t.rows_digest, export(&good).rows_digest);
    }

    #[test]
    fn deterministic_and_order_independent() {
        let a = promoted('1', 1.1);
        let b = promoted('2', 1.2);
        let r = rec('4', "winner", Err(refusal()));
        let t1 = export(&text(&[a.clone(), b.clone(), r.clone()]));
        let t1b = export(&text(&[a.clone(), b.clone(), r.clone()]));
        let t2 = export(&text(&[r, b, a]));
        let c = |t: &TableV1| crate::noise::canonical(&serde_json::to_value(t).unwrap());
        assert_eq!(c(&t1), c(&t1b));
        assert_eq!(t1.rows_digest, t2.rows_digest);
        assert_eq!(c(&t1), c(&t2));
    }

    #[test]
    fn later_duplicate_supersedes() {
        let t = export(&text(&[promoted('1', 1.1), promoted('1', 1.3)]));
        assert_eq!(t.rows.len(), 1);
        assert_eq!(t.rows[0].speedup, 1.3);
    }

    #[test]
    fn append_roundtrip() {
        let p = std::env::temp_dir().join(format!(
            "pm-census-store-test-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        let a = promoted('1', 1.1);
        let b = rec('2', "winner", Err(refusal()));
        append(&p, &a).unwrap();
        append(&p, &b).unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        let (recs, bad) = import(&s);
        assert!(bad.is_empty());
        assert_eq!(recs, vec![a, b]);
        let _ = ClaimPolicyV1::default();
    }
}
