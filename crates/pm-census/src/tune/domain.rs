// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4 domain enumeration: the ordered candidate set of one corpus node.
//!
//! Supported today: kernel `qwen4_moe_sym`, arch `gfx1201` (the only arch whose
//! shipped entries have NT1/NT4/NT8 gate/up and down siblings plus the
//! grouped-position `down_gp` twins). The node's launch symbol selects the
//! *kind* (`gate_up` or `down`) and the incumbent `--epi`; the candidates are
//! every `--epi` entry of that kind that `pm_loop` knows. The model family
//! selects the expert-geometry preset, and the node's `M` / `m_total` params are
//! checked against that preset so a corpus node can never be tuned at a shape it
//! did not run.
//!
//! This module decides **no legality**: a listed candidate may still be
//! rejected by the generator, by M7 or by `pm_loop`'s baseline resolution. That
//! screen is the offline phase of [`super::run`], which prunes with a recorded
//! reason. The incumbent's own epi is listed on purpose: it emits code
//! byte-identical to the shipped object and serves as the negative control.

use super::{canonical_sha256, CandidateSpecV1, DomainV1, DOMAIN_SCHEMA};
use hipfire_atlas::workload::WorkloadCaseV1;
use serde_json::json;
use std::collections::BTreeMap;

const KERNEL: &str = "qwen4_moe_sym";
const ARCH: &str = "gfx1201";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    GateUp,
    Down,
}

/// Expert-geometry preset of a model family (`pm_loop` `--shape` preset name).
struct Preset {
    name: &'static str,
    experts: u64,
    top: u64,
    hidden: u64,
    inter: u64,
}

fn preset(family: &str) -> Result<Preset, String> {
    match family {
        "a3b" => Ok(Preset {
            name: "a3b",
            experts: 256,
            top: 8,
            hidden: 2048,
            inter: 512,
        }),
        "flash-next" => Ok(Preset {
            name: "fn",
            experts: 512,
            top: 10,
            hidden: 2560,
            inter: 640,
        }),
        other => Err(format!(
            "model family {other:?} has no expert-geometry preset (a3b, flash-next)"
        )),
    }
}

/// `(kind, --epi)` of a corpus launch symbol.
fn classify(symbol: &str) -> Option<(Kind, &'static str)> {
    const TABLE: [(&str, Kind, &str); 8] = [
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
            Kind::GateUp,
            "gate_up",
        ),
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
            Kind::GateUp,
            "gate_up_nt4",
        ),
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
            Kind::GateUp,
            "gate_up_nt8",
        ),
        ("qwen4_moe_down_iu4_sym_pm_gfx1201", Kind::Down, "down"),
        (
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
            Kind::Down,
            "down_nt4",
        ),
        (
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
            Kind::Down,
            "down_nt8",
        ),
        (
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
            Kind::Down,
            "down_gp_nt4",
        ),
        (
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
            Kind::Down,
            "down_gp_nt8",
        ),
    ];
    TABLE
        .iter()
        .find(|(s, _, _)| *s == symbol)
        .map(|(_, k, e)| (*k, *e))
}

fn candidate_epis(kind: Kind) -> &'static [&'static str] {
    match kind {
        Kind::GateUp => &["gate_up", "gate_up_nt4", "gate_up_nt8"],
        Kind::Down => &["down", "down_nt4", "down_nt8", "down_gp_nt4", "down_gp_nt8"],
    }
}

fn ceil16(x: u64) -> u64 {
    x.div_ceil(16) * 16
}

fn spec(shape: &str, epi: &str) -> CandidateSpecV1 {
    CandidateSpecV1 {
        kernel: KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape: shape.to_owned(),
        knobs: BTreeMap::from([("epi".to_owned(), epi.to_owned())]),
    }
}

/// The ordered domain of `case.chain[node]` at `tokens` rows. No GPU probing and
/// no legality decision (see the module docs).
pub fn enumerate(case: &WorkloadCaseV1, node: usize, tokens: u64) -> Result<DomainV1, String> {
    if case.arch != ARCH {
        return Err(format!(
            "case arch is {:?}; only {ARCH} is enumerable",
            case.arch
        ));
    }
    let n = case.chain.get(node).ok_or_else(|| {
        format!(
            "node {node} out of range: case has {} chain nodes",
            case.chain.len()
        )
    })?;
    let (kind, incumbent_epi) = classify(&n.symbol).ok_or_else(|| {
        format!(
            "node {node} symbol {:?} is not a {KERNEL} {ARCH} entry",
            n.symbol
        )
    })?;
    let p = preset(&case.model.family)?;
    let mut derivation = vec![format!(
        "model family {:?} -> preset {} (E={}, TOP={}, H={}, I={})",
        case.model.family, p.name, p.experts, p.top, p.hidden, p.inter
    )];
    derivation.push(format!(
        "node {node} symbol {} -> kind {}, incumbent --epi {incumbent_epi}",
        n.symbol,
        if kind == Kind::GateUp {
            "gate_up"
        } else {
            "down"
        }
    ));

    let (want_m, m_rule) = match kind {
        Kind::GateUp => (2 * p.inter, "2*I"),
        Kind::Down => (p.hidden, "H"),
    };
    let m = *n
        .params
        .get("M")
        .ok_or_else(|| format!("node {node} has no param M"))?;
    if m != want_m {
        return Err(format!(
            "node {node} param M is {m}, preset {} needs {m_rule} = {want_m}",
            p.name
        ));
    }
    derivation.push(format!(
        "param M = {m} equals {m_rule} of preset {}",
        p.name
    ));

    let want_total = ceil16(
        tokens
            .saturating_mul(p.top)
            .saturating_add(p.experts.saturating_mul(15)),
    );
    let m_total = *n
        .params
        .get("m_total")
        .ok_or_else(|| format!("node {node} has no param m_total"))?;
    if m_total != want_total {
        return Err(format!(
            "node {node} param m_total is {m_total}, tokens={tokens} on preset {} needs \
             ceil16(T*TOP + E*15) = {want_total}",
            p.name
        ));
    }
    derivation.push(format!(
        "param m_total = {m_total} equals ceil16({tokens}*{} + {}*15) at T={tokens}",
        p.top, p.experts
    ));

    let shape = format!("{},T={tokens}", p.name);
    let incumbent = spec(&shape, incumbent_epi);
    let candidates: Vec<CandidateSpecV1> = candidate_epis(kind)
        .iter()
        .map(|e| spec(&shape, e))
        .collect();
    let domain_hash = canonical_sha256(&json!({
        "kernel": KERNEL,
        "arch": ARCH,
        "shape": shape,
        "incumbent": incumbent,
        "candidates": candidates,
    }));
    Ok(DomainV1 {
        schema: DOMAIN_SCHEMA.to_owned(),
        case_id: case.case_id.clone(),
        node_index: node,
        node_symbol: n.symbol.clone(),
        kernel: KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape,
        incumbent,
        candidates,
        derivation,
        domain_hash,
    })
}

/// A minimal sealed A3B/FN case with one node, for the tune tests.
#[cfg(test)]
pub(crate) fn test_case(family: &str, symbol: &str, params: &[(&str, u64)]) -> WorkloadCaseV1 {
    let params: serde_json::Map<String, serde_json::Value> = params
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    let v = json!({
        "schema": "", "case_id": "",
        "model": {"name": "m", "family": family, "weight_format": null,
                  "file_sha256": null, "file_bytes": null, "sidecar_sha256": null},
        "prompt": null, "arch": "gfx1201", "device": null, "config": {},
        "phase": "prefill",
        "rows": {"tokens": 8192, "context": null, "chunk": null, "ragged": []},
        "mode": "eager", "placement": "resident",
        "chain": [{
            "symbol": symbol,
            "object": {"origin": "pm", "module": null, "artifact": null,
                       "artifact_sha256": null, "join": "test"},
            "abi": null, "params": params, "params_basis": "model_geometry",
            "grid": null, "block": null, "dynamic_lds": null,
            "kernarg_sha256": null, "calls": 1, "time_ns": null
        }],
        "fixtures": [],
        "state": {"rollback": false, "prefix_commit": false, "graph_replay": false, "notes": []},
        "denominator": {"status": "unmeasured", "reason": "test"},
        "provenance": {"source_kind": "test", "source_path": "t", "source_sha256": "00",
                       "extracted_by": "test", "commit": null}
    });
    serde_json::from_value::<WorkloadCaseV1>(v)
        .expect("test case literal parses")
        .seal()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GU_NT8: &str = "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8";
    const DOWN_GP_NT4: &str = "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4";

    fn epis(d: &DomainV1) -> Vec<&str> {
        d.candidates
            .iter()
            .map(|c| c.knobs["epi"].as_str())
            .collect()
    }

    #[test]
    fn a3b_gate_up_nt8_domain() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let d = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(d.schema, DOMAIN_SCHEMA);
        assert_eq!(d.shape, "a3b,T=8192");
        assert_eq!(d.incumbent.knobs["epi"], "gate_up_nt8");
        assert_eq!(epis(&d), ["gate_up", "gate_up_nt4", "gate_up_nt8"]);
        assert!(d.candidates.contains(&d.incumbent));
        assert_eq!(d.node_symbol, GU_NT8);
        assert_eq!(d.case_id, case.case_id);
        assert_eq!(d.derivation.len(), 4);
    }

    #[test]
    fn down_gp_nt4_has_five_candidates() {
        let case = test_case("a3b", DOWN_GP_NT4, &[("M", 2048), ("m_total", 69376)]);
        let d = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(
            epis(&d),
            ["down", "down_nt4", "down_nt8", "down_gp_nt4", "down_gp_nt8"]
        );
        assert_eq!(d.incumbent.knobs["epi"], "down_gp_nt4");
    }

    #[test]
    fn flash_next_uses_fn_preset() {
        let case = test_case(
            "flash-next",
            "qwen4_moe_down_iu4_sym_pm_gfx1201",
            &[("M", 2560), ("m_total", 48640)],
        );
        let d = enumerate(&case, 0, 4096).unwrap();
        assert_eq!(d.shape, "fn,T=4096");
        assert_eq!(d.incumbent.knobs["epi"], "down");
    }

    #[test]
    fn refuses_mismatches() {
        let ok = |m: u64, mt: u64| test_case("a3b", GU_NT8, &[("M", m), ("m_total", mt)]);
        let e = enumerate(&ok(1024, 69377), 0, 8192).unwrap_err();
        assert!(e.contains("69377") && e.contains("69376"), "{e}");
        let e = enumerate(&ok(2048, 69376), 0, 8192).unwrap_err();
        assert!(e.contains("2048") && e.contains("1024"), "{e}");
        let e = enumerate(&ok(1024, 69376), 0, 4096).unwrap_err();
        assert!(e.contains("69376"), "{e}");
        let e = enumerate(&ok(1024, 69376), 3, 8192).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let unknown = test_case("a3b", "mystery_kernel", &[("M", 1024), ("m_total", 69376)]);
        assert!(enumerate(&unknown, 0, 8192).is_err());
        let fam = test_case("other", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        assert!(enumerate(&fam, 0, 8192).unwrap_err().contains("preset"));
        let mut gfx = ok(1024, 69376);
        gfx.arch = "gfx1100".into();
        assert!(enumerate(&gfx, 0, 8192).unwrap_err().contains("gfx1100"));
    }

    #[test]
    fn domain_hash_is_stable_and_shape_sensitive() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let a = enumerate(&case, 0, 8192).unwrap();
        let b = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(a.domain_hash, b.domain_hash);
        assert_eq!(a, b);
        let case4 = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 36608)]);
        let c = enumerate(&case4, 0, 4096).unwrap();
        assert_ne!(a.domain_hash, c.domain_hash);
    }
}
