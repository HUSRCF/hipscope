// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4 domain enumeration: the ordered candidate set of one corpus node.
//!
//! Two node families are supported, both on arch `gfx1201`:
//!
//! * kernel `qwen4_moe_sym` (the shipped entries have NT1/NT4/NT8 gate/up and down
//!   siblings plus the grouped-position `down_gp` twins). The node's launch symbol
//!   selects the *kind* (`gate_up` or `down`) and the incumbent `--epi`; the
//!   candidates are every `--epi` entry of that kind that `pm_loop` knows. The model
//!   family selects the expert-geometry preset, and the node's `M` / `m_total`
//!   params are checked against that preset so a corpus node can never be tuned at a
//!   shape it did not run.
//! * kernel `iu4_gemm`, the dense 27B prefill `gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`
//!   (the slab-activation silu gate/up GEMM; knobs `fold`/`tile`/`cacc`/`epi`/`alayout`
//!   as `pm_loop` names them). The rocprof-derived corpus node carries no params, only
//!   its launch grid, so the model-family preset (`dense-27b`: hidden K = 5120,
//!   intermediate M = 17408) is checked against the grid: `grid.x = 2*M/128` virtual
//!   128-row tiles and `grid.y = ceil(tokens/128)`. The candidates are the generator
//!   twin of the incumbent (the negative control), the one other tile (illegal for the
//!   slab silu symbol, so the offline screen prunes it with the generator's reason), the
//!   generator knob space `ring {2,3} x wait {after,mid,before}` (ring-major; the
//!   control point is the twin, the legal points carry their non-default `--ring` /
//!   `--wait` knobs, the illegal ones are recorded in the derivation with the reason),
//!   the opt-in numeric `--fold-op muladd` candidate ([`IU4_NUMERIC_VAR`]) and one
//!   candidate per `.rip` script, whose path is the `rip` knob (`pm_loop --rip`).
//!   The scripts are given to [`enumerate_with_rips`], or for the CLI (which calls
//!   [`enumerate`]) as the `*.rip` files of the directory named by the environment
//!   variable [`RIP_DIR_VAR`]. Their sha256 enter the derivation and the domain hash.
//!
//! This module decides **no legality** (the one exception is the `iu4_gemm` `ring`
//! axis, whose generator refusal is mirrored so the pruned points are recorded in the
//! derivation): a listed candidate may still be
//! rejected by the generator, by M7 or by `pm_loop`'s baseline resolution. That
//! screen is the offline phase of [`super::run`], which prunes with a recorded
//! reason. The incumbent's own knob set is listed on purpose: it emits code
//! byte-identical to the shipped object and serves as the negative control.

use super::{canonical_sha256, CandidateSpecV1, DomainV1, DOMAIN_SCHEMA};
use hipfire_atlas::workload::{ChainNodeV1, WorkloadCaseV1};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const KERNEL: &str = "qwen4_moe_sym";
const IU4_KERNEL: &str = "iu4_gemm";
const MQ6_KERNEL: &str = "mq6_i8_gemm";
const ARCH: &str = "gfx1201";

/// Environment variable naming a directory whose `*.rip` files [`enumerate`]
/// lists as the `rip` candidates of an `iu4_gemm` node.
pub const RIP_DIR_VAR: &str = "PM_CENSUS_RIP_DIR";

/// Environment variable that, set to `1`, adds the deliberately NUMERIC
/// `fold-op muladd` candidate (non-fused stage-2 accumulate; `pm_loop` declares
/// `numeric_contract.relation = "numeric"`) to an `iu4_gemm` node's domain. It
/// exists to exercise the KLD/state gate; it is never part of the default space.
pub const IU4_NUMERIC_VAR: &str = "PM_CENSUS_IU4_NUMERIC";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// `(kind, --epi)` of a corpus `qwen4_moe_sym` launch symbol.
fn classify_moe(symbol: &str) -> Option<(Kind, &'static str)> {
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

/// The shipped launch symbol of the dense 27B prefill gate/up silu GEMM on the
/// slab activation layout (module `gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s`).
pub const IU4_GATE_UP_SILU_B1S: &str = "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s";

/// The `pm_loop --kernel iu4_gemm` knob values that select a shipped symbol.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Iu4Knobs {
    fold: &'static str,
    tile: &'static str,
    cacc: &'static str,
    epi: &'static str,
    alayout: &'static str,
}

/// What a corpus launch symbol is tunable as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    /// `mq6_i8_gemm`: the entry the corpus profile observed.
    Mq6(Mq6Entry),
    /// `qwen4_moe_sym`: its kind and incumbent `--epi`.
    Moe(Kind, &'static str),
    /// `iu4_gemm`: the knob set of the shipped symbol.
    Iu4(Iu4Knobs),
}

/// The tunable identity of a corpus launch symbol (`None`: not an enumerable node).
fn classify(symbol: &str) -> Option<Node> {
    if symbol == IU4_GATE_UP_SILU_B1S {
        return Some(Node::Iu4(Iu4Knobs {
            fold: "k128",
            tile: "128x128x8",
            cacc: "1",
            epi: "silu",
            alayout: "slab",
        }));
    }
    match symbol {
        MQ6_SYM_128 => return Some(Node::Mq6(Mq6Entry::Narrow)),
        MQ6_SYM_WIDE => return Some(Node::Mq6(Mq6Entry::Wide)),
        _ => {}
    }
    classify_moe(symbol).map(|(k, e)| Node::Moe(k, e))
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

/// The `*.rip` files of `dir` as absolute paths, sorted by file name.
pub fn rips_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for ent in rd {
        let p = ent.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if p.extension().is_some_and(|x| x == "rip") && p.is_file() {
            out.push(
                p.canonicalize()
                    .map_err(|e| format!("{}: {e}", p.display()))?,
            );
        }
    }
    out.sort();
    Ok(out)
}

/// The ordered domain of `case.chain[node]` at `tokens` rows. No GPU probing and
/// no legality decision (see the module docs). An `iu4_gemm` node takes its `rip`
/// candidates from the `*.rip` files of the directory named by [`RIP_DIR_VAR`]
/// (none when unset); every other node ignores that variable.
pub fn enumerate(case: &WorkloadCaseV1, node: usize, tokens: u64) -> Result<DomainV1, String> {
    let is_iu4 = case
        .chain
        .get(node)
        .is_some_and(|n| matches!(classify(&n.symbol), Some(Node::Iu4(_))));
    let rips = match std::env::var_os(RIP_DIR_VAR) {
        Some(dir) if is_iu4 => rips_in_dir(Path::new(&dir))?,
        _ => Vec::new(),
    };
    let numeric = is_iu4 && std::env::var_os(IU4_NUMERIC_VAR).is_some_and(|v| v == "1");
    enumerate_inner(case, node, tokens, &rips, numeric)
}

/// [`enumerate`] with explicit `.rip` candidate scripts (`iu4_gemm` nodes only; a
/// node of another kernel refuses a non-empty list). The deliberately numeric
/// `iu4_gemm` candidate ([`IU4_NUMERIC_VAR`]) is not enumerated.
pub fn enumerate_with_rips(
    case: &WorkloadCaseV1,
    node: usize,
    tokens: u64,
    rips: &[PathBuf],
) -> Result<DomainV1, String> {
    enumerate_inner(case, node, tokens, rips, false)
}

/// The shared body of [`enumerate`] / [`enumerate_with_rips`]; `iu4_numeric`
/// adds the numeric `fold-op muladd` candidate to an `iu4_gemm` node.
fn enumerate_inner(
    case: &WorkloadCaseV1,
    node: usize,
    tokens: u64,
    rips: &[PathBuf],
    iu4_numeric: bool,
) -> Result<DomainV1, String> {
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
    match classify(&n.symbol) {
        Some(Node::Moe(kind, incumbent_epi)) => {
            if !rips.is_empty() {
                return Err(format!(
                    "node {node} symbol {} is a {KERNEL} entry: .rip candidates are enumerated \
                     for {IU4_KERNEL} nodes only",
                    n.symbol
                ));
            }
            enumerate_moe(case, n, node, tokens, kind, incumbent_epi)
        }
        Some(Node::Iu4(k)) => enumerate_iu4(case, n, node, tokens, k, rips, iu4_numeric),
        Some(Node::Mq6(entry)) => {
            if !rips.is_empty() {
                return Err(format!(
                    "node {node} symbol {} is an {MQ6_KERNEL} entry: .rip candidates are enumerated \
                     for {IU4_KERNEL} nodes only",
                    n.symbol
                ));
            }
            enumerate_mq6(case, n, node, tokens, entry)
        }
        None => Err(format!(
            "node {node} symbol {:?} is not an enumerable {ARCH} entry \
             ({KERNEL} entries, {IU4_GATE_UP_SILU_B1S}, {MQ6_SYM_128}, {MQ6_SYM_WIDE})",
            n.symbol
        )),
    }
}

/// `pm_loop` `iu4_gemm` knob set of `k` plus `overrides`.
fn iu4_spec(shape: &str, k: Iu4Knobs, overrides: &[(&str, &str)]) -> CandidateSpecV1 {
    let mut knobs: BTreeMap<String, String> = [
        ("fold", k.fold),
        ("tile", k.tile),
        ("cacc", k.cacc),
        ("epi", k.epi),
        ("alayout", k.alayout),
    ]
    .iter()
    .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
    .collect();
    for (a, b) in overrides {
        knobs.insert((*a).to_owned(), (*b).to_owned());
    }
    CandidateSpecV1 {
        kernel: IU4_KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape: shape.to_owned(),
        knobs,
    }
}

/// The dense 27B prefill gate/up GEMM: `M` intermediate rows (gate and up each),
/// `K` hidden, `N = tokens`.
struct DensePreset {
    hidden: u64,
    inter: u64,
}

fn dense_preset(family: &str) -> Result<DensePreset, String> {
    match family {
        "dense-27b" => Ok(DensePreset {
            hidden: 5120,
            inter: 17408,
        }),
        other => Err(format!(
            "model family {other:?} has no dense gate/up geometry preset (dense-27b)"
        )),
    }
}

/// Rows per workgroup tile of the shipped `128x128x8` iu4 tile (`pm_loop` `--tile`).
const IU4_TILE_ROWS: u64 = 128;
/// Tokens per workgroup tile of that tile.
const IU4_TILE_TOKENS: u64 = 128;

/// `pm_loop` `--ring` / `--wait` values the generator knows (`iu4_gemm::RingDepth`,
/// `iu4_gemm::WaitPlace`). The first of each is the shipped codegen.
const IU4_RINGS: [&str; 2] = ["2", "3"];
const IU4_WAITS: [&str; 3] = ["after", "mid", "before"];

/// Why a ring depth has no generator: the refusal `iu4_gemm::Spec::validate` gives (pm-census
/// does not link the generator, so the rule is mirrored here; `pm_loop --ring 3` surfaces the
/// generator's own text, and an iu4 candidate this table lists as legal that the generator
/// refuses is pruned by the offline screen with the generator's reason).
fn iu4_ring_refusal(ring: &str) -> Option<&'static str> {
    (ring == "3").then_some(
        "ring depth 3 has no generator: the typed LDS core has only two-buffer rings \
         (peacemaker-author Ring) and a 4-slab K-loop trip cannot rotate three slots \
         through immediate LDS addresses (a 12-slab period would be needed)",
    )
}

/// The generator knob points other than the control, `(legal candidates, pruned-point
/// derivation lines)`, in ring-major order. A point carries only its non-default knobs.
fn iu4_generator_space(shape: &str, k: Iu4Knobs) -> (Vec<CandidateSpecV1>, Vec<String>) {
    let (mut legal, mut pruned) = (Vec::new(), Vec::new());
    for ring in IU4_RINGS {
        for wait in IU4_WAITS {
            if (ring, wait) == (IU4_RINGS[0], IU4_WAITS[0]) {
                continue;
            }
            if let Some(why) = iu4_ring_refusal(ring) {
                pruned.push(format!("pruned ring={ring} wait={wait}: {why}"));
                continue;
            }
            let mut overrides = Vec::new();
            if ring != IU4_RINGS[0] {
                overrides.push(("ring", ring));
            }
            if wait != IU4_WAITS[0] {
                overrides.push(("wait", wait));
            }
            legal.push(iu4_spec(shape, k, &overrides));
        }
    }
    (legal, pruned)
}

fn enumerate_iu4(
    case: &WorkloadCaseV1,
    n: &ChainNodeV1,
    node: usize,
    tokens: u64,
    k: Iu4Knobs,
    rips: &[PathBuf],
    numeric: bool,
) -> Result<DomainV1, String> {
    let p = dense_preset(&case.model.family)?;
    let mut derivation = vec![
        format!(
            "model family {:?} -> preset dense-27b (K={}, M={} intermediate rows)",
            case.model.family, p.hidden, p.inter
        ),
        format!(
            "node {node} symbol {} -> kernel {IU4_KERNEL}, incumbent knobs fold={} tile={} \
             cacc={} epi={} alayout={}",
            n.symbol, k.fold, k.tile, k.cacc, k.epi, k.alayout
        ),
    ];
    if let Some(&m) = n.params.get("M") {
        if m != p.inter {
            return Err(format!(
                "node {node} param M is {m}, preset dense-27b needs {}",
                p.inter
            ));
        }
    }
    if tokens == 0 {
        return Err("tokens must be at least 1".to_owned());
    }
    let grid = n.grid.ok_or_else(|| {
        format!(
            "node {node} has no launch grid: the {IU4_KERNEL} shape is bound by the grid \
             (x = 2*M/{IU4_TILE_ROWS}, y = ceil(tokens/{IU4_TILE_TOKENS}))"
        )
    })?;
    let want_x = (2 * p.inter).div_ceil(IU4_TILE_ROWS);
    if u64::from(grid[0]) != want_x {
        return Err(format!(
            "node {node} grid.x is {}, preset dense-27b needs 2*M/{IU4_TILE_ROWS} = {want_x}",
            grid[0]
        ));
    }
    let want_y = tokens.div_ceil(IU4_TILE_TOKENS);
    if u64::from(grid[1]) != want_y {
        return Err(format!(
            "node {node} grid.y is {}, tokens={tokens} needs ceil(tokens/{IU4_TILE_TOKENS}) \
             = {want_y}",
            grid[1]
        ));
    }
    derivation.push(format!(
        "grid.x = {} equals 2*{}/{IU4_TILE_ROWS}; grid.y = {} equals ceil({tokens}/{IU4_TILE_TOKENS})",
        grid[0], p.inter, grid[1]
    ));

    let shape = format!("M={},N={tokens},K={}", p.inter, p.hidden);
    let incumbent = iu4_spec(&shape, k, &[]);
    // Deterministic order: the generator twin (control), the other tile, the generator
    // knob space (ring x wait, ring-major), the numeric candidate if asked, then the scripts.
    let mut candidates = vec![
        incumbent.clone(),
        iu4_spec(&shape, k, &[("tile", "256x128x16")]),
    ];
    let (legal, pruned) = iu4_generator_space(&shape, k);
    derivation.push(format!(
        "generator knob space ring {{{}}} x wait {{{}}} = {} points: control (ring=2 wait=after) \
         is the incumbent twin, {} further legal, {} pruned",
        IU4_RINGS.join(","),
        IU4_WAITS.join(","),
        IU4_RINGS.len() * IU4_WAITS.len(),
        legal.len(),
        pruned.len()
    ));
    candidates.extend(legal);
    derivation.extend(pruned);
    if numeric {
        derivation.push(format!(
            "numeric candidate fold-op=muladd ({IU4_NUMERIC_VAR}=1): non-fused stage-2 accumulate, \
             pm_loop declares numeric_contract relation numeric; never part of the default space"
        ));
        candidates.push(iu4_spec(&shape, k, &[("fold-op", "muladd")]));
    }
    let mut rip_shas = Vec::new();
    for path in rips {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let p = path
            .to_str()
            .ok_or_else(|| format!("{}: the rip path is not valid UTF-8", path.display()))?;
        derivation.push(format!("rip candidate {p} sha256 {sha}"));
        rip_shas.push(json!({"path": p, "sha256": sha}));
        candidates.push(iu4_spec(&shape, k, &[("rip", p)]));
    }
    let domain_hash = canonical_sha256(&json!({
        "kernel": IU4_KERNEL,
        "arch": ARCH,
        "shape": shape,
        "incumbent": incumbent,
        "candidates": candidates,
        "rips": rip_shas,
    }));
    Ok(DomainV1 {
        schema: DOMAIN_SCHEMA.to_owned(),
        case_id: case.case_id.clone(),
        node_index: node,
        node_symbol: n.symbol.clone(),
        kernel: IU4_KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape,
        incumbent,
        candidates,
        derivation,
        domain_hash,
    })
}

fn enumerate_moe(
    case: &WorkloadCaseV1,
    n: &ChainNodeV1,
    node: usize,
    tokens: u64,
    kind: Kind,
    incumbent_epi: &'static str,
) -> Result<DomainV1, String> {
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

/// The shipped F32 `set` entry of the 128 x 128 PM tile (`gemm_mq6_i8.rs` `MQ6_I8_PM_ENTRY_F32`).
pub const MQ6_SYM_128: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201";
/// The shipped F32 `set` entry of the 128 x 256 PM tile (`MQ6_I8_PM_ENTRY_F32_T256`).
pub const MQ6_SYM_WIDE: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256";

/// The PM tile a corpus launch symbol (the profile) ran.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mq6Entry {
    /// `MQ6_SYM_128`: 128 x 128, 256 threads, grid `[ceil(M/128), ceil(N/128)]`.
    Narrow,
    /// `MQ6_SYM_WIDE`: 128 x 256, 512 threads, grid `[ceil(M/128), ceil(N/256)]`.
    Wide,
}

/// CUs the runtime selector sees on the gfx1201 R9700: `cu_count_or_default`
/// (`rdna-compute/src/gemm.rs:32519`) = HIP's multiprocessor count (32 WGPs) x 2
/// (`profiler.rs:142` `hip_mp_count_to_cu_count`); the selector's own tests use 64
/// (`gemm_mq6_i8.rs:776`).
const MQ6_CU_COUNT: u64 = 64;

/// Flash-Next gfx1201 MQ6 PM shapes keyed by the narrow grid's x extent (`ceil(M/128)`):
/// `(grid.x, M, K)`, the `pm_loop mq6` `fn:` presets (`pm_loop/mq6.rs` `FN_MK`). The rocprof
/// corpus node has no params; its grid names M and the model geometry names K. Grid x values
/// that do not name one preset M (`1`: M = 48 or 128; `5`: M = 640) are refused.
const MQ6_FN_SHAPES: [(u64, u64, u64); 5] = [
    (80, 10240, 2560),
    (48, 6144, 2560),
    (96, 12288, 2560),
    (20, 2560, 6144),
    (4, 512, 2560),
];

/// Token-tile width the runtime launches for `m` rows x `n` tokens on `cu_count` CUs: a
/// transcription of `mq6_i8_pm_token_tile` (`rdna-compute/src/gemm_mq6_i8.rs:101-116`),
/// called by `mq6_i8_pm_launch` (`:118`) from `gemm_mq6g256v2_i8_prequant` (`:409`).
fn mq6_runtime_token_tile(m: u64, n: u64, cu_count: u64) -> u64 {
    let r = n % 256;
    if r != 0 && r <= 128 {
        return 128;
    }
    let wgps = (cu_count / 2).max(1);
    let rows = m.div_ceil(128);
    let rounds256 = (rows * n.div_ceil(256)).div_ceil(wgps);
    let rounds128 = (rows * n.div_ceil(128)).div_ceil(wgps);
    if 188 * rounds256 <= 100 * rounds128 {
        256
    } else {
        128
    }
}

/// `pm_loop mq6_i8_gemm` knob set of one candidate: F32 `set` epilogue, `raster` and `tile`, and
/// `base-shipped` = the runtime's tile, so every arm is measured against the shipped entry the
/// runtime launches at the shape.
fn mq6_spec(shape: &str, tile: &str, raster: &str, runtime_tile: &str) -> CandidateSpecV1 {
    CandidateSpecV1 {
        kernel: MQ6_KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape: shape.to_owned(),
        knobs: [
            ("epi", "set"),
            ("raster", raster),
            ("tile", tile),
            ("base-shipped", runtime_tile),
        ]
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect(),
    }
}

/// The launch symbol `pm_loop`'s baseline must resolve to for `d`: the entry the runtime
/// selects (the incumbent's `base-shipped` tile) for an `mq6_i8_gemm` domain, else the corpus
/// node's own symbol. The corpus symbol of an `mq6_i8_gemm` node can be a profile of an older
/// selector, so it is not the incumbent there.
pub fn baseline_symbol(d: &DomainV1) -> String {
    if d.kernel == MQ6_KERNEL {
        return match d.incumbent.knobs.get("base-shipped").map(String::as_str) {
            Some("128x256") => MQ6_SYM_WIDE.to_owned(),
            _ => MQ6_SYM_128.to_owned(),
        };
    }
    d.node_symbol.clone()
}

/// The legal-tile x raster domain of a Flash-Next `gemm_mq6g256v2_mmq_i8_pm_set` node at one
/// shape. The incumbent (and the code-identical control, listed first) is the tile the
/// runtime's selector takes at that shape, not the `128x128` tile `pm_loop` pairs by default.
/// Every candidate must reproduce the baseline bit for bit (`pm_loop` requires
/// `identical_to_baseline` for this workload), so all are numeric contract `exact`.
fn enumerate_mq6(
    case: &WorkloadCaseV1,
    n: &ChainNodeV1,
    node: usize,
    tokens: u64,
    entry: Mq6Entry,
) -> Result<DomainV1, String> {
    if case.model.family != "flash-next" {
        return Err(format!(
            "model family {:?} has no {MQ6_KERNEL} shape table (flash-next)",
            case.model.family
        ));
    }
    if tokens == 0 {
        return Err("tokens must be at least 1".to_owned());
    }
    let mut derivation = vec![format!(
        "node {node} symbol {} -> kernel {MQ6_KERNEL}, F32 epilogue `set`, corpus entry {}",
        n.symbol,
        if entry == Mq6Entry::Narrow {
            "128x128 (grid.y = ceil(N/128))"
        } else {
            "128x256 (grid.y = ceil(N/256))"
        }
    )];
    let grid = n.grid.ok_or_else(|| {
        format!("node {node} has no launch grid: the {MQ6_KERNEL} shape is bound by the grid")
    })?;
    let gx = u64::from(grid[0]);
    let &(_, m, k) = MQ6_FN_SHAPES
        .iter()
        .find(|(x, _, _)| *x == gx)
        .ok_or_else(|| {
            format!(
                "node {node} grid.x is {gx}: no unique Flash-Next MQ6 preset has \
                 ceil(M/128) = {gx} (one of {})",
                MQ6_FN_SHAPES
                    .iter()
                    .map(|(x, m, k)| format!("{x} -> M={m},K={k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    if let Some(&pm) = n.params.get("M") {
        if pm != m {
            return Err(format!(
                "node {node} param M is {pm}, grid.x = {gx} names M = {m}"
            ));
        }
    }
    let tile_tokens = if entry == Mq6Entry::Narrow { 128 } else { 256 };
    let want_y = tokens.div_ceil(tile_tokens);
    if u64::from(grid[1]) != want_y {
        return Err(format!(
            "node {node} grid.y is {}, tokens={tokens} needs ceil(tokens/{tile_tokens}) = {want_y}",
            grid[1]
        ));
    }
    derivation.push(format!(
        "grid [{gx}, {}] equals [ceil({m}/128), ceil({tokens}/{tile_tokens})]: M={m}, K={k} \
         (model geometry; the corpus node has no params), N={tokens}",
        grid[1]
    ));
    let sibling: Vec<String> = case
        .chain
        .iter()
        .enumerate()
        .filter(|(_, c)| c.symbol == n.symbol)
        .map(|(i, c)| {
            format!(
                "node {i} grid {:?} {} ns",
                c.grid,
                c.time_ns.map_or_else(|| "?".to_owned(), |t| t.to_string())
            )
        })
        .collect();
    if sibling.len() > 1 {
        derivation.push(format!(
            "same-symbol nodes of the case (pick the highest-time node for the headline shape): {}",
            sibling.join("; ")
        ));
    }

    let runtime = mq6_runtime_token_tile(m, tokens, MQ6_CU_COUNT);
    let runtime_tile = if runtime == 256 { "128x256" } else { "128x128" };
    derivation.push(format!(
        "runtime selector mq6_i8_pm_token_tile(m={m}, n={tokens}, cu={MQ6_CU_COUNT}) = {runtime} \
         (rdna-compute/src/gemm_mq6_i8.rs:101-116; launch at :409 via mq6_i8_pm_launch :118; \
         cu_count_or_default gemm.rs:32519, WGP x2 profiler.rs:142): the incumbent tile is \
         {runtime_tile} entry {}",
        if runtime == 256 {
            MQ6_SYM_WIDE
        } else {
            MQ6_SYM_128
        }
    ));
    let observed = if entry == Mq6Entry::Narrow { 128 } else { 256 };
    if observed != runtime {
        derivation.push(format!(
            "NOTE corpus symbol {} ran the {observed}-token tile, the current selector takes the \
             {runtime}-token tile at this shape: the profile predates the selector or ran another \
             device; the incumbent is the current selector's tile and pm_loop's baseline is \
             that shipped entry (--base-shipped {runtime_tile})",
            n.symbol
        ));
    }

    let shape = format!("M={m},N={tokens},K={k}");
    let incumbent = mq6_spec(&shape, runtime_tile, "banded", runtime_tile);
    // Control first (the incumbent's own knobs), then tile-major, banded before linear.
    let mut candidates = vec![incumbent.clone()];
    for tile in ["128x128", "256x128", "128x256"] {
        for raster in ["banded", "linear"] {
            let c = mq6_spec(&shape, tile, raster, runtime_tile);
            if c != incumbent {
                candidates.push(c);
            }
        }
    }
    derivation.push(
        "numerics: all tiles and rasters run the same exact integer dot and per-element fold \
         order; pm_loop requires identical_to_baseline for this workload, so every candidate is \
         exact (none routed to the KLD gate)"
            .to_owned(),
    );
    let domain_hash = canonical_sha256(&json!({
        "kernel": MQ6_KERNEL,
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
        kernel: MQ6_KERNEL.to_owned(),
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
    test_case_grid(family, symbol, params, None)
}

/// [`test_case`] with the node's launch grid (a rocprof-derived dense node has a grid, no params).
#[cfg(test)]
pub(crate) fn test_case_grid(
    family: &str,
    symbol: &str,
    params: &[(&str, u64)],
    grid: Option<[u32; 3]>,
) -> WorkloadCaseV1 {
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
            "grid": grid, "block": null, "dynamic_lds": null,
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

    // ---- iu4_gemm: dense 27B prefill gate/up silu (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`)

    fn iu4_case() -> WorkloadCaseV1 {
        test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 64, 1]))
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-domain-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_rips(dir: &Path, files: &[(&str, &str)]) -> Vec<PathBuf> {
        for (name, text) in files {
            std::fs::write(dir.join(name), text).unwrap();
        }
        rips_in_dir(dir).unwrap()
    }

    #[test]
    fn classify_knows_both_families() {
        assert_eq!(
            classify(IU4_GATE_UP_SILU_B1S),
            Some(Node::Iu4(Iu4Knobs {
                fold: "k128",
                tile: "128x128x8",
                cacc: "1",
                epi: "silu",
                alayout: "slab",
            }))
        );
        assert_eq!(
            classify(GU_NT8),
            Some(Node::Moe(Kind::GateUp, "gate_up_nt8"))
        );
        // The token-layout symbol and the other epilogues are not enumerable nodes.
        assert_eq!(classify("gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1"), None);
        assert_eq!(
            classify("gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s"),
            None
        );
    }

    #[test]
    fn iu4_domain_without_rips_is_control_and_tile() {
        let case = iu4_case();
        let d = enumerate_with_rips(&case, 0, 8192, &[]).unwrap();
        assert_eq!(d.schema, DOMAIN_SCHEMA);
        assert_eq!(d.kernel, "iu4_gemm");
        assert_eq!(d.arch, "gfx1201");
        assert_eq!(d.shape, "M=17408,N=8192,K=5120");
        assert_eq!(d.node_symbol, IU4_GATE_UP_SILU_B1S);
        assert_eq!(d.case_id, case.case_id);
        let k = &d.incumbent.knobs;
        assert_eq!(k.len(), 5);
        assert_eq!(
            [
                k["fold"].as_str(),
                k["tile"].as_str(),
                k["cacc"].as_str(),
                k["epi"].as_str(),
                k["alayout"].as_str()
            ],
            ["k128", "128x128x8", "1", "silu", "slab"]
        );
        assert!(!k.contains_key("rip"));
        // control, the other tile, then the legal (ring x wait) points: ring 2 with wait mid / before.
        assert_eq!(d.candidates.len(), 4);
        assert_eq!(d.candidates[0], d.incumbent);
        assert_eq!(d.candidates[1].knobs["tile"], "256x128x16");
        assert_eq!(d.candidates[1].knobs["alayout"], "slab");
        // A generator-knob point keeps every incumbent knob and adds only its non-default knobs.
        for (i, wait) in [(2, "mid"), (3, "before")] {
            let mut k = d.candidates[i].knobs.clone();
            assert_eq!(k.remove("wait").as_deref(), Some(wait));
            assert_eq!(k, d.incumbent.knobs);
        }
        // 3 header lines, the knob-space summary, and the three ring=3 points pruned with the reason.
        assert_eq!(d.derivation.len(), 7);
        assert!(
            d.derivation[3].contains("6 points") && d.derivation[3].contains("2 further legal")
        );
        for (line, wait) in d.derivation[4..].iter().zip(["after", "mid", "before"]) {
            assert!(
                line.starts_with(&format!(
                    "pruned ring=3 wait={wait}: ring depth 3 has no generator"
                )),
                "{line}"
            );
        }
    }

    #[test]
    fn iu4_numeric_candidate_is_opt_in() {
        let case = iu4_case();
        let off = enumerate_with_rips(&case, 0, 8192, &[]).unwrap();
        assert!(off
            .candidates
            .iter()
            .all(|c| !c.knobs.contains_key("fold-op")));
        let on = enumerate_inner(&case, 0, 8192, &[], true).unwrap();
        assert_eq!(on.candidates.len(), off.candidates.len() + 1);
        assert_eq!(on.candidates[..off.candidates.len()], off.candidates[..]);
        let last = on.candidates.last().unwrap();
        assert_eq!(last.knobs["fold-op"], "muladd");
        let mut k = last.knobs.clone();
        k.remove("fold-op");
        assert_eq!(k, on.incumbent.knobs);
        let args = last.pm_loop_args();
        let i = args.iter().position(|a| a == "--fold-op").unwrap();
        assert_eq!(args[i + 1], "muladd");
        assert!(on
            .derivation
            .iter()
            .any(|l| l.starts_with("numeric candidate fold-op=muladd")));
        assert_ne!(on.domain_hash, off.domain_hash);
    }

    #[test]
    fn iu4_generator_points_become_pm_loop_flags_in_ring_major_order() {
        let d = enumerate_with_rips(&iu4_case(), 0, 8192, &[]).unwrap();
        let flags = |c: &CandidateSpecV1| -> Vec<(String, String)> {
            let a = c.pm_loop_args();
            ["--ring", "--wait"]
                .iter()
                .filter_map(|f| {
                    a.iter()
                        .position(|x| x == f)
                        .map(|i| ((*f).to_owned(), a[i + 1].clone()))
                })
                .collect()
        };
        assert!(flags(&d.candidates[0]).is_empty());
        assert_eq!(
            flags(&d.candidates[2]),
            [("--wait".to_owned(), "mid".to_owned())]
        );
        assert_eq!(
            flags(&d.candidates[3]),
            [("--wait".to_owned(), "before".to_owned())]
        );
        let ids: std::collections::BTreeSet<_> = d.candidates.iter().map(|c| c.id()).collect();
        assert_eq!(ids.len(), d.candidates.len(), "candidate ids are distinct");
        // Deterministic.
        assert_eq!(d, enumerate_with_rips(&iu4_case(), 0, 8192, &[]).unwrap());
    }

    #[test]
    fn iu4_domain_lists_rip_candidates_in_name_order() {
        let dir = tmp_dir("rips");
        let rips = write_rips(
            &dir,
            &[
                ("b.rip", "fn emit() {}"),
                ("a.rip", "fn emit() { }"),
                ("notes.txt", "not a script"),
            ],
        );
        assert_eq!(rips.len(), 2);
        assert!(rips[0].ends_with("a.rip") && rips[1].ends_with("b.rip"));
        assert!(rips[0].is_absolute());
        let d = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(d.candidates.len(), 6);
        let rip_of = |i: usize| d.candidates[i].knobs["rip"].clone();
        assert_eq!(rip_of(4), rips[0].to_str().unwrap());
        assert_eq!(rip_of(5), rips[1].to_str().unwrap());
        // The rip candidate keeps every incumbent knob and adds only `rip`.
        let mut k = d.candidates[4].knobs.clone();
        k.remove("rip");
        assert_eq!(k, d.incumbent.knobs);
        // `pm_loop` gets `--rip PATH` and the five knob flags.
        let args = d.candidates[4].pm_loop_args();
        let at = |flag: &str| {
            let i = args.iter().position(|a| a == flag).unwrap();
            args[i + 1].clone()
        };
        assert_eq!(at("--kernel"), "iu4_gemm");
        assert_eq!(at("--shape"), "M=17408,N=8192,K=5120");
        assert_eq!(at("--rip"), rips[0].to_str().unwrap());
        assert_eq!(at("--fold"), "k128");
        assert_eq!(at("--tile"), "128x128x8");
        assert_eq!(at("--cacc"), "1");
        assert_eq!(at("--epi"), "silu");
        assert_eq!(at("--alayout"), "slab");
        assert_eq!(
            d.derivation
                .iter()
                .filter(|l| l.starts_with("rip candidate "))
                .count(),
            2
        );
        // Distinct ids: the resume key tells the scripts apart.
        assert_ne!(d.candidates[4].id(), d.candidates[5].id());
        assert_ne!(d.candidates[0].id(), d.candidates[4].id());
    }

    #[test]
    fn iu4_domain_hash_tracks_script_content_and_shape() {
        let dir = tmp_dir("hash");
        let rips = write_rips(&dir, &[("a.rip", "one")]);
        let a = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        let again = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(a, again);
        // Same path, edited content: the candidate spec is unchanged, the domain hash is not.
        write_rips(&dir, &[("a.rip", "two")]);
        let edited = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(a.candidates, edited.candidates);
        assert_ne!(a.domain_hash, edited.domain_hash);
        // Another token count is another shape.
        let case_half = test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 32, 1]));
        let half = enumerate_with_rips(&case_half, 0, 4096, &rips).unwrap();
        assert_eq!(half.shape, "M=17408,N=4096,K=5120");
        assert_ne!(edited.domain_hash, half.domain_hash);
        // A partial last token tile still rounds up (N = 8191 -> 64 tiles).
        assert!(enumerate_with_rips(&iu4_case(), 0, 8191, &[]).is_ok());
    }

    #[test]
    fn iu4_refusals() {
        let g = |x: u32, y: u32| {
            test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([x, y, 1]))
        };
        let e = enumerate_with_rips(&g(271, 64), 0, 8192, &[]).unwrap_err();
        assert!(
            e.contains("grid.x") && e.contains("271") && e.contains("272"),
            "{e}"
        );
        let e = enumerate_with_rips(&g(272, 32), 0, 8192, &[]).unwrap_err();
        assert!(
            e.contains("grid.y") && e.contains("32") && e.contains("64"),
            "{e}"
        );
        let none = test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], None);
        assert!(enumerate_with_rips(&none, 0, 8192, &[])
            .unwrap_err()
            .contains("no launch grid"));
        let fam = test_case_grid("a3b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 64, 1]));
        assert!(enumerate_with_rips(&fam, 0, 8192, &[])
            .unwrap_err()
            .contains("preset"));
        let m = test_case_grid(
            "dense-27b",
            IU4_GATE_UP_SILU_B1S,
            &[("M", 4096)],
            Some([272, 64, 1]),
        );
        let e = enumerate_with_rips(&m, 0, 8192, &[]).unwrap_err();
        assert!(e.contains("4096") && e.contains("17408"), "{e}");
        assert!(enumerate_with_rips(&iu4_case(), 0, 0, &[])
            .unwrap_err()
            .contains("tokens"));
        let mut gfx = iu4_case();
        gfx.arch = "gfx1100".into();
        assert!(enumerate_with_rips(&gfx, 0, 8192, &[])
            .unwrap_err()
            .contains("gfx1100"));
        assert!(enumerate_with_rips(&iu4_case(), 1, 8192, &[])
            .unwrap_err()
            .contains("out of range"));
        let missing =
            enumerate_with_rips(&iu4_case(), 0, 8192, &[PathBuf::from("/nonexistent/x.rip")]);
        assert!(missing.unwrap_err().contains("/nonexistent/x.rip"));
    }

    #[test]
    fn rips_are_refused_for_moe_nodes() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let e = enumerate_with_rips(&case, 0, 8192, &[PathBuf::from("/x/a.rip")]).unwrap_err();
        assert!(e.contains(".rip") && e.contains("iu4_gemm"), "{e}");
        // Without scripts the MoE path is unchanged.
        assert!(enumerate_with_rips(&case, 0, 8192, &[]).is_ok());
    }

    #[test]
    fn rips_in_dir_errors_on_missing_dir() {
        assert!(rips_in_dir(Path::new("/nonexistent/rip-dir")).is_err());
    }

    fn mq6_case(symbol: &str, grid: [u32; 3]) -> WorkloadCaseV1 {
        test_case_grid("flash-next", symbol, &[], Some(grid))
    }

    fn mq6_ids(d: &DomainV1) -> Vec<(String, String)> {
        d.candidates
            .iter()
            .map(|c| (c.knobs["tile"].clone(), c.knobs["raster"].clone()))
            .collect()
    }

    #[test]
    fn mq6_runtime_selector_matches_the_runtime_tests() {
        // `gemm_mq6_i8.rs` tests: (m, n, 64 CUs) -> token tile.
        for (m, n, want) in [
            (10240, 4096, 256),
            (10240, 16384, 256),
            (6144, 4096, 256),
            (2560, 4096, 256),
            (512, 4096, 256),
            (128, 4096, 128),
            (48, 4096, 128),
            (384, 4096, 128),
            (10240, 4097, 128),
            (10240, 4096 + 129, 256),
        ] {
            assert_eq!(mq6_runtime_token_tile(m, n, 64), want, "m={m} n={n}");
        }
    }

    #[test]
    fn mq6_node_is_classified_and_enumerated_control_first() {
        assert_eq!(classify(MQ6_SYM_128), Some(Node::Mq6(Mq6Entry::Narrow)));
        assert_eq!(classify(MQ6_SYM_WIDE), Some(Node::Mq6(Mq6Entry::Wide)));
        assert_eq!(classify("gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201"), None);
        // FN pp16384, node 14 of the corpus: grid [80, 32] = M 10240, K 2560, N 4096 (the
        // highest-time of the six shapes, 220.4 ms of 632.6 ms).
        let case = mq6_case(MQ6_SYM_128, [80, 32, 1]);
        let d = enumerate(&case, 0, 4096).unwrap();
        assert_eq!(d.kernel, MQ6_KERNEL);
        assert_eq!(d.shape, "M=10240,N=4096,K=2560");
        // The runtime takes 128x256 here: the incumbent is that tile, not pm_loop's 128x128.
        assert_eq!(d.incumbent.knobs["tile"], "128x256");
        assert_eq!(d.incumbent.knobs["base-shipped"], "128x256");
        assert_eq!(d.incumbent.knobs["raster"], "banded");
        assert_eq!(d.incumbent.knobs["epi"], "set");
        assert_eq!(d.candidates[0], d.incumbent);
        assert_eq!(
            mq6_ids(&d),
            [
                ("128x256", "banded"),
                ("128x128", "banded"),
                ("128x128", "linear"),
                ("256x128", "banded"),
                ("256x128", "linear"),
                ("128x256", "linear"),
            ]
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
        );
        assert!(d
            .candidates
            .iter()
            .all(|c| c.knobs["base-shipped"] == "128x256"));
        assert_eq!(baseline_symbol(&d), MQ6_SYM_WIDE);
        let text = d.derivation.join("\n");
        assert!(text.contains("gemm_mq6_i8.rs:101-116"), "{text}");
        assert!(
            text.contains("NOTE corpus symbol"),
            "stale-profile note: {text}"
        );
        let again = enumerate(&case, 0, 4096).unwrap();
        assert_eq!(again, d);
        assert_eq!(
            d.incumbent.pm_loop_args()[..6],
            [
                "--kernel",
                "mq6_i8_gemm",
                "--arch",
                "gfx1201",
                "--shape",
                "M=10240,N=4096,K=2560"
            ]
        );
    }

    #[test]
    fn mq6_narrow_runtime_tile_makes_the_128_entry_the_incumbent() {
        // M=128 at N=4096: one row tile, the selector keeps 128 x 128 (ratio 1.336 measured).
        let case = mq6_case(MQ6_SYM_128, [1, 32, 1]);
        assert!(
            enumerate(&case, 0, 4096).is_err(),
            "grid.x = 1 is ambiguous"
        );
        // M=512 (grid.x 4) at N=4097 is a ragged token tile: narrow.
        let case = mq6_case(MQ6_SYM_128, [4, 33, 1]);
        let d = enumerate(&case, 0, 4097).unwrap();
        assert_eq!(d.shape, "M=512,N=4097,K=2560");
        assert_eq!(d.incumbent.knobs["tile"], "128x128");
        assert_eq!(baseline_symbol(&d), MQ6_SYM_128);
        assert_eq!(d.candidates[0], d.incumbent);
        assert_eq!(d.candidates.len(), 6);
        assert!(!d.derivation.join("\n").contains("NOTE corpus symbol"));
    }

    #[test]
    fn mq6_wide_corpus_symbol_and_refusals() {
        let case = mq6_case(MQ6_SYM_WIDE, [80, 16, 1]);
        let d = enumerate(&case, 0, 4096).unwrap();
        assert_eq!(d.incumbent.knobs["tile"], "128x256");
        assert!(!d.derivation.join("\n").contains("NOTE corpus symbol"));
        // grid.y of the wide entry is ceil(N/256).
        assert!(enumerate(&mq6_case(MQ6_SYM_WIDE, [80, 32, 1]), 0, 4096).is_err());
        assert!(enumerate(&mq6_case(MQ6_SYM_128, [80, 31, 1]), 0, 4096).is_err());
        assert!(enumerate(&mq6_case(MQ6_SYM_128, [7, 32, 1]), 0, 4096).is_err());
        assert!(enumerate(&mq6_case(MQ6_SYM_128, [80, 32, 1]), 0, 0).is_err());
        let mut a3b = mq6_case(MQ6_SYM_128, [80, 32, 1]);
        a3b.model.family = "a3b".to_owned();
        assert!(enumerate(&a3b, 0, 4096).is_err());
        let rip = vec![PathBuf::from("/x.rip")];
        assert!(enumerate_with_rips(&mq6_case(MQ6_SYM_128, [80, 32, 1]), 0, 4096, &rip).is_err());
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
