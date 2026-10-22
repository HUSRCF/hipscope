// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! PM loop: one command that emits a PeaceMaker production-generator variant,
//! certifies it, checks it on the card, and ABBA-times it against the shipped
//! kernel (gfx1201). Prints ONE JSON object (schema 1) on stdout.
//!
//! ```text
//! pm_loop --kernel iu4_gemm|qwen4_moe_sym --shape <S> [kernel knobs]
//!         [--base-<knob> V ...] [--time N] [--rounds R] [--emu] [--out DIR]
//! ```
//!
//! Pipeline (every step lands in the JSON record):
//! 1. Emit + link the candidate with the native writer (the path of
//!    `peacemaker-isa emit --co`, no ROCm tools) and resolve the baseline: the
//!    SHIPPED production object (committed `.hxaco` that `rdna-compute` embeds;
//!    `baseline.source` names the selecting file:line and the artifact sha256),
//!    or, when any `--base-<knob>` is given, a second emitted variant whose
//!    unspecified knobs inherit the candidate's (time two arbitrary variants).
//! 2. Certify: M7 (`pm_check`, behind `peacemaker-isa emit --m7`) on EVERY
//!    export of the candidate object through the cert cache (`m7.cache` is
//!    `hit`, `miss` or `disabled`; `HIPFIRE_CERT_CACHE=0` disables it); the
//!    lifted kernel descriptor/metadata give VGPRs, SGPRs, LDS, scratch and code
//!    bytes per export, plus `cost_lint` loop statistics.
//! 3. Correctness on the card: deterministic synthetic inputs at `--shape`,
//!    candidate and baseline each launched once with the production kernarg
//!    packing; `correctness` reports candidate vs the CPU reference
//!    (`max_abs_err`, `tolerance`, `pass`), the baseline vs the same
//!    reference, and candidate vs baseline (`identical_to_baseline`,
//!    `max_abs_diff_vs_baseline`). An epilogue with no CPU reference (iu4_gemm
//!    silu) reports `reference: "none"` and `pass` is then
//!    `identical_to_baseline` (`pass_basis` says which). `--emu` runs the candidate on the peacemaker emulator
//!    when every instruction is qualified, else `refused (<reason>)`.
//! 4. Timing, exactly `pm_decode_twins time`: A = baseline, B = candidate; one
//!    checked warm launch per arm, then `--rounds R` (default 5) ABBA rounds of
//!    `--time N` (default 30) launches per block, per-launch hipEvents on the
//!    default stream, accumulating output state restored outside the event
//!    interval. `separable` is true only if the median separation exceeds the
//!    larger sample spread AND the sample ranges are disjoint (the twins rule);
//!    otherwise read the ratio as "not separable at this N". Sampling is
//!    refused when the candidate fails correctness.
//!
//! Exit status: 0 ok; 1 correctness failure (record still printed); 3 the
//! candidate could not be emitted/certified/launched (record printed with the
//! error); 2 usage error.
//!
//! Kernel knobs and shape grammar: see the doc comment of `pm_loop/iu4.rs`
//! (`--kernel iu4_gemm`) and `pm_loop/moe.rs` (`--kernel qwen4_moe_sym`).
//!
//! Use (build once, then take a gfx1201 card through the fabric; check
//! `fabric-run.sh --status` first and pick an idle hiptrx card A/B/C/E):
//! ```text
//! cargo build --release -p rdna-compute --features lab --example pm_loop
//! /home/kaden/qcal/locks/fabric/fabric-run.sh --class timing --card B --mem-gb 4 \
//!   --max-secs 900 --owner PmLoop -- target/release/examples/pm_loop \
//!   --kernel iu4_gemm --fold k128 --tile 256x128x16 --epi set \
//!   --shape M=17408,N=512,K=5120 --time 30 --rounds 5
//! ```
//! The fabric exports `ROCR_VISIBLE_DEVICES`/`FABRIC_CARDS`, recorded under
//! `device`. GPU-free tests: `cargo test -p rdna-compute --features lab --example pm_loop`.

#[path = "support/abba.rs"]
mod abba;
#[path = "pm_loop/iu4.rs"]
mod iu4;
#[path = "pm_loop/moe.rs"]
mod moe;
#[path = "pm_loop/workload.rs"]
mod workload;

use abba::{timing_relation, timing_schedule, timing_stats, TimingArm, TimingStats};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use workload::{Arm, Knobs, Object, Result, Workload};

const SCHEMA: u32 = 1;
const ARCH: &str = "gfx1201";
const DEFAULT_TIME: usize = 30;
const DEFAULT_ROUNDS: usize = 5;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// CLI

#[derive(Debug)]
struct Opts {
    kernel: String,
    shape: Option<String>,
    time: usize,
    rounds: usize,
    emu: bool,
    out: Option<PathBuf>,
    /// `--rip FILE`: candidate compiled from a `.rip` source (iu4_gemm only).
    rip: Option<PathBuf>,
    knobs: Knobs,
}

const DRIVER_FLAGS: &str =
    "--kernel K --shape S [--time N] [--rounds R] [--emu] [--out DIR] [--rip FILE.rip] [--<knob> V]... [--base-<knob> V]...";

fn parse_args(mut a: impl Iterator<Item = String>) -> Result<Opts> {
    let mut o = Opts {
        kernel: String::new(),
        shape: None,
        time: DEFAULT_TIME,
        rounds: DEFAULT_ROUNDS,
        emu: false,
        out: None,
        rip: None,
        knobs: Knobs::default(),
    };
    while let Some(arg) = a.next() {
        let key = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {arg} (usage: {DRIVER_FLAGS})"))?
            .to_owned();
        if key == "emu" {
            o.emu = true;
            continue;
        }
        let value = a
            .next()
            .ok_or_else(|| format!("--{key} needs a value (usage: {DRIVER_FLAGS})"))?;
        match key.as_str() {
            "kernel" => o.kernel = value,
            "shape" => o.shape = Some(value),
            "time" => o.time = abba::positive_count(&value).map_err(|e| format!("--time: {e}"))?,
            "rounds" => {
                o.rounds = abba::positive_count(&value).map_err(|e| format!("--rounds: {e}"))?
            }
            "out" => o.out = Some(PathBuf::from(value)),
            "rip" => o.rip = Some(PathBuf::from(value)),
            _ => {
                if o.knobs.0.insert(key.clone(), value).is_some() {
                    return Err(format!("--{key} given twice"));
                }
            }
        }
    }
    if o.kernel.is_empty() {
        return Err(format!("missing --kernel (usage: {DRIVER_FLAGS})"));
    }
    if o.rip.is_some() && o.kernel != "iu4_gemm" {
        return Err(format!(
            "--rip is only supported with --kernel iu4_gemm: only iu4_gemm blocks are exposed to .rip so far (got --kernel {})",
            o.kernel
        ));
    }
    abba::timing_counts(o.time, o.rounds)?;
    Ok(o)
}

fn build_workload(o: &mut Opts) -> Result<Box<dyn Workload>> {
    let shape = o.shape.clone().unwrap_or_default();
    let mut knobs = std::mem::take(&mut o.knobs);
    let w = match o.kernel.as_str() {
        "iu4_gemm" => iu4::new_with_rip(&shape, &mut knobs, o.rip.as_deref())?,
        "qwen4_moe_sym" => moe::new(&shape, &mut knobs)?,
        other => return Err(format!("unknown --kernel {other} (iu4_gemm|qwen4_moe_sym)")),
    };
    knobs.finish()?;
    Ok(w)
}

// ---------------------------------------------------------------------------
// Object facts: M7, resources, cost lint

fn write_object(dir: &Path, tag: &str, obj: &Object) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{tag}.{}.co", &sha256_hex(&obj.bytes)[..16]));
    std::fs::write(&path, &obj.bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// Descriptor/metadata resources of every kernel of a lifted object, keyed by
/// symbol. `code_bytes` is the re-encoded instruction stream length.
fn resources(program: &peacemaker_ir::Program, symbols: &[String]) -> Result<Vec<Value>> {
    use peacemaker_ir::inst::Abi;
    let mut rows = Vec::new();
    for symbol in symbols {
        let kernel = program
            .kernels
            .iter()
            .find(|k| &k.symbol.0 == symbol)
            .ok_or_else(|| format!("object has no kernel {symbol}"))?;
        let Abi::Hsa {
            descriptor,
            metadata,
        } = &kernel.abi
        else {
            return Err(format!("{symbol}: not an HSA kernel"));
        };
        let m = &metadata.parsed;
        let code = peacemaker_lift::emit::bytes(kernel, program.target.arch).map_err(err)?;
        rows.push(json!({
            "symbol": symbol,
            "vgprs": m.vgpr_count,
            "sgprs": m.sgpr_count,
            "lds_bytes": descriptor.group_segment_fixed_size,
            "scratch_bytes": descriptor.private_segment_fixed_size,
            "code_bytes": code.len(),
            "kernarg_bytes": descriptor.kernarg_size,
            "vgpr_spills": m.vgpr_spill_count,
            "sgpr_spills": m.sgpr_spill_count,
            "max_flat_workgroup_size": m.max_flat_workgroup_size,
        }));
    }
    Ok(rows)
}

fn lift(bytes: &[u8]) -> Result<peacemaker_ir::Program> {
    peacemaker_lift::lift_object(
        bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map(|l| l.program)
    .map_err(err)
}

/// `cost_lint` loop statistics per kernel, `{symbol: [loop..]}`; `{"error": ..}`
/// if the static analysis refuses the object.
fn cost_lint(program: &peacemaker_ir::Program) -> Value {
    match peacemaker_isa::cost_lint::analyze(program, &Default::default()) {
        Ok(report) => {
            let mut out = serde_json::Map::new();
            for k in report.kernels {
                out.insert(
                    k.symbol,
                    serde_json::to_value(k.loops).unwrap_or(Value::Null),
                );
            }
            Value::Object(out)
        }
        Err(e) => json!({"error": e}),
    }
}

/// M7 on every export through the cert cache. Err only when M7 itself rejects;
/// the cache status of each export is kept alongside its receipt.
fn certify(path: &Path, symbols: &[String]) -> Value {
    let mut cache = Vec::new();
    let mut receipts = serde_json::Map::new();
    let mut status = "pass";
    let mut error = Value::Null;
    for symbol in symbols {
        match peacemaker_isa::pm_check::m7_status(path, ARCH, symbol) {
            Ok((receipt, c)) => {
                cache.push(json!({"symbol": symbol, "cache": c.as_str()}));
                receipts.insert(symbol.clone(), receipt);
            }
            Err(e) => {
                status = "fail";
                error = json!({"symbol": symbol, "error": e});
                break;
            }
        }
    }
    let summary = if !cache.is_empty() && cache.iter().all(|c| c["cache"] == "hit") {
        "hit"
    } else if cache.iter().any(|c| c["cache"] == "miss") {
        "miss"
    } else if cache.iter().any(|c| c["cache"] == "disabled") {
        "disabled"
    } else {
        "hit"
    };
    json!({"status": status, "cache": summary, "per_export": cache,
        "receipts": receipts, "error": error})
}

// ---------------------------------------------------------------------------
// Device

struct Loaded {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
}

fn load(gpu: &Gpu, obj: &Object) -> Result<Loaded> {
    let module = gpu.hip.module_load_data(&obj.bytes).map_err(err)?;
    let func = gpu
        .hip
        .module_get_function(&module, &obj.launch_symbol)
        .map_err(err)?;
    Ok(Loaded {
        _module: module,
        func,
    })
}

fn launch(gpu: &Gpu, w: &dyn Workload, arm: Arm, f: &Loaded) -> Result<()> {
    let mut l = w.launch(arm);
    // SAFETY: the workload packs the kernarg per the kernel's declared ABI
    // and every pointer addresses a live allocation it owns.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&f.func, l.grid, l.block, l.lds, None, &mut l.kernarg)
    }
    .map_err(err)
}

fn device_json(gpu: &Gpu) -> Value {
    let env = |k: &str| std::env::var(k).ok();
    let cards: Vec<String> = env("FABRIC_CARDS")
        .map(|s| s.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    json!({
        "arch": gpu.arch, "device_id": gpu.device_id,
        "uuid": env("ROCR_VISIBLE_DEVICES").or_else(|| env("HIPFIRE_DEVICES")),
        "fabric_cards": cards,
        "fabric_job": env("FABRIC_JOB"),
    })
}

// ---------------------------------------------------------------------------
// Timing record

fn us(v: f64) -> f64 {
    v / 1e3
}

fn arm_json(s: &TimingStats) -> Value {
    json!({"median": us(s.median), "min": us(s.min), "max": us(s.max),
        "p90": us(s.p90), "mean": us(s.mean)})
}

/// Pure timing record from the two sample sets (nanoseconds), with the twins'
/// separability rule. `ratio` = candidate/baseline median.
fn timing_record(base_ns: &[f64], cand_ns: &[f64], n: usize, rounds: usize) -> Result<Value> {
    let a = timing_stats(base_ns)?;
    let b = timing_stats(cand_ns)?;
    let (ratio, spread, delta, separable) = timing_relation(&a, &b);
    Ok(json!({
        "n": n, "rounds": rounds, "calls_per_arm": a.count,
        "baseline_us": arm_json(&a), "candidate_us": arm_json(&b),
        "ratio": ratio, "separable": separable,
        "observed_spread_us": us(spread), "median_delta_us": us(delta),
        "unit": "us per launch (hipEvent, default stream)",
        "rule": "separable = |median delta| > larger full spread AND ranges disjoint",
    }))
}

/// An emulator refusal carries the faulting instruction's full register dump
/// after ` | `; keep the reason, drop the dump.
fn tidy_emu(v: Value) -> Value {
    match v {
        Value::String(s) if s.starts_with("refused (") && s.contains(" | ") => {
            let reason = s.split(" | ").next().unwrap_or(&s);
            Value::String(format!("{reason})"))
        }
        other => other,
    }
}

struct Sampled {
    base_ns: Vec<f64>,
    cand_ns: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn sample(
    gpu: &Gpu,
    w: &dyn Workload,
    base: &Loaded,
    cand: &Loaded,
    n: usize,
    rounds: usize,
) -> Result<Sampled> {
    let (calls, _) = abba::timing_counts(n, rounds)?;
    let mut s = Sampled {
        base_ns: Vec::with_capacity(calls),
        cand_ns: Vec::with_capacity(calls),
    };
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = match gpu.hip.event_create() {
        Ok(e) => e,
        Err(e) => {
            let _ = gpu.hip.event_destroy(start);
            return Err(err(e));
        }
    };
    let measured = (|| -> Result<()> {
        for which in timing_schedule(n, rounds)? {
            let (arm, f) = match which {
                TimingArm::Incumbent => (Arm::Baseline, base),
                TimingArm::Candidate => (Arm::Candidate, cand),
            };
            w.reset(gpu, arm)?;
            gpu.hip.event_record(&start, None).map_err(err)?;
            launch(gpu, w, arm, f)?;
            gpu.hip.event_record(&stop, None).map_err(err)?;
            gpu.hip.event_synchronize(&stop).map_err(err)?;
            let ns = f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6;
            match which {
                TimingArm::Incumbent => s.base_ns.push(ns),
                TimingArm::Candidate => s.cand_ns.push(ns),
            }
        }
        Ok(())
    })();
    let d1 = gpu.hip.event_destroy(start).map_err(err);
    let d2 = gpu.hip.event_destroy(stop).map_err(err);
    measured?;
    d1?;
    d2?;
    Ok(s)
}

// ---------------------------------------------------------------------------
// Driver

/// Exit classification of a run.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok,
    CorrectnessFailed,
    CandidateUnusable,
}

fn object_json(obj: &Object, resources: Value, extra: serde_json::Map<String, Value>) -> Value {
    let mut v = json!({
        "co_sha256": sha256_hex(&obj.bytes), "co_bytes": obj.bytes.len(),
        "launch_symbol": obj.launch_symbol, "symbols": obj.exports,
        "resources": resources,
    });
    let o = v.as_object_mut().unwrap();
    if let Some(d) = obj.desc.as_object() {
        o.extend(d.clone());
    }
    o.extend(extra);
    v
}

fn run(o: &mut Opts) -> (Value, Outcome) {
    let mut rec = json!({"schema": SCHEMA, "kernel": o.kernel, "arch": ARCH});
    let fail = |mut rec: Value, stage: &str, e: String| {
        rec["error"] = json!({"stage": stage, "message": e});
        (rec, Outcome::CandidateUnusable)
    };
    let mut w = match build_workload(o) {
        Ok(w) => w,
        Err(e) => return fail(rec, "args", e),
    };
    rec["shape"] = w.shape_json();
    rec["candidate"] = json!({"knobs": w.candidate_knobs()});
    let out_dir = o
        .out
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("pm_loop-{}", std::process::id())));

    // 1. emit + link
    let cand = match w.candidate() {
        Ok(c) => c,
        Err(e) => return fail(rec, if o.rip.is_some() { "rip" } else { "emit" }, e),
    };
    let base = match w.baseline() {
        Ok(b) => b,
        Err(e) => return fail(rec, "baseline", e),
    };

    // 2. certify + resources
    let cand_path = match write_object(&out_dir, "candidate", &cand) {
        Ok(p) => p,
        Err(e) => return fail(rec, "write", e),
    };
    let m7 = certify(&cand_path, &cand.exports);
    let (cand_res, cand_lint) = match lift(&cand.bytes).and_then(|p| {
        let r = resources(&p, &cand.exports)?;
        Ok((r, cost_lint(&p)))
    }) {
        Ok((r, l)) => (json!(r), l),
        Err(e) => (json!({"error": e}), Value::Null),
    };
    let mut cextra = serde_json::Map::new();
    cextra.insert("m7".into(), {
        let mut v = m7.clone();
        v.as_object_mut().map(|o| o.remove("receipts"));
        v
    });
    cextra.insert("m7_receipts".into(), m7["receipts"].clone());
    cextra.insert("cost_lint".into(), cand_lint);
    cextra.insert("object_path".into(), json!(cand_path.display().to_string()));
    cextra.insert("knobs".into(), w.candidate_knobs());
    rec["candidate"] = object_json(&cand, cand_res, cextra);
    if m7["status"] != "pass" {
        let msg = m7["error"].to_string();
        return fail(rec, "m7", msg);
    }
    let (base_res, base_lint) = match lift(&base.bytes).and_then(|p| {
        let r = resources(&p, std::slice::from_ref(&base.launch_symbol))?;
        Ok((r, cost_lint(&p)))
    }) {
        Ok((r, l)) => (json!(r), l),
        Err(e) => (json!({"error": e}), Value::Null),
    };
    let base_path = write_object(&out_dir, "baseline", &base);
    let base_m7 = match &base_path {
        Ok(p) => certify(p, std::slice::from_ref(&base.launch_symbol)),
        Err(e) => json!({"status": "skipped", "error": e}),
    };
    let mut bextra = serde_json::Map::new();
    bextra.insert("source".into(), base.source.clone());
    bextra.insert("m7".into(), {
        let mut v = base_m7;
        v.as_object_mut().map(|o| o.remove("receipts"));
        v
    });
    // The shipped bundle carries every epilogue symbol: keep the launched one.
    bextra.insert(
        "cost_lint".into(),
        match base_lint.get(&base.launch_symbol) {
            Some(loops) => json!({ base.launch_symbol.clone(): loops }),
            None => base_lint,
        },
    );
    rec["baseline"] = object_json(&base, base_res, bextra);

    // 3. correctness on the card
    let gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => return fail(rec, "gpu_init", err(e)),
    };
    rec["device"] = device_json(&gpu);
    if gpu.arch != ARCH {
        let msg = format!("device is {}, pm_loop requires {ARCH}", gpu.arch);
        return fail(rec, "device", msg);
    }
    let staged = (|| -> Result<(Loaded, Loaded)> {
        w.setup(&gpu)?;
        Ok((load(&gpu, &base)?, load(&gpu, &cand)?))
    })();
    let (bf, cf) = match staged {
        Ok(v) => v,
        Err(e) => return fail(rec, "setup", e),
    };
    let warm = (|| -> Result<(Vec<u8>, Vec<u8>)> {
        for arm in [Arm::Baseline, Arm::Candidate] {
            w.reset(&gpu, arm)?;
        }
        launch(&gpu, &*w, Arm::Baseline, &bf)?;
        launch(&gpu, &*w, Arm::Candidate, &cf)?;
        gpu.hip.device_synchronize().map_err(err)?;
        Ok((
            w.output(&gpu, Arm::Baseline)?,
            w.output(&gpu, Arm::Candidate)?,
        ))
    })();
    let (base_out, cand_out) = match warm {
        Ok(v) => v,
        Err(e) => return fail(rec, "launch", e),
    };
    let reference = w.check(&cand_out);
    let base_report = w.check(&base_out);
    let diff = match w.max_abs_diff(&base_out, &cand_out) {
        Ok(d) => d,
        Err(e) => return fail(rec, "compare", e),
    };
    let identical = base_out == cand_out;
    // Without a CPU reference (e.g. an epilogue whose device intrinsics are not
    // reproducible) bit identity with the shipped baseline is the only
    // admissible evidence: `pass` is then `identical_to_baseline`.
    let pass = match &reference {
        Ok(r) => r.pass,
        Err(_) => identical,
    };
    rec["correctness"] = json!({
        "reference": reference.as_ref().map(|r| r.reference.as_str()).unwrap_or("none"),
        "reference_error": reference.as_ref().err(),
        "max_abs_err": reference.as_ref().ok().map(|r| r.max_abs_err),
        "tolerance": reference.as_ref().ok().map(|r| r.tolerance),
        "pass": pass,
        "pass_basis": if reference.is_ok() { "cpu_reference" } else { "identical_to_baseline (no CPU reference)" },
        "baseline_max_abs_err": base_report.as_ref().ok().map(|r| r.max_abs_err),
        "baseline_pass": base_report.as_ref().ok().map(|r| r.pass),
        "baseline_reference_error": base_report.as_ref().err(),
        "identical_to_baseline": identical,
        "max_abs_diff_vs_baseline": diff,
        "output_bytes": cand_out.len(),
        "emu": if o.emu { tidy_emu(w.emu()) } else { json!("not requested (--emu)") },
    });
    if !pass {
        rec["timing"] = json!({"status": "refused: candidate failed correctness"});
        return (rec, Outcome::CorrectnessFailed);
    }

    // 4. ABBA timing
    let sampled = sample(&gpu, &*w, &bf, &cf, o.time, o.rounds);
    let post = (|| -> Result<bool> {
        // The last timed launch of each arm must still equal its warm output.
        gpu.hip.device_synchronize().map_err(err)?;
        Ok(w.output(&gpu, Arm::Baseline)? == base_out
            && w.output(&gpu, Arm::Candidate)? == cand_out)
    })();
    match (sampled, post) {
        (Ok(s), Ok(true)) => match timing_record(&s.base_ns, &s.cand_ns, o.time, o.rounds) {
            Ok(mut t) => {
                t["post_outputs_unchanged"] = json!(true);
                rec["timing"] = t;
            }
            Err(e) => return fail(rec, "timing", e),
        },
        (Ok(_), Ok(false)) => {
            rec["timing"] =
                json!({"status": "post-sampling outputs changed; no performance claim"});
            return (rec, Outcome::CorrectnessFailed);
        }
        (Err(e), _) | (_, Err(e)) => return fail(rec, "timing", e),
    }
    rec["device_after"] = device_json(&gpu);
    (rec, Outcome::Ok)
}

fn main() {
    let mut opts = match parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("pm_loop: {e}");
            std::process::exit(2);
        }
    };
    let (rec, outcome) = run(&mut opts);
    println!(
        "{}",
        serde_json::to_string(&rec).expect("record serializes")
    );
    if let Some(ts) = rec.get("timing").filter(|t| t["ratio"].is_number()) {
        let verdict = if ts["separable"] == true {
            format!(
                "candidate/baseline median ratio {:.6}",
                ts["ratio"].as_f64().unwrap_or(f64::NAN)
            )
        } else {
            "not separable at this N".to_owned()
        };
        eprintln!("pm_loop: {verdict}");
    }
    std::process::exit(match outcome {
        Outcome::Ok => 0,
        Outcome::CorrectnessFailed => 1,
        Outcome::CandidateUnusable => 3,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> impl Iterator<Item = String> + '_ {
        s.split_whitespace().map(str::to_owned)
    }

    #[test]
    fn defaults_and_knob_passthrough() {
        let o = parse_args(args(
            "--kernel iu4_gemm --shape M=128,N=128,K=256 --fold k128 --base-tile 128x128x8",
        ))
        .unwrap();
        assert_eq!(o.kernel, "iu4_gemm");
        assert_eq!(o.shape.as_deref(), Some("M=128,N=128,K=256"));
        assert_eq!(
            (o.time, o.rounds, o.emu),
            (DEFAULT_TIME, DEFAULT_ROUNDS, false)
        );
        assert_eq!(o.knobs.0.get("fold").map(String::as_str), Some("k128"));
        assert_eq!(
            o.knobs.0.get("base-tile").map(String::as_str),
            Some("128x128x8")
        );
        assert_eq!(o.knobs.0.len(), 2, "driver flags are not knobs");
    }

    #[test]
    fn rip_flag_parses_and_is_iu4_only() {
        let o = parse_args(args("--kernel iu4_gemm --rip /tmp/a.rip --fold k128")).unwrap();
        assert_eq!(o.rip, Some(PathBuf::from("/tmp/a.rip")));
        assert_eq!(o.knobs.0.len(), 1, "--rip is not a knob");
        assert!(parse_args(args("--kernel iu4_gemm")).unwrap().rip.is_none());
        let e = parse_args(args("--kernel qwen4_moe_sym --rip /tmp/a.rip")).unwrap_err();
        assert!(e.contains("only iu4_gemm"), "{e}");
        assert!(parse_args(args("--kernel iu4_gemm --rip")).is_err());
    }

    #[test]
    fn driver_flags_and_emu() {
        let o = parse_args(args(
            "--kernel qwen4_moe_sym --time 7 --rounds 2 --emu --out /tmp/x --epi down_nt4",
        ))
        .unwrap();
        assert_eq!((o.time, o.rounds, o.emu), (7, 2, true));
        assert_eq!(o.out, Some(PathBuf::from("/tmp/x")));
        assert!(o.shape.is_none());
        assert_eq!(o.knobs.0.len(), 1);
    }

    #[test]
    fn bad_args_are_refused() {
        for bad in [
            "",
            "--shape M=1",
            "--kernel",
            "--kernel k --time 0",
            "--kernel k --rounds 0",
            "--kernel k --time x",
            "--kernel k positional",
            "--kernel k --fold a --fold b",
            "--kernel k --time 18446744073709551615",
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn unknown_kernel_and_leftover_knobs_are_refused() {
        let mut o = parse_args(args("--kernel nope")).unwrap();
        assert!(build_workload(&mut o)
            .err()
            .is_some_and(|e| e.contains("unknown --kernel")));
        let mut k = Knobs::default();
        k.0.insert("bogus".into(), "1".into());
        assert!(k.finish().unwrap_err().contains("--bogus"));
    }

    #[test]
    fn abba_schedule_is_balanced_and_blocked() {
        let seq: Vec<_> = timing_schedule(3, 5).unwrap().collect();
        assert_eq!(seq.len(), 3 * 5 * 4);
        assert_eq!(
            seq.iter().filter(|a| **a == TimingArm::Incumbent).count(),
            30
        );
        // A A A | B B B | B B B | A A A, repeated per round.
        assert_eq!(
            &seq[..12],
            &[
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
            ]
        );
    }

    #[test]
    fn timing_record_units_ratio_and_separability() {
        let base = [100_000.0, 101_000.0, 102_000.0, 100_500.0];
        let slow = [200_000.0, 201_000.0, 202_000.0, 200_500.0];
        let t = timing_record(&base, &slow, 2, 1).unwrap();
        assert_eq!(t["separable"], true);
        assert_eq!(t["baseline_us"]["median"], 100.75);
        assert_eq!(t["candidate_us"]["min"], 200.0);
        let ratio = t["ratio"].as_f64().unwrap();
        assert!((ratio - 200.75 / 100.75).abs() < 1e-12);
        let same = timing_record(&base, &base, 2, 1).unwrap();
        assert_eq!(same["separable"], false);
        assert_eq!(same["ratio"], 1.0);
        assert!(timing_record(&[], &base, 1, 1).is_err());
    }

    #[test]
    fn emu_refusals_drop_register_dumps() {
        let long = json!(
            "refused (wg (0,0): k inst=3 v_div_scale_f32: not measured | arch=Gfx1201 w0=[0x1])"
        );
        assert_eq!(
            tidy_emu(long),
            json!("refused (wg (0,0): k inst=3 v_div_scale_f32: not measured)")
        );
        let short = json!("refused (no semantics for s_setprio)");
        assert_eq!(tidy_emu(short.clone()), short);
        let ran = json!({"status": "pass"});
        assert_eq!(tidy_emu(ran.clone()), ran);
    }

    #[test]
    fn m7_cache_summary_vocabulary() {
        use peacemaker_isa::pm_check::M7Cache;
        let all = [M7Cache::Hit, M7Cache::Miss, M7Cache::Disabled].map(M7Cache::as_str);
        assert_eq!(all, ["hit", "miss", "disabled"]);
    }
}
