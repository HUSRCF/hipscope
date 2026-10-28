// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! hip_loop: the PM-KRSI judge for hipcc incumbents on gfx1100 (7900 XTX). A
//! candidate is a complete HIP source file that defines the incumbent's kernel
//! symbol with the identical signature and launch contract. Prints ONE JSON
//! object (schema 1) on stdout.
//!
//! ```text
//! hip_loop --node mq6_res|hc_mix|qsa_hg4 [--shape k=v,...] --cand FILE.hip
//!          [--offline] [--null-control aa] [--time N] [--rounds R] [--seed S] [--out DIR]
//! ```
//!
//! * Nodes: `mq6_res` (`gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds`; shapes
//!   `qkv|z|out|qg|ba`), `hc_mix` (`gemm_wmma_lds_64_64_32_64_k64_p`; shapes
//!   `mix|sel`), `qsa_hg4` (`indexed_attention_attention_q8_batched_hg4`). See
//!   `hip_loop/{mq6,hc,qsa}.rs` for the shape grammar and the production launch each
//!   one mirrors.
//! * Candidate header: lines `// hip_loop: key=value ...` in the candidate source
//!   choose its launch geometry (keys per node, e.g. `tile_m tile_n block
//!   lds_dynamic`; no header = the production launch exactly). Both arms'
//!   geometries are recorded; a geometry that does not cover the output is refused.
//!   Per-kernel compiler flags use the runtime's `// HIPFIRE_COMPILER_FLAGS: ...`.
//! * `--cand-object FILE.co [--cand-geometry "tile_m=.. tile_n=.. block=.. lds_dynamic=.."]`: a
//!   prebuilt candidate code object (ELF or offload bundle; PM twins, lifted objects) instead
//!   of a source. It must export the incumbent's symbol with the same explicit kernarg layout
//!   (AMDGPU metadata `.args`: offset, size, pointer-ness; refused with stage `abi` otherwise);
//!   `.text` sha256 and resources are reported as for a compiled candidate, static LDS comes
//!   from the object. The geometry string uses the `// hip_loop:` keys (default: production).
//! * `--base-object FILE.co`: use FILE as the incumbent instead of compiling the shipped source.
//!   `FILE.co.json` (written next to `DIR/baseline.co` by every run that compiles the incumbent)
//!   must certify the object: module, the sha256 of the shipped source it came from, the object
//!   sha256 and the launched kernel's `.text` sha256 and explicit kernarg layout. `--offline`
//!   also recompiles the production source and requires the same `.text` sha256; on the device the
//!   sidecar is trusted and nothing is compiled for the incumbent (`--null-control aa` then loads
//!   the object twice as separate modules, `--cand-object` makes the whole run compile-free).
//!   Every run that compiles writes `DIR/baseline.co` (+`.json`) and, for a `--cand` source,
//!   `DIR/candidate.co` (+`.json`) with `candidate_geometry_arg` (the header as a `--cand-geometry`
//!   string) in the record. hip_loop's own compiles never read the JIT hot cache; offline runs keep
//!   the runtime compiler's cache dir inside `--out` unless `HIPFIRE_KERNEL_CACHE`/`HIPFIRE_HOME`
//!   are set (then those are honoured), GPU runs leave the device runtime's cache alone.
//! * `--offline`: compile the incumbent (from the shipped source) and the
//!   candidate through rdna-compute's `KernelCompiler` for gfx1100 (the runtime
//!   JIT's own argv, module flags and scheduler profile), no device. The record has
//!   `baseline`/`candidate` = `{code_sha256 (the kernel symbol's .text), vgprs,
//!   sgprs, lds_bytes, scratch_bytes, spill, code_bytes, occupancy_waves_per_simd,
//!   geometry, ...}`.
//! * GPU mode (a gfx1100 device; anything else is refused before any timing):
//!   seeded valid inputs, an identity run (outputs reset from a pristine copy
//!   before every dispatch; bit-identical bytes required, `identical_to_baseline`),
//!   then `--rounds R` ABBA rounds of `--time N` launches per block, hipEvent
//!   timed per launch with outputs reset outside the interval. Per round ratio
//!   candidate/baseline, `ratio_median`, one-sided exact sign tests `p_faster`
//!   (P[>= faster rounds | fair coin], ties and the other sign count against) and
//!   `p_slower`. `--null-control aa`: the incumbent source compiled and loaded as
//!   a SEPARATE module is the candidate (no `--cand`). With `R` rounds the best
//!   possible p is `2^-R`: use R >= 11 for p < 1e-3.
//!
//! Exit status: 0 ok; 1 identity failure (record still printed); 2 usage; 3 the
//! candidate (or the device) could not compile/launch (record printed with `error`).
//!
//! ```text
//! cargo build --release -p rdna-compute --example hip_loop
//! hip_loop --node mq6_res --shape qkv --cand copy_of_shipped.hip --offline
//! ```
//! GPU-free tests: `cargo test --release -p rdna-compute --example hip_loop`.

#[path = "support/abba.rs"]
mod abba;
#[path = "hip_loop/elf.rs"]
mod elf;
#[path = "hip_loop/hc.rs"]
mod hc;
#[path = "hip_loop/jit.rs"]
mod jit;
#[path = "hip_loop/mq6.rs"]
mod mq6;
#[path = "hip_loop/node.rs"]
mod node;
#[path = "hip_loop/qsa.rs"]
mod qsa;

use node::{err, Geometry, Header, Incumbent, Node, Result, ShapeSpec, ARCH};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const SCHEMA: u32 = 1;
const DEFAULT_TIME: usize = 50;
const DEFAULT_ROUNDS: usize = 12;
const NODES: [&str; 3] = ["mq6_res", "hc_mix", "qsa_hg4"];
const USAGE: &str = "--node mq6_res|hc_mix|qsa_hg4 [--shape k=v,...] \
(--cand FILE.hip | --cand-object FILE.co [--cand-geometry \"tile_m=.. tile_n=.. block=.. lds_dynamic=..\"] | --null-control aa) \
[--base-object FILE.co] [--offline] [--time N] [--rounds R] [--seed S] [--out DIR]";

// ---------------------------------------------------------------------------
// CLI

#[derive(Debug, Clone)]
struct Opts {
    node: String,
    shape: Option<String>,
    cand: Option<PathBuf>,
    offline: bool,
    null_control: bool,
    cand_object: Option<PathBuf>,
    base_object: Option<PathBuf>,
    cand_geometry: Option<String>,
    time: usize,
    rounds: usize,
    seed: u64,
    out: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Opts> {
    let mut o = Opts {
        node: String::new(),
        shape: None,
        cand: None,
        offline: false,
        null_control: false,
        cand_object: None,
        cand_geometry: None,
        base_object: None,
        time: DEFAULT_TIME,
        rounds: DEFAULT_ROUNDS,
        seed: 1,
        out: None,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| -> Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("--{name} needs a value"))
        };
        match a.as_str() {
            "--node" => o.node = value("node")?,
            "--shape" => o.shape = Some(value("shape")?),
            "--cand" => o.cand = Some(PathBuf::from(value("cand")?)),
            "--cand-object" => o.cand_object = Some(PathBuf::from(value("cand-object")?)),
            "--base-object" => o.base_object = Some(PathBuf::from(value("base-object")?)),
            "--cand-geometry" => o.cand_geometry = Some(value("cand-geometry")?),
            "--offline" => o.offline = true,
            "--null-control" => {
                let v = value("null-control")?;
                if v != "aa" {
                    return Err(format!("--null-control {v}: the only null control is `aa`"));
                }
                o.null_control = true;
            }
            "--time" => o.time = abba::positive_count(&value("time")?)?,
            "--rounds" => o.rounds = abba::positive_count(&value("rounds")?)?,
            "--seed" => o.seed = value("seed")?.parse::<u64>().map_err(err)?,
            "--out" => o.out = Some(PathBuf::from(value("out")?)),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if !NODES.contains(&o.node.as_str()) {
        return Err(format!("--node must be one of {}", NODES.join(", ")));
    }
    let sources = usize::from(o.cand.is_some())
        + usize::from(o.cand_object.is_some())
        + usize::from(o.null_control);
    if sources != 1 {
        return Err("exactly one of --cand FILE.hip, --cand-object FILE.co, --null-control aa is required".into());
    }
    if o.cand_geometry.is_some() && o.cand_object.is_none() {
        return Err("--cand-geometry needs --cand-object (a source candidate carries a `// hip_loop:` header)".into());
    }
    abba::timing_counts(o.time, o.rounds)?;
    Ok(o)
}

fn build_node(name: &str, shape: Option<&str>) -> Result<Box<dyn Node>> {
    let spec = ShapeSpec::parse(shape)?;
    match name {
        "mq6_res" => Ok(Box::new(mq6::Mq6Res::new(&spec)?)),
        "hc_mix" => Ok(Box::new(hc::HcMix::new(&spec)?)),
        "qsa_hg4" => Ok(Box::new(qsa::QsaHg4::new(&spec)?)),
        other => Err(format!("unknown node `{other}`")),
    }
}

// ---------------------------------------------------------------------------
// Records

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok,
    IdentityFailed,
    Unusable,
}

impl Outcome {
    fn code(&self) -> i32 {
        match self {
            Outcome::Ok => 0,
            Outcome::IdentityFailed => 1,
            Outcome::Unusable => 3,
        }
    }
}

fn fail(mut rec: Value, stage: &str, message: impl std::fmt::Display) -> (Value, Outcome) {
    rec["error"] = json!({"stage": stage, "message": message.to_string()});
    (rec, Outcome::Unusable)
}

fn geo_json(g: &Geometry) -> Value {
    json!({
        "grid_workgroups": g.grid,
        "block": g.block,
        "lds_dynamic_bytes": g.lds,
        "grid_work_items": g.threads(),
    })
}

fn arm_json(b: &jit::Built, symbol: &str, geo: &Geometry) -> Result<Value> {
    let mut v = b.kernel(symbol)?.json();
    v["geometry"] = geo_json(geo);
    v["object_sha256"] = json!(b.image_sha256);
    v["object_bytes"] = json!(b.image.len());
    v["object_path"] = json!(b.image_path.display().to_string());
    v["source_path"] = json!(b.source_path.display().to_string());
    v["kernels_in_object"] = json!(b.kernels.len());
    v["hipcc_argv"] = json!(b.argv);
    Ok(v)
}

fn out_dir_of(o: &Opts) -> PathBuf {
    o.out
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("hip_loop-{}", std::process::id())))
}

/// Everything the offline stage produces.
struct Prep {
    inc: Incumbent,
    base: jit::Built,
    cand: jit::Built,
    base_geo: Geometry,
    cand_geo: Geometry,
}

/// Resolve the candidate, its geometry, compile both arms, and fill the offline
/// part of the record.
fn prepare(o: &Opts, node: &dyn Node, workdir: &Path, rec: &mut Value) -> std::result::Result<Prep, (String, String)> {
    let stage = |s: &str| {
        let s = s.to_owned();
        move |e: String| (s.clone(), e)
    };
    let inc = node.incumbent().map_err(stage("incumbent"))?;
    rec["symbol"] = json!(inc.symbol);
    rec["module"] = json!(inc.module);
    rec["selection"] = json!(inc.selection);
    // A source candidate (`--cand`), the incumbent's own source (`--null-control aa`) or
    // a prebuilt object (`--cand-object`, no source).
    let cand_source = match (&o.cand, &o.cand_object) {
        (Some(p), _) => Some(
            std::fs::read_to_string(p)
                .map_err(|e| ("args".to_owned(), format!("--cand {}: {e}", p.display())))?,
        ),
        (None, Some(_)) => None,
        (None, None) => Some(inc.source.clone()),
    };
    let mut header = match &cand_source {
        Some(s) => Header::parse(s),
        None => Header::parse_list(o.cand_geometry.as_deref().unwrap_or("")),
    }
    .map_err(stage("header"))?;
    rec["candidate_header"] = json!(header.kv);
    let header_kv = header.kv.clone();
    let cand_geo = node.geometry_with(&mut header).map_err(stage("geometry"))?;
    let base_geo = node.geometry();
    if o.null_control && cand_geo != base_geo {
        return Err(("geometry".into(), "null control geometry differs".into()));
    }
    if o.offline {
        jit::isolate_caches(workdir).map_err(stage("compile"))?;
    }
    // Compile only what is needed: the incumbent unless `--base-object` supplies it (offline
    // still recompiles it, to verify the object), the candidate unless it is a prebuilt object
    // or an A/A against `--base-object` on the device (both arms load that object).
    let aa_from_object = o.null_control && o.base_object.is_some() && !o.offline;
    let want_base = o.base_object.is_none() || o.offline;
    let want_cand = cand_source.is_some() && !aa_from_object;
    let (compiled_base, compiled_cand) = std::thread::scope(|s| {
        let b = want_base.then(|| s.spawn(|| jit::compile(workdir, "baseline", &inc, &inc.source)));
        let c = want_cand.then(|| {
            s.spawn(|| jit::compile(workdir, "candidate", &inc, cand_source.as_deref().unwrap_or_default()))
        });
        let join = |h: Option<std::thread::ScopedJoinHandle<'_, Result<jit::Built>>>| -> Result<Option<jit::Built>> {
            match h {
                None => Ok(None),
                Some(h) => h.join().map_err(|_| "compile thread panicked".to_owned())?.map(Some),
            }
        };
        let (b, c) = (join(b), join(c));
        Ok::<_, String>((b?, c?))
    })
    .map_err(stage("compile"))?;
    let base = match &o.base_object {
        Some(p) => {
            let b = jit::from_object("baseline", p).map_err(stage("base_object"))?;
            rec["base_object"] = verify_base_object(p, &inc, &b, compiled_base.as_ref()).map_err(stage("base_object"))?;
            b
        }
        None => compiled_base.ok_or(("compile".to_owned(), "internal: no baseline".to_owned()))?,
    };
    let cand = if let Some(p) = &o.cand_object {
        jit::from_object("candidate", p).map_err(stage("object"))?
    } else if let Some(c) = compiled_cand {
        c
    } else {
        let p = o.base_object.as_ref().expect("an A/A without a candidate build has --base-object");
        jit::from_object("candidate", p).map_err(stage("base_object"))?
    };
    if o.cand_object.is_some() {
        // The object must be launchable with the incumbent's kernargs and fit the LDS.
        let (bk, ck) = (
            base.kernel(&inc.symbol).map_err(stage("compile"))?,
            cand.kernel(&inc.symbol).map_err(stage("object"))?,
        );
        if let Some(why) = ck.kernarg_mismatch(bk) {
            return Err(("abi".into(), format!("{}: {why}", inc.symbol)));
        }
        let lds = ck.lds_bytes + u64::from(cand_geo.lds);
        if lds > 64 * 1024 {
            return Err(("abi".into(), format!("static + dynamic LDS is {lds} B, over the 64 KiB workgroup limit")));
        }
        rec["candidate_object"] = json!({
            "path": o.cand_object.as_ref().map(|p| p.display().to_string()),
            "sha256": cand.image_sha256,
            "geometry_arg": o.cand_geometry,
            "kernarg_size_matches_incumbent": ck.kernarg_bytes == bk.kernarg_bytes,
        });
    }
    // The compiled objects, reusable on a card without recompiling: `--base-object
    // DIR/baseline.co`, `--cand-object DIR/candidate.co --cand-geometry <candidate_geometry_arg>`.
    let publish = |name: &str, b: &jit::Built, source: &str, kv: &std::collections::BTreeMap<String, String>| {
        let co = workdir.join(name);
        let side = jit::sidecar_json(&o.node, &inc, b, source, kv);
        jit::publish(&co, b, &side)
            .map(|()| json!({"path": co.display().to_string(), "sha256": b.image_sha256,
                "sidecar": jit::sidecar_path(&co).display().to_string()}))
    };
    if o.base_object.is_none() {
        rec["baseline_co"] = publish("baseline.co", &base, &inc.source, &Default::default()).map_err(stage("write"))?;
    }
    if let (true, Some(src)) = (want_cand && o.cand_object.is_none(), &cand_source) {
        rec["candidate_co"] = publish("candidate.co", &cand, src, &header_kv).map_err(stage("write"))?;
        let arg: Vec<String> = header_kv.iter().map(|(k, v)| format!("{k}={v}")).collect();
        rec["candidate_geometry_arg"] = json!(arg.join(" "));
    }
    rec["baseline"] = arm_json(&base, &inc.symbol, &base_geo).map_err(stage("compile"))?;
    if o.base_object.is_some() {
        rec["baseline"]["kind"] = json!("object");
    }
    let cand_stage = if o.cand_object.is_some() { "object" } else { "compile" };
    let ck = arm_json(&cand, &inc.symbol, &cand_geo).map_err(stage(cand_stage))?;
    rec["candidate"] = ck;
    if o.cand_object.is_some() {
        rec["candidate"]["kind"] = json!("object");
    }
    let same = rec["baseline"]["code_sha256"] == rec["candidate"]["code_sha256"];
    rec["identical_code"] = json!(same);
    rec["same_geometry"] = json!(base_geo == cand_geo);
    if o.null_control {
        rec["null_control"] = json!({
            "kind": "aa",
            "identical_code": same,
            "identical_objects": base.image_sha256 == cand.image_sha256,
            "baseline_object_sha256": base.image_sha256,
            "candidate_object_sha256": cand.image_sha256,
            "note": if aa_from_object {
                "both arms load --base-object as separate modules (no compile on the device)"
            } else {
                "the candidate is the incumbent source compiled separately and loaded as a separate module"
            },
        });
    }
    Ok(Prep { inc, base, cand, base_geo, cand_geo })
}

/// `--base-object FILE.co` as the incumbent: its sidecar `FILE.co.json` (written by an
/// earlier `--offline --out DIR`) must certify the object, the shipped source it was
/// compiled from and the `.text` sha256 of the launched kernel. Offline, `recompiled` (the
/// production-source compile just made) must also agree; on the device the sidecar is trusted.
fn verify_base_object(
    path: &Path,
    inc: &Incumbent,
    b: &jit::Built,
    recompiled: Option<&jit::Built>,
) -> Result<Value> {
    let side = jit::sidecar_path(path);
    let text = std::fs::read_to_string(&side).map_err(|e| {
        format!(
            "{}: {e} (write it with `hip_loop --offline --out DIR`, which produces DIR/baseline.co and DIR/baseline.co.json)",
            side.display()
        )
    })?;
    let sc: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", side.display()))?;
    if sc["tool"] != "hip_loop" || sc["schema"] != SCHEMA {
        return Err(format!("{}: not a hip_loop schema {SCHEMA} sidecar", side.display()));
    }
    if sc["module"] != inc.module.as_str() || sc["arch"] != ARCH {
        return Err(format!(
            "{}: sidecar is for module {} on {}, the incumbent is {} on {ARCH}",
            side.display(), sc["module"], sc["arch"], inc.module
        ));
    }
    let src_sha = elf::sha256_hex(inc.source.as_bytes());
    if sc["source_sha256"] != src_sha.as_str() {
        return Err(format!(
            "{}: the object was compiled from a different shipped source (sidecar {}, this build {src_sha}): stale object",
            path.display(), sc["source_sha256"]
        ));
    }
    if sc["object_sha256"] != b.image_sha256.as_str() {
        return Err(format!("{}: the object changed after its sidecar was written", path.display()));
    }
    let k = b.kernel(&inc.symbol)?;
    let sk = &sc["kernels"][&inc.symbol];
    if sk.is_null() || sk["code_sha256"] != k.code_sha256.as_str() {
        return Err(format!("{}: `{}` .text sha256 {} does not match the sidecar's {}", path.display(), inc.symbol, k.code_sha256, sk["code_sha256"]));
    }
    if sk["explicit_args"] != k.json()["explicit_args"] {
        return Err(format!("{}: `{}` explicit kernel arguments differ from the sidecar's", path.display(), inc.symbol));
    }
    if let Some(r) = recompiled {
        let fresh = &r.kernel(&inc.symbol)?.code_sha256;
        if *fresh != k.code_sha256 {
            return Err(format!(
                "{}: `{}` .text sha256 {} differs from a fresh compile of the production source ({fresh})",
                path.display(), inc.symbol, k.code_sha256
            ));
        }
    }
    let local = rdna_compute::KernelCompiler::local_toolchain_id();
    Ok(json!({
        "path": path.display().to_string(),
        "sha256": b.image_sha256,
        "sidecar": side.display().to_string(),
        "verified": if recompiled.is_some() { "sidecar + recompiled production source" } else { "sidecar (source sha256, object sha256, kernel .text sha256, kernel ABI)" },
        "source_sha256": src_sha,
        "toolchain_recorded": sc["toolchain_id"],
        "toolchain_local": local,
        "toolchain_matches": sc["toolchain_id"] == json!(local),
    }))
}

fn run(o: &Opts) -> (Value, Outcome) {
    let mut rec = json!({
        "schema": SCHEMA, "tool": "hip_loop", "node": o.node, "arch": ARCH,
        "shape": Value::Null, "symbol": Value::Null, "error": Value::Null,
        "offline": o.offline,
    });
    let mut node = match build_node(&o.node, o.shape.as_deref()) {
        Ok(n) => n,
        Err(e) => return fail(rec, "args", e),
    };
    rec["shape"] = json!(node.shape_label());
    rec["shape_dims"] = node.shape_json();
    let w = node.work();
    rec["work"] = json!({"flops": w.flops, "bytes": w.bytes});
    if let Value::Object(extra) = node.record_extra() {
        for (k, v) in extra {
            rec[k] = v;
        }
    }
    let workdir = out_dir_of(o);
    if let Err(e) = std::fs::create_dir_all(&workdir) {
        return fail(rec, "args", format!("--out {}: {e}", workdir.display()));
    }
    rec["out_dir"] = json!(workdir.display().to_string());
    let prep = match prepare(o, node.as_ref(), &workdir, &mut rec) {
        Ok(p) => p,
        Err((stage, e)) => return fail(rec, &stage, e),
    };
    if o.offline {
        return (rec, Outcome::Ok);
    }
    gpu_stage(o, node.as_mut(), &prep, rec)
}

// ---------------------------------------------------------------------------
// GPU

struct Arm {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
    geo: Geometry,
    karg: Vec<u8>,
}

fn load(gpu: &rdna_compute::Gpu, b: &jit::Built, symbol: &str, geo: Geometry, karg: Vec<u8>) -> Result<Arm> {
    // A fresh module per arm: no name-keyed function cache sits between the two
    // arms, which share the kernel symbol.
    let module = gpu.hip.module_load_data(&b.image).map_err(err)?;
    let func = gpu.hip.module_get_function(&module, symbol).map_err(err)?;
    Ok(Arm { _module: module, func, geo, karg })
}

fn sync(gpu: &rdna_compute::Gpu) -> Result<()> {
    gpu.hip.device_synchronize().map_err(err)
}

fn run_arm(gpu: &rdna_compute::Gpu, node: &mut dyn Node, arm: &mut Arm) -> Result<Vec<u8>> {
    node.reset(gpu)?;
    sync(gpu)?;
    node.dispatch(gpu, &arm.func, &arm.geo, &mut arm.karg)?;
    sync(gpu)?;
    node.read_output(gpu)
}

/// First differing byte and the number of differing bytes.
fn diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), a.len().abs_diff(b.len())));
    }
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some((first, a.iter().zip(b).filter(|(x, y)| x != y).count()))
}

fn clock_note(gpu: &rdna_compute::Gpu) -> Value {
    let bus = gpu.hip.device_pci_bus_id(gpu.device_id).ok().map(|s| s.to_lowercase());
    let read = |name: &str| -> Option<String> {
        let bus = bus.as_ref()?;
        std::fs::read_to_string(format!("/sys/bus/pci/devices/{bus}/{name}"))
            .ok()
            .map(|s| s.trim().replace('\n', " | "))
    };
    json!({
        "bus_id": bus,
        "sclk_dpm": read("pp_dpm_sclk"),
        "perf_level": read("power_dpm_force_performance_level"),
    })
}

/// P[X >= k], X ~ Binomial(n, 1/2): the one-sided exact sign test.
fn p_upper(k: usize, n: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    // sum_{i=0}^{n-k} C(n, i) / 2^n, by symmetry.
    let mut term = 0.5f64.powi(n as i32);
    let mut sum = term;
    for i in 0..(n - k) {
        term *= (n - i) as f64 / (i + 1) as f64;
        sum += term;
    }
    sum.min(1.0)
}

fn stats_json(s: &abba::TimingStats) -> Value {
    json!({"median": s.median / 1e3, "min": s.min / 1e3, "max": s.max / 1e3,
        "p90": s.p90 / 1e3, "mean": s.mean / 1e3})
}

/// The timing record from the arms' samples (ns, schedule order, whole rounds).
fn timing_record(base_ns: &[f64], cand_ns: &[f64], n: usize, work: node::Work) -> Result<Value> {
    let bs = abba::timing_stats(base_ns)?;
    let cs = abba::timing_stats(cand_ns)?;
    let bm = abba::block_medians(base_ns, n)?;
    let cm = abba::block_medians(cand_ns, n)?;
    let ratios = abba::round_ratios(base_ns, cand_ns, n)?;
    let rounds: Vec<Value> = ratios
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let b = (bm[2 * i] + bm[2 * i + 1]) / 2.0;
            let c = (cm[2 * i] + cm[2 * i + 1]) / 2.0;
            json!({"round": i, "base_us": b / 1e3, "cand_us": c / 1e3, "ratio": r})
        })
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let r = sorted.len();
    let median = if r % 2 == 0 {
        (sorted[r / 2 - 1] + sorted[r / 2]) / 2.0
    } else {
        sorted[r / 2]
    };
    let faster = ratios.iter().filter(|x| **x < 1.0).count();
    let slower = ratios.iter().filter(|x| **x > 1.0).count();
    let tflops = |ns: f64| work.flops / ns / 1e3;
    let gbps = |ns: f64| work.bytes / ns;
    Ok(json!({
        "rounds": rounds,
        "ratio_median": median,
        "ratio_min": sorted[0],
        "ratio_max": sorted[r - 1],
        "rounds_faster": faster,
        "rounds_slower": slower,
        "rounds_tied": r - faster - slower,
        "p_faster": p_upper(faster, r),
        "p_slower": p_upper(slower, r),
        "timing": {
            "n": n, "rounds": r, "calls_per_arm": bs.count,
            "baseline_us": stats_json(&bs), "candidate_us": stats_json(&cs),
            "baseline_tflops": tflops(bs.median), "candidate_tflops": tflops(cs.median),
            "baseline_gbps": gbps(bs.median), "candidate_gbps": gbps(cs.median),
            "unit": "us per launch: hipEvent pair on the default stream around one dispatch; \
                     device idle at the start (output reset + sync outside the interval), \
                     so launch latency is inside the interval for both arms",
        },
    }))
}

fn sample(
    gpu: &rdna_compute::Gpu,
    node: &mut dyn Node,
    base: &mut Arm,
    cand: &mut Arm,
    n: usize,
    rounds: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (per_arm, _) = abba::timing_counts(n, rounds)?;
    let (mut b, mut c) = (Vec::with_capacity(per_arm), Vec::with_capacity(per_arm));
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = match gpu.hip.event_create() {
        Ok(e) => e,
        Err(e) => {
            let _ = gpu.hip.event_destroy(start);
            return Err(err(e));
        }
    };
    let measured = (|| -> Result<()> {
        for which in abba::timing_schedule(n, rounds)? {
            let (arm, sink) = match which {
                abba::TimingArm::Incumbent => (&mut *base, &mut b),
                abba::TimingArm::Candidate => (&mut *cand, &mut c),
            };
            node.reset(gpu)?;
            sync(gpu)?;
            gpu.hip.event_record(&start, None).map_err(err)?;
            node.dispatch(gpu, &arm.func, &arm.geo, &mut arm.karg)?;
            gpu.hip.event_record(&stop, None).map_err(err)?;
            gpu.hip.event_synchronize(&stop).map_err(err)?;
            sink.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6);
        }
        Ok(())
    })();
    let d1 = gpu.hip.event_destroy(start).map_err(err);
    let d2 = gpu.hip.event_destroy(stop).map_err(err);
    measured?;
    d1?;
    d2?;
    Ok((b, c))
}

fn gpu_stage(o: &Opts, node: &mut dyn Node, prep: &Prep, mut rec: Value) -> (Value, Outcome) {
    let gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => return fail(rec, "device", e),
    };
    let arch = gpu.arch.split(':').next().unwrap_or("").to_owned();
    rec["device"] = json!({
        "arch": gpu.arch, "device_id": gpu.device_id,
        "uuid": std::env::var("ROCR_VISIBLE_DEVICES").ok().or_else(|| std::env::var("HIPFIRE_DEVICES").ok()),
        "fabric_job": std::env::var("FABRIC_JOB").ok(),
    });
    if arch != ARCH {
        return fail(rec, "device", format!("device is {arch}, hip_loop requires {ARCH}"));
    }
    if let Err(e) = node.setup(&gpu, o.seed) {
        return fail(rec, "setup", e);
    }
    let res = gpu_run(o, node, prep, &gpu, &mut rec);
    node.teardown(&gpu);
    match res {
        Ok(outcome) => (rec, outcome),
        Err((stage, e)) => fail(rec, &stage, e),
    }
}

fn gpu_run(
    o: &Opts,
    node: &mut dyn Node,
    prep: &Prep,
    gpu: &rdna_compute::Gpu,
    rec: &mut Value,
) -> std::result::Result<Outcome, (String, String)> {
    let stage = |s: &str| {
        let s = s.to_owned();
        move |e: String| (s.clone(), e)
    };
    let karg = node.kernarg().map_err(stage("setup"))?;
    let sym = &prep.inc.symbol;
    let mut base = load(gpu, &prep.base, sym, prep.base_geo, karg.clone()).map_err(stage("load"))?;
    let mut cand = load(gpu, &prep.cand, sym, prep.cand_geo, karg).map_err(stage("load"))?;

    // Identity: every dispatch starts from the pristine output.
    node.reset(gpu).map_err(stage("identity"))?;
    sync(gpu).map_err(stage("identity"))?;
    let pristine = node.read_output(gpu).map_err(stage("identity"))?;
    let b1 = run_arm(gpu, node, &mut base).map_err(stage("identity"))?;
    let b2 = run_arm(gpu, node, &mut base).map_err(stage("identity"))?;
    let c1 = run_arm(gpu, node, &mut cand).map_err(stage("identity"))?;
    let c2 = run_arm(gpu, node, &mut cand).map_err(stage("identity"))?;
    let changed = diff(&pristine, &b1).is_some();
    let base_repeat = b1 == b2;
    let cand_repeat = c1 == c2;
    let d = diff(&b1, &c1);
    let identical = d.is_none();
    rec["identical_to_baseline"] = json!(identical);
    rec["identity"] = json!({
        "output_bytes": b1.len(),
        "dispatch_changed_output": changed,
        "baseline_repeatable": base_repeat,
        "candidate_repeatable": cand_repeat,
        "first_diff_byte": d.map(|x| x.0),
        "differing_bytes": d.map(|x| x.1),
    });
    if !changed {
        return Err(("identity".into(), "the baseline dispatch did not change the output: the launch is not exercised".into()));
    }
    if !base_repeat {
        return Err(("identity".into(), "the baseline output is not repeatable from a pristine state".into()));
    }
    if !identical || !cand_repeat {
        return Ok(Outcome::IdentityFailed);
    }

    let before = clock_note(gpu);
    let (b_ns, c_ns) = sample(gpu, node, &mut base, &mut cand, o.time, o.rounds).map_err(stage("timing"))?;
    let after = clock_note(gpu);
    let t = timing_record(&b_ns, &c_ns, o.time, node.work()).map_err(stage("timing"))?;
    if let Value::Object(m) = t {
        for (k, v) in m {
            rec[k] = v;
        }
    }
    rec["clock_note"] = json!({
        "clocks_pinned": false,
        "method": "ABBA-paired rounds cancel slow drift; sclk dpm table sampled before/after",
        "before": before, "after": after,
    });
    Ok(Outcome::Ok)
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: hip_loop {USAGE}");
        return;
    }
    let o = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("hip_loop: {e}\nusage: hip_loop {USAGE}");
            std::process::exit(2);
        }
    };
    let (rec, outcome) = run(&o);
    println!("{}", serde_json::to_string_pretty(&rec).expect("record is JSON"));
    if let Some(e) = rec["error"].as_object() {
        eprintln!("hip_loop: {}: {}", e["stage"], e["message"]);
    }
    if outcome == Outcome::IdentityFailed {
        eprintln!("hip_loop: candidate output is not bit-identical to the baseline");
    }
    std::process::exit(outcome.code());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn cli_parses_and_refuses() {
        let o = parse_args(&args("--node mq6_res --shape qkv --cand a.hip --offline --time 5 --rounds 3 --seed 9")).unwrap();
        assert!(o.offline && !o.null_control);
        assert_eq!((o.time, o.rounds, o.seed), (5, 3, 9));
        assert_eq!(o.shape.as_deref(), Some("qkv"));
        let o = parse_args(&args("--node hc_mix --null-control aa")).unwrap();
        assert!(o.null_control && o.cand.is_none());
        for bad in [
            "--node nope --cand a",
            "--node mq6_res",
            "--node mq6_res --cand a --null-control aa",
            "--node mq6_res --cand a --null-control bb",
            "--node mq6_res --cand a --time 0",
            "--node mq6_res --cand a --wat",
            "--node mq6_res --cand",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn one_sided_sign_test() {
        assert_eq!(p_upper(0, 10), 1.0);
        assert_eq!(p_upper(11, 10), 0.0);
        assert!((p_upper(10, 10) - 1.0 / 1024.0).abs() < 1e-15);
        assert!((p_upper(8, 10) - 56.0 / 1024.0).abs() < 1e-15);
        assert!((p_upper(5, 10) - 638.0 / 1024.0).abs() < 1e-15);
    }

    #[test]
    fn timing_record_pairs_rounds_and_signs() {
        // n = 2 launches per block, 3 rounds; the candidate is 10% faster each round.
        let n = 2;
        let base: Vec<f64> = vec![100.0; 2 * n * 3 * 1];
        let cand: Vec<f64> = vec![90.0; 2 * n * 3];
        let w = node::Work { flops: 1e6, bytes: 1e6 };
        let t = timing_record(&base, &cand, n, w).unwrap();
        assert_eq!(t["rounds"].as_array().unwrap().len(), 3);
        assert!((t["ratio_median"].as_f64().unwrap() - 0.9).abs() < 1e-12);
        assert_eq!(t["rounds_faster"], 3);
        assert!((t["p_faster"].as_f64().unwrap() - 0.125).abs() < 1e-15);
        assert_eq!(t["p_slower"].as_f64().unwrap(), 1.0);
        assert!((t["timing"]["baseline_tflops"].as_f64().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn diff_reports_first_byte_and_count() {
        assert_eq!(diff(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(diff(&[1, 2, 3, 4], &[1, 9, 3, 8]), Some((1, 2)));
        assert!(diff(&[1], &[1, 2]).is_some());
    }

    #[test]
    fn registered_nodes_build_and_reproduce_production_grids() {
        // (node, shape, PF-1 profiler grid)
        for (n, s, grid) in [
            ("mq6_res", "qkv", [163840u64, 6, 1]),
            ("mq6_res", "z", [98304, 6, 1]),
            ("mq6_res", "out", [40960, 6, 1]),
            ("mq6_res", "qg", [196608, 6, 1]),
            ("mq6_res", "ba", [768, 6, 1]),
            ("hc_mix", "mix", [320, 12, 1]),
        ] {
            let node = build_node(n, Some(s)).unwrap();
            assert_eq!(node.geometry().threads(), grid, "{n} {s}");
            assert_eq!(node.name(), n);
        }
    }

    /// Run `--offline` for `node` with the incumbent's own source (or `source`).
    fn offline(node: &str, shape: &str, source: Option<&str>, tag: &str) -> Value {
        let dir = tmp(tag);
        let cand = dir.join("cand.hip");
        let n = build_node(node, Some(shape)).unwrap();
        std::fs::write(&cand, source.map(str::to_owned).unwrap_or(n.incumbent().unwrap().source)).unwrap();
        let o = Opts {
            node: node.into(),
            shape: Some(shape.into()),
            cand: Some(cand),
            offline: true,
            null_control: false,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        rec
    }

    #[test]
    fn offline_record_for_every_mq6_shape_matches_pf1_resources() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let mut shas = std::collections::BTreeSet::new();
        for (i, shape) in ["qkv", "z", "out", "qg", "ba"].iter().enumerate() {
            let rec = offline("mq6_res", shape, None, &format!("mq6-{i}"));
            assert_eq!(rec["node"], "mq6_res");
            assert_eq!(rec["shape"], *shape);
            assert_eq!(rec["symbol"], "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds");
            assert!(rec["error"].is_null());
            assert_eq!(rec["identical_code"], true);
            assert_eq!(rec["baseline"]["code_sha256"], rec["candidate"]["code_sha256"]);
            for arm in ["baseline", "candidate"] {
                let a = &rec[arm];
                assert_eq!(a["code_sha256"].as_str().unwrap().len(), 64);
                // PF-1: the MW8 kernel is 256 threads, 8 KiB LDS, 96 VGPR.
                assert_eq!(a["lds_bytes"], 8192, "{a}");
                // The metadata count is 92; the allocation granule (8) gives the 96 PF-1 reports.
                assert_eq!(a["vgprs_allocated"], 96, "{a}");
                assert!(a["vgprs"].as_u64().unwrap() <= 96, "{a}");
                assert_eq!(a["max_flat_workgroup_size"], 256);
                assert_eq!(a["scratch_bytes"], 0);
                assert_eq!(a["spill"], 0);
                assert!(a["code_bytes"].as_u64().unwrap() > 0);
                assert!(a["occupancy_waves_per_simd"].as_u64().unwrap() >= 1);
                assert_eq!(a["geometry"]["block"], json!([256, 1, 1]));
            }
            shas.insert(rec["baseline"]["code_sha256"].as_str().unwrap().to_owned());
        }
        // Shape does not change the code: one kernel, one sha.
        assert_eq!(shas.len(), 1);
    }

    #[test]
    fn code_sha_is_stable_across_compiles_and_changes_with_the_source() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let n = build_node("mq6_res", Some("qkv")).unwrap();
        let src = n.incumbent().unwrap().source;
        assert!(src.contains("#pragma unroll"));
        let a = offline("mq6_res", "qkv", None, "stab-a");
        let b = offline("mq6_res", "qkv", None, "stab-b");
        assert_eq!(a["baseline"]["code_sha256"], b["baseline"]["code_sha256"]);
        assert_eq!(a["baseline"]["object_sha256"], b["baseline"]["object_sha256"]);
        // Perturb the source: the unrolled loops become rolled.
        let perturbed = src.replace("#pragma unroll\n", "#pragma unroll 1\n");
        assert_ne!(perturbed, src);
        let p = offline("mq6_res", "qkv", Some(&perturbed), "pert");
        assert_eq!(p["identical_code"], false);
        assert_ne!(p["baseline"]["code_sha256"], p["candidate"]["code_sha256"]);
        assert_eq!(p["baseline"]["code_sha256"], a["baseline"]["code_sha256"]);
    }

    #[test]
    fn candidate_header_changes_geometry_not_code() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let n = build_node("mq6_res", Some("qkv")).unwrap();
        let src = format!("// hip_loop: tile_m=32 tile_n=128 block=256\n{}", n.incumbent().unwrap().source);
        let r = offline("mq6_res", "qkv", Some(&src), "hdr");
        assert_eq!(r["candidate"]["geometry"]["grid_workgroups"], json!([320, 6, 1]));
        assert_eq!(r["baseline"]["geometry"]["grid_workgroups"], json!([640, 6, 1]));
        assert_eq!(r["same_geometry"], false);
        assert_eq!(r["candidate_header"]["tile_m"], "32");
    }

    #[test]
    fn missing_symbol_and_bad_header_are_loud() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("loud");
        let cand = dir.join("c.hip");
        std::fs::write(&cand, "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void other(){}\n").unwrap();
        let mut o = Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: Some(cand.clone()),
            offline: true,
            null_control: false,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "compile");
        assert!(rec["error"]["message"].as_str().unwrap().contains("does not define kernel"), "{rec}");
        std::fs::write(&cand, "// hip_loop: tile_m=0\n").unwrap();
        o.out = Some(dir.join("out2"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "geometry");
    }

    #[test]
    fn null_control_offline_compiles_the_incumbent_twice() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("aa");
        let o = Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: None,
            offline: true,
            null_control: true,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        assert_eq!(rec["null_control"]["kind"], "aa");
        assert_eq!(rec["null_control"]["identical_objects"], true);
        assert_ne!(rec["baseline"]["source_path"], rec["candidate"]["source_path"]);
    }

    #[test]
    fn offline_record_for_hc_mix() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let rec = &offline("hc_mix", "mix", None, "hc");
        assert_eq!(rec["symbol"], "gemm_wmma_lds_64_64_32_64_k64_p");
        assert_eq!(rec["identical_code"], true);
        assert_eq!(rec["baseline"]["geometry"]["grid_work_items"], json!([320, 12, 1]));
        assert_eq!(rec["baseline"]["max_flat_workgroup_size"], 64);
        assert_eq!(rec["baseline"]["spill"], 0);
        assert_eq!(rec["baseline"]["code_sha256"].as_str().unwrap().len(), 64);
    }
}

#[cfg(test)]
mod object_tests {
    use super::*;

    fn opts(dir: &Path, object: PathBuf, geometry: Option<&str>) -> Opts {
        Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: None,
            offline: true,
            null_control: false,
            cand_object: Some(object),
            cand_geometry: geometry.map(str::to_owned),
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-objtest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn cli_takes_exactly_one_candidate_source() {
        let a = |s: &str| parse_args(&s.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
        let o = a("--node mq6_res --cand-object x.co --cand-geometry tile_m=32,tile_n=256").unwrap();
        assert_eq!(o.cand_geometry.as_deref(), Some("tile_m=32,tile_n=256"));
        for bad in [
            "--node mq6_res --cand-object x.co --cand y.hip",
            "--node mq6_res --cand-object x.co --null-control aa",
            "--node mq6_res --cand-geometry tile_m=32 --cand y.hip",
            "--node mq6_res",
        ] {
            assert!(a(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prebuilt_object_is_judged_like_a_compiled_candidate() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("same");
        jit::isolate_caches(&dir).unwrap();
        let n = build_node("mq6_res", Some("ba")).unwrap();
        let inc = n.incumbent().unwrap();
        let built = jit::compile(&dir, "prebuilt", &inc, &inc.source).unwrap();
        let (rec, outcome) = run(&opts(
            &dir,
            built.image_path.clone(),
            Some("tile_m=32 tile_n=128 block=256"),
        ));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        assert_eq!(rec["candidate"]["kind"], "object");
        assert_eq!(rec["identical_code"], true);
        assert_eq!(rec["candidate"]["geometry"]["grid_workgroups"], json!([2, 6, 1]));
        assert_eq!(rec["candidate"]["lds_bytes"], 8192);
        assert_eq!(rec["candidate_object"]["sha256"], json!(built.image_sha256));
        assert_eq!(rec["candidate_object"]["kernarg_size_matches_incumbent"], true);
        assert!(rec["candidate"]["hipcc_argv"].as_array().unwrap().is_empty());
    }

    #[test]
    fn object_with_another_kernarg_layout_or_symbol_is_refused() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("abi");
        jit::isolate_caches(&dir).unwrap();
        let n = build_node("mq6_res", Some("ba")).unwrap();
        let mut inc = n.incumbent().unwrap();
        let good = inc.clone();
        inc.source = format!(
            "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void {}(int* a) {{ a[0] = 1; }}\n",
            inc.symbol
        );
        let wrong_abi = jit::compile(&dir, "wrong-abi", &inc, &inc.source).unwrap();
        let (rec, outcome) = run(&opts(&dir, wrong_abi.image_path.clone(), None));
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "abi", "{rec}");
        inc.source = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void other(int* a) { a[0] = 1; }\n".into();
        let other = jit::compile(&dir, "other", &inc, &inc.source).unwrap();
        let mut o = opts(&dir, other.image_path.clone(), None);
        o.out = Some(dir.join("out2"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "object");
        assert!(rec["error"]["message"].as_str().unwrap().contains("does not define kernel"), "{rec}");
        // A geometry outside the header grammar is refused before any compile.
        let built = jit::compile(&dir, "ok", &good, &good.source).unwrap();
        let mut o = opts(&dir, built.image_path.clone(), Some("block=100"));
        o.out = Some(dir.join("out3"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "geometry");
    }
}

#[cfg(test)]
mod base_object_tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-basetest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn opts(out: PathBuf, cand: Option<PathBuf>, base: Option<PathBuf>, aa: bool) -> Opts {
        Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand,
            offline: true,
            null_control: aa,
            cand_object: None,
            cand_geometry: None,
            base_object: base,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(out),
        }
    }

    fn inc() -> Incumbent {
        build_node("mq6_res", Some("ba")).unwrap().incumbent().unwrap()
    }

    #[test]
    fn cli_accepts_base_object_with_any_candidate() {
        let a = |s: &str| parse_args(&s.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
        let o = a("--node mq6_res --base-object b.co --cand c.hip").unwrap();
        assert_eq!(o.base_object.as_deref(), Some(Path::new("b.co")));
        assert!(a("--node mq6_res --base-object b.co --null-control aa").is_ok());
        assert!(a("--node mq6_res --base-object b.co --cand-object c.co").is_ok());
        assert!(a("--node mq6_res --base-object").is_err());
    }

    #[test]
    fn offline_writes_both_objects_and_sidecars_and_base_object_round_trips() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("rt");
        let cand = dir.join("cand.hip");
        std::fs::write(&cand, format!("// hip_loop: tile_m=32 tile_n=128\n{}", inc().source)).unwrap();
        let (rec, outcome) = run(&opts(dir.join("o1"), Some(cand.clone()), None, false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        for (key, name) in [("baseline_co", "baseline.co"), ("candidate_co", "candidate.co")] {
            let p = dir.join("o1").join(name);
            assert_eq!(rec[key]["path"], json!(p.display().to_string()));
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(rec[key]["sha256"], json!(elf::sha256_hex(&bytes)));
            let side: Value = serde_json::from_str(&std::fs::read_to_string(jit::sidecar_path(&p)).unwrap()).unwrap();
            assert_eq!(side["object_sha256"], rec[key]["sha256"]);
            let src = if key == "baseline_co" { inc().source } else { std::fs::read_to_string(&cand).unwrap() };
            assert_eq!(side["source_sha256"], json!(elf::sha256_hex(src.as_bytes())));
            assert_eq!(
                side["kernels"][&inc().symbol]["code_sha256"],
                rec["baseline"]["code_sha256"]
            );
        }
        assert_eq!(rec["candidate_geometry_arg"], "tile_m=32 tile_n=128");

        // Offline: the prebuilt incumbent is verified against a fresh production compile.
        let base = dir.join("o1").join("baseline.co");
        let (rec2, outcome) = run(&opts(dir.join("o2"), Some(cand), Some(base.clone()), false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec2["error"]);
        assert_eq!(rec2["baseline"]["kind"], "object");
        assert_eq!(rec2["base_object"]["verified"], "sidecar + recompiled production source");
        assert_eq!(rec2["baseline"]["code_sha256"], rec["baseline"]["code_sha256"]);
        assert!(rec2.get("baseline_co").is_none());

        // A/A with --base-object offline recompiles the candidate; both arms identical.
        let (rec3, outcome) = run(&opts(dir.join("o3"), None, Some(base.clone()), true));
        assert_eq!(outcome, Outcome::Ok, "{}", rec3["error"]);
        assert_eq!(rec3["null_control"]["identical_objects"], true);

        // Device-mode verification (no recompile) trusts the sidecar and checks everything else.
        let b = jit::from_object("baseline", &base).unwrap();
        let v = verify_base_object(&base, &inc(), &b, None).unwrap();
        assert_eq!(v["verified"].as_str().unwrap().starts_with("sidecar ("), true);
    }

    #[test]
    fn base_object_is_refused_when_the_sidecar_does_not_certify_it() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("bad");
        let cand = dir.join("cand.hip");
        std::fs::write(&cand, inc().source).unwrap();
        let (rec, outcome) = run(&opts(dir.join("o1"), Some(cand), None, false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        let good = dir.join("o1").join("baseline.co");
        let side = jit::sidecar_path(&good);
        let sidecar: Value = serde_json::from_str(&std::fs::read_to_string(&side).unwrap()).unwrap();
        let load = |p: &Path| jit::from_object("baseline", p).unwrap();
        let sym = inc().symbol;

        // Missing sidecar.
        let lone = dir.join("lone.co");
        std::fs::copy(&good, &lone).unwrap();
        let e = verify_base_object(&lone, &inc(), &load(&lone), None).unwrap_err();
        assert!(e.contains("lone.co.json"), "{e}");

        let tampered = |name: &str, edit: &dyn Fn(&mut Value)| -> PathBuf {
            let p = dir.join(name);
            std::fs::copy(&good, &p).unwrap();
            let mut s = sidecar.clone();
            edit(&mut s);
            std::fs::write(jit::sidecar_path(&p), serde_json::to_string(&s).unwrap()).unwrap();
            p
        };
        // Wrong kernel .text sha, stale source, wrong module, object sha.
        let p = tampered("sha.co", &|s| s["kernels"][&sym]["code_sha256"] = json!("00"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None).unwrap_err().contains(".text sha256"));
        let p = tampered("src.co", &|s| s["source_sha256"] = json!("11"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None).unwrap_err().contains("stale"));
        let p = tampered("mod.co", &|s| s["module"] = json!("other_module"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None).unwrap_err().contains("other_module"));
        let p = tampered("obj.co", &|s| s["object_sha256"] = json!("22"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None).unwrap_err().contains("changed"));
        let p = tampered("abi.co", &|s| s["kernels"][&sym]["explicit_args"] = json!([]));
        assert!(verify_base_object(&p, &inc(), &load(&p), None).unwrap_err().contains("kernel arguments"));

        // Offline recompile disagreement: the sidecar and object agree with each other
        // but not with the production compile.
        let other = dir.join("other.co");
        let mut other_inc = inc();
        other_inc.source = inc().source.replace("#pragma unroll\n", "#pragma unroll 1\n");
        let built = jit::compile(&dir.join("o1"), "perturbed", &other_inc, &other_inc.source).unwrap();
        std::fs::copy(&built.image_path, &other).unwrap();
        let mut s = jit::sidecar_json("mq6_res", &inc(), &built, &inc().source, &Default::default());
        s["source_sha256"] = json!(elf::sha256_hex(inc().source.as_bytes()));
        std::fs::write(jit::sidecar_path(&other), serde_json::to_string(&s).unwrap()).unwrap();
        let fresh = jit::compile(&dir.join("o1"), "fresh", &inc(), &inc().source).unwrap();
        let e = verify_base_object(&other, &inc(), &load(&other), Some(&fresh)).unwrap_err();
        assert!(e.contains("fresh compile"), "{e}");
        // And through the driver: stage base_object, exit 3.
        let (rec, outcome) = run(&opts(dir.join("o9"), None, Some(other), true));
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "base_object");
    }
}
