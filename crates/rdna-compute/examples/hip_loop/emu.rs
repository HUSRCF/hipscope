// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The pre-card emulator gate of `hip_loop --emu`: no candidate reaches the XTX before its code
//! object has run, at a small shape and over the FULL launch grid, on the PeaceMaker CPU emulator
//! (`peacemaker-ir::emu`, objects lifted by `peacemaker_lift::lift_object`) next to the
//! incumbent's object, from the same host-side seeded inputs.
//!
//! What fails the gate (every one of these is a card fault or a silent wrong answer there):
//! * any global access outside a mapped buffer: each buffer is its own exact-size emulator region
//!   with [`GUARD_BYTES`] of unmapped space on both sides, so a one-byte overrun is an error;
//! * LDS access beyond `group_segment_fixed_size + dynamic LDS`;
//! * a read of an LDS byte this workgroup never wrote (`Emulator::strict_lds`);
//! * a write into an input buffer or the kernarg segment (the kernel's contract is `out` only);
//! * a barrier deadlock, an instruction-limit overrun, an instruction the emulator has no
//!   semantics for (the error names the opcode: the gate fails closed), a panic in the emulator;
//! * output bytes not bit-identical to the emulated incumbent's at the same shape and inputs.
//!
//! The kernarg segment is mapped as its own region at the object's own `kernarg_segment_size`:
//! the explicit arguments at the metadata offsets, the hidden arguments the runtime fills
//! (block counts, group sizes, remainders, `hidden_dynamic_lds_size`) from the arm's launch
//! geometry, everything else zero (a zero pointer a kernel dereferences is an unmapped access).
//!
//! Included with `#[path = "hip_loop/emu.rs"] mod emu;`.
#![allow(dead_code)]

use crate::elf::sha256_hex;
use crate::node::{EmuArg, EmuCase, Geometry, Result};
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Abi, Frontend};
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

/// Unmapped gap below and above every region (>= 64 KiB required by the gate's contract).
pub const GUARD_BYTES: u64 = 1 << 20;

/// What one emulated arm produced.
pub struct ArmRun {
    pub out: Vec<u8>,
    pub instructions: u64,
    pub workgroups: u64,
    pub json: Value,
    /// Bytes of the output buffer the arm wrote.
    pub out_written: usize,
}

/// An arm that failed, with the work done before it did.
pub struct ArmFail {
    pub reason: String,
    pub instructions: u64,
    pub workgroups: u64,
    pub json: Value,
}

fn fail(reason: impl Into<String>, json: Value) -> ArmFail {
    ArmFail { reason: reason.into(), instructions: 0, workgroups: 0, json }
}

/// The kernarg segment of one arm: explicit arguments from `case` (addresses of the mapped
/// buffers) at the object's metadata offsets, hidden arguments from the launch geometry.
fn kernarg_bytes(
    case: &EmuCase,
    addrs: &[u64],
    args: &[peacemaker_ir::metadata::Kernarg],
    segment: usize,
    geo: &Geometry,
) -> Result<Vec<u8>> {
    let explicit: Vec<_> = args.iter().filter(|a| !a.value_kind.starts_with("hidden_")).collect();
    if explicit.len() != case.args.len() {
        return Err(format!(
            "abi: the object declares {} explicit kernel arguments, the node passes {}",
            explicit.len(),
            case.args.len()
        ));
    }
    let end = args.iter().map(|a| (a.offset + a.size) as usize).max().unwrap_or(0);
    let mut k = vec![0u8; segment.max(end)];
    for (i, (a, v)) in explicit.iter().zip(&case.args).enumerate() {
        let (off, size) = (a.offset as usize, a.size as usize);
        let bytes: Vec<u8> = match v {
            EmuArg::Buf(b) => addrs.get(*b).ok_or("abi: argument names a missing buffer")?.to_le_bytes().to_vec(),
            EmuArg::Null => 0u64.to_le_bytes().to_vec(),
            EmuArg::I32(x) => x.to_le_bytes().to_vec(),
            EmuArg::U32(x) => x.to_le_bytes().to_vec(),
            EmuArg::F32(x) => x.to_le_bytes().to_vec(),
        };
        if bytes.len() != size {
            return Err(format!(
                "abi: explicit argument {i} `{}` is {size} B in the object, {} B in the node",
                a.name,
                bytes.len()
            ));
        }
        k[off..off + size].copy_from_slice(&bytes);
    }
    let [bx, by, bz] = geo.block;
    let [gx, gy, gz] = geo.grid;
    for a in args.iter().filter(|a| a.value_kind.starts_with("hidden_")) {
        let (off, size) = (a.offset as usize, a.size as usize);
        let value: u64 = match a.value_kind.as_str() {
            "hidden_block_count_x" => u64::from(gx),
            "hidden_block_count_y" => u64::from(gy),
            "hidden_block_count_z" => u64::from(gz),
            "hidden_group_size_x" => u64::from(bx),
            "hidden_group_size_y" => u64::from(by),
            "hidden_group_size_z" => u64::from(bz),
            // The grid is a whole number of workgroups.
            "hidden_remainder_x" | "hidden_remainder_y" | "hidden_remainder_z" => 0,
            "hidden_grid_dims" => 3,
            "hidden_dynamic_lds_size" => u64::from(geo.lds),
            // Global offsets, queue/printf/hostcall/heap pointers, ...: zero.
            _ => 0,
        };
        k[off..off + size].copy_from_slice(&value.to_le_bytes()[..size.min(8)]);
    }
    Ok(k)
}

/// Lift `image` (hipcc first, then the builder frontend for PM twins) and return the program and
/// the frontend that accepted it.
fn lift(image: &[u8]) -> Result<(peacemaker_ir::Program, &'static str)> {
    let mut errs = Vec::new();
    for (name, frontend) in [("hipcc", Frontend::Hipcc), ("builder", Frontend::Builder)] {
        match peacemaker_lift::lift_object(image, peacemaker_lift::Options { frontend }) {
            Ok(l) => return Ok((l.program, name)),
            Err(e) => errs.push(format!("{name}: {e:?}")),
        }
    }
    Err(format!("lift refused the object ({})", errs.join("; ")))
}

/// Emulate every workgroup of `geo` for `symbol` in `image` on a fresh copy of `case`.
pub fn run_arm(
    tag: &str,
    image: &[u8],
    symbol: &str,
    geo: &Geometry,
    case: &EmuCase,
) -> std::result::Result<ArmRun, ArmFail> {
    let started = Instant::now();
    let mut info = json!({"tag": tag, "symbol": symbol, "object_sha256": sha256_hex(image)});
    let (program, frontend) = lift(image).map_err(|e| fail(format!("lift: {e}"), info.clone()))?;
    info["frontend"] = json!(frontend);
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| fail(format!("the object does not define `{symbol}`"), info.clone()))?;
    let Abi::Hsa { descriptor, metadata } = &kernel.abi else {
        return Err(fail("not an HSA kernel", info));
    };
    info["group_segment_fixed_size"] = json!(descriptor.group_segment_fixed_size);
    info["private_segment_fixed_size"] = json!(descriptor.private_segment_fixed_size);
    info["kernarg_segment_size"] = json!(metadata.parsed.kernarg_segment_size);
    // Every opcode of the kernel the emulator cannot execute, named at once (fail closed).
    let mut missing = std::collections::BTreeMap::<&str, usize>::new();
    for id in &kernel.body.layout {
        let inst = kernel.body.insts.get(*id).expect("layout names an instruction");
        match peacemaker_ir::isa::lookup(program.target.arch, inst.op, inst.form) {
            Some(row) if peacemaker_ir::emu::semantic_class(row.name).is_some() => {}
            Some(row) => *missing.entry(row.name).or_default() += 1,
            None => *missing.entry("<undecodable>").or_default() += 1,
        }
    }
    if !missing.is_empty() {
        let list: Vec<String> = missing.iter().map(|(n, c)| format!("{n} x{c}")).collect();
        return Err(fail(format!("no emulator semantics for: {}", list.join(", ")), info));
    }
    let emu = Emulator::new(&program, symbol)
        .map_err(|e| fail(format!("emulator: {e}"), info.clone()))?
        .strict_lds()
        .gfx1100_as_gfx1151();

    let mut mem = Memory::default();
    let mut addrs = Vec::with_capacity(case.bufs.len());
    for b in &case.bufs {
        addrs.push(
            mem.map_guarded(b.name.to_owned(), b.bytes.clone(), GUARD_BYTES)
                .map_err(|e| fail(format!("map {}: {e}", b.name), info.clone()))?,
        );
    }
    let segment = metadata.parsed.kernarg_segment_size.max(descriptor.kernarg_size) as usize;
    let kern = kernarg_bytes(case, &addrs, &metadata.parsed.args, segment, geo)
        .map_err(|e| fail(e, info.clone()))?;
    // The compiler merges neighbouring kernarg dwords into one 16-byte `s_load_b128`, which can span past
    // the declared segment end (the runtime pads kernarg allocations): zero-pad the region to 16 bytes.
    let mut kern = kern;
    kern.resize(kern.len().next_multiple_of(16), 0);
    info["kernarg_bytes_mapped"] = json!(kern.len());
    let kva = mem
        .map_guarded("kernarg".to_owned(), kern, GUARD_BYTES)
        .map_err(|e| fail(format!("map kernarg: {e}"), info.clone()))?;

    let (mut instructions, mut workgroups) = (0u64, 0u64);
    for z in 0..geo.grid[2] {
        for y in 0..geo.grid[1] {
            for x in 0..geo.grid[0] {
                let launch = Launch {
                    kernarg_ptr: kva,
                    workgroup: [x, y, z],
                    block: geo.block,
                    dynamic_lds: geo.lds,
                };
                let r = catch_unwind(AssertUnwindSafe(|| emu.run_workgroup_counted(&launch, &mut mem)));
                match r {
                    Ok(Ok(n)) => {
                        instructions += n;
                        workgroups += 1;
                    }
                    Ok(Err(e)) => {
                        return Err(ArmFail {
                            reason: format!("workgroup ({x},{y},{z}): {e}"),
                            instructions,
                            workgroups,
                            json: info,
                        });
                    }
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                            .unwrap_or_default();
                        return Err(ArmFail {
                            reason: format!("workgroup ({x},{y},{z}): emulator panic: {msg}"),
                            instructions,
                            workgroups,
                            json: info,
                        });
                    }
                }
            }
        }
    }
    let mut violations = Vec::new();
    for (i, b) in case.bufs.iter().enumerate() {
        let r = mem
            .regions
            .iter()
            .find(|r| r.name == b.name)
            .expect("every mapped region is still mapped");
        if i != case.out && r.written.count() != 0 {
            violations.push(format!("wrote {} byte(s) of the read-only input buffer `{}`", r.written.count(), b.name));
        }
    }
    if let Some(r) = mem.regions.iter().find(|r| r.name == "kernarg") {
        if r.written.count() != 0 {
            violations.push(format!("wrote {} byte(s) of the kernarg segment", r.written.count()));
        }
    }
    if !violations.is_empty() {
        return Err(ArmFail { reason: violations.join("; "), instructions, workgroups, json: info });
    }
    let out_region = mem
        .regions
        .iter()
        .find(|r| r.name == case.bufs[case.out].name)
        .expect("the output region is mapped");
    let out = out_region.bytes.clone();
    let out_written = out_region.written.count();
    info["workgroups"] = json!(workgroups);
    info["instructions"] = json!(instructions);
    info["geometry"] = json!({"grid_workgroups": geo.grid, "block": geo.block, "lds_dynamic_bytes": geo.lds});
    info["output_written_bytes"] = json!(out_written);
    info["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    Ok(ArmRun { out, instructions, workgroups, json: info, out_written })
}

/// First differing byte and the number of differing bytes.
fn diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), a.len().abs_diff(b.len())));
    }
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some((first, a.iter().zip(b).filter(|(x, y)| x != y).count()))
}

/// Run both arms (concurrently) and build the `emu_gate` receipt. `pass` is the verdict.
pub fn gate(
    symbol: &str,
    case: &EmuCase,
    base: (&[u8], &Geometry),
    cand: (&[u8], &Geometry),
    shape: Value,
    seed: u64,
) -> (bool, Value) {
    let (b, c) = std::thread::scope(|s| {
        let hb = std::thread::Builder::new()
            .name("emu-baseline".into())
            .stack_size(256 << 20)
            .spawn_scoped(s, || run_arm("baseline", base.0, symbol, base.1, case))
            .expect("spawn baseline emulation");
        let hc = std::thread::Builder::new()
            .name("emu-candidate".into())
            .stack_size(256 << 20)
            .spawn_scoped(s, || run_arm("candidate", cand.0, symbol, cand.1, case))
            .expect("spawn candidate emulation");
        (hb.join(), hc.join())
    });
    let unpack = |r: std::thread::Result<std::result::Result<ArmRun, ArmFail>>, tag: &str| match r {
        Ok(r) => r,
        Err(_) => Err(fail(format!("{tag} emulation thread panicked"), json!({"tag": tag}))),
    };
    let (b, c) = (unpack(b, "baseline"), unpack(c, "candidate"));
    let count = |r: &std::result::Result<ArmRun, ArmFail>| match r {
        Ok(a) => (a.instructions, a.workgroups),
        Err(f) => (f.instructions, f.workgroups),
    };
    let arm_json = |r: &std::result::Result<ArmRun, ArmFail>| match r {
        Ok(a) => a.json.clone(),
        Err(f) => f.json.clone(),
    };
    let (bn, cn) = (count(&b), count(&c));
    let mut rec = json!({
        "pass": false,
        "reason": Value::Null,
        "shape": shape,
        "seed": seed,
        "strict_lds": true,
        "numerics_model": "gfx1100 program decoded with gfx1100 tables; WMMA/transcendental/NaN rules of the gfx1151 model (both arms identical; no claim about gfx1100 hardware ulps)",
        "guard_bytes": GUARD_BYTES,
        "instructions": {"baseline": bn.0, "candidate": cn.0},
        "workgroups": {"baseline": bn.1, "candidate": cn.1},
        "arms": {"baseline": arm_json(&b), "candidate": arm_json(&c)},
        "output_bytes": case.bufs[case.out].bytes.len(),
        "output_sha256": {"baseline": Value::Null, "candidate": Value::Null},
        "identical_to_baseline": Value::Null,
    });
    if let Ok(a) = &b {
        rec["output_sha256"]["baseline"] = json!(sha256_hex(&a.out));
    }
    if let Ok(a) = &c {
        rec["output_sha256"]["candidate"] = json!(sha256_hex(&a.out));
    }
    let (b, c) = match (b, c) {
        (Ok(b), Ok(c)) => (b, c),
        (b, c) => {
            let mut why = Vec::new();
            if let Err(f) = &b {
                why.push(format!("baseline: {}", f.reason));
            }
            if let Err(f) = &c {
                why.push(format!("candidate: {}", f.reason));
            }
            rec["reason"] = json!(why.join(" | "));
            return (false, rec);
        }
    };
    if b.out_written == 0 {
        rec["reason"] = json!("the baseline wrote none of the output buffer: the launch is not exercised");
        return (false, rec);
    }
    let pristine = &case.bufs[case.out].bytes;
    if diff(pristine, &b.out).is_none() {
        rec["reason"] = json!("the baseline left the output buffer unchanged: the launch is not exercised");
        return (false, rec);
    }
    match diff(&b.out, &c.out) {
        None => {
            rec["identical_to_baseline"] = json!(true);
            rec["pass"] = json!(true);
            (true, rec)
        }
        Some((first, n)) => {
            rec["identical_to_baseline"] = json!(false);
            rec["reason"] = json!(format!(
                "output differs from the emulated incumbent: first differing byte {first}, {n} differing byte(s) of {}",
                b.out.len()
            ));
            (false, rec)
        }
    }
}
