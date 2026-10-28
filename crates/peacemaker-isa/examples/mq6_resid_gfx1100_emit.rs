// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emitter + certifier of the gfx1100 MQ6 residual twin
//! (`qwen4_mq6_x4_gfx11` builder, `Kind::Resid`): one single-kernel object per `wr` (4, 8), each
//! exporting hipcc's incumbent symbol `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds` so it is a
//! drop-in candidate object for `hip_loop --node mq6_res --cand-object`.
//!
//! ```text
//! nice -n 19 cargo run --release -j 8 -p peacemaker-isa --features toolchain --example mq6_resid_gfx1100_emit -- <out-dir>
//! ```
//!
//! Writes `<out-dir>/w{4,8}/mq6_resid_w{4,8}.{s,proof.json,co,m7.json}` and `<out-dir>/emit.json`
//! (resources, `.text` sha256 per object). Every stage is a hard stop: wait replay, empty hazard
//! replay, native assembly, M7 on the native object, ROCm `llvm-mc` + `ld.lld` oracle build of the
//! same source with M7 on it, and a `.text` byte comparison of native vs oracle.
#![allow(dead_code)]

#[cfg(not(feature = "toolchain"))]
fn main() {
    println!(
        "{}",
        serde_json::json!({"verdict": "kill", "stage": "features", "reason": "build with --features toolchain (M7, ROCm oracle build)"})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
use peacemaker_isa::{
    kernels::qwen4_mq6_x4_gfx11::{self as mq6, Cfg, Fmt, Kind},
    ledger_replay, native, pm_check,
    toolchain::{build, Toolchain},
    Arch,
};
#[cfg(feature = "toolchain")]
use serde_json::{json, Value};
#[cfg(feature = "toolchain")]
use sha2::{Digest, Sha256};
#[cfg(feature = "toolchain")]
use std::path::{Path, PathBuf};

#[cfg(feature = "toolchain")]
const ARCH: Arch = Arch::Gfx1100;

#[cfg(feature = "toolchain")]
fn kill(stage: &str, reason: impl std::fmt::Display) -> ! {
    println!(
        "{}",
        json!({"verdict": "kill", "stage": stage, "reason": reason.to_string()})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "toolchain")]
fn put(stage: &str, path: &Path, bytes: impl AsRef<[u8]>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| kill(stage, format!("{}: {e}", parent.display())))
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| kill(stage, format!("{}: {e}", path.display())));
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
#[cfg(feature = "toolchain")]
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes
        .windows(4)
        .position(|w| w == b"\x7fELF")
        .unwrap_or_else(|| kill("text", "no ELF"));
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap_or_else(|| kill("text", "no .text"))
}

/// First `.amdhsa_<key> N` value of the source.
#[cfg(feature = "toolchain")]
fn directive(s: &str, key: &str) -> u32 {
    s.lines()
        .find_map(|l| l.trim().strip_prefix(key))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| kill("resources", format!("missing {key}")))
}

#[cfg(feature = "toolchain")]
fn main() {
    let usage = "usage: mq6_resid_gfx1100_emit <out-dir>";
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| kill("usage", usage)));
    let toolchain = Toolchain::oracle();
    let mut rec = serde_json::Map::new();
    for wr in Cfg::RESID {
        let cfg = Cfg {
            arch: ARCH,
            wr,
            kind: Kind::Resid,
            fmt: Fmt::Mq6,
        };
        let symbol = cfg.symbol();
        let stem = format!("mq6_resid_w{wr}");
        let dir = out.join(format!("w{wr}"));
        let e = mq6::emit_cfg(cfg).unwrap_or_else(|e| kill("emit", format!("w{wr}: {e}")));
        ledger_replay::replay_waits(&e.s_text, ARCH)
            .unwrap_or_else(|er| kill("wait_replay", format!("w{wr}: {er}")));
        let hz = ledger_replay::replay_hazards(&e.s_text, ARCH)
            .unwrap_or_else(|er| kill("hazard_replay", format!("w{wr}: {er}")));
        if !hz.is_empty() {
            kill("hazard_replay", format!("w{wr}: {} hazards: {hz:?}", hz.len()))
        }
        put("output", &dir.join(format!("{stem}.s")), &e.s_text);
        put(
            "output",
            &dir.join(format!("{stem}.proof.json")),
            serde_json::to_vec_pretty(&e.proof).unwrap(),
        );
        let elf = native::assemble(&e.s_text, ARCH).unwrap_or_else(|er| kill("native_assemble", er));
        let co = dir.join(format!("{stem}.co"));
        put("output", &co, &elf);
        let m7n = pm_check::m7(&co, ARCH.name(), &symbol)
            .unwrap_or_else(|er| kill("m7_native", format!("w{wr} {symbol}: {er}")));
        if m7n["lift"].as_str() != Some("byte-exact") || m7n["obligations"] != json!({}) {
            kill("m7_native", format!("w{wr}: {m7n}"))
        }
        // ROCm oracle build of the same source.
        let odir = dir.join("oracle");
        let os = odir.join(format!("{stem}.s"));
        put("output", &os, &e.s_text);
        let built = build(&toolchain, &os, &odir.join(format!("{stem}.hsaco")), ARCH.name())
            .unwrap_or_else(|er| kill("oracle_build", format!("w{wr}: {er}")));
        let m7o = pm_check::m7(&built.elf, ARCH.name(), &symbol)
            .unwrap_or_else(|er| kill("m7_oracle", format!("w{wr} {symbol}: {er}")));
        if m7o["lift"].as_str() != Some("byte-exact") || m7o["obligations"] != json!({}) {
            kill("m7_oracle", format!("w{wr}: {m7o}"))
        }
        let (tn, to) = (
            text_section(&elf),
            text_section(&std::fs::read(&built.elf).unwrap()),
        );
        if tn != to {
            kill("text_identity", format!("w{wr}: native and ROCm .text differ"))
        }
        put(
            "output",
            &dir.join(format!("{stem}.m7.json")),
            serde_json::to_vec_pretty(&json!({"native": m7n, "oracle": m7o})).unwrap(),
        );
        rec.insert(
            format!("w{wr}"),
            json!({
                "symbol": symbol, "co": co.display().to_string(), "co_sha256": sha256(&elf),
                "text_sha256": sha256(&tn), "text_bytes": tn.len(),
                "s_text_sha256": e.proof.s_text_sha256,
                "vgprs": directive(&e.s_text, ".amdhsa_next_free_vgpr"),
                "sgprs": directive(&e.s_text, ".amdhsa_next_free_sgpr"),
                "lds_bytes": directive(&e.s_text, ".amdhsa_group_segment_fixed_size"),
                "workgroup_size": 32 * u32::from(wr),
                "geometry": format!("tile_m={} tile_n=256 block={}", 16 * u32::from(wr), 32 * u32::from(wr)),
                "m7_native": m7n, "m7_oracle_text_identical": true,
            }),
        );
    }
    let rec = Value::Object(rec);
    put(
        "output",
        &out.join("emit.json"),
        serde_json::to_vec_pretty(&rec).unwrap(),
    );
    println!("{}", json!({"verdict": "pass", "records": rec}));
}
