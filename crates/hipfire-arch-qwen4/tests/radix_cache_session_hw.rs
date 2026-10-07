// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 radix prefix-cache session oracle on a real Flash-Next artifact.
//!
//! `#[ignore]`d: it needs a real HIP GPU with the VMM QSA context backend, the
//! canonical `qwen3.8-flash-next-gptq3.mq4` artifact and the canonical
//! 17-token state corpus manifest. It never skips to a pass: a missing or
//! unreadable environment variable, model, corpus or GPU backend fails the
//! test.
//!
//! Environment:
//!
//! - `HIPFIRE_RADIX_SESSION_MODEL` (required): path of the `.mq4` artifact.
//! - `HIPFIRE_RADIX_SESSION_CORPUS` (required): absolute path of
//!   `benchmarks/prompts/qwen4-teacher-forced.tokens.json` (its
//!   `.tokens.u32le` payload sits beside it).
//! - `HIPFIRE_RADIX_SESSION_OUT` (optional): directory for `report.json`,
//!   `child.out` and `child.err` (default: a fresh directory under the system
//!   temp dir).
//!
//! Run the parent test only (the `--ignored` filter alone would also start the
//! child entry, which does nothing without the parent's marker):
//!
//! ```text
//! HIPFIRE_RADIX_SESSION_MODEL=/path/qwen3.8-flash-next-gptq3.mq4 \
//! HIPFIRE_RADIX_SESSION_CORPUS=/path/to/benchmarks/prompts/qwen4-teacher-forced.tokens.json \
//! cargo test --release -p hipfire-arch-qwen4 --features reference-parity \
//!   --test radix_cache_session_hw -- --ignored --nocapture --test-threads=1 \
//!   --exact radix_cache_session_matches_oracle_on_flash_next
//! ```
//!
//! The `reference-parity` feature provides `state_parity`; without it the
//! test fails with that instruction instead of vanishing from the run.
//!
//! The oracle (`state_parity::run_radix_cache_session_parity`) loads fresh
//! bundles with the shipped state formats, the product's default context and
//! the VMM backend, attaches the prefix cache and a radix store under explicit
//! limits, and drives requests as production does (`select_prefix_plan`,
//! `begin_prefix` through the AR or native-MTP prefill, a short decode,
//! `commit_prefix`). For AR and native MTP it reports, per mode:
//!
//! - `cold_identity`: the first cold request equals a bundle with no prefix
//!   cache attached;
//! - `cross_session` and `a_b_a`: a checkpoint published by one session
//!   serves a later one after an unrelated request, equal to "cold prefill of
//!   the checkpoint prompt committed with no decode, then the live
//!   continuation of the suffix" on every state family, the final logits or
//!   seed and the next greedy ids;
//! - `anchor`: the first cold session splits nothing, the second observation
//!   of a shared turn splits and captures at it, the third hits it;
//! - `siblings`: repeated forks of one checkpoint alias its granules, never
//!   repeat a base VA, repeat byte for byte and leave the source reusable;
//! - `boundary_fork`: end-of-prompt checkpoints at 1, 3, 4, 5, 127, 128, 129,
//!   4095, 4096, 4097, 8191, 8192 and 8193 tokens, each forked with a 1-token
//!   and a 513-token suffix;
//! - `pressure`: two checkpoints and a small device cap: GDSF eviction, the
//!   ledger within the cap, a pinned checkpoint surviving, and the live VMM
//!   granule bytes returning to their pre-load value after unload.
//!
//! The child process runs with `HIPFIRE_QWEN_CACHE_TRACE=1`; the parent checks
//! the `[qwen4-radix] begin source=Radix` and fork lines and the
//! `stage at=.. kind=anchor` line so the radix restore, the fork bank and the
//! anchor split are proven to have been taken, not just reported by the oracle.
//!
//! The oracle runs in a fresh child process of this test binary, like the
//! prefix-cache session test: route knobs are read from a process-start
//! snapshot and a GPU fault stays out of the harness.

#[cfg(feature = "reference-parity")]
use std::fs;
#[cfg(feature = "reference-parity")]
use std::path::PathBuf;
#[cfg(feature = "reference-parity")]
use std::process::{Command, Stdio};

#[cfg(feature = "reference-parity")]
const MODEL_ENV: &str = "HIPFIRE_RADIX_SESSION_MODEL";
#[cfg(feature = "reference-parity")]
const CORPUS_ENV: &str = "HIPFIRE_RADIX_SESSION_CORPUS";
#[cfg(feature = "reference-parity")]
const OUT_ENV: &str = "HIPFIRE_RADIX_SESSION_OUT";
#[cfg(feature = "reference-parity")]
const CHILD_ENV: &str = "HIPFIRE_RADIX_SESSION_CHILD";
#[cfg(feature = "reference-parity")]
const CHILD_TEST: &str = "radix_cache_session_child";

#[cfg(feature = "reference-parity")]
/// Cases per mode: `cold_identity`, `cross_session`, `a_b_a`, `anchor`,
/// `siblings`, one `boundary_fork` per (13 prompt lengths x 2 suffixes) and
/// `pressure`.
const EXPECTED_CASES: usize = 2 * (5 + 13 * 2 + 1);

#[cfg(feature = "reference-parity")]
fn required_file(name: &str, what: &str) -> PathBuf {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must name {what}"));
    assert!(path.is_file(), "{name}={} is not a file", path.display());
    path
}

#[cfg(feature = "reference-parity")]
fn out_dir() -> PathBuf {
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen4-radix-session-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// Lines of `stderr` that start with `prefix` and contain every needle.
#[cfg(feature = "reference-parity")]
fn trace_lines(stderr: &str, prefix: &str, needles: &[&str]) -> usize {
    stderr
        .lines()
        .filter(|line| line.starts_with(prefix) && needles.iter().all(|n| line.contains(n)))
        .count()
}

#[cfg(feature = "reference-parity")]
#[test]
#[ignore = "requires a real HIP GPU, HIPFIRE_RADIX_SESSION_MODEL (qwen3.8-flash-next-gptq3.mq4) and HIPFIRE_RADIX_SESSION_CORPUS"]
fn radix_cache_session_matches_oracle_on_flash_next() {
    let model = required_file(MODEL_ENV, "qwen3.8-flash-next-gptq3.mq4");
    let corpus = required_file(
        CORPUS_ENV,
        "the canonical benchmarks/prompts/qwen4-teacher-forced.tokens.json",
    );
    let dir = out_dir();
    let report_path = dir.join("report.json");
    let _ = fs::remove_file(&report_path);
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(&exe)
        .args([
            "--exact",
            CHILD_TEST,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env(MODEL_ENV, &model)
        .env(CORPUS_ENV, &corpus)
        .env(OUT_ENV, &dir)
        .env("HIPFIRE_QWEN_CACHE_TRACE", "1")
        .env_remove("HIPFIRE_QWEN_PROMPT_CACHE")
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("spawn session child: {e}"));
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::write(dir.join("child.out"), &output.stdout).unwrap();
    fs::write(dir.join("child.err"), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "session child failed ({}); logs in {}",
        output.status,
        dir.display()
    );
    let report: serde_json::Value = serde_json::from_slice(
        &fs::read(&report_path).unwrap_or_else(|e| panic!("read {}: {e}", report_path.display())),
    )
    .expect("session report json");
    let session = &report["session"];
    let failed: Vec<&str> = session["failed"]
        .as_array()
        .expect("session report has no failed list")
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert!(
        failed.is_empty() && report["passed"] == true && report["status"] == "pass",
        "radix-cache session oracle failed {} case(s) (report {}):\n{}",
        failed.len(),
        report_path.display(),
        failed.join("\n")
    );
    assert_eq!(
        session["total"].as_u64(),
        Some(EXPECTED_CASES as u64),
        "session oracle ran an unexpected number of cases (report {})",
        report_path.display()
    );
    for (what, prefix, needles) in [
        (
            "radix restore (`[qwen4-radix] begin source=Radix`)",
            "[qwen4-radix] begin ",
            &["source=Radix "][..],
        ),
        (
            "fork bank (`[qwen4-radix] begin ... fork=true`)",
            "[qwen4-radix] begin ",
            &["fork=true"][..],
        ),
        (
            "anchor capture (`[qwen4-radix] stage at=.. kind=anchor`)",
            "[qwen4-radix] stage at=",
            &["kind=anchor"][..],
        ),
    ] {
        assert!(
            trace_lines(&stderr, prefix, needles) > 0,
            "no {what} trace line in the child's stderr; logs in {}",
            dir.display()
        );
    }
    for mode in ["Ar", "NativeMtp"] {
        let mode = format!("mode={mode}");
        assert!(
            trace_lines(&stderr, "[qwen4-radix] begin ", &["source=Radix ", &mode]) > 0,
            "no `[qwen4-radix] begin source=Radix ... {mode}` trace line; logs in {}",
            dir.display()
        );
    }
    eprintln!(
        "radix-cache session oracle: {} cases passed on {} (report {})",
        EXPECTED_CASES,
        report["gpu_arch"],
        report_path.display()
    );
}

/// The oracle in a fresh process; spawned by the test above.
#[cfg(feature = "reference-parity")]
#[test]
#[ignore = "child process of radix_cache_session_matches_oracle_on_flash_next"]
fn radix_cache_session_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let model = required_file(MODEL_ENV, "qwen3.8-flash-next-gptq3.mq4");
    let corpus = required_file(
        CORPUS_ENV,
        "the canonical benchmarks/prompts/qwen4-teacher-forced.tokens.json",
    );
    let report = hipfire_arch_qwen4::state_parity::run_radix_cache_session_parity(&model, &corpus)
        .unwrap_or_else(|e| panic!("radix-cache session oracle: {e}"));
    report
        .write(&out_dir().join("report.json"))
        .unwrap_or_else(|e| panic!("write session report: {e}"));
}

/// Without `reference-parity` the oracle is not compiled: fail loudly rather
/// than leave the test out of the run.
#[cfg(not(feature = "reference-parity"))]
#[test]
#[ignore = "requires --features reference-parity, a real HIP GPU, HIPFIRE_RADIX_SESSION_MODEL and HIPFIRE_RADIX_SESSION_CORPUS"]
fn radix_cache_session_matches_oracle_on_flash_next() {
    panic!(
        "radix_cache_session_hw needs the `reference-parity` feature: \
         cargo test --release -p hipfire-arch-qwen4 --features reference-parity \
         --test radix_cache_session_hw -- --ignored --exact radix_cache_session_matches_oracle_on_flash_next"
    );
}
