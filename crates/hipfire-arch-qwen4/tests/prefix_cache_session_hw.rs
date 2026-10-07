// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 prefix-cache session oracle on a real Flash-Next artifact.
//!
//! `#[ignore]`d: it needs a real HIP GPU, the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact and the canonical 17-token state
//! corpus manifest. It never skips to a pass: a missing or unreadable
//! environment variable, model, corpus or GPU backend fails the test.
//!
//! Environment:
//!
//! - `HIPFIRE_PREFIX_SESSION_MODEL` (required): path of the `.mq4` artifact.
//! - `HIPFIRE_PREFIX_SESSION_CORPUS` (required): absolute path of
//!   `benchmarks/prompts/qwen4-teacher-forced.tokens.json` (its
//!   `.tokens.u32le` payload sits beside it).
//! - `HIPFIRE_PREFIX_SESSION_OUT` (optional): directory for `report.json`,
//!   `child.out` and `child.err` (default: a fresh directory under the system
//!   temp dir).
//!
//! Run the parent test only (the `--ignored` filter alone would also start the
//! child entry, which does nothing without the parent's marker):
//!
//! ```text
//! HIPFIRE_PREFIX_SESSION_MODEL=/path/qwen3.8-flash-next-gptq3.mq4 \
//! HIPFIRE_PREFIX_SESSION_CORPUS=/path/to/benchmarks/prompts/qwen4-teacher-forced.tokens.json \
//! cargo test --release -p hipfire-arch-qwen4 --features reference-parity \
//!   --test prefix_cache_session_hw -- --ignored --nocapture --test-threads=1 \
//!   --exact prefix_cache_session_matches_oracle_on_flash_next
//! ```
//!
//! The `reference-parity` feature provides `state_parity`; without it the
//! test fails with that instruction instead of vanishing from the run.
//!
//! The oracle (`state_parity::run_prefix_cache_session_parity`) loads one
//! bundle with the shipped state formats and the product's default context,
//! then runs sequential arms on it for AR and native MTP over prompt lengths
//! {1, 5, 63, 64, 65, 511, 513, 1025, 1900, 4097, 8193} and suffix lengths
//! {1, 3, 64, 513, two chunks plus one}:
//!
//! - `prompt_divergence`: a request that decoded and committed, then a prompt
//!   diverging at the end of the first prompt, restores the end-of-prompt
//!   checkpoint. The result must equal, byte for byte, a cold prefill that
//!   committed with no decode and continues live: every target and head state
//!   family, the pending hidden row, the final logits or seed, and the next
//!   greedy ids.
//! - `live_continuation`: committed live state extended by a suffix, run twice
//!   from reset, must be identical; `live_begin_noop` requires `begin_prefix`
//!   with a Live receipt to leave every state family untouched.
//! - Negatives: misaligned commits never bind Live, a checkpoint refuses the
//!   other mode, stale receipts are refused before any write.
//!
//! The child process runs with `HIPFIRE_QWEN_CACHE_TRACE=1`; the parent checks
//! the `[qwen4-prefix] begin source=... mode=...` lines so every
//! source/mode pair (Cold, Live and Prompt, each for Ar and NativeMtp) is
//! proven to have been taken, not just reported by the oracle.
//!
//! The oracle runs in a fresh child process of this test binary, like the
//! greedy and sampled MTP hardware tests: route knobs are read from a
//! process-start snapshot and a GPU fault stays out of the harness.

#[cfg(feature = "reference-parity")]
use std::fs;
#[cfg(feature = "reference-parity")]
use std::path::PathBuf;
#[cfg(feature = "reference-parity")]
use std::process::{Command, Stdio};

#[cfg(feature = "reference-parity")]
const MODEL_ENV: &str = "HIPFIRE_PREFIX_SESSION_MODEL";
#[cfg(feature = "reference-parity")]
const CORPUS_ENV: &str = "HIPFIRE_PREFIX_SESSION_CORPUS";
#[cfg(feature = "reference-parity")]
const OUT_ENV: &str = "HIPFIRE_PREFIX_SESSION_OUT";
#[cfg(feature = "reference-parity")]
const CHILD_ENV: &str = "HIPFIRE_PREFIX_SESSION_CHILD";
#[cfg(feature = "reference-parity")]
const CHILD_TEST: &str = "prefix_cache_session_child";

#[cfg(feature = "reference-parity")]
/// Cases per mode: 3 negatives, then per prompt length one `live_begin_noop`
/// plus a divergence and a determinism case for each of 5 suffix lengths.
const EXPECTED_CASES: usize = 2 * (3 + 11 * (1 + 5 * 2));

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
            std::env::temp_dir().join(format!("qwen4-prefix-session-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

#[cfg(feature = "reference-parity")]
fn trace_count(stderr: &str, source: &str, mode: &str) -> usize {
    let source = format!("source={source} ");
    let mode = format!("mode={mode}");
    stderr
        .lines()
        .filter(|line| {
            line.starts_with("[qwen4-prefix] begin ")
                && line.contains(&source)
                && line.trim_end().ends_with(&mode)
        })
        .count()
}

#[cfg(feature = "reference-parity")]
#[test]
#[ignore = "requires a real HIP GPU, HIPFIRE_PREFIX_SESSION_MODEL (qwen3.8-flash-next-gptq3.mq4) and HIPFIRE_PREFIX_SESSION_CORPUS"]
fn prefix_cache_session_matches_oracle_on_flash_next() {
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
        failed.is_empty() && report["status"] == "pass",
        "prefix-cache session oracle failed {} case(s) (report {}):\n{}",
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
    for source in ["Cold", "Live", "Prompt"] {
        for mode in ["Ar", "NativeMtp"] {
            assert!(
                trace_count(&stderr, source, mode) > 0,
                "no `[qwen4-prefix] begin source={source} ... mode={mode}` trace line; logs in {}",
                dir.display()
            );
        }
    }
    eprintln!(
        "prefix-cache session oracle: {} cases passed on {} (report {})",
        EXPECTED_CASES,
        report["gpu_arch"],
        report_path.display()
    );
}

/// The oracle in a fresh process; spawned by the test above.
#[cfg(feature = "reference-parity")]
#[test]
#[ignore = "child process of prefix_cache_session_matches_oracle_on_flash_next"]
fn prefix_cache_session_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let model = required_file(MODEL_ENV, "qwen3.8-flash-next-gptq3.mq4");
    let corpus = required_file(
        CORPUS_ENV,
        "the canonical benchmarks/prompts/qwen4-teacher-forced.tokens.json",
    );
    let report = hipfire_arch_qwen4::state_parity::run_prefix_cache_session_parity(&model, &corpus)
        .unwrap_or_else(|e| panic!("prefix-cache session oracle: {e}"));
    report
        .write(&out_dir().join("report.json"))
        .unwrap_or_else(|e| panic!("write session report: {e}"));
}

/// Without `reference-parity` the oracle is not compiled: fail loudly rather
/// than leave the test out of the run.
#[cfg(not(feature = "reference-parity"))]
#[test]
#[ignore = "requires --features reference-parity, a real HIP GPU, HIPFIRE_PREFIX_SESSION_MODEL and HIPFIRE_PREFIX_SESSION_CORPUS"]
fn prefix_cache_session_matches_oracle_on_flash_next() {
    panic!(
        "prefix_cache_session_hw needs the `reference-parity` feature: \
         cargo test --release -p hipfire-arch-qwen4 --features reference-parity \
         --test prefix_cache_session_hw -- --ignored --exact prefix_cache_session_matches_oracle_on_flash_next"
    );
}
