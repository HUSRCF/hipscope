// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! In-TUI "doctor": runs `hipfire diag --json` on a background thread and parses
//! its diagnostics object into pass/fail checks the System tab renders.
//!
//! On-demand only (the live-GPU probe spawns the daemon, so it is slow) — never
//! per-frame. The single report arrives on an mpsc channel the App drains, the
//! same pattern as serve control. Honest: a missing field is a failed/!ok check
//! with a reason, never a fabricated pass.

use std::sync::mpsc::{self, Receiver};
use std::thread;

use serde_json::Value;

use crate::hipfire::cli_command;

/// One diagnostic line: a name, whether it passed, and a short detail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoctorCheck {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// Parsed `hipfire diag --json` result, or a transport-level `error`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DoctorReport {
    pub checks: Vec<DoctorCheck>,
    pub error: Option<String>,
}

/// Spawn `bun cli/index.ts diag --json` on a background thread; the single
/// report arrives on the returned receiver.
pub fn run() -> Receiver<DoctorReport> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(run_inner());
    });
    rx
}

fn run_inner() -> DoctorReport {
    let mut cmd = match cli_command() {
        Some(c) => c,
        None => {
            return DoctorReport {
                checks: Vec::new(),
                error: Some(
                    "cli/index.ts not found (set HIPFIRE_CLI_SCRIPT or run from the repo root)"
                        .into(),
                ),
            }
        }
    };
    match cmd.arg("diag").arg("--json").output() {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            // The JSON is the last non-empty line (diag may log warnings first).
            match stdout.lines().rev().find(|l| l.trim_start().starts_with('{')) {
                Some(json) => parse_diag_json(json),
                None => DoctorReport {
                    checks: Vec::new(),
                    error: Some(format!(
                        "diag produced no JSON: {}",
                        String::from_utf8_lossy(&o.stderr)
                            .lines()
                            .next_back()
                            .unwrap_or("")
                            .trim()
                    )),
                },
            }
        }
        Err(e) => DoctorReport {
            checks: Vec::new(),
            error: Some(format!("diag spawn failed: {e}")),
        },
    }
}

fn check(name: &str, ok: bool, detail: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        ok,
        detail: detail.into(),
    }
}

/// Derive pass/fail checks from the `hipfire diag --json` object. Defensive: any
/// missing field becomes a failed check with an honest reason.
pub fn parse_diag_json(body: &str) -> DoctorReport {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return DoctorReport {
                checks: Vec::new(),
                error: Some(format!("diag JSON parse failed: {e}")),
            }
        }
    };

    let mut checks = Vec::new();

    // Platform — informational, always shown ok.
    checks.push(check(
        "platform",
        true,
        v["platform"].as_str().unwrap_or("unknown"),
    ));

    let amdgpu = v["amdgpu_loaded"].as_bool().unwrap_or(false);
    checks.push(check(
        "amdgpu module",
        amdgpu,
        if amdgpu { "loaded" } else { "not loaded" },
    ));

    let kfd = v["kfd"].as_bool().unwrap_or(false);
    checks.push(check(
        "/dev/kfd",
        kfd,
        if kfd { "present" } else { "missing" },
    ));

    let hipcc = v["rocm"]["hipcc"].as_str().filter(|s| !s.is_empty());
    checks.push(check(
        "hipcc (ROCm)",
        hipcc.is_some(),
        hipcc.unwrap_or("not found"),
    ));

    let daemon = v["daemon"].as_str();
    checks.push(check(
        "daemon binary",
        daemon == Some("found"),
        daemon.unwrap_or("missing"),
    ));

    let n_gpus = v["gpus"].as_array().map(|a| a.len()).unwrap_or(0);
    checks.push(check(
        "GPU (PCI)",
        n_gpus > 0,
        format!("{n_gpus} detected"),
    ));

    let n_models = v["models"].as_array().map(|a| a.len()).unwrap_or(0);
    checks.push(check(
        "local models",
        n_models > 0,
        format!("{n_models} present"),
    ));

    // Live GPU probe (daemon one-shot): ok iff an arch came back with no error.
    let gpu = &v["gpu"];
    let gpu_err = gpu.get("error").and_then(Value::as_str);
    let gpu_arch = gpu.get("arch").and_then(Value::as_str);
    let (gpu_ok, gpu_detail) = match (gpu_err, gpu_arch) {
        (Some(e), _) => (false, e.to_string()),
        (None, Some(arch)) => (true, arch.to_string()),
        (None, None) => (false, "no live probe".to_string()),
    };
    checks.push(check("live GPU probe", gpu_ok, gpu_detail));

    DoctorReport {
        checks,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_diag_extracts_pass_fail() {
        let body = r#"{
            "platform": "linux",
            "amdgpu_loaded": true,
            "kfd": true,
            "rocm": { "hipcc": "HIP version 6.2" },
            "daemon": "found",
            "gpus": ["AMD Radeon RX 7900 XTX"],
            "models": [{"name":"q","tag":"q","size":"5 GB"}],
            "gpu": { "arch": "gfx1100", "vram_total_mb": 24560 }
        }"#;
        let r = parse_diag_json(body);
        assert!(r.error.is_none());
        let by = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap().ok;
        assert!(by("amdgpu module"));
        assert!(by("/dev/kfd"));
        assert!(by("hipcc (ROCm)"));
        assert!(by("daemon binary"));
        assert!(by("GPU (PCI)"));
        assert!(by("local models"));
        assert!(by("live GPU probe"));
    }

    #[test]
    fn parse_diag_marks_missing_as_failures() {
        // Honest: absent fields fail, the live probe surfaces its error.
        let body = r#"{
            "platform": "wsl2",
            "amdgpu_loaded": false,
            "kfd": false,
            "rocm": { "hipcc": null },
            "daemon": null,
            "gpus": [],
            "models": [],
            "gpu": { "error": "no device" }
        }"#;
        let r = parse_diag_json(body);
        let c = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap();
        assert!(!c("amdgpu module").ok);
        assert!(!c("hipcc (ROCm)").ok);
        assert!(!c("daemon binary").ok);
        assert!(!c("GPU (PCI)").ok);
        assert!(!c("live GPU probe").ok);
        assert_eq!(c("live GPU probe").detail, "no device");
        assert!(c("platform").ok, "platform is informational");
    }

    #[test]
    fn parse_diag_bad_json_is_error() {
        let r = parse_diag_json("not json");
        assert!(r.error.is_some());
        assert!(r.checks.is_empty());
    }
}
