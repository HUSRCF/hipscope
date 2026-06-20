// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background runners for Models-tab actions: pull (with live progress) and rm.
//!
//! Both shell out to `bun cli/index.ts <...>` on a dedicated thread so the UI
//! thread never blocks. `pull` streams the CLI's stderr progress line (which is
//! carriage-return-updated, so we split on '\r' AND '\n') and reports a parsed
//! percentage; `remove` runs `rm <tag> --yes` (the TUI does its own y/n confirm
//! first) and reports a single outcome. Events arrive on mpsc channels the App
//! drains each frame.

use std::{
    env,
    io::Read,
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

/// Streamed events from a `pull` run.
#[derive(Clone, Debug, PartialEq)]
pub enum PullEvent {
    /// A progress update: parsed percent (None if not yet sampled) + the raw
    /// CLI progress line (bar / rate / ETA / bytes) for display.
    Progress { percent: Option<f64>, line: String },
    Done,
    Failed(String),
}

/// Outcome of a one-shot model action (currently `remove`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RmOutcome {
    Ok(String),
    Failed(String),
}

/// Spawn `bun cli/index.ts pull <tag>`, streaming progress on the returned
/// receiver and finishing with Done or Failed.
pub fn pull(tag: String) -> Receiver<PullEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || pull_inner(tag, tx));
    rx
}

/// Spawn `bun cli/index.ts rm <tag> --yes`; the single outcome arrives on the
/// returned receiver.
pub fn remove(tag: String) -> Receiver<RmOutcome> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(remove_inner(&tag));
    });
    rx
}

/// Extract the percentage from a CLI progress line like
/// `[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB` — find the '%' and
/// read the trailing number before it (robust to the bar's internal spaces).
pub fn parse_percent(line: &str) -> Option<f64> {
    let pct = line.find('%')?;
    let num: String = line[..pct]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    num.parse::<f64>().ok()
}

fn cli_script() -> Result<(std::path::PathBuf, std::path::PathBuf), String> {
    let cwd = env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    let script = cwd.join("cli/index.ts");
    if !script.exists() {
        return Err("cli/index.ts not found; run hipfire from the repo root".into());
    }
    Ok((cwd, script))
}

fn pull_inner(tag: String, tx: Sender<PullEvent>) {
    let (cwd, script) = match cli_script() {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(e));
            return;
        }
    };
    let mut child = match Command::new("bun")
        .arg(&script)
        .arg("pull")
        .arg(&tag)
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(format!("spawn: {e}")));
            return;
        }
    };

    // Drain stdout on its own thread so a full pipe can't deadlock the child.
    let stdout_handle = child.stdout.take().map(|mut so| {
        thread::spawn(move || {
            let mut s = String::new();
            let _ = so.read_to_string(&mut s);
            s
        })
    });

    // Read stderr byte-by-byte, flushing a "line" on each '\r' or '\n' (the CLI
    // overwrites one line with '\r' and only emits '\n' at completion).
    if let Some(mut stderr) = child.stderr.take() {
        let mut buf: Vec<u8> = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match stderr.read(&mut byte) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    if byte[0] == b'\r' || byte[0] == b'\n' {
                        flush_progress(&mut buf, &tx);
                    } else {
                        buf.push(byte[0]);
                    }
                }
                Err(_) => break,
            }
        }
        flush_progress(&mut buf, &tx);
    }

    let status = child.wait();
    let stdout = stdout_handle
        .map(|h| h.join().unwrap_or_default())
        .unwrap_or_default();
    match status {
        Ok(s) if s.success() => {
            let _ = tx.send(PullEvent::Done);
        }
        Ok(_) => {
            let msg = stdout
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("pull failed")
                .trim()
                .to_string();
            let _ = tx.send(PullEvent::Failed(msg));
        }
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(format!("wait: {e}")));
        }
    }
}

fn flush_progress(buf: &mut Vec<u8>, tx: &Sender<PullEvent>) {
    if buf.is_empty() {
        return;
    }
    let line = String::from_utf8_lossy(buf).trim().to_string();
    buf.clear();
    if !line.is_empty() {
        let percent = parse_percent(&line);
        let _ = tx.send(PullEvent::Progress { percent, line });
    }
}

fn remove_inner(tag: &str) -> RmOutcome {
    let (cwd, script) = match cli_script() {
        Ok(v) => v,
        Err(e) => return RmOutcome::Failed(e),
    };
    let output = Command::new("bun")
        .arg(&script)
        .arg("rm")
        .arg(tag)
        .arg("--yes")
        .current_dir(&cwd)
        .output();
    match output {
        Ok(o) if o.status.success() => RmOutcome::Ok(format!("removed {tag}")),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail = err
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("rm failed")
                .trim();
            RmOutcome::Failed(format!("rm {tag}: {tail}"))
        }
        Err(e) => RmOutcome::Failed(format!("rm {tag}: spawn failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_percent_handles_real_line() {
        assert_eq!(
            parse_percent("[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB"),
            Some(45.2)
        );
        assert_eq!(parse_percent("[          ]   0.0%   —   123/272 MB"), Some(0.0));
        assert_eq!(parse_percent("[██████████] 100.0% done"), Some(100.0));
        assert_eq!(parse_percent("no percent here"), None);
    }
}
