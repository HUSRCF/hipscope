// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background tailer for `~/.hipfire/serve.log` — the daemon's stdout+stderr.
//!
//! A dedicated thread re-reads the tail of the file (bounded to the last 64 KiB,
//! so a huge log never blocks) only while the Logs tab is focused, exposing the
//! last N lines via a snapshot the UI thread clones each frame. Mirrors the
//! DashboardWorker pattern (Arc<Mutex<snapshot>> + atomic flags + wake channel +
//! Drop-shutdown). Honest states: Missing / Empty / Error are surfaced, never
//! faked.

use std::{
    fs::File,
    io::{ErrorKind, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const MAX_LINES: usize = 400;
const TAIL_WINDOW: u64 = 64 * 1024;
const ACTIVE_POLL: Duration = Duration::from_millis(1000);
const IDLE_POLL: Duration = Duration::from_millis(4000);

/// Why the log isn't showing content (or that it is).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LogStatus {
    #[default]
    Pending,
    Ok,
    Missing,
    Empty,
    Error(String),
}

/// The latest tail snapshot, mirrored to the UI thread.
#[derive(Clone, Debug, Default)]
pub struct LogSnapshot {
    pub lines: Vec<String>,
    pub status: LogStatus,
}

pub struct LogTailer {
    snapshot: Arc<Mutex<LogSnapshot>>,
    active: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    wake: Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl LogTailer {
    pub fn spawn(path: PathBuf) -> Self {
        let snapshot = Arc::new(Mutex::new(LogSnapshot::default()));
        let active = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let (wake, wake_rx) = mpsc::channel::<()>();
        let handle = {
            let snapshot = Arc::clone(&snapshot);
            let active = Arc::clone(&active);
            let shutdown = Arc::clone(&shutdown);
            thread::spawn(move || worker_loop(path, snapshot, active, shutdown, wake_rx))
        };
        Self {
            snapshot,
            active,
            shutdown,
            wake,
            handle: Some(handle),
        }
    }

    /// Mark the Logs tab focused/unfocused; waking the worker on activation so
    /// the tail is fresh immediately.
    pub fn set_active(&self, active: bool) {
        let was = self.active.swap(active, Ordering::SeqCst);
        if active && !was {
            let _ = self.wake.send(());
        }
    }

    /// Cheap clone of the latest tail snapshot for the UI thread.
    pub fn snapshot(&self) -> LogSnapshot {
        self.snapshot
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

impl Drop for LogTailer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    path: PathBuf,
    snapshot: Arc<Mutex<LogSnapshot>>,
    active: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    wake_rx: mpsc::Receiver<()>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        let is_active = active.load(Ordering::SeqCst);
        if is_active {
            let snap = read_tail(&path, MAX_LINES);
            if let Ok(mut g) = snapshot.lock() {
                *g = snap;
            }
        }
        let wait = if is_active { ACTIVE_POLL } else { IDLE_POLL };
        let _ = wake_rx.recv_timeout(wait);
    }
}

/// Read the last `max_lines` lines of `path`, bounded to the final `TAIL_WINDOW`
/// bytes so a large log never causes a big read. Pure (no shared state) so it is
/// directly unit-testable.
pub fn read_tail(path: &Path, max_lines: usize) -> LogSnapshot {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Missing,
            }
        }
        Err(e) => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Error(e.to_string()),
            }
        }
    };
    let len = match f.metadata() {
        Ok(m) => m.len(),
        Err(e) => {
            return LogSnapshot {
                lines: Vec::new(),
                status: LogStatus::Error(e.to_string()),
            }
        }
    };
    if len == 0 {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Empty,
        };
    }
    let from = len.saturating_sub(TAIL_WINDOW);
    if let Err(e) = f.seek(SeekFrom::Start(from)) {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Error(e.to_string()),
        };
    }
    let mut bytes = Vec::new();
    if let Err(e) = f.read_to_end(&mut bytes) {
        return LogSnapshot {
            lines: Vec::new(),
            status: LogStatus::Error(e.to_string()),
        };
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut lines: Vec<&str> = text.lines().collect();
    // If we started mid-file, the first line is a partial fragment — drop it.
    if from > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    let start = lines.len().saturating_sub(max_lines);
    let tail: Vec<String> = lines[start..].iter().map(|s| s.to_string()).collect();
    LogSnapshot {
        lines: tail,
        status: LogStatus::Ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hipfire-logtail-{}-{name}", std::process::id()))
    }

    #[test]
    fn missing_file_is_honest() {
        let p = tmp("nope.log");
        let _ = std::fs::remove_file(&p);
        let s = read_tail(&p, 100);
        assert_eq!(s.status, LogStatus::Missing);
        assert!(s.lines.is_empty());
    }

    #[test]
    fn empty_file_is_empty_status() {
        let p = tmp("empty.log");
        File::create(&p).unwrap();
        let s = read_tail(&p, 100);
        assert_eq!(s.status, LogStatus::Empty);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn tails_last_n_lines() {
        let p = tmp("tail.log");
        let mut f = File::create(&p).unwrap();
        for i in 0..50 {
            writeln!(f, "line {i}").unwrap();
        }
        let s = read_tail(&p, 5);
        assert_eq!(s.status, LogStatus::Ok);
        assert_eq!(s.lines.len(), 5);
        assert_eq!(s.lines.last().unwrap(), "line 49");
        assert_eq!(s.lines.first().unwrap(), "line 45");
        let _ = std::fs::remove_file(&p);
    }
}
