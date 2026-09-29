// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! `HIPFIRE_REDLINE_GAP_TIMING=1`: per-token host breakdown of the plain-AR
//! decode step (embedding launch, position copy, retained-PM4 patch and
//! submit/wait with its in-IB GPU span, or the HIP-graph launch), plus the
//! host time between consecutive decode steps. Diagnostic only; the disabled
//! path reads no clocks. One summary line per 128 recorded tokens on stderr.

use std::cell::RefCell;
use std::sync::LazyLock;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub enum Slot {
    Embed = 0,
    PosCopy,
    Patch,
    SubmitWait,
    GpuSpan,
    GraphLaunch,
    Forward,
    Between,
}

const SLOTS: usize = 8;
const NAMES: [&str; SLOTS] = [
    "embed", "pos_copy", "patch", "submit_wait", "gpu_span", "graph_launch", "forward", "between",
];
const WINDOW: usize = 128;

struct State {
    current: [u64; SLOTS],
    tokens: Vec<[u64; SLOTS]>,
    last_end: Option<Instant>,
}

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::process_value("HIPFIRE_REDLINE_GAP_TIMING").as_deref() == Some("1")
});
thread_local! {
    // The decode step runs on one thread; per-thread state needs no lock.
    static STATE: RefCell<State> = const {
        RefCell::new(State {
            current: [0; SLOTS],
            tokens: Vec::new(),
            last_end: None,
        })
    };
}

pub fn enabled() -> bool {
    *ENABLED
}

pub fn now() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub fn add_since(slot: Slot, start: Option<Instant>) {
    if let Some(start) = start {
        add_ns(slot, start.elapsed().as_nanos() as u64);
    }
}

pub fn add_ns(slot: Slot, ns: u64) {
    if !enabled() {
        return;
    }
    STATE.with_borrow_mut(|state| state.current[slot as usize] += ns);
}

/// Start of one decode step: records the host time since the previous step ended.
pub fn begin_forward() -> Option<Instant> {
    let start = now()?;
    STATE.with_borrow_mut(|state| {
        state.current = [0; SLOTS];
        if let Some(last) = state.last_end {
            state.current[Slot::Between as usize] = start.duration_since(last).as_nanos() as u64;
        }
    });
    Some(start)
}

/// End of one decode step on `route`; prints a summary every 128 steps.
pub fn end_forward(start: Option<Instant>, route: &str) {
    let Some(start) = start else {
        return;
    };
    let end = Instant::now();
    let tokens = STATE.with_borrow_mut(|state| {
        state.current[Slot::Forward as usize] = end.duration_since(start).as_nanos() as u64;
        let record = state.current;
        state.tokens.push(record);
        state.last_end = Some(end);
        (state.tokens.len() >= WINDOW).then(|| std::mem::take(&mut state.tokens))
    });
    let Some(tokens) = tokens else {
        return;
    };
    let mut line = format!("[gap-timing] route={route} n={}", tokens.len());
    for (slot, name) in NAMES.iter().enumerate() {
        let mut values = tokens.iter().map(|t| t[slot]).collect::<Vec<_>>();
        values.sort_unstable();
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        let median = values[values.len() / 2];
        line.push_str(&format!(
            " {name}_us={:.2}/{:.2}",
            mean / 1e3,
            median as f64 / 1e3
        ));
    }
    eprintln!("{line}  (mean/median; between excludes the first step)");
}
