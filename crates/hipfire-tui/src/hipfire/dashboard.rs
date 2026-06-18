// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! O3c-1 live serve Dashboard data layer.
//!
//! The PARSE logic (HTTP JSON → struct, rocm-smi text → VRAM) is factored into
//! pure, I/O-free helpers so it can be unit-tested on CPU without a TTY, a
//! running serve, or a GPU. The fetchers ([`fetch_dashboard`]) do the network /
//! subprocess I/O and then hand raw text to those pure helpers.
//!
//! Honesty contract: every field is either REAL data parsed from the live
//! serve / rocm-smi, or an explicit unavailable/offline state. We NEVER
//! fabricate placeholder numbers and present them as live.

use std::{
    io::Read,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::Value;

use super::config::ConfigState;

/// Hard wall-clock bound on a single `rocm-smi` invocation. Even on the
/// background thread a zombie / hung rocm-smi must not accumulate forever, so
/// we spawn it, poll for completion, and `kill()` it past this deadline.
const ROCM_SMI_TIMEOUT: Duration = Duration::from_millis(2000);

/// Per-GPU VRAM reading parsed from `rocm-smi`. Values are bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMem {
    /// rocm-smi card index (the `card0`/GPU[0] ordinal).
    pub index: u32,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

impl GpuMem {
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / 1e9
    }
    pub fn free_gb(&self) -> f64 {
        self.free_bytes() as f64 / 1e9
    }
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1e9
    }
}

/// VRAM availability state — distinguishes "we have real per-GPU numbers" from
/// the honest "rocm-smi is absent / not Linux / returned garbage" case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VramState {
    /// Real per-GPU readings parsed from rocm-smi.
    Available(Vec<GpuMem>),
    /// rocm-smi missing, non-Linux, or unparseable. Carries an honest reason.
    Unavailable(String),
}

/// Stats parsed from the serve's GET /stats endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeStats {
    pub model: Option<String>,
    pub uptime_s: u64,
    pub queue_depth: u64,
    pub requests_served: u64,
    pub recent_tok_s: Option<f64>,
}

/// A full Dashboard snapshot. `serve_up` is the single source of truth for the
/// online/offline split the UI renders.
#[derive(Clone, Debug)]
pub struct Dashboard {
    pub serve_up: bool,
    pub endpoint: String,
    /// Loaded model tag, from /stats or /health (None when idle / serve down).
    pub model: Option<String>,
    pub stats: Option<ServeStats>,
    /// Models advertised by GET /v1/models (ids).
    pub model_ids: Vec<String>,
    pub vram: VramState,
    /// When serve is down: an honest hint, e.g. how to start it.
    pub offline_hint: Option<String>,
}

impl Dashboard {
    /// Offline dashboard: serve unreachable. Carries a start hint and whatever
    /// VRAM we could still read (rocm-smi works without serve).
    pub fn offline(endpoint: String, vram: VramState) -> Self {
        Self {
            serve_up: false,
            endpoint,
            model: None,
            stats: None,
            model_ids: Vec::new(),
            vram,
            offline_hint: Some("serve offline — start it with `hipfire serve -d`".into()),
        }
    }
}

// ── Pure parse helpers (no I/O) ─────────────────────────────────────────────

/// Parse the GET /health JSON body. Returns the loaded-model field (which may
/// be JSON null → None). Returns None for garbage / non-object input rather
/// than fabricating a value.
pub fn parse_health_model(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Parse the GET /stats JSON body into [`ServeStats`]. Returns None on garbage
/// / non-object input. Missing numeric fields default to 0 (the serve always
/// emits them, but be defensive against an older build); `recent_tok_s` stays
/// None unless present and finite/positive.
pub fn parse_stats(body: &str) -> Option<ServeStats> {
    let v: Value = serde_json::from_str(body).ok()?;
    if !v.is_object() {
        return None;
    }
    let model = match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let as_u64 = |key: &str| v.get(key).and_then(|x| x.as_u64()).unwrap_or(0);
    let recent_tok_s = v
        .get("recent_tok_s")
        .and_then(|x| x.as_f64())
        .filter(|f| f.is_finite() && *f > 0.0);
    Some(ServeStats {
        model,
        uptime_s: as_u64("uptime_s"),
        queue_depth: as_u64("queue_depth"),
        requests_served: as_u64("requests_served"),
        recent_tok_s,
    })
}

/// Parse the GET /v1/models JSON body into the list of model ids. Empty vec on
/// garbage input.
pub fn parse_model_ids(body: &str) -> Vec<String> {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `rocm-smi --showmeminfo vram --json` output. ROCm emits a JSON object
/// keyed by `card0`, `card1`, … with `"VRAM Total Memory (B)"` and
/// `"VRAM Total Used Memory (B)"` string fields. Returns an honest
/// [`VramState::Unavailable`] when nothing parses.
pub fn parse_rocm_smi_json(body: &str) -> VramState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return VramState::Unavailable(format!("rocm-smi JSON parse failed: {e}")),
    };
    let Some(obj) = v.as_object() else {
        return VramState::Unavailable("rocm-smi JSON not an object".into());
    };
    let mut gpus: Vec<GpuMem> = Vec::new();
    // Iterate cardN keys in numeric order.
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    for (idx, card) in cards {
        let total = field_u64(card, "VRAM Total Memory (B)");
        let used = field_u64(card, "VRAM Total Used Memory (B)");
        if let (Some(total), Some(used)) = (total, used) {
            if total > 0 {
                gpus.push(GpuMem {
                    index: idx,
                    used_bytes: used,
                    total_bytes: total,
                });
            }
        }
    }
    if gpus.is_empty() {
        VramState::Unavailable("rocm-smi returned no VRAM fields".into())
    } else {
        VramState::Available(gpus)
    }
}

/// rocm-smi reports numbers as JSON strings; read either a string-encoded or a
/// raw number.
fn field_u64(card: &Value, key: &str) -> Option<u64> {
    match card.get(key)? {
        Value::String(s) => s.trim().parse::<u64>().ok(),
        Value::Number(n) => n.as_u64(),
        _ => None,
    }
}

// ── Fetchers (I/O; thin wrappers around the pure parsers) ───────────────────

fn get_text(agent: &ureq::Agent, url: &str) -> Option<String> {
    match agent.get(url).call() {
        Ok(resp) if resp.status() < 400 => resp.into_string().ok(),
        _ => None,
    }
}

/// Outcome of a bounded rocm-smi run. Separated from parsing so the timeout /
/// kill path stays testable in isolation.
enum RocmSmiRun {
    /// Process exited within the deadline; carries captured stdout.
    Output(String),
    /// Binary not present on PATH.
    NotFound,
    /// Spawn or wait failed for some other reason.
    SpawnError(String),
    /// Process exceeded [`ROCM_SMI_TIMEOUT`] and was killed.
    TimedOut,
}

/// Run `rocm-smi --showmeminfo vram --json` with a hard wall-clock deadline.
/// Spawns the child, polls `try_wait`, and `kill()`s it if it blows past
/// [`ROCM_SMI_TIMEOUT`] so a hung rocm-smi can never wedge the caller (even on
/// the background fetch thread, where it would otherwise leak a zombie probe).
fn run_rocm_smi_bounded() -> RocmSmiRun {
    let mut cmd = Command::new("rocm-smi");
    cmd.args(["--showmeminfo", "vram", "--json"]);
    run_bounded(cmd, ROCM_SMI_TIMEOUT)
}

/// Spawn `cmd`, capturing stdout, and enforce a hard `timeout`. Polls
/// `try_wait`; on deadline it `kill()`s and reaps the child so no zombie leaks.
/// Generic over the command so the timeout/kill path is unit-testable with a
/// stand-in (`sleep`) without depending on rocm-smi.
fn run_bounded(mut cmd: Command, timeout: Duration) -> RocmSmiRun {
    let mut child = match cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RocmSmiRun::NotFound,
        Err(e) => return RocmSmiRun::SpawnError(e.to_string()),
    };

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                // Exited within the deadline — drain stdout.
                let mut buf = String::new();
                if let Some(mut out) = child.stdout.take() {
                    let _ = out.read_to_string(&mut buf);
                }
                return RocmSmiRun::Output(buf);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Hung: kill and reap so we don't leak a zombie. Best-effort
                    // — even if kill races the exit, we report TimedOut.
                    let _ = child.kill();
                    let _ = child.wait();
                    return RocmSmiRun::TimedOut;
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return RocmSmiRun::SpawnError(e.to_string()),
        }
    }
}

/// Probe rocm-smi for per-GPU VRAM. Honest [`VramState::Unavailable`] when the
/// platform is not Linux, the binary is absent, the output doesn't parse, or
/// the call exceeds [`ROCM_SMI_TIMEOUT`] (hard-killed). Never blocks
/// indefinitely.
pub fn fetch_vram() -> VramState {
    if !cfg!(target_os = "linux") {
        return VramState::Unavailable("VRAM unavailable: rocm-smi is Linux-only".into());
    }
    match run_rocm_smi_bounded() {
        RocmSmiRun::Output(text) => {
            let parsed = parse_rocm_smi_json(&text);
            // If JSON failed but the binary ran, surface a clean reason.
            match parsed {
                VramState::Available(_) => parsed,
                VramState::Unavailable(_) if text.trim().is_empty() => {
                    VramState::Unavailable("VRAM unavailable: rocm-smi produced no output".into())
                }
                other => other,
            }
        }
        RocmSmiRun::NotFound => {
            VramState::Unavailable("VRAM unavailable: rocm-smi not installed".into())
        }
        RocmSmiRun::TimedOut => VramState::Unavailable(format!(
            "VRAM unavailable: rocm-smi timed out after {}ms (killed)",
            ROCM_SMI_TIMEOUT.as_millis()
        )),
        RocmSmiRun::SpawnError(e) => {
            VramState::Unavailable(format!("VRAM unavailable: rocm-smi error: {e}"))
        }
    }
}

/// Build a full live Dashboard snapshot: probes /health, /stats, /v1/models,
/// and rocm-smi. Tolerant of serve being down (returns [`Dashboard::offline`])
/// and of rocm-smi being absent (VramState::Unavailable). All timeouts are
/// short so the refresh tick never wedges the UI.
pub fn fetch_dashboard(config: &ConfigState) -> Dashboard {
    let endpoint = format!("{}:{}", config.probe_host(), config.port);
    let vram = fetch_vram();

    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(450))
        .build();
    let base = format!("http://{endpoint}");

    let health = get_text(&agent, &format!("{base}/health"));
    if health.is_none() {
        // Serve unreachable — honest offline state (VRAM may still be present).
        return Dashboard::offline(endpoint, vram);
    }
    let health = health.unwrap();
    let health_model = parse_health_model(&health);

    let stats = get_text(&agent, &format!("{base}/stats")).and_then(|b| parse_stats(&b));
    let model_ids = get_text(&agent, &format!("{base}/v1/models"))
        .map(|b| parse_model_ids(&b))
        .unwrap_or_default();

    // Prefer the model from /stats (authoritative), fall back to /health.
    let model = stats
        .as_ref()
        .and_then(|s| s.model.clone())
        .or(health_model);

    Dashboard {
        serve_up: true,
        endpoint,
        model,
        stats,
        model_ids,
        vram,
        offline_hint: None,
    }
}

// ── Background fetch worker ─────────────────────────────────────────────────

/// How often the background thread re-fetches a snapshot while the Dashboard
/// tab is active.
const WORKER_REFRESH: Duration = Duration::from_millis(1500);
/// Poll cadence for the worker's "is the tab active?" / "should I exit?" flags
/// while it is idle (Dashboard tab not focused).
const WORKER_IDLE_POLL: Duration = Duration::from_millis(150);

/// Owns the background fetch thread. The UI thread NEVER calls
/// [`fetch_dashboard`] / rocm-smi / HTTP directly; it only reads the latest
/// snapshot via [`DashboardWorker::snapshot`]. The worker fetches on its own
/// thread every [`WORKER_REFRESH`] while the Dashboard tab is active, writing
/// the result into a shared `Arc<Mutex<Option<Dashboard>>>`, so a hung
/// rocm-smi or slow HTTP probe can stall only the worker — never render/input.
pub struct DashboardWorker {
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    /// Whether the Dashboard tab is currently focused. The worker only fetches
    /// when this is true (plus an immediate fetch when it flips on).
    active: Arc<AtomicBool>,
    /// Set by the UI to force an immediate re-fetch (manual `r` refresh).
    force: Arc<AtomicBool>,
    /// Live config snapshot the worker reads each cycle (host/port). Updated by
    /// the UI on reload so a port change is picked up without restarting.
    config: Arc<Mutex<ConfigState>>,
    /// Signals the worker to exit; also wakes it via `wake`.
    shutdown: Arc<AtomicBool>,
    wake: mpsc::Sender<()>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DashboardWorker {
    /// Spawn the background fetch thread. It starts idle (no fetch) until the
    /// Dashboard tab is marked active via [`DashboardWorker::set_active`].
    pub fn spawn(config: ConfigState) -> Self {
        let snapshot = Arc::new(Mutex::new(None));
        let active = Arc::new(AtomicBool::new(false));
        let force = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let config = Arc::new(Mutex::new(config));
        let (wake, wake_rx) = mpsc::channel::<()>();

        let handle = {
            let snapshot = Arc::clone(&snapshot);
            let active = Arc::clone(&active);
            let force = Arc::clone(&force);
            let shutdown = Arc::clone(&shutdown);
            let config = Arc::clone(&config);
            thread::spawn(move || {
                worker_loop(snapshot, active, force, shutdown, config, wake_rx);
            })
        };

        Self {
            snapshot,
            active,
            force,
            config,
            shutdown,
            wake,
            handle: Some(handle),
        }
    }

    /// Read the latest snapshot (cheap clone). `None` until the first fetch
    /// completes. This is the ONLY path the UI thread uses for dashboard data.
    pub fn snapshot(&self) -> Option<Dashboard> {
        self.snapshot.lock().unwrap().clone()
    }

    /// Mark the Dashboard tab focused / unfocused. Flipping to active wakes the
    /// worker so the first fetch starts immediately instead of after an idle
    /// poll.
    pub fn set_active(&self, active: bool) {
        let was = self.active.swap(active, Ordering::SeqCst);
        if active && !was {
            let _ = self.wake.send(());
        }
    }

    /// Request an immediate re-fetch (manual refresh). No-op cost if the worker
    /// is mid-cycle; it picks the flag up at the next loop turn.
    pub fn force_refresh(&self) {
        self.force.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
    }

    /// Update the config the worker fetches against (e.g. after a port change
    /// on reload).
    pub fn update_config(&self, config: ConfigState) {
        *self.config.lock().unwrap() = config;
    }
}

impl Drop for DashboardWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    active: Arc<AtomicBool>,
    force: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    config: Arc<Mutex<ConfigState>>,
    wake_rx: mpsc::Receiver<()>,
) {
    let mut last_fetch: Option<Instant> = None;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        let is_active = active.load(Ordering::SeqCst);
        let forced = force.swap(false, Ordering::SeqCst);
        let due = is_active
            && (forced
                || last_fetch
                    .map(|t| t.elapsed() >= WORKER_REFRESH)
                    .unwrap_or(true));

        if due {
            // Clone the config out of the lock so the (potentially slow) fetch
            // never holds it.
            let cfg = config.lock().unwrap().clone();
            let dash = fetch_dashboard(&cfg);
            *snapshot.lock().unwrap() = Some(dash);
            last_fetch = Some(Instant::now());
        }

        // Sleep until the next refresh is due (when active) or a short idle
        // poll (when inactive), but wake early on force/active/shutdown signals.
        let wait = if is_active {
            last_fetch
                .map(|t| WORKER_REFRESH.saturating_sub(t.elapsed()))
                .unwrap_or(WORKER_REFRESH)
                .max(Duration::from_millis(20))
        } else {
            WORKER_IDLE_POLL
        };
        // Drain a single wake (coalesces bursts) or time out.
        let _ = wake_rx.recv_timeout(wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_model_present() {
        let b = r#"{"status":"ok","model":"/home/u/.hipfire/models/q.mq4","pid":42}"#;
        assert_eq!(
            parse_health_model(b),
            Some("/home/u/.hipfire/models/q.mq4".to_string())
        );
    }

    #[test]
    fn health_model_null_or_idle() {
        assert_eq!(parse_health_model(r#"{"status":"ok","model":null}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok","model":""}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok"}"#), None);
    }

    #[test]
    fn health_garbage_is_none_not_panic() {
        assert_eq!(parse_health_model("not json"), None);
        assert_eq!(parse_health_model(""), None);
        assert_eq!(parse_health_model("[1,2,3]"), None);
    }

    #[test]
    fn stats_full_shape() {
        let b = r#"{"model":"q.mq4","uptime_s":63,"queue_depth":2,"requests_served":42,"recent_tok_s":151.24}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model.as_deref(), Some("q.mq4"));
        assert_eq!(s.uptime_s, 63);
        assert_eq!(s.queue_depth, 2);
        assert_eq!(s.requests_served, 42);
        assert_eq!(s.recent_tok_s, Some(151.24));
    }

    #[test]
    fn stats_idle_no_model_no_tok_s() {
        let b = r#"{"model":null,"uptime_s":5,"queue_depth":0,"requests_served":0}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model, None);
        assert_eq!(s.queue_depth, 0);
        assert_eq!(s.requests_served, 0);
        assert_eq!(s.recent_tok_s, None);
    }

    #[test]
    fn stats_garbage_is_none() {
        assert_eq!(parse_stats("not json"), None);
        assert_eq!(parse_stats("[1,2]"), None);
        assert_eq!(parse_stats(""), None);
    }

    #[test]
    fn stats_rejects_nonfinite_tok_s() {
        // serde won't parse NaN, but a zero/negative must be dropped.
        let b = r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":0}"#;
        assert_eq!(parse_stats(b).unwrap().recent_tok_s, None);
        let b2 = r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":-3}"#;
        assert_eq!(parse_stats(b2).unwrap().recent_tok_s, None);
    }

    #[test]
    fn model_ids_parsed() {
        let b = r#"{"data":[{"id":"qwen3.5:9b"},{"id":"qwen3.5:27b"}]}"#;
        assert_eq!(parse_model_ids(b), vec!["qwen3.5:9b", "qwen3.5:27b"]);
        assert!(parse_model_ids("garbage").is_empty());
        assert!(parse_model_ids(r#"{"data":[]}"#).is_empty());
    }

    #[test]
    fn rocm_smi_two_gpus() {
        let b = r#"{
            "card1": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "10000000000"},
            "card0": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "2000000000"}
        }"#;
        let vram = parse_rocm_smi_json(b);
        match vram {
            VramState::Available(gpus) => {
                assert_eq!(gpus.len(), 2);
                // Sorted by card index.
                assert_eq!(gpus[0].index, 0);
                assert_eq!(gpus[0].total_bytes, 34208743424);
                assert_eq!(gpus[0].used_bytes, 2000000000);
                assert_eq!(gpus[0].free_bytes(), 32208743424);
                assert_eq!(gpus[1].index, 1);
            }
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    fn rocm_smi_numeric_fields_also_work() {
        let b = r#"{"card0":{"VRAM Total Memory (B)": 8000000000, "VRAM Total Used Memory (B)": 1000000000}}"#;
        match parse_rocm_smi_json(b) {
            VramState::Available(g) => assert_eq!(g[0].total_bytes, 8000000000),
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_kills_hung_process() {
        // A process that sleeps far longer than the timeout must be killed and
        // reported as TimedOut within ~the timeout (not blocked for 60s).
        let mut cmd = Command::new("sleep");
        cmd.arg("60");
        let start = Instant::now();
        let run = run_bounded(cmd, Duration::from_millis(200));
        let elapsed = start.elapsed();
        assert!(matches!(run, RocmSmiRun::TimedOut), "expected TimedOut");
        assert!(
            elapsed < Duration::from_secs(5),
            "kill must be prompt, took {elapsed:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_captures_fast_output() {
        let mut cmd = Command::new("printf");
        cmd.arg("hello");
        match run_bounded(cmd, Duration::from_secs(5)) {
            RocmSmiRun::Output(s) => assert_eq!(s, "hello"),
            other => panic!("expected Output, got {:?}", run_kind(&other)),
        }
    }

    #[test]
    fn bounded_run_missing_binary_is_notfound() {
        let cmd = Command::new("definitely-not-a-real-binary-xyz-123");
        assert!(matches!(
            run_bounded(cmd, Duration::from_secs(1)),
            RocmSmiRun::NotFound
        ));
    }

    #[cfg(unix)]
    fn run_kind(r: &RocmSmiRun) -> &'static str {
        match r {
            RocmSmiRun::Output(_) => "Output",
            RocmSmiRun::NotFound => "NotFound",
            RocmSmiRun::SpawnError(_) => "SpawnError",
            RocmSmiRun::TimedOut => "TimedOut",
        }
    }

    #[test]
    fn worker_spawns_and_drops_cleanly() {
        // Spawn the worker, leave it idle (Dashboard tab not active), and drop
        // it — Drop must signal shutdown and join without hanging. No GPU /
        // serve needed; idle worker performs no fetch.
        let cfg = ConfigState::load(&crate::hipfire::HipfirePaths::discover());
        let worker = DashboardWorker::spawn(cfg);
        // No snapshot before any fetch.
        assert!(worker.snapshot().is_none());
        // set_active toggling must not panic; we don't assert a fetch result
        // (no serve in CI) — just that the worker stays responsive and Drop
        // joins promptly.
        worker.set_active(true);
        worker.set_active(false);
        drop(worker); // joins; would hang the test if shutdown were broken.
    }

    #[test]
    fn rocm_smi_garbage_is_unavailable_not_panic() {
        assert!(matches!(
            parse_rocm_smi_json("not json"),
            VramState::Unavailable(_)
        ));
        assert!(matches!(
            parse_rocm_smi_json("{}"),
            VramState::Unavailable(_)
        ));
        // Object without the VRAM fields → unavailable, not a fake 0.
        assert!(matches!(
            parse_rocm_smi_json(r#"{"card0":{"GPU use (%)":"50"}}"#),
            VramState::Unavailable(_)
        ));
    }
}
