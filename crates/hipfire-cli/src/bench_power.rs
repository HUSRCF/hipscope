// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU power state for `hipfire bench`.
//!
//! Every measured run is sampled every [`SAMPLE_PERIOD`] from world-readable
//! sysfs only: the card's `gpu_metrics` blob, its amdgpu hwmon, k10temp
//! (Tctl), `power_dpm_force_performance_level` and `pp_od_clk_voltage`. The
//! blob is decoded by its header's format/content revision; a revision this
//! file does not know, or a missing file, leaves only the hwmon and sysfs
//! fields and marks the `gpu_metrics` fields `"unavailable"`. Nothing here
//! can fail a bench.

use hipfire_config::devices::{
    enumerate_gpus, resolve_device_selectors, Claim, DeviceRoot, GpuDevice,
};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Sampling period inside a measured run.
pub const SAMPLE_PERIOD: Duration = Duration::from_millis(500);

/// A run whose PPT/STAPM/THM/PROCHOT residency exceeds this share of its
/// wall time draws the bench's throttle warning.
pub const THROTTLE_WARN_PCT: f64 = 10.0;

const UNAVAILABLE: &str = "unavailable";

/// `gpu_metrics_v3_0` (amdgpu `kgd_pp_interface.h`): a naturally aligned
/// 264-byte struct, little-endian.
mod v3_0 {
    pub const SIZE: usize = 264;
    pub const TEMPERATURE_GFX: usize = 4; // u16, centi-°C
    pub const AVERAGE_SOCKET_POWER: usize = 112; // u32, mW
    pub const AVERAGE_GFX_POWER: usize = 124; // u32, mW
    pub const AVERAGE_GFXCLK_FREQUENCY: usize = 174; // u16, MHz
    pub const AVERAGE_FCLK_FREQUENCY: usize = 182; // u16, MHz
    pub const CURRENT_GFX_MAXFREQ: usize = 224; // u16, MHz
    /// `throttle_residency_{prochot,spl,fppt,sppt,thm_core,thm_gfx,thm_soc}`:
    /// cumulative u32 counters in milliseconds (a limiter held for the whole
    /// of one second adds 1000).
    pub const THROTTLE_RESIDENCY: usize = 228;
}

/// Throttle residency counters in `gpu_metrics_v3_0` order, by report name.
/// `stapm` is the SMU's SPL (sustained power limit) counter.
const THROTTLERS: [&str; 7] = [
    "prochot", "stapm", "fast_ppt", "slow_ppt", "thm_core", "thm_gfx", "thm_soc",
];

/// One decoded `gpu_metrics_v3_0` sample. `None` is a field the SMU leaves
/// at its all-ones "not supported" value.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MetricsV3 {
    pub gfx_temp_c: Option<f64>,
    pub socket_power_w: Option<f64>,
    pub gfx_power_w: Option<f64>,
    pub gfxclk_mhz: Option<f64>,
    pub fclk_mhz: Option<f64>,
    pub gfx_maxfreq_mhz: Option<f64>,
    pub throttle_ms: [Option<u32>; 7],
}

/// A `gpu_metrics` read, classified by its header.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Metrics {
    /// The file is absent or unreadable.
    Missing,
    /// A header this decoder does not know (`"format.content"`), or a blob
    /// whose size disagrees with its header (`"malformed"`).
    Unsupported(String),
    V3_0(MetricsV3),
}

impl Metrics {
    fn format(&self) -> String {
        match self {
            Self::Missing => UNAVAILABLE.into(),
            Self::Unsupported(format) => format.clone(),
            Self::V3_0(_) => "3.0".into(),
        }
    }
}

/// Decode a `gpu_metrics` blob by its `metrics_table_header`
/// (`u16 structure_size, u8 format_revision, u8 content_revision`).
pub(crate) fn decode_gpu_metrics(blob: &[u8]) -> Metrics {
    let &[s0, s1, format, content, ..] = blob else {
        return Metrics::Unsupported("malformed".into());
    };
    let size = usize::from(u16::from_le_bytes([s0, s1]));
    if size > blob.len() {
        return Metrics::Unsupported("malformed".into());
    }
    match (format, content) {
        (3, 0) if size == v3_0::SIZE => {
            let u16_at = |at: usize| {
                let value = u16::from_le_bytes([blob[at], blob[at + 1]]);
                (value != u16::MAX).then_some(f64::from(value))
            };
            let u32_at = |at: usize| {
                let value =
                    u32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]]);
                (value != u32::MAX).then_some(value)
            };
            let mut throttle_ms = [None; 7];
            for (slot, counter) in throttle_ms.iter_mut().enumerate() {
                *counter = u32_at(v3_0::THROTTLE_RESIDENCY + 4 * slot);
            }
            Metrics::V3_0(MetricsV3 {
                gfx_temp_c: u16_at(v3_0::TEMPERATURE_GFX).map(|c| c / 100.0),
                socket_power_w: u32_at(v3_0::AVERAGE_SOCKET_POWER).map(|mw| f64::from(mw) / 1000.0),
                gfx_power_w: u32_at(v3_0::AVERAGE_GFX_POWER).map(|mw| f64::from(mw) / 1000.0),
                gfxclk_mhz: u16_at(v3_0::AVERAGE_GFXCLK_FREQUENCY),
                fclk_mhz: u16_at(v3_0::AVERAGE_FCLK_FREQUENCY),
                gfx_maxfreq_mhz: u16_at(v3_0::CURRENT_GFX_MAXFREQ),
                throttle_ms,
            })
        }
        _ => Metrics::Unsupported(format!("{format}.{content}")),
    }
}

/// `pp_od_clk_voltage`'s `OD_SCLK` levels and `OD_RANGE` SCLK bounds, MHz.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OdSclk {
    pub min_mhz: Option<f64>,
    pub max_mhz: Option<f64>,
    pub range_min_mhz: Option<f64>,
    pub range_max_mhz: Option<f64>,
}

fn mhz(token: &str) -> Option<f64> {
    let digits = token
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_alphabetic());
    digits.parse().ok()
}

/// Parse the `OD_SCLK:` block (`0: 600Mhz`, `1: 2900Mhz`) and the
/// `SCLK: 600Mhz 2900Mhz` line under `OD_RANGE:`. `None` without either.
pub(crate) fn parse_od_sclk(text: &str) -> Option<OdSclk> {
    let mut od = OdSclk::default();
    let mut section = "";
    for line in text.lines().map(str::trim) {
        if line.ends_with(':') && !line.contains(' ') {
            section = line;
            continue;
        }
        match section {
            "OD_SCLK:" => {
                let Some((level, value)) = line.split_once(':') else {
                    continue;
                };
                match level.trim() {
                    "0" => od.min_mhz = mhz(value),
                    "1" => od.max_mhz = mhz(value),
                    _ => {}
                }
            }
            "OD_RANGE:" => {
                let Some(bounds) = line.strip_prefix("SCLK:") else {
                    continue;
                };
                let mut bounds = bounds.split_whitespace();
                od.range_min_mhz = bounds.next().and_then(mhz);
                od.range_max_mhz = bounds.next().and_then(mhz);
            }
            _ => {}
        }
    }
    (od != OdSclk::default()).then_some(od)
}

/// The card hipfire runs on, for logical device 0.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PowerDevice {
    /// `cardN` under `/sys/class/drm`, when one points at the PCI device.
    pub card: Option<String>,
    pub pci: String,
    pub arch: String,
    /// How the card was chosen, for the report.
    pub selected_by: String,
    /// The PCI device's sysfs directory (`gpu_metrics`, `hwmon/`, ...).
    pub dir: PathBuf,
}

/// Pick the card behind the bench's logical device 0, as the daemon does:
/// `hardware.devices` (`spec`) first, then an inherited
/// `ROCR_VISIBLE_DEVICES`, else ROCr agent order. Without a filter the
/// daemon's reported `arch` narrows the candidates. `root` is `/` outside
/// tests.
pub(crate) fn resolve_device(
    root: &Path,
    spec: Option<&str>,
    rocr_visible: Option<&str>,
    arch: Option<&str>,
) -> Result<PowerDevice, String> {
    let gpus =
        enumerate_gpus(&DeviceRoot::at(root).kfd_topology()).map_err(|err| err.to_string())?;
    let (gpu, selected_by): (GpuDevice, String) = if let Some(spec) = spec {
        // The daemon claimed its card already; resolve without claiming.
        let mut no_claim = |_: &GpuDevice| Ok(Claim::Claimed);
        let chosen =
            resolve_device_selectors(spec, &gpus, &mut no_claim).map_err(|err| err.to_string())?;
        let first = chosen
            .into_iter()
            .next()
            .ok_or("hardware.devices selects no GPU")?;
        (first, format!("hardware.devices={spec}"))
    } else if let Some(rocr) = rocr_visible.filter(|value| !value.trim().is_empty()) {
        let token = rocr.split(',').next().unwrap_or_default().trim();
        let gpu = gpus
            .iter()
            .find(|gpu| {
                gpu.uuid()
                    .is_some_and(|uuid| uuid.eq_ignore_ascii_case(token))
                    || token.parse::<usize>() == Ok(gpu.rocr_index)
            })
            .ok_or_else(|| format!("ROCR_VISIBLE_DEVICES={rocr} matches no KFD GPU"))?;
        (gpu.clone(), format!("ROCR_VISIBLE_DEVICES={rocr}"))
    } else {
        let gpu = gpus
            .iter()
            .filter(|gpu| arch.is_none_or(|arch| gpu.arch == arch))
            .min_by_key(|gpu| gpu.rocr_index)
            .ok_or_else(|| format!("no KFD GPU of arch {}", arch.unwrap_or("any")))?;
        (
            gpu.clone(),
            "unfiltered: first ROCr agent of the daemon's arch".into(),
        )
    };
    let pci = gpu.bdf.to_string();
    let card = drm_card(root, &pci);
    let dir = match &card {
        Some(card) => root.join("sys/class/drm").join(card).join("device"),
        None => root.join("sys/bus/pci/devices").join(&pci),
    };
    if !dir.is_dir() {
        return Err(format!("no sysfs device directory for PCI {pci}"));
    }
    Ok(PowerDevice {
        card,
        pci,
        arch: gpu.arch,
        selected_by,
        dir,
    })
}

/// `cardN` whose `device` link resolves to PCI `pci`.
fn drm_card(root: &Path, pci: &str) -> Option<String> {
    let mut cards: Vec<String> = std::fs::read_dir(root.join("sys/class/drm"))
        .ok()?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let digits = name.strip_prefix("card")?;
            (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(name)
        })
        .collect();
    cards.sort();
    cards.into_iter().find(|card| {
        std::fs::canonicalize(root.join("sys/class/drm").join(card).join("device"))
            .ok()
            .and_then(|target| target.file_name().map(|name| name == pci))
            .unwrap_or(false)
    })
}

/// One labelled hwmon reading: `key`, file, and the factor to SI-ish units
/// (°C, W, MHz).
#[derive(Clone, Debug)]
struct HwmonInput {
    key: String,
    path: PathBuf,
    scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HwmonKind {
    Temp,
    Power,
    Freq,
}

impl HwmonKind {
    const ALL: [Self; 3] = [Self::Temp, Self::Power, Self::Freq];

    fn prefix(self) -> &'static str {
        match self {
            Self::Temp => "temp",
            Self::Power => "power",
            Self::Freq => "freq",
        }
    }

    fn scale(self) -> f64 {
        match self {
            Self::Temp => 1e-3,  // m°C
            Self::Power => 1e-6, // µW
            Self::Freq => 1e-6,  // Hz
        }
    }

    fn json_key(self) -> &'static str {
        match self {
            Self::Temp => "temp_c",
            Self::Power => "power_w",
            Self::Freq => "freq_mhz",
        }
    }
}

/// The labelled temp/power/freq inputs of one hwmon directory. Power prefers
/// the driver's `powerN_average` over `powerN_input`.
fn hwmon_inputs(dir: &Path) -> Vec<(HwmonKind, HwmonInput)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let mut inputs = Vec::new();
    for kind in HwmonKind::ALL {
        for name in &names {
            let Some(index) = name
                .strip_prefix(kind.prefix())
                .and_then(|rest| rest.strip_suffix("_input"))
                .filter(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
            else {
                continue;
            };
            let stem = format!("{}{index}", kind.prefix());
            let label = std::fs::read_to_string(dir.join(format!("{stem}_label")))
                .map(|label| label.trim().to_ascii_lowercase())
                .ok()
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| stem.clone());
            let average = dir.join(format!("{stem}_average"));
            let path = if kind == HwmonKind::Power && average.exists() {
                average
            } else {
                dir.join(name)
            };
            inputs.push((
                kind,
                HwmonInput {
                    key: label,
                    path,
                    scale: kind.scale(),
                },
            ));
        }
    }
    inputs
}

fn read_number(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The k10temp `Tctl` input, or its first temperature when unlabelled.
fn find_tctl(root: &Path) -> Option<PathBuf> {
    let mut hwmons: Vec<PathBuf> = std::fs::read_dir(root.join("sys/class/hwmon"))
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    hwmons.sort();
    let k10 = hwmons
        .into_iter()
        .find(|dir| read_trimmed(&dir.join("name")).as_deref() == Some("k10temp"))?;
    let temps = hwmon_inputs(&k10);
    temps
        .iter()
        .find(|(kind, input)| *kind == HwmonKind::Temp && input.key == "tctl")
        .or_else(|| temps.iter().find(|(kind, _)| *kind == HwmonKind::Temp))
        .map(|(_, input)| input.path.clone())
}

/// Everything sampled once per tick.
#[derive(Clone, Debug)]
struct Sample {
    at: Instant,
    metrics: Metrics,
    hwmon: Vec<Option<f64>>,
    tctl_c: Option<f64>,
}

/// The files a run samples, resolved once per bench.
#[derive(Clone, Debug)]
struct Sources {
    gpu_metrics: Option<PathBuf>,
    hwmon: Vec<(HwmonKind, HwmonInput)>,
    tctl: Option<PathBuf>,
}

impl Sources {
    fn sample(&self) -> Sample {
        let metrics = match &self.gpu_metrics {
            Some(path) => match std::fs::read(path) {
                Ok(blob) => decode_gpu_metrics(&blob),
                Err(_) => Metrics::Missing,
            },
            None => Metrics::Missing,
        };
        Sample {
            at: Instant::now(),
            metrics,
            hwmon: self
                .hwmon
                .iter()
                .map(|(_, input)| read_number(&input.path).map(|raw| raw * input.scale))
                .collect(),
            tctl_c: self
                .tctl
                .as_deref()
                .and_then(read_number)
                .map(|millis| millis / 1000.0),
        }
    }
}

/// Per-bench sampler: resolves the device and its files once.
pub(crate) struct PowerProbe {
    root: PathBuf,
    device: Result<PowerDevice, String>,
    sources: Sources,
}

impl PowerProbe {
    /// Never fails: an unresolvable device keeps only Tctl.
    pub(crate) fn new(spec: Option<&str>, rocr_visible: Option<&str>, arch: Option<&str>) -> Self {
        Self::at(Path::new("/"), spec, rocr_visible, arch)
    }

    fn at(root: &Path, spec: Option<&str>, rocr_visible: Option<&str>, arch: Option<&str>) -> Self {
        let device = resolve_device(root, spec, rocr_visible, arch);
        let (gpu_metrics, hwmon) = match &device {
            Ok(device) => {
                let hwmon_dir =
                    std::fs::read_dir(device.dir.join("hwmon"))
                        .ok()
                        .and_then(|entries| {
                            let mut dirs: Vec<PathBuf> = entries
                                .filter_map(|entry| Some(entry.ok()?.path()))
                                .collect();
                            dirs.sort();
                            dirs.into_iter().next()
                        });
                (
                    Some(device.dir.join("gpu_metrics")),
                    hwmon_dir.as_deref().map(hwmon_inputs).unwrap_or_default(),
                )
            }
            Err(_) => (None, Vec::new()),
        };
        Self {
            sources: Sources {
                gpu_metrics,
                hwmon,
                tctl: find_tctl(root),
            },
            device,
            root: root.to_owned(),
        }
    }

    /// The `power_device` report object.
    pub(crate) fn device_json(&self) -> Value {
        match &self.device {
            Ok(device) => json!({
                "card": device.card,
                "pci": device.pci,
                "arch": device.arch,
                "selected_by": device.selected_by,
                "sample_period_ms": SAMPLE_PERIOD.as_millis() as u64,
            }),
            Err(reason) => json!({ "status": UNAVAILABLE, "reason": reason }),
        }
    }

    /// One line for the bench banner.
    pub(crate) fn describe(&self) -> String {
        match &self.device {
            Ok(device) => format!(
                "{} PCI {} {} ({})",
                device.card.as_deref().unwrap_or("no DRM card"),
                device.pci,
                device.arch,
                device.selected_by
            ),
            Err(reason) => format!("{UNAVAILABLE}: {reason}"),
        }
    }

    /// Start sampling one measured run. Call [`RunSampler::finish`] right
    /// after the run's last event.
    pub(crate) fn start_run(&self) -> RunSampler {
        let (perf_level, od_sclk) = match &self.device {
            Ok(device) => (
                read_trimmed(&device.dir.join("power_dpm_force_performance_level")),
                std::fs::read_to_string(device.dir.join("pp_od_clk_voltage"))
                    .ok()
                    .and_then(|text| parse_od_sclk(&text)),
            ),
            Err(_) => (None, None),
        };
        let sources = self.sources.clone();
        let (stop, stopped) = mpsc::channel::<()>();
        let first = sources.sample();
        let worker = std::thread::Builder::new()
            .name("bench-power".into())
            .spawn(move || {
                let mut samples = vec![first];
                loop {
                    match stopped.recv_timeout(SAMPLE_PERIOD) {
                        Err(RecvTimeoutError::Timeout) => samples.push(sources.sample()),
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                            samples.push(sources.sample());
                            return samples;
                        }
                    }
                }
            })
            .ok();
        RunSampler {
            stop,
            worker,
            perf_level,
            od_sclk,
            memory: memory_config(
                &self.root,
                self.device.as_ref().ok().map(|device| device.dir.as_path()),
            ),
            hwmon: self.sources.hwmon.clone(),
        }
    }
}

/// One measured run being sampled.
pub(crate) struct RunSampler {
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<Vec<Sample>>>,
    perf_level: Option<String>,
    od_sclk: Option<OdSclk>,
    memory: Value,
    hwmon: Vec<(HwmonKind, HwmonInput)>,
}

impl RunSampler {
    pub(crate) fn finish(self) -> RunPower {
        let _ = self.stop.send(());
        let samples = self
            .worker
            .and_then(|worker| worker.join().ok())
            .unwrap_or_default();
        let mut run = summarize(&samples, self.perf_level, self.od_sclk, &self.hwmon);
        run.json["memory"] = self.memory;
        run
    }
}

/// One run's power report plus its throttle residencies for the warning.
#[derive(Clone, Debug)]
pub(crate) struct RunPower {
    pub json: Value,
    /// Percent of the run at each limiter, in [`THROTTLERS`] order.
    pub throttle_pct: Option<[Option<f64>; 7]>,
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn or_unavailable(value: Option<f64>) -> Value {
    value.map_or_else(
        || Value::from(UNAVAILABLE),
        |value| Value::from(round1(value)),
    )
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    avg: f64,
    min: f64,
    max: f64,
}

fn stats(values: impl Iterator<Item = f64>) -> Option<Stats> {
    let (mut sum, mut count, mut min, mut max) = (0.0, 0usize, f64::INFINITY, f64::NEG_INFINITY);
    for value in values {
        sum += value;
        count += 1;
        min = min.min(value);
        max = max.max(value);
    }
    (count > 0).then(|| Stats {
        avg: sum / count as f64,
        min,
        max,
    })
}

fn stats_json(stats: Option<Stats>, fields: &[&str]) -> Value {
    let Some(stats) = stats else {
        return UNAVAILABLE.into();
    };
    let mut object = Map::new();
    for field in fields {
        let value = match *field {
            "avg" => stats.avg,
            "min" => stats.min,
            _ => stats.max,
        };
        object.insert((*field).into(), round1(value).into());
    }
    Value::Object(object)
}

/// Residency percent per limiter from the first and last decoded samples.
/// The SMU counters are cumulative milliseconds, so the increase over the
/// run's wall time is the share of it spent at that limit.
fn throttle_residency(decoded: &[(Instant, &MetricsV3)]) -> Option<[Option<f64>; 7]> {
    let (&(t0, first), &(t1, last)) = (decoded.first()?, decoded.last()?);
    let elapsed_ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
    if elapsed_ms <= 0.0 {
        return None;
    }
    let mut pct = [None; 7];
    for (slot, value) in pct.iter_mut().enumerate() {
        if let (Some(a), Some(b)) = (first.throttle_ms[slot], last.throttle_ms[slot]) {
            *value = Some((f64::from(b.wrapping_sub(a)) / elapsed_ms * 100.0).min(100.0));
        }
    }
    Some(pct)
}

/// `/proc/meminfo` field `name` (`kB`), in MiB.
fn meminfo_mib(meminfo: &str, name: &str) -> Option<f64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix(name)?.strip_prefix(':')?;
        let kib: f64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(kib / 1024.0)
    })
}

/// The memory configuration a run sees, in MiB: the card's VRAM carve-out
/// and GTT (`mem_info_{vram,gtt}_{total,used}`, bytes; free is total minus
/// used) and the host's `MemTotal`/`MemAvailable`. Read at run start.
pub(crate) fn memory_config(root: &Path, device_dir: Option<&Path>) -> Value {
    let mib = |name: &str| {
        device_dir
            .and_then(|dir| read_number(&dir.join(name)))
            .map(|bytes| bytes / (1024.0 * 1024.0))
    };
    let mut object = Map::new();
    for pool in ["vram", "gtt"] {
        let total = mib(&format!("mem_info_{pool}_total"));
        let used = mib(&format!("mem_info_{pool}_used"));
        let free = total.zip(used).map(|(total, used)| total - used);
        object.insert(format!("{pool}_total_mib"), or_unavailable(total));
        object.insert(format!("{pool}_used_mib"), or_unavailable(used));
        object.insert(format!("{pool}_free_mib"), or_unavailable(free));
    }
    let meminfo = std::fs::read_to_string(root.join("proc/meminfo")).unwrap_or_default();
    object.insert(
        "mem_total_mib".into(),
        or_unavailable(meminfo_mib(&meminfo, "MemTotal")),
    );
    object.insert(
        "mem_available_mib".into(),
        or_unavailable(meminfo_mib(&meminfo, "MemAvailable")),
    );
    Value::Object(object)
}

fn summarize(
    samples: &[Sample],
    perf_level: Option<String>,
    od_sclk: Option<OdSclk>,
    hwmon: &[(HwmonKind, HwmonInput)],
) -> RunPower {
    let duration_s = match (samples.first(), samples.last()) {
        (Some(first), Some(last)) => last.at.duration_since(first.at).as_secs_f64(),
        _ => 0.0,
    };
    // Report the format seen most recently; a decoded sample anywhere wins.
    let format = samples
        .iter()
        .find(|sample| matches!(sample.metrics, Metrics::V3_0(_)))
        .or(samples.last())
        .map_or_else(|| UNAVAILABLE.to_owned(), |sample| sample.metrics.format());
    let decoded: Vec<(Instant, &MetricsV3)> = samples
        .iter()
        .filter_map(|sample| match &sample.metrics {
            Metrics::V3_0(metrics) => Some((sample.at, metrics)),
            _ => None,
        })
        .collect();
    let field =
        |get: fn(&MetricsV3) -> Option<f64>| stats(decoded.iter().filter_map(|(_, m)| get(m)));

    let throttle_pct = throttle_residency(&decoded);
    let throttle_json = match &throttle_pct {
        Some(pct) => Value::Object(
            THROTTLERS
                .iter()
                .zip(pct)
                .map(|(name, value)| ((*name).to_owned(), or_unavailable(*value)))
                .collect(),
        ),
        None => UNAVAILABLE.into(),
    };

    let hwmon_json = if hwmon.is_empty() {
        Value::from(UNAVAILABLE)
    } else {
        let mut groups = Map::new();
        for kind in HwmonKind::ALL {
            let mut group = Map::new();
            for (index, (_, input)) in hwmon.iter().enumerate().filter(|(_, (k, _))| *k == kind) {
                let values = samples
                    .iter()
                    .filter_map(|sample| sample.hwmon.get(index).copied().flatten());
                let fields: &[&str] = if kind == HwmonKind::Temp {
                    &["max"]
                } else {
                    &["avg", "min", "max"]
                };
                group.insert(input.key.clone(), stats_json(stats(values), fields));
            }
            if !group.is_empty() {
                groups.insert(kind.json_key().into(), Value::Object(group));
            }
        }
        Value::Object(groups)
    };

    let tctl = stats(samples.iter().filter_map(|sample| sample.tctl_c));
    let od_json = od_sclk.map_or_else(
        || Value::from(UNAVAILABLE),
        |od| {
            json!({
                "min_mhz": or_unavailable(od.min_mhz),
                "max_mhz": or_unavailable(od.max_mhz),
                "range_min_mhz": or_unavailable(od.range_min_mhz),
                "range_max_mhz": or_unavailable(od.range_max_mhz),
            })
        },
    );

    let json = json!({
        "samples": samples.len(),
        "duration_s": (duration_s * 100.0).round() / 100.0,
        "gpu_metrics_format": format,
        "gfxclk_mhz": stats_json(field(|m| m.gfxclk_mhz), &["avg", "min", "max"]),
        "gfx_maxfreq_mhz_min": or_unavailable(field(|m| m.gfx_maxfreq_mhz).map(|s| s.min)),
        "socket_power_w": stats_json(field(|m| m.socket_power_w), &["avg", "max"]),
        "gfx_power_w": stats_json(field(|m| m.gfx_power_w), &["avg", "max"]),
        "gfx_temp_c_max": or_unavailable(field(|m| m.gfx_temp_c).map(|s| s.max)),
        "fclk_mhz_avg": or_unavailable(field(|m| m.fclk_mhz).map(|s| s.avg)),
        "throttle_pct": throttle_json,
        "tctl_c_max": or_unavailable(tctl.map(|s| s.max)),
        "perf_level": perf_level.map_or_else(|| Value::from(UNAVAILABLE), Value::from),
        "od_sclk": od_json,
        "hwmon": hwmon_json,
    });
    RunPower { json, throttle_pct }
}

/// One line when any run spent more than [`THROTTLE_WARN_PCT`] of its time
/// at a power or thermal limit, naming each such limiter's worst run.
pub(crate) fn throttle_warning(runs: &[RunPower]) -> Option<String> {
    let mut worst = [0.0f64; 7];
    let mut throttled_runs = 0;
    for run in runs {
        let Some(pct) = run.throttle_pct else {
            continue;
        };
        let mut throttled = false;
        for (slot, value) in pct.iter().enumerate() {
            if let Some(value) = *value {
                if value > THROTTLE_WARN_PCT {
                    throttled = true;
                    worst[slot] = worst[slot].max(value);
                }
            }
        }
        throttled_runs += usize::from(throttled);
    }
    if throttled_runs == 0 {
        return None;
    }
    let limiters = THROTTLERS
        .iter()
        .zip(worst)
        .filter(|(_, pct)| *pct > THROTTLE_WARN_PCT)
        .map(|(name, pct)| format!("{name} {pct:.0}%"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "GPU throttled for more than {THROTTLE_WARN_PCT:.0}% of {throttled_runs}/{} measured runs (worst: {limiters}); these numbers are power/thermal-limited — compare only against runs with matching power fields",
        runs.len()
    ))
}

/// Per-run `power` array for a report.
pub(crate) fn runs_json(runs: &[RunPower]) -> Value {
    Value::Array(runs.iter().map(|run| run.json.clone()).collect())
}
