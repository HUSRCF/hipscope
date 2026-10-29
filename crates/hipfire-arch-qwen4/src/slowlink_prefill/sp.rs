// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SP: the short-prompt prefill hook's load owner, window selection data and
//! per-layer execution (design `screens/SP/HOOK-DESIGN.md` §3, §6, §7, §9 F1/F2).
//!
//! SP is a per-window mode of S1. Everything here is default off
//! (`developer.qwen4_sp` / `HIPFIRE_QWEN4_SP`): with the knob off no fact is
//! gathered, no owner is bound, no window is selected for SP and every S1 path
//! is the one it was. With it on:
//!
//! * **load** ([`sp_load_decision`], [`SpRuntime::bind`]): the pure admission
//!   ([`crate::slowlink_sp::admit_sp`]) over the facts of this load, then the
//!   `LoadOwner::SpHost` owner (pinned host buffers, the xfer stream, the four
//!   events, the CPU scratch);
//! * **selection** ([`sp_window_verdicts`]): one pre-mutation verdict per
//!   window;
//! * **execution** ([`Run::sp_host_layer`]): the per-layer protocol of §3.

use super::*;
use crate::gpu_forward::panel_stage::{RingSchedule, SpSchedule};
use crate::slowlink_admission::qualify_request_sp;
use crate::slowlink_sp::{
    admit_sp, qualify_sp_window, SpBypass, SpDecline, SpEventBackend, SpLoadFacts, SpProfile,
    SpProtocol, SpProtocolError, SP_MAX_ROWS,
};
use hipfire_dispatch::cpu_exec::{
    cpu_quant_for, moe_cpu_grouped_rows_bf16, CpuMoeExpert, SpCpuExpert, SpCpuScratch, SpCpuStats,
};
use hipfire_dispatch::pipeline::sp_layer::{exact_counts, plan_sp_layer, SpLayerPlan, SpLimits};
use hipfire_dispatch::sp_plan::SplitCost;
use rayon::prelude::*;

/// Slots one `moe_slots_compute` chunk of the CPU arm holds (design §3.8, §6:
/// 4,096 slots x 17,920 B = 70 MiB of scratch, independent of the window).
pub(crate) const SP_CPU_CHUNK_SLOTS: usize = 4096;

/// Elements one widening task converts.
const WIDEN_CHUNK: usize = 1 << 14;

fn proto_err(error: SpProtocolError) -> Qwen4GpuForwardError {
    invalid(format!("slow-link {error}"))
}

// ── lab placement ─────────────────────────────────────────────────────

/// Where the layer's experts go. Production is always [`Self::Auto`]; the lab
/// knob `HIPFIRE_QWEN4_SP_FORCE_K` selects the other two (V2a / V2b gates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(any(test, feature = "lab")), allow(dead_code))]
pub(crate) enum SpPlacement {
    /// The planner's argmin.
    Auto,
    /// k = U: every touched expert streams (no CPU slot).
    All,
    /// k = 0: every touched expert runs on the CPU.
    Cpu,
}

/// The placement of this process: `Auto` outside the lab feature.
#[cfg(feature = "lab")]
pub(crate) fn lab_placement() -> Result<SpPlacement, String> {
    use crate::slowlink_sp::{sp_force_k, SpForceK};
    Ok(match sp_force_k()? {
        SpForceK::All => SpPlacement::All,
        SpForceK::Cpu => SpPlacement::Cpu,
        SpForceK::Auto => SpPlacement::Auto,
    })
}

#[cfg(not(feature = "lab"))]
pub(crate) fn lab_placement() -> Result<SpPlacement, String> {
    Ok(SpPlacement::Auto)
}

/// `HIPFIRE_QWEN4_SP_TRACE=1` (lab only).
#[cfg(feature = "lab")]
pub(crate) fn lab_trace() -> bool {
    crate::slowlink_sp::sp_trace()
}

#[cfg(not(feature = "lab"))]
pub(crate) fn lab_trace() -> bool {
    false
}

/// The placement cost of one layer: the frozen hook-transport model
/// ([`SplitCost::sp_hook_xtx`]) with the lab override applied as a cost, so
/// the planner itself stays one function.
///
/// * `All`: no CPU slot is feasible, so the only feasible `k` is `U`;
/// * `Cpu`: a streamed expert costs more than any CPU time, so `k = 0` wins
///   (a layer whose CPU slots exceed the staging cap is then refused by
///   [`plan_sp_layer`], a lab-only outcome).
pub(crate) fn sp_layer_cost(
    rows: usize,
    hidden: usize,
    workers: usize,
    placement: SpPlacement,
) -> SplitCost {
    let mut cost = SplitCost::sp_hook_xtx(rows as u32, hidden, workers as u32);
    match placement {
        SpPlacement::Auto => {}
        SpPlacement::All => cost.cpu_slot_cap = Some(0),
        SpPlacement::Cpu => {
            cost.stream_ms_per_expert = 1.0e9;
            cost.cpu_slot_cap = None;
        }
    }
    cost
}

// ── load decision ─────────────────────────────────────────────────────

/// SP's pinned host budget: the host RAM the unmovable pinned allocations can
/// still take (`MemAvailable - CMA` plus the TTM page pool, less the
/// [`crate::expert_residency::HOST_RAM_HEADROOM_BYTES`] the experts' own check
/// keeps) less the pinned bytes S1's own page pool is about to stage. `0` when
/// `MemAvailable` is unreadable (SP is then declined by name).
pub(crate) fn sp_pinned_budget(
    mem_available: Option<u64>,
    ttm_pool_bytes: u64,
    s1_pinned_bytes: u64,
) -> u64 {
    mem_available
        .map(|available| {
            available
                .saturating_add(ttm_pool_bytes)
                .saturating_sub(crate::expert_residency::HOST_RAM_HEADROOM_BYTES)
                .saturating_sub(s1_pinned_bytes)
        })
        .unwrap_or(0)
}

/// `serve.prefix_cache` of this process (default off). SP is declined while it
/// is on: prefix-cache exactness is a non-goal of the hook.
pub(crate) fn serve_prefix_cache() -> bool {
    matches!(
        hipfire_config::active_or_local_process_config()
            .values
            .get("serve.prefix_cache"),
        Some(hipfire_config::ConfigValue::Bool(true))
    )
}

/// The one `(gate_up, down)` dtype pair every expert of every host layer
/// shares, when each has a CPU quant. `None` for no experts, a mixed layer set
/// or a dtype the CPU engine cannot read.
pub(crate) fn uniform_cpu_dtypes(
    pairs: impl IntoIterator<Item = (DType, DType)>,
) -> Option<(DType, DType)> {
    let mut common: Option<(DType, DType)> = None;
    for pair in pairs {
        match common {
            None => common = Some(pair),
            Some(known) if known == pair => {}
            Some(_) => return None,
        }
    }
    common.filter(|&(gate_up, down)| {
        cpu_quant_for(gate_up).is_some() && cpu_quant_for(down).is_some()
    })
}

/// Largest window rows whose SP layout fits the S1 route region (the
/// `census.route_storage` charge), at most [`SP_MAX_ROWS`] and the admitted
/// window rows: the pure `region_fit_rows` of [`SpLoadFacts`].
pub(crate) fn sp_region_fit_rows(profile: &SlowlinkProfile) -> usize {
    let hi = SP_MAX_ROWS.min(profile.route_dims().rows);
    let (mut lo, mut hi) = (0usize, hi);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if sp_window_layout(profile, mid).is_ok() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The SP route layout of a `rows`-row window, checked against the admitted
/// route region before anything is carved from it.
pub(crate) fn sp_window_layout(
    profile: &SlowlinkProfile,
    rows: usize,
) -> R<dsp::PromptRouteLayout> {
    let admitted = profile.route_dims();
    if rows == 0 || rows > admitted.rows {
        return Err(invalid(format!(
            "slow-link SP window of {rows} rows is outside the admitted 1..={} rows",
            admitted.rows
        )));
    }
    let dims = dsp::PromptRouteDims {
        rows,
        k1_rows: 0,
        tail_rows: 0,
        ..admitted
    };
    let layout = dsp::PromptRouteLayout::with_sp_rows(&dims).map_err(dsp_err("SP route layout"))?;
    if layout.total_bytes() as u64 > profile.census.route_storage {
        return Err(invalid(format!(
            "slow-link SP window layout is {} bytes, the admitted census holds {}",
            layout.total_bytes(),
            profile.census.route_storage
        )));
    }
    Ok(layout)
}

/// What the load admitted: the plan [`SpRuntime::bind`] needs, or the sticky
/// named decline.
pub(crate) type SpOutcome = Result<SpBindPlan, SpDecline>;

/// Everything [`SpRuntime::bind`] needs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpBindPlan {
    pub(crate) profile: SpProfile,
    pub(crate) gate_up: DType,
    pub(crate) down: DType,
    pub(crate) hidden: usize,
    pub(crate) mi: usize,
    pub(crate) n_exp: usize,
    pub(crate) pages: usize,
    pub(crate) placement: SpPlacement,
    pub(crate) trace: bool,
}

/// Decide SP at load (knob on), after S1's own runtime plan passed and before
/// any owner is bound. Reads only; allocates nothing. The outer `Err` is a lab
/// placement typo (a named load failure, never a silent default).
pub(crate) fn sp_load_decision(
    forward: &Qwen4GpuForward,
    gpu: &Gpu,
    profile: &SlowlinkProfile,
    pages: usize,
) -> Result<SpOutcome, String> {
    let knob = true;
    let placement = lab_placement()?;
    let dtypes = uniform_cpu_dtypes(forward.band_layers().flat_map(|layer| {
        forward
            .moe
            .get(layer)
            .and_then(Option::as_ref)
            .filter(|runtime| runtime.s1_host_mapped())
            .into_iter()
            .flat_map(|runtime| {
                (0..runtime.s1_num_experts()).filter_map(|id| runtime.s1_expert_dtypes(id))
            })
    }));
    let route = profile.geometry.route;
    let workers = (gpu.arch == crate::slowlink_sp::SP_ARCH && route == dsp::WindowRoute::Aggregate)
        .then(hipfire_dispatch::cpu_exec::splice_pool_threads);
    let facts = SpLoadFacts {
        knob,
        s1_route: Some(route),
        arch: &gpu.arch,
        n_exp: profile.geometry.num_experts,
        cpu_gate_up_quant: dtypes.is_some(),
        cpu_down_quant: dtypes.is_some(),
        splice_workers: workers,
        region_fit_rows: sp_region_fit_rows(profile),
        hidden: profile.geometry.hidden,
        pinned_budget_bytes: sp_pinned_budget(
            rdna_compute::kv_slots::mem_available_bytes(),
            crate::expert_residency::ttm_pool_estimate_on(gpu),
            profile.host_pinned_bytes(),
        ),
        prefix_cache: serve_prefix_cache(),
    };
    Ok(admit_sp(&facts).map(|sp| {
        // `admit_sp` returned Ok, so `dtypes` is Some.
        let (gate_up, down) = dtypes.unwrap_or((DType::MQ4G256V2, DType::MQ4G128V2));
        SpBindPlan {
            profile: sp,
            gate_up,
            down,
            hidden: profile.geometry.hidden,
            mi: profile.geometry.moe_intermediate,
            n_exp: profile.geometry.num_experts,
            pages,
            placement,
            trace: lab_trace(),
        }
    }))
}

// ── per-window selection ──────────────────────────────────────────────

/// One pre-mutation SP verdict per window of the plan ([`qualify_sp_window`]).
/// Empty when SP is not armed: no window is SP and no reason is named.
pub(crate) fn sp_window_verdicts(
    sp: Option<&SpProfile>,
    plan: &Qwen4PromptPlan,
    capture: CaptureKind,
) -> Vec<Result<(), SpBypass>> {
    let Some(sp) = sp else {
        return Vec::new();
    };
    plan.windows
        .iter()
        .map(|window| qualify_sp_window(sp, window, plan.window_tiles(window), capture))
        .collect()
}

/// Whether a request of `rows` rows is one tile below S1's own multi-chunk
/// floor, the shape whose window SP must take or the incumbent keeps.
pub(crate) fn sp_single_tile(tile_rows: usize, rows: usize) -> bool {
    (crate::slowlink_sp::SP_MIN_ROWS..=tile_rows).contains(&rows)
}

/// The shape [`qualify_request_sp`](crate::slowlink_admission::qualify_request_sp)
/// reads: the session cuts count only for a single-tile request (so every
/// longer request is judged exactly as without SP).
pub(crate) fn sp_request_shape(tile_rows: usize, shape: RequestShape, cuts: usize) -> RequestShape {
    RequestShape {
        session_cuts: if sp_single_tile(tile_rows, shape.rows) {
            cuts
        } else {
            shape.session_cuts
        },
        ..shape
    }
}

/// The request-level verdict before the plan exists: S1's own
/// ([`qualify_request`]), or with SP armed [`qualify_request_sp`] (a request
/// of one tile is S1's only when its window is SP's).
pub(crate) fn sp_request_verdict(
    mode: SlowlinkMode,
    profile: &SlowlinkProfile,
    sp: Option<&SpProfile>,
    shape: RequestShape,
    cuts: usize,
) -> Result<(), RequestBypass> {
    match sp {
        Some(sp) => qualify_request_sp(
            mode,
            profile,
            sp,
            &sp_request_shape(profile.tile_rows, shape, cuts),
        )
        .map(|_| ()),
        None => qualify_request(mode, profile, &shape).map(|_| ()),
    }
}

/// The plan-level rule below S1's multi-chunk floor: a single-tile request
/// whose window SP does not take is the incumbent's (never a stream-all
/// sweep), with the named reason. Longer requests are unaffected.
pub(crate) fn sp_floor_verdict(
    tile_rows: usize,
    rows: usize,
    verdicts: &[Result<(), SpBypass>],
) -> Result<(), RequestBypass> {
    if sp_single_tile(tile_rows, rows) {
        if let Some(&Err(bypass)) = verdicts.first() {
            return Err(RequestBypass::Sp(bypass));
        }
    }
    Ok(())
}

/// Pinned bytes of the four SP host buffers `[x16, sorted, cpu_rows, inv]`
/// (design §6); their sum is the admission's `pinned_bytes`.
pub(crate) fn sp_owner_bytes(
    hidden: usize,
    n_exp: usize,
    profile: &SpProfile,
) -> Option<[usize; 4]> {
    let grouped = crate::slowlink_sp::sp_grouped_bound(profile.sp_max_rows, n_exp)?;
    Some([
        profile.sp_max_rows.checked_mul(hidden)?.checked_mul(2)?,
        grouped.checked_mul(4)?,
        profile.s_cap.checked_mul(hidden)?.checked_mul(2)?,
        profile
            .sp_max_rows
            .checked_mul(dsp::K_TOP)?
            .checked_mul(4)?,
    ])
}

// ── pure helpers of the CPU arm ───────────────────────────────────────

/// Exact F16 -> F32 widening of one IEEE half bit pattern.
#[inline]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = (u32::from(h) >> 10) & 0x1f;
    let man = u32::from(h) & 0x3ff;
    let bits = match exp {
        0 => {
            // Zero or subnormal: man * 2^-24 is exact in F32.
            let magnitude = man as f32 * f32::from_bits(0x3380_0000);
            return f32::from_bits(magnitude.to_bits() | sign);
        }
        0x1f => sign | 0x7f80_0000 | (man << 13) | if man != 0 { 0x0040_0000 } else { 0 },
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Widen F16 bit patterns to F32 on the rayon pool (exact, order free).
pub(crate) fn widen_f16_rows(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "widen_f16_rows: length mismatch");
    dst.par_chunks_mut(WIDEN_CHUNK)
        .zip(src.par_chunks(WIDEN_CHUNK))
        .for_each(|(out, input)| {
            for (o, &h) in out.iter_mut().zip(input) {
                *o = f16_bits_to_f32(h);
            }
        });
}

/// The CPU arm's work list of one layer: every CPU expert's rows, ordered as
/// [`SpLayerPlan::cpu_rows`] (hybrid row `G_gpu + j` is list index `j`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CpuWork {
    /// `(token row, output index)` of every CPU slot.
    pub(crate) rows: Vec<(u32, u32)>,
    /// Expert id and its range of [`Self::rows`], ascending ids, no empty range.
    pub(crate) experts: Vec<(usize, Range<usize>)>,
}

/// Build the [`CpuWork`] of `plan` over the layer's grouping. Refuses a work
/// list that disagrees with the plan's own `cpu_rows` (a planner/hook bug).
pub(crate) fn build_cpu_work(
    plan: &SpLayerPlan,
    sorted: &[i32],
    offsets: &[usize],
) -> Result<CpuWork, String> {
    let n_exp = plan.split.cpu_mask.len();
    if offsets.len() != n_exp + 1 {
        return Err(format!(
            "the grouping has {} expert starts, the plan {n_exp} experts",
            offsets.len().saturating_sub(1)
        ));
    }
    let mut rows: Vec<(u32, u32)> = Vec::with_capacity(plan.cpu_rows.len());
    let mut experts: Vec<(usize, Range<usize>)> = Vec::with_capacity(plan.split.n_cpu);
    for expert in 0..n_exp {
        if plan.split.cpu_mask[expert] == 0 {
            continue;
        }
        let start = rows.len();
        for row in offsets[expert]..offsets[expert + 1] {
            let slot = *sorted
                .get(row)
                .ok_or_else(|| format!("grouped row {row} is outside the sorted map"))?;
            if slot >= 0 {
                rows.push(((slot / dsp::K_TOP as i32) as u32, rows.len() as u32));
            }
        }
        if rows.len() > start {
            experts.push((expert, start..rows.len()));
        }
    }
    if rows.len() != plan.cpu_rows.len() {
        return Err(format!(
            "the CPU work lists {} rows, the plan {}",
            rows.len(),
            plan.cpu_rows.len()
        ));
    }
    for (j, (&row, &(token, out))) in plan.cpu_rows.iter().zip(&rows).enumerate() {
        let slot = sorted.get(row as usize).copied().unwrap_or(-1);
        if out as usize != j || slot < 0 || (slot / dsp::K_TOP as i32) as u32 != token {
            return Err(format!(
                "CPU row {j} (grouped row {row}) disagrees with the plan"
            ));
        }
    }
    Ok(CpuWork { rows, experts })
}

/// The anchor tile of an SP window's panel scratch: the first K2-class tile,
/// else `None` (the stand-in tile of the admitted tile rows anchors it, as S1's
/// all-decode windows do). Never an F32-tail tile: its own call is not the K2
/// arm the panels run.
pub(crate) fn sp_anchor_tile(
    route: dsp::WindowRoute,
    tiles: &[TilePlan],
    few_row_limit: usize,
) -> Option<usize> {
    tiles.iter().position(|tile| {
        tile_cohort(route, tile.routing.route_rows, few_row_limit) == TileCohort::K2
    })
}

// ── receipt ───────────────────────────────────────────────────────────

/// What the slot knows of SP at receipt time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpSummary {
    pub(crate) knob: bool,
    pub(crate) admitted: Option<SpProfile>,
    pub(crate) decline: Option<SpDecline>,
    /// A window failed: SP is disabled for the process.
    pub(crate) unavailable: Option<String>,
}

/// Measured quantities of one SP window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpWindowReport {
    /// Host layers that ran the SP flow.
    pub(crate) layers: u64,
    /// Streamed payload bytes submitted (the ring's layouts).
    pub(crate) streamed_bytes: u64,
    /// Experts streamed, summed over layers.
    pub(crate) streamed_experts: u64,
    /// Experts run on the CPU, summed over layers.
    pub(crate) cpu_experts: u64,
    /// Routed slots run on the CPU, summed over layers.
    pub(crate) cpu_slots: u64,
}

/// One window of the receipt's `sp.windows`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpWindowEntry {
    pub(crate) rows: usize,
    /// The pre-mutation verdict (`Err` = the named `sp_bypass`).
    pub(crate) verdict: Result<(), SpBypass>,
    pub(crate) report: SpWindowReport,
}

/// The receipt's `sp` object. `None` when the knob is off: the receipt is then
/// the one it was, with no `sp` key.
///
/// `armed` is "admitted at load and not disabled since"; `decline` is the
/// sticky load-time reason (`null` when admitted), `unavailable` the runtime
/// failure that disabled SP. Every window names whether SP ran it and, when it
/// did not, the bypass reason (`null` when SP ran it or SP is not armed).
pub(crate) fn sp_receipt_json(summary: &SpSummary, windows: &[SpWindowEntry]) -> Option<Value> {
    if !summary.knob {
        return None;
    }
    let armed = summary.admitted.is_some() && summary.unavailable.is_none();
    let windows: Vec<Value> = windows
        .iter()
        .map(|entry| {
            let engaged = armed && entry.verdict.is_ok() && entry.report.layers > 0;
            let bypass = match (&entry.verdict, armed) {
                (Err(bypass), true) => Some(bypass.to_string()),
                _ => None,
            };
            json!({
                "rows": entry.rows,
                "engaged": engaged,
                "bypass": bypass,
                "layers": entry.report.layers,
                "streamed_bytes": entry.report.streamed_bytes,
                "streamed_experts": entry.report.streamed_experts,
                "cpu_experts": entry.report.cpu_experts,
                "cpu_slots": entry.report.cpu_slots,
            })
        })
        .collect();
    let engaged_windows = windows
        .iter()
        .filter(|window| window["engaged"] == json!(true))
        .count();
    Some(json!({
        "knob": true,
        "armed": armed,
        "decline": summary.decline.as_ref().map(ToString::to_string),
        "decline_reason": summary.decline.as_ref().map(|decline| decline.label()),
        "unavailable": summary.unavailable,
        "sp_max_rows": summary.admitted.map(|sp| sp.sp_max_rows),
        "s_cap": summary.admitted.map(|sp| sp.s_cap),
        "workers": summary.admitted.map(|sp| sp.workers),
        "pinned_bytes": summary.admitted.map(|sp| sp.pinned_bytes),
        "engaged_windows": engaged_windows,
        "windows": windows,
    }))
}

// ── host owners ───────────────────────────────────────────────────────

/// A `hipHostMalloc` slab the SP runtime owns.
struct Pinned(hip_bridge::PinnedHostBuffer);

// SAFETY: `Pinned` is the unique owner of its hipHostMalloc allocation. The
// host touches it only through `&mut self`/`&self` accessors on the driver
// thread; device-side reads and writes are fenced by the SP events (I3/I4) and
// by the xfer drain before the runtime frees or quarantines it (a failed drain
// leaks the allocation, never frees it). Deliberately not `Sync`.
unsafe impl Send for Pinned {}

impl Pinned {
    fn new(gpu: &Gpu, bytes: usize) -> Result<Self, String> {
        gpu.hip
            .host_buffer(bytes)
            .map(Self)
            .map_err(|error| format!("pinned host allocation of {bytes} bytes failed: {error}"))
    }

    fn bytes(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }

    fn u16s(&mut self) -> &mut [u16] {
        // SAFETY: hipHostMalloc storage is page aligned and any length is a
        // multiple of 2 for the buffers sized here; the checks below fire
        // otherwise. `u16` has no invalid bit pattern.
        let (head, body, tail) = unsafe { self.0.as_bytes_mut().align_to_mut::<u16>() };
        assert!(head.is_empty() && tail.is_empty());
        body
    }

    fn i32s(&mut self) -> &mut [i32] {
        // SAFETY: as `u16s`, for `i32`.
        let (head, body, tail) = unsafe { self.0.as_bytes_mut().align_to_mut::<i32>() };
        assert!(head.is_empty() && tail.is_empty());
        body
    }
}

/// The i32 view of a host mapping (page aligned; lengths here are multiples of 4).
fn mapped_i32s(bytes: &[u8]) -> &[i32] {
    // SAFETY: `i32` has no invalid bit pattern; alignment is checked below.
    let (head, body, tail) = unsafe { bytes.align_to::<i32>() };
    assert!(head.is_empty() && tail.is_empty());
    body
}

/// The four events of the per-layer protocol (§3.1).
struct SpEvents {
    /// compute -> xfer: tile `t`'s rows are prepared.
    x_ready: hip_bridge::Event,
    /// xfer -> host: the layer's x16 D2H is complete.
    x_done: hip_bridge::Event,
    /// xfer -> compute: the CPU rows and the inverse are uploaded.
    rows_ready: hip_bridge::Event,
    /// compute -> xfer: the layer's combine no longer reads the hybrid table.
    combine_done: hip_bridge::Event,
}

/// [`SpEventBackend`] over the real events: queries are host reads of the HIP
/// event state; `combine_done` is "recorded" once the host enqueued it.
struct SpEventView<'a> {
    hip: &'a hip_bridge::HipRuntime,
    ev: &'a SpEvents,
    combine_recorded: Option<usize>,
}

impl SpEventBackend for SpEventView<'_> {
    fn x_done_complete(&self, _layer: usize) -> bool {
        self.hip.event_query(&self.ev.x_done).unwrap_or(false)
    }

    fn rows_ready_complete(&self, _layer: usize) -> bool {
        self.hip.event_query(&self.ev.rows_ready).unwrap_or(false)
    }

    fn combine_done_recorded(&self, layer: usize) -> bool {
        self.combine_recorded == Some(layer)
    }
}

/// Everything SP owns after load: the `LoadOwner::SpHost` owner.
pub(crate) struct SpRuntime {
    pub(crate) profile: SpProfile,
    gate_up: DType,
    down: DType,
    mi: usize,
    pages: usize,
    placement: SpPlacement,
    trace: bool,
    xfer: hip_bridge::Stream,
    ev: SpEvents,
    /// Pinned `[sp_max_rows, hidden]` u16: the window's rotated F16 rows.
    x16: Pinned,
    /// Host-mapped coherent `[G_max]` i32: the layer's grouped-row -> slot
    /// map, written by a copy kernel on compute (never an SDMA copy).
    sorted: hip_bridge::MappedHostBuffer,
    /// Pinned `[s_cap, hidden]` u16: the CPU arm's BF16 rows.
    cpu_rows: Pinned,
    /// Pinned `[sp_max_rows * 10]` i32: the hybrid inverse.
    inv: Pinned,
    /// Pageable `[sp_max_rows, hidden]` f32: the widened activations.
    x_wide: Vec<f32>,
    scratch: SpCpuScratch,
    protocol: SpProtocol,
    /// The last layer whose `combine_done` was enqueued this window.
    combine_recorded: Option<usize>,
    /// The window in flight.
    acc: SpWindowReport,
    /// Streamed payload bytes the window's schedules submitted.
    submitted: u64,
    /// Every executed window's report, by window index; cleared per request.
    reports: Vec<(usize, SpWindowReport)>,
    /// Why SP was disabled (the sticky `sp_unavailable`).
    failure: Option<String>,
    /// The xfer drain failed: nothing SP owns, and nothing it reads or writes,
    /// is ever freed.
    quarantined: bool,
}

impl SpRuntime {
    /// Allocate the owner (§6): pinned host buffers, the xfer stream, the four
    /// events, the CPU scratch. Everything allocated before a failure is
    /// released (nothing was queued on it yet).
    pub(crate) fn bind(gpu: &mut Gpu, plan: &SpBindPlan) -> Result<Self, String> {
        let profile = plan.profile;
        let hidden = plan.hidden;
        let [x16_bytes, sorted_bytes, cpu_bytes, inv_bytes] =
            sp_owner_bytes(hidden, plan.n_exp, &profile).ok_or("SP owner sizes overflow")?;
        let charged = (x16_bytes + sorted_bytes + cpu_bytes + inv_bytes) as u64;
        if charged != profile.pinned_bytes {
            return Err(format!(
                "SP pinned owners hold {charged} bytes, the admission charged {}",
                profile.pinned_bytes
            ));
        }
        let x16 = Pinned::new(gpu, x16_bytes)?;
        let sorted = gpu
            .hip
            .mapped_host_buffer(sorted_bytes)
            .map_err(|error| format!("SP mapped sorted buffer of {sorted_bytes} bytes: {error}"))?;
        let cpu_rows = Pinned::new(gpu, cpu_bytes)?;
        let inv = Pinned::new(gpu, inv_bytes)?;
        let wide = profile
            .sp_max_rows
            .checked_mul(hidden)
            .ok_or("SP widened activation size overflows")?;
        let mut x_wide: Vec<f32> = Vec::new();
        x_wide
            .try_reserve_exact(wide)
            .map_err(|error| format!("SP widened activations ({wide} f32): {error}"))?;
        x_wide.resize(wide, 0.0);
        let mut scratch = SpCpuScratch::new();
        scratch
            .reserve(hidden, plan.mi, SP_CPU_CHUNK_SLOTS)
            .map_err(|error| format!("SP CPU scratch: {error:?}"))?;
        let xfer = gpu
            .hip
            .stream_create_non_blocking()
            .map_err(|error| format!("SP xfer stream: {error}"))?;
        let mut events: Vec<hip_bridge::Event> = Vec::with_capacity(4);
        for _ in 0..4 {
            match gpu
                .hip
                .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)
            {
                Ok(event) => events.push(event),
                Err(error) => {
                    for event in events {
                        let _ = gpu.hip.event_destroy(event);
                    }
                    let _ = gpu.hip.stream_destroy(xfer);
                    return Err(format!("SP event: {error}"));
                }
            }
        }
        let mut events = events.into_iter();
        let (Some(x_ready), Some(x_done), Some(rows_ready), Some(combine_done)) =
            (events.next(), events.next(), events.next(), events.next())
        else {
            return Err("SP events were not created".to_string());
        };
        Ok(Self {
            profile,
            gate_up: plan.gate_up,
            down: plan.down,
            mi: plan.mi,
            pages: plan.pages,
            placement: plan.placement,
            trace: plan.trace,
            xfer,
            ev: SpEvents {
                x_ready,
                x_done,
                rows_ready,
                combine_done,
            },
            x16,
            sorted,
            cpu_rows,
            inv,
            x_wide,
            scratch,
            protocol: SpProtocol::new(),
            combine_recorded: None,
            acc: SpWindowReport::default(),
            submitted: 0,
            reports: Vec::new(),
            failure: None,
            quarantined: false,
        })
    }

    /// Release the owner. The xfer stream is drained first; a failed drain
    /// QUARANTINES everything (leaked, never freed: the device may still read
    /// the pinned buffers or write the route region) and reports it.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> R<()> {
        if self.quarantined {
            std::mem::forget(self);
            return Err(invalid(
                "SP xfer drain had failed earlier: the SP owners stay quarantined",
            ));
        }
        if let Err(error) = gpu.hip.stream_synchronize(&self.xfer) {
            std::mem::forget(self);
            return Err(invalid(format!(
                "SP xfer drain failed ({error}): the SP owners are quarantined, not freed"
            )));
        }
        let Self { xfer, ev, .. } = self;
        let mut first: Option<Qwen4GpuForwardError> = None;
        for event in [ev.x_ready, ev.x_done, ev.rows_ready, ev.combine_done] {
            if let Err(error) = gpu.hip.event_destroy(event) {
                first.get_or_insert(error.into());
            }
        }
        if let Err(error) = gpu.hip.stream_destroy(xfer) {
            first.get_or_insert(error.into());
        }
        first.map_or(Ok(()), Err)
    }

    /// Whether a failed window disabled SP for the process.
    pub(crate) fn unavailable(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Every executed window's report, by window index.
    pub(crate) fn reports(&self) -> &[(usize, SpWindowReport)] {
        &self.reports
    }

    /// Forget the previous request's window reports.
    pub(crate) fn begin_request(&mut self) {
        self.reports.clear();
    }

    fn begin_window(&mut self) -> Result<(), SpProtocolError> {
        self.acc = SpWindowReport::default();
        self.submitted = 0;
        self.combine_recorded = None;
        self.protocol.begin_window()
    }
}

// ── the layer flow ────────────────────────────────────────────────────

/// One layer's trace record (lab `HIPFIRE_QWEN4_SP_TRACE`).
#[derive(Clone, Debug)]
struct LayerTrace {
    layer: usize,
    rows: usize,
    k: usize,
    active: usize,
    groups: usize,
    cpu_slots: usize,
    cpu_experts: usize,
    stream_bytes: u64,
    y_bytes: u64,
    predicted_ms: f64,
    plan_ms: f64,
    cpu_ms: f64,
    phase_ms: f64,
    cpu: SpCpuStats,
}

impl LayerTrace {
    fn line(&self) -> String {
        json!({
            "sp_layer": self.layer,
            "rows": self.rows,
            "k": self.k,
            "u": self.active,
            "groups": self.groups,
            "cpu_slots": self.cpu_slots,
            "cpu_experts": self.cpu_experts,
            "stream_bytes": self.stream_bytes,
            "y_bytes": self.y_bytes,
            "predicted_ms": self.predicted_ms,
            "plan_ms": self.plan_ms,
            "cpu_ms": self.cpu_ms,
            "e1_compute_ms": self.cpu.compute_ns as f64 / 1e6,
            "e1_validate_ms": self.cpu.validate_ns as f64 / 1e6,
            "e1_scatter_ms": self.cpu.scatter_ns as f64 / 1e6,
            "measured_ms": self.phase_ms,
        })
        .to_string()
    }
}

impl Run<'_, '_> {
    /// SP runs this window: the load armed it, no failure disabled it and the
    /// pre-mutation verdict qualified it.
    pub(super) fn sp_window_on(&self, w: &WinCtx<'_>) -> bool {
        self.rt.sp.is_some()
            && self
                .sp_verdicts
                .get(w.window.index)
                .is_some_and(|verdict| verdict.is_ok())
    }

    /// The route layout of the window: the S1 one, or the SP one.
    pub(super) fn window_layout(&self, w: &WinCtx<'_>, sp_on: bool) -> R<dsp::PromptRouteLayout> {
        if sp_on {
            sp_window_layout(self.profile, w.window.rows)
        } else {
            window_route_layout(
                self.profile,
                w.window.rows,
                w.window.k1_rows,
                w.window.tail_rows,
            )
        }
    }

    /// Window entry of an SP window: arm the route storage, start the protocol
    /// and the ring with zero enqueues.
    pub(super) fn sp_begin_window(
        &mut self,
        routes: &mut dsp::PromptRouteStorage,
        progress: &mut Progress,
    ) -> R<()> {
        progress.sp_armed = true;
        routes.set_sp(true);
        let sp = self
            .rt
            .sp
            .as_mut()
            .ok_or_else(|| invalid("slow-link SP runtime is not bound"))?;
        sp.begin_window().map_err(proto_err)?;
        let mut schedule = SpSchedule::new(&self.rt.schedule);
        self.rt.ring.prime(&*self.gpu, &mut schedule)?;
        Ok(())
    }

    /// Window exit of an SP window (after the ring drain and the timeline):
    /// the submitted payload must equal the planned groups' layouts (I6), the
    /// xfer stream is joined, the protocol closes.
    pub(super) fn sp_end_window(&mut self, stats: &PanelStats, window: usize) -> R<()> {
        let sp = self
            .rt
            .sp
            .as_mut()
            .ok_or_else(|| invalid("slow-link SP runtime is not bound"))?;
        if stats.submitted_payload_bytes != sp.submitted {
            return Err(invalid(format!(
                "slow-link SP window submitted {} payload bytes, its planned groups are {}",
                stats.submitted_payload_bytes, sp.submitted
            )));
        }
        self.gpu.hip.stream_synchronize(&sp.xfer)?;
        sp.protocol.end_window().map_err(proto_err)?;
        sp.acc.streamed_bytes = sp.submitted;
        let report = sp.acc;
        sp.reports.push((window, report));
        Ok(())
    }

    /// The SP part of a failed window's cleanup, after the ring drain: join the
    /// xfer stream (quarantining the SP owners when that fails), leave the
    /// poisoned protocol and disable SP for the process.
    pub(super) fn sp_abort(&mut self, error: &str, extra: &mut Vec<String>) {
        let Some(sp) = self.rt.sp.as_mut() else {
            return;
        };
        sp.protocol.fail();
        sp.failure
            .get_or_insert_with(|| format!("sp_unavailable: a window failed: {error}"));
        if let Err(drain) = self.gpu.hip.stream_synchronize(&sp.xfer) {
            sp.quarantined = true;
            extra.push(format!(
                "SP xfer drain failed: {drain}; the SP owners and the route region are quarantined"
            ));
        }
        let _ = sp.protocol.fail_window();
    }

    /// One host layer of an SP window (§3): tiles, one barrier, the plan, the
    /// streamed GPU arm, the CPU arm, the upload, the one canonical combine and
    /// the unchanged tails.
    pub(super) fn sp_host_layer(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        routes: &mut dsp::PromptRouteStorage,
        txn: &mut PromptTxn,
        report: &mut WindowReport,
    ) -> R<()> {
        let geo = self.geo;
        let hidden = geo.hidden;
        let rows = w.window.rows;
        let tile_count = w.tiles.len();
        routes.reset_for_layer();

        // §3.3 tiles: mixer prefix + route preparation, then x16 D2H on xfer.
        for (ti, tile) in w.tiles.iter().enumerate() {
            if layer == geo.band.start {
                self.embed_tile(w, ti)?;
            }
            self.tile_layer(w, ti, layer, Some(&mut *routes))?;
            self.cut_if(txn, tile, layer)?;
            let sp = self
                .rt
                .sp
                .as_mut()
                .ok_or_else(|| invalid("slow-link SP runtime is not bound"))?;
            sp.protocol
                .stage_tile(layer, ti, tile_count)
                .map_err(proto_err)?;
            let source = routes
                .x_f16_rows(tile.window_row, tile.rows)
                .map_err(dsp_err("SP x rows"))?;
            let compute = self
                .gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("slow-link SP needs the compute stream"))?;
            self.gpu.hip.event_record(&sp.ev.x_ready, Some(compute))?;
            self.gpu.hip.stream_wait_event(&sp.xfer, &sp.ev.x_ready)?;
            let (offset, bytes) = (tile.window_row * hidden * 2, tile.rows * hidden * 2);
            let target = &mut sp.x16.bytes()[offset..offset + bytes];
            self.gpu
                .hip
                .memcpy_dtoh_async(target, &source.buf, Some(&sp.xfer))?;
            if ti + 1 == tile_count {
                self.gpu.hip.event_record(&sp.ev.x_done, Some(&sp.xfer))?;
            }
        }

        // §3.4 barrier: the unchanged count barrier plus the sorted download.
        let sp = self
            .rt
            .sp
            .as_mut()
            .ok_or_else(|| invalid("slow-link SP runtime is not bound"))?;
        let mut sorted_words = 0usize;
        let route_plan = dsp::finish_route_plan_with(
            &mut *self.gpu,
            routes,
            dsp::PanelLimits::S1,
            &mut |gpu, routes| {
                let sorted = routes.sorted();
                sorted_words = sorted.buf.size() / 4;
                gpu.slowlink_copy_i32_to_mapped(&sorted, &sp.sorted, sorted_words)
                    .map_err(|error| {
                        hipfire_dispatch::types::DispatchError::Hip(format!(
                            "slow-link SP sorted download: {error}"
                        ))
                    })
            },
        )
        .map_err(dsp_err("route plan"))?;
        report.route_barrier_ms += route_plan.barrier.sync_ms;
        let phase = Instant::now();

        // §3.5 plan: exact counts, then the placement and the layer plan.
        // SAFETY: the copy kernel ran on the compute stream before the count
        // barrier, whose wait on that stream just returned; no GPU write to
        // this buffer is in flight until the next layer's barrier.
        let sorted_all: &[i32] = mapped_i32s(unsafe { sp.sorted.host_bytes() });
        let sorted: &[i32] = &sorted_all[..sorted_words];
        let exact = exact_counts(sorted, &route_plan);
        // A readback defect poisons the protocol and is reported by name.
        let barrier = sp.protocol.barrier(exact.is_ok());
        exact.map_err(dsp_err("exact counts"))?;
        barrier.map_err(proto_err)?;
        let cost = sp_layer_cost(rows, hidden, sp.profile.workers, sp.placement);
        let limits = SpLimits {
            cpu_slot_cap: sp.profile.s_cap,
            ..SpLimits::S1_XTX
        };
        let plan = plan_sp_layer(&route_plan, sorted, rows, &cost, limits)
            .map_err(dsp_err("SP layer plan"))?;
        sp.protocol
            .plan(
                plan.groups.len(),
                sp.pages,
                plan.cpu_rows.len(),
                sp.profile.s_cap,
            )
            .map_err(proto_err)?;
        let plan_ms = phase.elapsed().as_secs_f64() * 1e3;

        // §3.6-3.7 selective streaming and the GPU arm, all enqueued at once.
        let by_group = panels_by_group(
            plan.gpu_panels.iter().map(|panel| panel.binding),
            plan.groups.len(),
        )?;
        let work: Vec<GroupWork> = (0..plan.groups.len())
            .map(|index| GroupWork {
                group: index as u32,
                bound: by_group[index].start < by_group[index].end,
            })
            .collect();
        let mut schedule = SpSchedule::new(&self.rt.schedule);
        schedule.plan_layer(layer, &plan.groups)?;
        self.rt.ring.enter_layer(&*self.gpu, &mut schedule, layer)?;
        let layer_stream_bytes = schedule.submitted_bytes();
        sp.submitted += layer_stream_bytes;
        // Streamed bytes the routed rows read: every streamed expert has rows.
        for &expert in plan.groups.iter().flatten() {
            report.useful_bytes += self
                .rt
                .schedule
                .expert_extent_bytes(layer, usize::from(expert))
                .unwrap_or(0) as u64;
        }
        sp.protocol.enqueue_gpu().map_err(proto_err)?;
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        let first_bound = work.iter().position(|group| group.bound);
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        let inject = self
            .bundle
            .slowlink_take_injection(SlowlinkFailPoint::DuringPanel);
        let anchor_index = sp_anchor_tile(self.profile.geometry.route, w.tiles, geo.few_row_limit);
        let stand_in;
        let anchor: &TilePlan = match anchor_index {
            Some(ti) => &w.tiles[ti],
            None => {
                let anchor_rows = self.profile.tile_rows;
                stand_in = TilePlan {
                    index: 0,
                    window: w.window.index,
                    prompt_row: 0,
                    window_row: 0,
                    stream_offset: 0,
                    rows: anchor_rows,
                    position: w.tiles[0].position,
                    routing: tile_routing(&*self.forward, &mut *self.gpu, anchor_rows),
                    ends_at_cut: None,
                    last_in_window: false,
                };
                &stand_in
            }
        };
        let hybrid = routes.sp_rows();
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, anchor.rows);
        let walked = with_tile_frames(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            &[anchor],
            layer,
            |gpu, frames| {
                let frame = &frames[0];
                let (tile_scopes, _) = self.forward.enter_tile_scopes(gpu, anchor.rows);
                let sealed = (|| -> R<_> {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a host layer's FFN is not MoE"));
                    };
                    Ok(frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?)
                })();
                Qwen4GpuForward::leave_tile_scopes(gpu, tile_scopes);
                let sealed = sealed?;
                let (tile_scopes, _) = self.forward.enter_tile_scopes(gpu, anchor.rows);
                let call = dsp::PreparedChunkCall::new(
                    &*gpu,
                    &sealed,
                    anchor.window_row,
                    anchor.position,
                    self.profile.geometry.route,
                );
                Qwen4GpuForward::leave_tile_scopes(gpu, tile_scopes);
                let call = call.map_err(dsp_err("admission"))?;
                let mut scratch = dsp::PanelScratch::new(&call, dsp::PanelLimits::S1)
                    .map_err(dsp_err("panel scratch"))?;
                let mut ring = HipRing {
                    ring: &mut self.rt.ring,
                    schedule: &mut schedule,
                };
                let route_plan = &route_plan;
                let routes = &*routes;
                walk_layer_groups(&mut ring, gpu, layer, &work, &mut |gpu, index, parts| {
                    let parts = parts
                        .ok_or_else(|| invalid("slow-link SP panel lease carries no binding"))?;
                    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
                    if inject && Some(index) == first_bound {
                        return Err(invalid("injected slow-link failure during a panel"));
                    }
                    let ids = &plan.groups[index];
                    let (Some(&lo), Some(&last)) = (ids.first(), ids.last()) else {
                        return Err(invalid("an empty SP expert group"));
                    };
                    let binding = dsp::ExpertPanelBinding {
                        layer,
                        epoch: parts.epoch,
                        expert_lo: usize::from(lo),
                        expert_hi: usize::from(last) + 1,
                        gate_up_table: parts.gate_up_table,
                        down_table: parts.down_table,
                        codec: dsp::PanelCodec::Mq4,
                        source: dsp::BindingSource::Staged {
                            slot: parts.ticket.slot as u32,
                        },
                    };
                    for panel_index in by_group[index].clone() {
                        let panel = &plan.gpu_panels[panel_index];
                        let y_rows = view(
                            &hybrid,
                            plan.gpu_base[panel_index] * hidden,
                            panel.rows * hidden,
                        );
                        dsp::execute_panel_rows(
                            gpu,
                            binding,
                            route_plan.view(routes, panel),
                            &mut scratch,
                            &y_rows,
                        )
                        .map_err(dsp_err("SP panel"))?;
                    }
                    Ok(())
                })
            },
        );
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        let walked = walked?;
        for _ in 0..walked.acquired + walked.discarded {
            sp.protocol.retire_group().map_err(proto_err)?;
        }

        // §3.8 CPU arm on the driver thread while the GPU and the link work.
        let cpu_start = Instant::now();
        self.gpu.hip.event_synchronize(&sp.ev.x_done)?;
        let SpRuntime {
            protocol,
            ev,
            xfer,
            x16,
            sorted: sorted_host,
            cpu_rows,
            inv,
            x_wide,
            scratch,
            combine_recorded,
            gate_up,
            down,
            mi,
            ..
        } = &mut *sp;
        let events = SpEventView {
            hip: &self.gpu.hip,
            ev,
            combine_recorded: *combine_recorded,
        };
        protocol.begin_cpu(&events).map_err(proto_err)?;
        protocol.host_write_cpu_rows(&events).map_err(proto_err)?;
        let cpu_slots = plan.cpu_rows.len();
        let mut cpu_stats = SpCpuStats::default();
        let cpu_result: R<()> = (|| {
            let inverse = &mut inv.i32s()[..rows * dsp::K_TOP];
            inverse.copy_from_slice(&plan.inverse_hybrid);
            if cpu_slots == 0 {
                return Ok(());
            }
            let n = rows * hidden;
            widen_f16_rows(&x16.u16s()[..n], &mut x_wide[..n]);
            // SAFETY: as at the barrier above; the next write is the next
            // layer's barrier, after this host read in program order.
            let sorted: &[i32] = &mapped_i32s(unsafe { sorted_host.host_bytes() })[..sorted_words];
            let cpu_work = build_cpu_work(&plan, sorted, &route_plan.offsets)
                .map_err(|error| invalid(format!("slow-link SP CPU work: {error}")))?;
            let sources = self
                .rt
                .schedule
                .sources_of(layer)
                .ok_or_else(|| invalid(format!("SP layer {layer} has no expert sources")))?;
            let (gate_up_quant, down_quant) = (
                cpu_quant_for(*gate_up).ok_or_else(|| invalid("SP gate_up has no CPU quant"))?,
                cpu_quant_for(*down).ok_or_else(|| invalid("SP down has no CPU quant"))?,
            );
            let mut experts: Vec<SpCpuExpert<'_>> = Vec::with_capacity(cpu_work.experts.len());
            for (expert, range) in &cpu_work.experts {
                let source = sources.expert(*expert).ok_or_else(|| {
                    invalid(format!(
                        "SP layer {layer} expert {expert} has no host source"
                    ))
                })?;
                // SAFETY: the extents lie inside the layer's pinned host
                // ranges (checked by `LayerExpertSources`) and stay valid and
                // unmodified while the sources' owner lease lives; the
                // schedule that holds it outlives this call (I9).
                let (gate_up_bytes, down_bytes) = unsafe {
                    (
                        std::slice::from_raw_parts(
                            source.gate_up.addr as *const u8,
                            source.gate_up.len,
                        ),
                        std::slice::from_raw_parts(source.down.addr as *const u8, source.down.len),
                    )
                };
                experts.push(SpCpuExpert {
                    spans: CpuMoeExpert {
                        gate_up_quant,
                        gate_up: gate_up_bytes,
                        down_quant,
                        down: down_bytes,
                    },
                    rows: &cpu_work.rows[range.clone()],
                });
            }
            let out = &mut cpu_rows.u16s()[..cpu_slots * hidden];
            cpu_stats = moe_cpu_grouped_rows_bf16(
                hidden,
                *mi,
                &x_wide[..n],
                &experts,
                SP_CPU_CHUNK_SLOTS,
                scratch,
                out,
            )
            .map_err(dsp_err("SP CPU arm"))?;
            Ok(())
        })();
        protocol
            .cpu_returned(cpu_result.is_ok())
            .map_err(proto_err)?;
        cpu_result?;
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

        // Upload on xfer: wait the previous combine (I5), two H2D, rows_ready.
        let xfer: &hip_bridge::Stream = xfer;
        if combine_recorded.is_some() {
            self.gpu.hip.stream_wait_event(xfer, &ev.combine_done)?;
        }
        if cpu_slots > 0 {
            let target = hybrid.sub_offset(plan.gpu_rows() * hidden, cpu_slots * hidden);
            let bytes = &cpu_rows.bytes()[..cpu_slots * hidden * 2];
            self.gpu.hip.memcpy_htod_async(&target.buf, bytes, xfer)?;
        }
        let inverse_target = routes.sp_inverse();
        let inverse_bytes = &inv.bytes()[..rows * dsp::K_TOP * 4];
        self.gpu
            .hip
            .memcpy_htod_async(&inverse_target.buf, inverse_bytes, xfer)?;
        self.gpu.hip.event_record(&ev.rows_ready, Some(xfer))?;
        let events = SpEventView {
            hip: &self.gpu.hip,
            ev,
            combine_recorded: *combine_recorded,
        };
        protocol.upload(&events).map_err(proto_err)?;

        // §3.9 compute waits rows_ready, one canonical combine, combine_done.
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, anchor.rows);
        let combined = (|| -> R<()> {
            let compute = self
                .gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("slow-link SP needs the compute stream"))?;
            self.gpu.hip.stream_wait_event(compute, &ev.rows_ready)?;
            dsp::sp_finish_combine(&mut *self.gpu, routes, plan.hybrid_rows)
                .map_err(dsp_err("SP combine"))?;
            let compute = self
                .gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("slow-link SP needs the compute stream"))?;
            self.gpu.hip.event_record(&ev.combine_done, Some(compute))?;
            Ok(())
        })();
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        combined?;
        *combine_recorded = Some(layer);
        protocol.combine().map_err(proto_err)?;
        let phase_ms = phase.elapsed().as_secs_f64() * 1e3;

        sp.acc.layers += 1;
        sp.acc.streamed_experts += plan.split.k as u64;
        sp.acc.cpu_experts += plan.split.n_cpu as u64;
        sp.acc.cpu_slots += cpu_slots as u64;
        if sp.trace {
            let trace = LayerTrace {
                layer,
                rows,
                k: plan.split.k,
                active: plan.split.n_active,
                groups: plan.groups.len(),
                cpu_slots,
                cpu_experts: plan.split.n_cpu,
                stream_bytes: layer_stream_bytes,
                y_bytes: (cpu_slots * hidden * 2 + rows * dsp::K_TOP * 4) as u64,
                predicted_ms: plan.split.predicted_ms,
                plan_ms,
                cpu_ms,
                phase_ms,
                cpu: cpu_stats,
            };
            eprintln!("{}", trace.line());
        }

        // The unchanged per-tile tails.
        for ti in 0..tile_count {
            self.tile_tail(w, ti, layer, routes)?;
            if layer == geo.last_layer {
                self.tile_output(w, ti)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::slowlink_prompt::{plan_for_profile, product_profile_on};
    use super::*;
    use crate::slowlink_admission::RequestVerdict;
    use crate::slowlink_sp::{SpBypass, SP_MIN_ROWS};
    use hipfire_dispatch::sp_plan::{plan_layer_split, NUM_EXPERTS};

    const AGG: dsp::WindowRoute = dsp::WindowRoute::Aggregate;

    fn sp_profile() -> SpProfile {
        SpProfile {
            sp_max_rows: SP_MAX_ROWS,
            s_cap: 5 * SP_MAX_ROWS,
            workers: 16,
            pinned_bytes: 0,
        }
    }

    fn gfx1100_routing(rows: usize) -> TileRouting {
        TileRouting {
            route_rows: rows,
            bf16_streams: rows >= dsp::AGGREGATE_MIN_TILE_ROWS,
        }
    }

    fn plan_agg(rows: usize, cuts: &[usize]) -> Qwen4PromptPlan {
        plan_for_profile(&product_profile_on(AGG), 0, rows, cuts, &gfx1100_routing, 8)
            .unwrap()
            .expect("the aggregate request plans")
    }

    fn shape(rows: usize, capture: CaptureKind) -> RequestShape {
        RequestShape {
            rows,
            entry_pos: 0,
            session_cuts: 0,
            capture,
        }
    }

    // ── selection truth table ─────────────────────────────────────────

    #[test]
    fn sp_off_request_selection_is_exactly_the_s1_qualification() {
        let profile = product_profile_on(AGG);
        for mode in [SlowlinkMode::Auto, SlowlinkMode::Force, SlowlinkMode::Off] {
            for rows in [1, 4, 5, 300, 768, 769, 1537, 16_896, 20_000] {
                for capture in [
                    CaptureKind::None,
                    CaptureKind::Bounded,
                    CaptureKind::AllRows,
                    CaptureKind::External,
                ] {
                    // Cuts are irrelevant without SP: the shape's own count rules.
                    for cuts in [0, 3] {
                        let s = shape(rows, capture);
                        assert_eq!(
                            sp_request_verdict(mode, &profile, None, s, cuts),
                            qualify_request(mode, &profile, &s).map(|_: RequestVerdict| ()),
                            "{mode:?} rows {rows} {capture:?} cuts {cuts}"
                        );
                    }
                }
            }
        }
        // No SP: no window verdict is made and nothing is named.
        let plan = plan_agg(1000, &[]);
        assert!(sp_window_verdicts(None, &plan, CaptureKind::None).is_empty());
    }

    #[test]
    fn sp_on_request_selection_is_all_or_incumbent_below_the_s1_floor() {
        let profile = product_profile_on(AGG);
        let sp = sp_profile();
        let tile = profile.tile_rows;
        assert_eq!(tile, 768);
        let auto = |rows: usize, cuts: usize, capture: CaptureKind| {
            sp_request_verdict(
                SlowlinkMode::Auto,
                &profile,
                Some(&sp),
                shape(rows, capture),
                cuts,
            )
        };
        // 1..=4 rows: D2's windows, never SP's and never S1's: unchanged.
        for rows in 1..SP_MIN_ROWS {
            assert!(matches!(
                auto(rows, 0, CaptureKind::None),
                Err(RequestBypass::TooShort { .. })
            ));
        }
        // 5..=768 rows: S1 admits the one-tile request only through SP.
        for rows in [5, 6, 300, 511, 512, 767, 768] {
            assert_eq!(auto(rows, 0, CaptureKind::None), Ok(()), "rows {rows}");
            assert_eq!(auto(rows, 0, CaptureKind::Bounded), Ok(()), "rows {rows}");
            assert_eq!(
                auto(rows, 1, CaptureKind::None),
                Err(RequestBypass::Sp(SpBypass::InteriorCuts { cuts: 1 })),
                "rows {rows}: an interior session cut leaves the request to the incumbent"
            );
            assert_eq!(
                auto(rows, 0, CaptureKind::AllRows),
                Err(RequestBypass::AllRowCapture)
            );
            assert_eq!(
                auto(rows, 0, CaptureKind::External),
                Err(RequestBypass::ExternalCapture)
            );
        }
        // Above one tile the request is judged exactly as without SP, cuts included.
        for rows in [769, 1537, 5376, 5377, 16_897, 40_000] {
            for cuts in [0, 1, 3] {
                assert_eq!(
                    auto(rows, cuts, CaptureKind::None),
                    qualify_request(
                        SlowlinkMode::Auto,
                        &profile,
                        &shape(rows, CaptureKind::None)
                    )
                    .map(|_: RequestVerdict| ()),
                    "rows {rows} cuts {cuts}"
                );
            }
        }
        // The kill switch still wins.
        assert_eq!(
            sp_request_verdict(
                SlowlinkMode::Off,
                &profile,
                Some(&sp),
                shape(300, CaptureKind::None),
                0
            ),
            Err(RequestBypass::KillSwitch)
        );
        // Forced S1 below the floor: SP's rules replace the forced minimum only
        // for the SP-sized one-tile request.
        assert_eq!(
            sp_request_verdict(
                SlowlinkMode::Force,
                &profile,
                Some(&sp),
                shape(5, CaptureKind::None),
                0
            ),
            Ok(())
        );
        assert_eq!(
            sp_request_verdict(
                SlowlinkMode::Force,
                &profile,
                Some(&sp),
                shape(5, CaptureKind::None),
                2
            ),
            Err(RequestBypass::Sp(SpBypass::InteriorCuts { cuts: 2 }))
        );
    }

    #[test]
    fn sp_window_verdicts_name_every_window_before_any_mutation() {
        let sp = sp_profile();
        let verdicts = |rows: usize, cuts: &[usize]| {
            let plan = plan_agg(rows, cuts);
            (
                sp_window_verdicts(Some(&sp), &plan, CaptureKind::None),
                plan,
            )
        };
        // One-tile requests: the single window is SP's.
        for rows in [5, 300, 768] {
            let (v, plan) = verdicts(rows, &[]);
            assert_eq!(v, vec![Ok(())], "rows {rows}");
            assert_eq!(
                sp_floor_verdict(plan.tile_rows, rows, &v),
                Ok(()),
                "rows {rows}"
            );
        }
        // Several tiles in one window, up to the SP row cap.
        for rows in [1536, 2304, 5376] {
            assert_eq!(verdicts(rows, &[]).0, vec![Ok(())], "rows {rows}");
        }
        // Past the cap the window stays on the stream-all sweep (not the incumbent).
        let (v, plan) = verdicts(5377, &[]);
        assert_eq!(
            v,
            vec![Err(SpBypass::TooManyRows {
                rows: 5377,
                maximum: SP_MAX_ROWS
            })]
        );
        assert_eq!(sp_floor_verdict(plan.tile_rows, 5377, &v), Ok(()));
        // 769 rows: the 1-row tail tile is a decode tile; the window stays S1's.
        let (v, plan) = verdicts(769, &[]);
        assert_eq!(v, vec![Err(SpBypass::OneRowTile { tile: 1 })]);
        assert_eq!(sp_floor_verdict(plan.tile_rows, 769, &v), Ok(()));
        // A short last window of a multi-window request is SP's; the long one is not.
        let (v, plan) = verdicts(16_896 + 300, &[]);
        assert_eq!(plan.windows.len(), 2);
        assert_eq!(
            v,
            vec![
                Err(SpBypass::TooManyRows {
                    rows: 16_896,
                    maximum: SP_MAX_ROWS
                }),
                Ok(())
            ]
        );
        assert_eq!(sp_floor_verdict(plan.tile_rows, plan.rows, &v), Ok(()));
        // A last window of one row is below the SP minimum (a decode row).
        let (v, _) = verdicts(16_896 + 1, &[]);
        assert_eq!(
            v[1],
            Err(SpBypass::TooFewRows {
                rows: 1,
                minimum: SP_MIN_ROWS
            })
        );
        // An interior session cut leaves the window to the stream-all sweep ...
        let (v, _) = verdicts(1000, &[500]);
        assert_eq!(v, vec![Err(SpBypass::InteriorCuts { cuts: 1 })]);
        // ... but below the floor the request is the incumbent's.
        let (v, plan) = verdicts(300, &[100]);
        assert_eq!(
            sp_floor_verdict(plan.tile_rows, 300, &v),
            Err(RequestBypass::Sp(SpBypass::InteriorCuts { cuts: 1 }))
        );
        // All-row capture never reaches the plan, but the verdict still names it.
        let plan = plan_agg(1000, &[]);
        assert_eq!(
            sp_window_verdicts(Some(&sp), &plan, CaptureKind::AllRows),
            vec![Err(SpBypass::Capture(CaptureKind::AllRows))]
        );
    }

    #[test]
    fn sp_anchor_is_the_first_k2_tile_and_never_an_f32_tail() {
        let few = 8;
        let route = AGG;
        let tail_only = plan_agg(300, &[]);
        assert_eq!(
            sp_anchor_tile(route, plan_tiles(&tail_only), few),
            None,
            "a window of one F32-tail tile is anchored by the stand-in"
        );
        let mixed = plan_agg(768 + 300, &[]);
        assert_eq!(sp_anchor_tile(route, plan_tiles(&mixed), few), Some(0));
        let short_first = plan_agg(600, &[]);
        assert_eq!(
            sp_anchor_tile(route, plan_tiles(&short_first), few),
            Some(0)
        );
    }

    fn plan_tiles(plan: &Qwen4PromptPlan) -> &[TilePlan] {
        plan.window_tiles(&plan.windows[0])
    }

    // ── load facts ────────────────────────────────────────────────────

    #[test]
    fn sp_region_fit_is_the_largest_layout_inside_the_route_region() {
        let profile = product_profile_on(AGG);
        let fit = sp_region_fit_rows(&profile);
        assert!(fit > 0);
        assert!(sp_window_layout(&profile, fit).is_ok());
        let cap = SP_MAX_ROWS.min(profile.route_dims().rows);
        assert!(fit <= cap);
        if fit < cap {
            assert!(
                sp_window_layout(&profile, fit + 1).is_err(),
                "the fit is the largest"
            );
        }
        // The layout is the SP one: no cohort rows, SP fields present.
        let layout = sp_window_layout(&profile, fit.min(300)).unwrap();
        assert!(layout.has_sp());
        assert!(sp_window_layout(&profile, 0).is_err());
        assert!(sp_window_layout(&profile, profile.route_dims().rows + 1).is_err());
    }

    #[test]
    fn sp_pinned_budget_is_available_host_ram_less_headroom_and_s1_staging() {
        let headroom = crate::expert_residency::HOST_RAM_HEADROOM_BYTES;
        assert_eq!(sp_pinned_budget(None, 1 << 40, 0), 0);
        assert_eq!(
            sp_pinned_budget(Some(headroom + (200 << 20)), 0, 50 << 20),
            150 << 20
        );
        assert_eq!(
            sp_pinned_budget(Some(headroom), 100, 0),
            100,
            "the TTM page pool counts"
        );
        assert_eq!(sp_pinned_budget(Some(headroom - 1), 0, 0), 0);
        assert_eq!(sp_pinned_budget(Some(u64::MAX), u64::MAX, u64::MAX), 0);
    }

    #[test]
    fn sp_owner_bytes_sum_to_the_admitted_pinned_charge() {
        let profile = SpProfile {
            pinned_bytes: crate::slowlink_sp::sp_pinned_bytes(2560, 512, SP_MAX_ROWS).unwrap(),
            ..sp_profile()
        };
        let parts = sp_owner_bytes(2560, 512, &profile).unwrap();
        assert_eq!(parts.iter().sum::<usize>() as u64, profile.pinned_bytes);
        assert_eq!(parts[0], SP_MAX_ROWS * 2560 * 2);
        assert_eq!(parts[2], 5 * SP_MAX_ROWS * 2560 * 2);
        assert_eq!(parts[3], SP_MAX_ROWS * 40);
    }

    #[test]
    fn sp_uniform_cpu_dtypes_need_one_cpu_executable_pair() {
        let mq4 = (DType::MQ4G256V2, DType::MQ4G128V2);
        assert_eq!(uniform_cpu_dtypes([mq4, mq4, mq4]), Some(mq4));
        assert_eq!(uniform_cpu_dtypes([]), None, "no host expert");
        assert_eq!(
            uniform_cpu_dtypes([mq4, (DType::MQ4G256V2, DType::MQ4G256V2)]),
            None,
            "a mixed layer set is refused"
        );
        assert_eq!(
            uniform_cpu_dtypes([(DType::Raw, DType::Raw)]),
            None,
            "a dtype the CPU engine cannot read is refused"
        );
    }

    // ── placement and the CPU work ────────────────────────────────────

    #[test]
    fn sp_layer_cost_applies_the_lab_placement_as_a_cost() {
        let auto = sp_layer_cost(1024, 2560, 16, SpPlacement::Auto);
        assert_eq!(
            auto,
            SplitCost::sp_hook_xtx(1024, 2560, 16),
            "production is the frozen hook transport model"
        );
        let mut counts = [0u32; NUM_EXPERTS];
        for (i, c) in counts.iter_mut().enumerate().take(300) {
            *c = if i < 40 { 120 } else { 1 + (i as u32 % 5) };
        }
        let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        let active = counts.iter().filter(|&&c| c > 0).count();
        let mixed = plan_layer_split(&counts, &auto);
        assert!(
            mixed.k > 0 && mixed.k < active,
            "the model splits this layer"
        );
        let all = plan_layer_split(&counts, &sp_layer_cost(1024, 2560, 16, SpPlacement::All));
        assert_eq!((all.k, all.n_cpu, all.cpu_slots), (active, 0, 0));
        let cpu = plan_layer_split(&counts, &sp_layer_cost(1024, 2560, 16, SpPlacement::Cpu));
        assert_eq!((cpu.k, cpu.n_cpu, cpu.cpu_slots), (0, active, total));
    }

    /// Deterministic skewed routes: `rows` tokens x 10 distinct experts.
    fn routes(rows: usize, seed: u64) -> Vec<[u16; 10]> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        (0..rows)
            .map(|_| {
                let mut ids = [0u16; 10];
                let mut taken = 0;
                while taken < 10 {
                    // Heavy tail: half the picks land in 16 hot experts.
                    let e = if next() % 2 == 0 {
                        (next() % 16) as u16
                    } else {
                        (next() % 300) as u16
                    };
                    if !ids[..taken].contains(&e) {
                        ids[taken] = e;
                        taken += 1;
                    }
                }
                ids
            })
            .collect()
    }

    /// The stable (expert, slot) pad-16 grouping the device scatter produces.
    fn grouping(routes: &[[u16; 10]]) -> (dsp::LayerRoutePlan, Vec<i32>) {
        let mut per: Vec<Vec<i32>> = vec![Vec::new(); NUM_EXPERTS];
        for (token, ids) in routes.iter().enumerate() {
            for (rank, &expert) in ids.iter().enumerate() {
                per[usize::from(expert)].push((token * 10 + rank) as i32);
            }
        }
        let (mut sorted, mut counts) = (Vec::new(), Vec::new());
        for slots in &per {
            let padded = slots.len().div_ceil(16) * 16;
            counts.push(padded as i32);
            sorted.extend(slots);
            sorted.extend(std::iter::repeat_n(-1, padded - slots.len()));
        }
        let plan =
            dsp::LayerRoutePlan::from_counts(&counts, Vec::new(), dsp::BarrierCost::default())
                .unwrap();
        // The device array is the grouped-row bound, longer than the used rows.
        sorted.extend(std::iter::repeat_n(-1, 512 * 15));
        (plan, sorted)
    }

    fn plan_with(
        routes: &[[u16; 10]],
        placement: SpPlacement,
    ) -> (dsp::LayerRoutePlan, Vec<i32>, SpLayerPlan) {
        let (plan, sorted) = grouping(routes);
        let cost = sp_layer_cost(routes.len(), 2560, 16, placement);
        let sp = plan_sp_layer(&plan, &sorted, routes.len(), &cost, SpLimits::S1_XTX).unwrap();
        (plan, sorted, sp)
    }

    #[test]
    fn sp_cpu_work_lists_every_cpu_slot_once_in_plan_order() {
        for (rows, placement) in [
            (64, SpPlacement::Auto),
            (300, SpPlacement::Auto),
            (300, SpPlacement::Cpu),
            (300, SpPlacement::All),
            (1024, SpPlacement::Auto),
        ] {
            let routes = routes(rows, 0x5eed ^ rows as u64);
            let (plan, sorted, sp) = plan_with(&routes, placement);
            let work = build_cpu_work(&sp, &sorted, &plan.offsets).unwrap();
            assert_eq!(work.rows.len(), sp.cpu_rows.len(), "{rows} {placement:?}");
            // Output indices are the hybrid-row order, a permutation of 0..S.
            assert!(work
                .rows
                .iter()
                .enumerate()
                .all(|(j, &(_, out))| out as usize == j));
            // Expert ranges are ascending, disjoint and cover every row.
            let mut next = 0usize;
            let mut last: Option<usize> = None;
            for (expert, range) in &work.experts {
                assert!(last.is_none_or(|l| *expert > l));
                last = Some(*expert);
                assert_eq!(range.start, next);
                assert!(range.end > range.start);
                next = range.end;
                assert_eq!(sp.split.cpu_mask[*expert], 1);
                // Every listed token routes to that expert.
                for &(token, _) in &work.rows[range.clone()] {
                    assert!(routes[token as usize].contains(&(*expert as u16)));
                }
            }
            assert_eq!(next, work.rows.len());
            // Every routed slot is streamed or on the CPU: nothing is dropped.
            let streamed: usize = sp
                .gpu_panels
                .iter()
                .map(|panel| {
                    (panel.row_base..panel.row_base + panel.rows)
                        .filter(|&r| sorted[r] >= 0)
                        .count()
                })
                .sum();
            assert_eq!(
                streamed + work.rows.len(),
                rows * 10,
                "{rows} {placement:?}"
            );
            match placement {
                SpPlacement::Cpu => {
                    assert!(sp.groups.is_empty() && sp.gpu_panels.is_empty());
                    assert_eq!(work.rows.len(), rows * 10);
                }
                SpPlacement::All => {
                    assert!(work.rows.is_empty() && work.experts.is_empty());
                    assert!(sp.cpu_rows.is_empty() && !sp.groups.is_empty());
                }
                SpPlacement::Auto => {}
            }
        }
    }

    #[test]
    fn sp_cpu_work_refuses_a_disagreeing_plan() {
        let routes = routes(64, 7);
        let (plan, sorted, sp) = plan_with(&routes, SpPlacement::Cpu);
        let mut shifted = sp.clone();
        shifted.cpu_rows.rotate_left(1);
        assert!(build_cpu_work(&shifted, &sorted, &plan.offsets).is_err());
        let mut short = sp.clone();
        short.cpu_rows.pop();
        assert!(build_cpu_work(&short, &sorted, &plan.offsets).is_err());
        assert!(build_cpu_work(&sp, &sorted, &plan.offsets[..plan.offsets.len() - 1]).is_err());
    }

    #[test]
    fn sp_f16_widening_is_exact_for_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let ours = f16_bits_to_f32(bits);
            let reference = half::f16::from_bits(bits).to_f32();
            if reference.is_nan() {
                assert!(ours.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(ours.to_bits(), reference.to_bits(), "{bits:#06x}");
            }
        }
        let src: Vec<u16> = (0..70_000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 16) as u16)
            .collect();
        let mut wide = vec![0f32; src.len()];
        widen_f16_rows(&src, &mut wide);
        for (&h, &w) in src.iter().zip(&wide) {
            let reference = half::f16::from_bits(h).to_f32();
            assert!(w.to_bits() == reference.to_bits() || (w.is_nan() && reference.is_nan()));
        }
    }

    // ── receipt ───────────────────────────────────────────────────────

    #[test]
    fn sp_receipt_object_is_absent_with_the_knob_off() {
        assert_eq!(sp_receipt_json(&SpSummary::default(), &[]), None);
        let off = SpSummary {
            decline: Some(SpDecline::KnobOff),
            ..SpSummary::default()
        };
        assert_eq!(sp_receipt_json(&off, &[]), None, "knob false: no sp key");
    }

    #[test]
    fn sp_receipt_object_names_decline_armed_windows_and_failure() {
        let declined = SpSummary {
            knob: true,
            decline: Some(SpDecline::ArchNotCalibrated {
                arch: "gfx1201".into(),
            }),
            ..SpSummary::default()
        };
        let window = SpWindowEntry {
            rows: 300,
            verdict: Ok(()),
            report: SpWindowReport::default(),
        };
        let sp = sp_receipt_json(&declined, &[window]).unwrap();
        assert_eq!(sp["knob"], json!(true));
        assert_eq!(sp["armed"], json!(false));
        assert_eq!(sp["decline_reason"], json!("arch not calibrated"));
        assert!(sp["decline"].as_str().unwrap().contains("gfx1201"));
        assert_eq!(sp["windows"][0]["engaged"], json!(false));
        assert_eq!(sp["windows"][0]["bypass"], Value::Null);
        assert_eq!(sp["engaged_windows"], json!(0));

        let armed = SpSummary {
            knob: true,
            admitted: Some(sp_profile()),
            ..SpSummary::default()
        };
        let ran = SpWindowEntry {
            rows: 1024,
            verdict: Ok(()),
            report: SpWindowReport {
                layers: 40,
                streamed_bytes: 123,
                streamed_experts: 3400,
                cpu_experts: 9000,
                cpu_slots: 96_000,
            },
        };
        let bypassed = SpWindowEntry {
            rows: 16_896,
            verdict: Err(SpBypass::TooManyRows {
                rows: 16_896,
                maximum: SP_MAX_ROWS,
            }),
            report: SpWindowReport::default(),
        };
        let sp = sp_receipt_json(&armed, &[bypassed, ran]).unwrap();
        assert_eq!(sp["armed"], json!(true));
        assert_eq!(sp["decline"], Value::Null);
        assert_eq!(sp["sp_max_rows"], json!(SP_MAX_ROWS));
        assert_eq!(sp["engaged_windows"], json!(1));
        assert_eq!(sp["windows"][0]["engaged"], json!(false));
        assert!(sp["windows"][0]["bypass"]
            .as_str()
            .unwrap()
            .contains("exceeds"));
        assert_eq!(sp["windows"][1]["engaged"], json!(true));
        assert_eq!(sp["windows"][1]["cpu_slots"], json!(96_000));
        assert_eq!(sp["windows"][1]["bypass"], Value::Null);

        // A failed window disables SP for the process: no longer armed.
        let failed = SpSummary {
            unavailable: Some("sp_unavailable: a window failed: x".into()),
            ..armed
        };
        let sp = sp_receipt_json(&failed, &[ran]).unwrap();
        assert_eq!(sp["armed"], json!(false));
        assert_eq!(
            sp["unavailable"],
            json!("sp_unavailable: a window failed: x")
        );
        assert_eq!(sp["windows"][0]["engaged"], json!(false));
    }

    #[test]
    fn sp_receipt_key_is_added_to_a_layer_major_receipt_without_touching_the_rest() {
        let profile = product_profile_on(AGG);
        let plan = plan_agg(1024, &[]);
        let base = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Auto,
            plan: &plan,
            resident_layers: 10,
            resident_layer_set: &[],
            host_layers: 38,
            tile_scratch_bytes: 0,
            reports: &[WindowReport::default()],
            peak_vram_bytes: None,
            status: "ok".into(),
            moe_iu4: json!({}),
        });
        assert!(base.get("sp").is_none(), "no sp key by default");
        let armed = SpSummary {
            knob: true,
            admitted: Some(sp_profile()),
            ..SpSummary::default()
        };
        let mut with_sp = base.clone();
        with_sp["sp"] = sp_receipt_json(
            &armed,
            &[SpWindowEntry {
                rows: plan.rows,
                verdict: Ok(()),
                report: SpWindowReport::default(),
            }],
        )
        .unwrap();
        let mut stripped = with_sp.clone();
        stripped.as_object_mut().unwrap().remove("sp");
        assert_eq!(stripped, base, "every other key is unchanged");
        let _ = profile;
    }
}
