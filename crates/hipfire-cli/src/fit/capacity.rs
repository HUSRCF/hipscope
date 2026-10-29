// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-GPU context capacity for `hipfire fit`: the wording of the context
//! finding and the plan's capacity line, the best-GPU choice, and the quant-tier
//! summary. Pure functions over the admission projection
//! ([`Qwen35CardProjection`], never recomputed here) and the machine facts
//! `fit/host.rs` already read; nothing opens HIP.

use hipfire_loader::admission::{Qwen35CardProjection, QWEN35_MIN_AUTO_CONTEXT};
use serde::Serialize;
use serde_json::Value;

use super::types::{gib, Finding, FitDevice, Verdict};

/// Smallest projected context worth running on. A GPU whose free memory leaves
/// room for fewer KV tokens than this (down to none at all) is reported as
/// not fitting the model, and no fix ever suggests lowering the context below
/// it: the loader's own minimum (`QWEN35_MIN_AUTO_CONTEXT`) is smaller still.
pub(crate) const USEFUL_CONTEXT_FLOOR: usize = 4096;

/// What the missing bytes of a GPU are, as far as sysfs can say: the amdgpu
/// counters carry no per-process split.
const HELD_BY: &str = "held by other processes or the desktop";

/// The context a model would get on one visible GPU.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GpuCapacity {
    pub index: usize,
    /// The admission projection against this GPU's free memory; `None` when it
    /// cannot be made (free counter unreadable).
    pub projection: Option<Qwen35CardProjection>,
    /// `HIP_VISIBLE_DEVICES` value that selects this GPU.
    pub env_selector: String,
    /// `hardware.devices` value that selects this GPU.
    pub config_selector: String,
}

impl GpuCapacity {
    pub(crate) fn new(
        device: &FitDevice,
        projection: Result<Qwen35CardProjection, String>,
    ) -> Self {
        Self {
            index: device.index,
            projection: projection.ok(),
            env_selector: device.env_selector.clone(),
            config_selector: device.config_selector.clone(),
        }
    }

    /// Projected tokens of context; `0` when the weights do not fit.
    pub(crate) fn capacity(&self) -> Option<usize> {
        self.projection.map(|p| p.card_cap)
    }
}

/// `plan.best_device` in the JSON report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct BestDevice {
    pub index: usize,
    pub free_bytes: u64,
    pub capacity: u64,
}

/// The GPU with the largest projected capacity; ties go to the lowest index.
pub(crate) fn best_gpu(gpus: &[GpuCapacity]) -> Option<&GpuCapacity> {
    let mut best: Option<(&GpuCapacity, usize)> = None;
    for gpu in gpus {
        let Some(capacity) = gpu.capacity() else {
            continue;
        };
        best = match best {
            Some((held, held_capacity))
                if capacity < held_capacity
                    || (capacity == held_capacity && gpu.index >= held.index) =>
            {
                best
            }
            _ => Some((gpu, capacity)),
        };
    }
    best.map(|(gpu, _)| gpu)
}

pub(crate) fn best_device(best: &GpuCapacity) -> Option<BestDevice> {
    let p = best.projection?;
    Some(BestDevice {
        index: best.index,
        free_bytes: p.free as u64,
        capacity: p.card_cap as u64,
    })
}

/// Whether the load's own context check passes: an explicit context must fit
/// the projection, an automatic one needs the loader's minimum.
pub(crate) fn context_fits(p: &Qwen35CardProjection, requested: usize, automatic: bool) -> bool {
    if automatic {
        p.card_cap >= QWEN35_MIN_AUTO_CONTEXT
    } else {
        requested <= p.card_cap
    }
}

/// `2.7 GiB free of 31.9 GiB; 29.2 GiB held by other processes or the desktop`.
/// A unified-memory GPU has no private pool to subtract from, so it states the
/// free figure alone.
pub(crate) fn memory_phrase(free: u64, device: Option<&FitDevice>) -> String {
    match device {
        Some(device) if !device.unified_memory => match device.vram_total {
            Some(total) => format!(
                "{} free of {}; {} {HELD_BY}",
                gib(free),
                gib(total),
                gib(total.saturating_sub(free))
            ),
            None => format!("{} free", gib(free)),
        },
        _ => format!("{} free", gib(free)),
    }
}

fn exceeds_free(p: &Qwen35CardProjection) -> bool {
    p.weights
        .saturating_add(p.scratch)
        .saturating_add(p.page_rounding)
        >= p.free
}

fn charges(p: &Qwen35CardProjection) -> String {
    format!(
        "free {} - weights {} - prefill scratch {} - page rounding {}",
        gib(p.free as u64),
        gib(p.weights as u64),
        gib(p.scratch as u64),
        gib(p.page_rounding as u64)
    )
}

/// `a, b, or c`.
fn or_list(options: &[String]) -> String {
    match options {
        [] => String::new(),
        [only] => only.clone(),
        [first, last] => format!("{first}, or {last}"),
        [init @ .., last] => format!("{}, or {last}", init.join(", ")),
    }
}

/// The fix options every shortage names besides lowering the context: free
/// memory, another GPU when one is better, a smaller tier.
fn memory_fixes(better: Option<&GpuCapacity>) -> Vec<String> {
    let mut options = vec!["free GPU memory (e.g. stop other jobs using this GPU)".to_string()];
    if let Some((better, p)) = better.and_then(|b| Some((b, b.projection?))) {
        options.push(format!(
            "use another GPU (best: GPU{}, {} free, up to {} tokens; see `best GPU`)",
            better.index,
            gib(p.free as u64),
            p.card_cap
        ));
    }
    options.push("pick a smaller tier".to_string());
    options
}

/// The fix for a context the memory holds, but only below the requested one.
fn lower_context_fix(card_cap: usize, better: Option<&GpuCapacity>) -> String {
    let at_most = card_cap.max(QWEN35_MIN_AUTO_CONTEXT);
    let mut options = vec![format!(
        "lower --ctx to at most {at_most} (`hipfire config set memory.max_seq {at_most}`)"
    )];
    options.extend(memory_fixes(better));
    or_list(&options)
}

/// The fix for a context the projection does not hold: lower it when a useful
/// context still fits, otherwise only memory, another GPU or a smaller tier help.
pub(crate) fn context_fix(card_cap: usize, better: Option<&GpuCapacity>) -> String {
    if card_cap >= USEFUL_CONTEXT_FLOOR {
        lower_context_fix(card_cap, better)
    } else {
        or_list(&memory_fixes(better))
    }
}

/// The context finding of a Qwen3.5-family load admitted against `device`.
/// `better` is a visible GPU that holds more context than `device`.
pub(crate) fn context_finding(
    p: Qwen35CardProjection,
    requested: usize,
    automatic: bool,
    device: &FitDevice,
    better: Option<&GpuCapacity>,
) -> Finding {
    let charges = charges(&p);
    if context_fits(&p, requested, automatic) {
        return if automatic {
            Finding::fits("context", format!("automatic context: min(trained window {}, card capacity measured after weights load); projected {} tokens ({charges})", p.model_ctx, p.card_cap))
        } else {
            Finding::fits("context", format!("a {requested}-token context needs {requested} x {} bytes of KV; the projection leaves room for {} tokens ({charges})", p.bytes_per_token, p.card_cap))
        };
    }
    if p.card_cap >= USEFUL_CONTEXT_FLOOR {
        return Finding::fails(
            "context",
            format!("a {requested}-token context needs {requested} x {} bytes of KV; the projection leaves room for {} tokens ({charges})", p.bytes_per_token, p.card_cap),
            lower_context_fix(p.card_cap, better),
        );
    }
    // Too little room for any useful context: the model does not fit this
    // GPU's free memory, and lowering the context cannot fix that.
    let memory = memory_phrase(p.free as u64, Some(device));
    let need = p
        .weights
        .saturating_add(p.scratch)
        .saturating_add(p.page_rounding);
    let shortfall = if exceeds_free(&p) {
        format!(
            "weights {} + prefill scratch {} + page rounding {} need {}, more than the free memory",
            gib(p.weights as u64),
            gib(p.scratch as u64),
            gib(p.page_rounding as u64),
            gib(need as u64)
        )
    } else if p.card_cap == 0 {
        format!(
            "after weights {}, prefill scratch {} and page rounding {} no KV token fits",
            gib(p.weights as u64),
            gib(p.scratch as u64),
            gib(p.page_rounding as u64)
        )
    } else {
        format!(
            "after weights {}, prefill scratch {} and page rounding {} only {} tokens of KV fit, below {}",
            gib(p.weights as u64),
            gib(p.scratch as u64),
            gib(p.page_rounding as u64),
            p.card_cap,
            if automatic {
                format!("the minimum automatic context of {QWEN35_MIN_AUTO_CONTEXT} tokens")
            } else {
                format!("the {USEFUL_CONTEXT_FLOOR}-token useful floor")
            },
        )
    };
    Finding::fails(
        "context",
        format!(
            "the model does not fit in this GPU's currently free memory (GPU{}: {memory}): {shortfall}",
            device.index
        ),
        or_list(&memory_fixes(better)),
    )
}

/// The plan's `capacity` lines (first line follows the label, the rest are
/// continuations) from the JSON projection.
pub(crate) fn capacity_lines(p: &Value, device: Option<&FitDevice>) -> Vec<String> {
    let n = |key: &str| p[key].as_u64().unwrap_or(0);
    let estimate = format!(
        "pre-load estimate: {} free - {} weights - {} prefill scratch - {} rounding; {} B/token",
        gib(n("free")),
        gib(n("weights")),
        gib(n("scratch")),
        gib(n("page_rounding")),
        n("bytes_per_token")
    );
    if n("card_cap") > 0 {
        return vec![format!("up to {} tokens ({estimate})", n("card_cap"))];
    }
    let needed = n("weights")
        .saturating_add(n("scratch"))
        .saturating_add(n("page_rounding"));
    let why = if needed >= n("free") {
        "weights + prefill scratch exceed free memory"
    } else {
        "no room for KV after weights + prefill scratch in free memory"
    };
    vec![
        format!("{why} ({})", memory_phrase(n("free"), device)),
        estimate,
    ]
}

/// The `best GPU` plan line. `pinned`: `hardware.devices` is set, which wins
/// over `HIP_VISIBLE_DEVICES` (hipfire_config::devices), so only the config
/// form selects.
pub(crate) fn best_gpu_line(best: &GpuCapacity, pinned: bool) -> Option<String> {
    let p = best.projection?;
    let how = if pinned {
        format!(
            "select it with `hipfire config set hardware.devices {}` (hardware.devices is set, so HIP_VISIBLE_DEVICES is ignored)",
            best.config_selector
        )
    } else {
        format!(
            "select it with HIP_VISIBLE_DEVICES={} (or `hipfire config set hardware.devices {}`)",
            best.env_selector, best.config_selector
        )
    };
    let room = if p.card_cap == 0 {
        "weights don't fit".to_string()
    } else {
        format!("up to {} tokens", p.card_cap)
    };
    Some(format!(
        "GPU{} ({} free): {room} — {how}",
        best.index,
        gib(p.free as u64)
    ))
}

/// A GPU's capacity for the `this machine` rows.
pub(crate) fn device_capacity_text(capacity: u64) -> String {
    if capacity == 0 {
        "weights don't fit".into()
    } else {
        format!("up to {capacity} tokens")
    }
}

/// One quant tier on its best GPU.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TierCapacity {
    pub gpu: usize,
    pub summary: String,
    /// The context check on that GPU.
    pub verdict: Verdict,
}

/// A tier's row from its per-GPU capacities: the largest context on the best
/// GPU, or `weights don't fit`. `None` when no GPU could be projected.
pub(crate) fn tier_on_best_gpu(
    gpus: &[GpuCapacity],
    requested: usize,
    automatic: bool,
    multi_gpu: bool,
) -> Option<TierCapacity> {
    let best = best_gpu(gpus)?;
    let p = best.projection?;
    let fits = context_fits(&p, requested, automatic);
    let summary = if p.card_cap == 0 {
        "weights don't fit".to_string()
    } else {
        let on = if multi_gpu {
            format!(" on GPU{}", best.index)
        } else {
            String::new()
        };
        let short = if fits || automatic {
            String::new()
        } else {
            format!(" (below the requested {requested}-token context)")
        };
        format!("up to {} tokens{on}{short}", p.card_cap)
    };
    Some(TierCapacity {
        gpu: best.index,
        summary,
        verdict: if fits {
            Verdict::Fits
        } else {
            Verdict::DoesNotFit
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;
    const GIB_US: usize = 1 << 30;

    fn device(index: usize, free_gib: u64) -> FitDevice {
        FitDevice {
            index,
            arch: "gfx1201".into(),
            unified_memory: false,
            vram_total: Some(32 << 30),
            free: Some(free_gib << 30),
            link: None,
            capacity: None,
            env_selector: index.to_string(),
            config_selector: format!("GPU-{index:016x}"),
        }
    }

    /// 4B-class model: 3.2 GiB weights, 1.2 GiB scratch, 17408 B/token.
    fn projection(free_bytes: usize) -> Qwen35CardProjection {
        let weights = 3 * GIB_US + 200 * MIB;
        let scratch = GIB_US + 200 * MIB;
        let page_rounding = 128 * MIB;
        let bytes_per_token = 17_408;
        Qwen35CardProjection {
            free: free_bytes,
            weights,
            scratch,
            page_rounding,
            bytes_per_token,
            card_cap: free_bytes
                .saturating_sub(weights)
                .saturating_sub(scratch)
                .saturating_sub(page_rounding)
                / bytes_per_token,
            model_ctx: 262_144,
            kv_mode: "q8",
        }
    }

    fn with_cap(card_cap: usize) -> Qwen35CardProjection {
        Qwen35CardProjection {
            card_cap,
            ..projection(32 * GIB_US)
        }
    }

    fn gpu(index: usize, free_gib: usize) -> GpuCapacity {
        GpuCapacity::new(
            &device(index, free_gib as u64),
            Ok(projection(free_gib * GIB_US)),
        )
    }

    fn assert_never_lowers_below_floor(finding: &Finding) {
        let text = format!(
            "{} {}",
            finding.detail,
            finding.fix.clone().unwrap_or_default()
        );
        for banned in ["max_seq 0", "at most 0", "memory.max_seq"] {
            assert!(!text.contains(banned), "{banned:?} in {text}");
        }
    }

    #[test]
    fn no_room_names_free_total_and_held_and_never_lowers_the_context() {
        // 2.7 GiB free of 32 GiB: weights alone do not fit.
        let p = projection(2_900 * MIB);
        assert_eq!(p.card_cap, 0);
        let finding = context_finding(p, 262_144, false, &device(0, 3), None);
        assert_eq!(finding.verdict, Verdict::DoesNotFit);
        assert!(finding
            .detail
            .contains("does not fit in this GPU's currently free memory"));
        assert!(finding.detail.contains("2.8 GiB free of 32.0 GiB"));
        assert!(finding
            .detail
            .contains("held by other processes or the desktop"));
        assert!(finding.detail.contains("need"));
        let fix = finding.fix.as_deref().unwrap();
        assert!(fix.contains("free GPU memory") && fix.contains("pick a smaller tier"));
        assert!(!fix.contains("another GPU"));
        assert_never_lowers_below_floor(&finding);
        // The automatic context refuses the same way.
        let automatic = context_finding(p, 0, true, &device(0, 3), None);
        assert_eq!(automatic.verdict, Verdict::DoesNotFit);
        assert_never_lowers_below_floor(&automatic);
    }

    #[test]
    fn capacity_below_the_floor_gets_the_same_treatment() {
        let free = 3 * GIB_US + 200 * MIB + GIB_US + 200 * MIB + 128 * MIB + 2_000 * 17_408;
        let p = projection(free);
        assert_eq!(p.card_cap, 2_000);
        let finding = context_finding(p, 8_192, false, &device(0, 6), None);
        assert_eq!(finding.verdict, Verdict::DoesNotFit);
        assert!(finding
            .detail
            .contains("does not fit in this GPU's currently free memory"));
        assert!(finding.detail.contains("only 2000 tokens of KV fit"));
        assert!(finding.detail.contains("held by other processes"));
        assert!(!finding.fix.as_deref().unwrap().contains("lower --ctx"));
        assert_never_lowers_below_floor(&finding);
        // Room for the requested context stays a pass, however small.
        assert_eq!(
            context_finding(p, 1_024, false, &device(0, 6), None).verdict,
            Verdict::Fits
        );
    }

    #[test]
    fn capacity_between_floor_and_request_keeps_lowering_the_context() {
        let p = with_cap(100_000);
        let finding = context_finding(p, 262_144, false, &device(0, 20), None);
        assert_eq!(finding.verdict, Verdict::DoesNotFit);
        let fix = finding.fix.unwrap();
        assert!(fix.starts_with("lower --ctx to at most 100000 "), "{fix}");
        assert!(fix.contains("`hipfire config set memory.max_seq 100000`"));
        // At the floor itself the suggestion stays valid and never drops under 512.
        let floor = context_finding(
            with_cap(USEFUL_CONTEXT_FLOOR),
            262_144,
            false,
            &device(0, 20),
            None,
        );
        assert!(floor.fix.unwrap().contains("lower --ctx to at most 4096 "));
        assert!(lower_context_fix(100, None).contains("at most 512 "));
    }

    #[test]
    fn best_gpu_is_the_largest_capacity_with_ties_to_the_lowest_index() {
        let gpus = [gpu(0, 3), gpu(1, 12), gpu(2, 31), gpu(3, 31)];
        let best = best_gpu(&gpus).unwrap();
        assert_eq!(best.index, 2);
        assert_eq!(best_device(best).unwrap().index, 2);
        assert_eq!(
            best_device(best).unwrap().capacity,
            best.capacity().unwrap() as u64
        );
        // Nothing fits anywhere: GPU0 stays the answer, the plan names no other.
        let none = [gpu(0, 2), gpu(1, 2)];
        assert_eq!(best_gpu(&none).unwrap().index, 0);
        // Unreadable counters are skipped.
        let unknown = GpuCapacity::new(&device(0, 1), Err("no free".into()));
        assert_eq!(best_gpu(&[unknown, gpu(1, 8)]).unwrap().index, 1);

        let line = best_gpu_line(best, false).unwrap();
        assert!(line.starts_with("GPU2 (31.0 GiB free): up to "));
        assert!(line.contains("select it with HIP_VISIBLE_DEVICES=2"));
        assert!(line.contains("`hipfire config set hardware.devices GPU-0000000000000002`"));
        let pinned = best_gpu_line(best, true).unwrap();
        assert!(!pinned.contains("HIP_VISIBLE_DEVICES=2"));
        assert!(pinned.contains("hipfire config set hardware.devices GPU-0000000000000002"));

        // The busy GPU's finding points at it.
        let finding = context_finding(
            projection(3 * GIB_US),
            262_144,
            false,
            &device(0, 3),
            Some(best),
        );
        let fix = finding.fix.unwrap();
        assert!(fix.contains("use another GPU (best: GPU2, 31.0 GiB free, up to "));
        assert!(fix.contains("pick a smaller tier"));
    }

    #[test]
    fn tier_rows_state_the_best_gpus_capacity_or_that_weights_do_not_fit() {
        let busy = [gpu(0, 2), gpu(1, 2)];
        let row = tier_on_best_gpu(&busy, 8_192, false, true).unwrap();
        assert_eq!(row.summary, "weights don't fit");
        assert_eq!(row.verdict, Verdict::DoesNotFit);
        let mixed = [gpu(0, 2), gpu(1, 2), gpu(2, 31)];
        let row = tier_on_best_gpu(&mixed, 8_192, false, true).unwrap();
        assert_eq!(row.gpu, 2);
        assert!(row.summary.starts_with("up to "), "{}", row.summary);
        assert!(row.summary.ends_with(" tokens on GPU2"), "{}", row.summary);
        assert_eq!(row.verdict, Verdict::Fits);
        let single = tier_on_best_gpu(&mixed[2..], 8_192, false, false).unwrap();
        assert!(single.summary.ends_with(" tokens") && !single.summary.contains("GPU"));
        let over = tier_on_best_gpu(&mixed, 2_000_000, false, true).unwrap();
        assert_eq!(over.verdict, Verdict::DoesNotFit);
        assert!(over
            .summary
            .contains("below the requested 2000000-token context"));
        assert!(tier_on_best_gpu(&[], 8_192, false, true).is_none());
    }

    #[test]
    fn capacity_line_states_the_shortage_and_what_holds_the_memory() {
        let p = projection(2_900 * MIB);
        let json = serde_json::json!({"free": p.free, "weights": p.weights, "scratch": p.scratch,
            "page_rounding": p.page_rounding, "bytes_per_token": p.bytes_per_token,
            "card_cap": p.card_cap});
        let lines = capacity_lines(&json, Some(&device(0, 3)));
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("weights + prefill scratch exceed free memory (2.8 GiB free of 32.0 GiB; 29.2 GiB held by other processes or the desktop)"));
        assert!(lines[1].starts_with("pre-load estimate: "));
        assert!(!lines[0].contains("up to 0 tokens"));
        let roomy = projection(30 * GIB_US);
        let json = serde_json::json!({"free": roomy.free, "weights": roomy.weights,
            "scratch": roomy.scratch, "page_rounding": roomy.page_rounding,
            "bytes_per_token": roomy.bytes_per_token, "card_cap": roomy.card_cap});
        let lines = capacity_lines(&json, None);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with(&format!(
            "up to {} tokens (pre-load estimate: ",
            roomy.card_cap
        )));
    }
}
