// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 MTP acceptance and transaction-shape helpers.
//!
//! The runtime owns the speculative loop and its pending-seed contract.  This
//! module records the native MTP count convention and lowers a verified
//! window (greedy prefix match, or with `speculation.mtp_sampled` a sampled
//! verify from [`hipfire_runtime::spec_sampling`]: speculative rejection
//! sampling, or SpecInfer naive sampling under
//! `HIPFIRE_MTP_SAMPLED_MODE=naive`) onto the canonical runtime types.  In
//! particular, the seed is never copied into `MtpWindow::committed` or
//! `SpecStep::emit`.
//! Target rollback counts accepted drafts only; the position helpers take the
//! consumed-row count, which adds the seed.

use crate::bundle::{Qwen4Bundle, Qwen4PrefixMode, Qwen4PrefixPlan};
use crate::mtp_gpu::{MtpAppendScratch, MTP_FILL_ROWS};
#[cfg(any(test, feature = "reference-parity"))]
use crate::reference_mtp::{MtpError, Qwen4MtpState};
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::spec::{
    accept_greedy_prefix, GreedyAccept, MtpDrafter, MtpRequestStats, MtpSpeculator, MtpWindow,
    SpecAdvance, SpecGrammar, SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use hipfire_runtime::spec_sampling::{
    accept_naive_prefix, accept_sampled_prefix, naive_target_sampler, verify_sampled_draft,
    DraftVerdict, SampleSpec, SparseDist, SpecRng,
};
use rdna_compute::profile::{unix_micros, Span, SpanProfiler};
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

/// Device-side phase timing for one native MTP window.
///
/// Enabled by `HIPFIRE_MTP_PHASE_TIMING=1`.  Each phase is one `hipEvent` pair
/// recorded on the null stream and resolved once at window end, so the
/// instrument never synchronizes mid-window and never changes launch order:
/// the number it prints is the GPU time the phase actually occupies.
struct MtpPhaseTimers {
    spans: SpanProfiler,
    open: Option<Span>,
}

impl MtpPhaseTimers {
    fn new() -> Self {
        Self {
            spans: SpanProfiler::new(
                hipfire_config::developer_var("HIPFIRE_MTP_PHASE_TIMING")
                    .is_ok_and(|value| value == "1"),
            ),
            open: None,
        }
    }

    fn enabled(&self) -> bool {
        self.spans.enabled()
    }

    /// Close the open phase and open `label`.
    fn mark(&mut self, gpu: &Gpu, label: &'static str) {
        self.spans.end(&gpu.hip, None, self.open.take());
        self.open = self.spans.begin(&gpu.hip, None, label);
    }

    /// Resolve every recorded pair and print one line.  `fields` carries the
    /// per-window scalars (position, k, accepted drafts).  Repeated labels are
    /// summed, so a per-row phase reads as its total across the window.
    fn finish(mut self, gpu: &Gpu, tag: &str, fields: &str) {
        self.spans.end(&gpu.hip, None, self.open.take());
        let totals = self.spans.resolve(&gpu.hip, None);
        if totals.is_empty() {
            return;
        }
        let phases = totals
            .iter()
            .map(|total| {
                format!(
                    "\"{}\":{{\"us\":{:.1},\"t0\":{},\"t1\":{}}}",
                    total.label, total.us, total.first_start_unix_us, total.last_end_unix_us
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("{tag} {{\"event\":\"mtp_phase\",{fields},\"phases_us\":{{{phases}}}}}");
    }
}

/// Accepted drafts the target commits before the pending seed.
///
/// An accepted EOS draft (no bonus follows it) stays the pending seed, so
/// neither target nor MTP state consumes it until the terminal flush; a bonus
/// EOS is already predicted after the accepted prefix. This is also the
/// captured target-hidden row that produced the next pending seed.
pub fn target_commit_accept_len(accepted: &GreedyAccept) -> usize {
    let accepted_eos = accepted.hit_eos && accepted.committed.len() == accepted.accepted;
    accepted.accepted - usize::from(accepted_eos)
}

/// With `speculation.mtp_sampled` off (`HIPFIRE_MTP_SAMPLED=0`; on by
/// default), native MTP verifies greedy target picks only, so a sampled
/// request fails closed here (the router keeps such requests on AR).
pub fn require_native_greedy(temp: f32) -> Result<(), String> {
    if !temp.is_finite() || temp.abs() > 1.0e-6 {
        return Err(
            "Qwen4 native MTP verifies greedy requests only while speculation.mtp_sampled is off (HIPFIRE_MTP_SAMPLED=0)"
                .to_string(),
        );
    }
    Ok(())
}

/// Validate a native MTP prefill request before touching either owner.
///
/// A cold fill is the complete prompt from position zero. A cache hit fills
/// exactly `prompt_tokens[start_pos..]` after the bundle restored its
/// checkpoint at `start_pos` (`Qwen4Bundle::begin_prefix` re-validates
/// it), so target and MTP resume at the same absolute position.
pub fn validate_native_mtp_prefill_request(
    prompt_tokens: &[u32],
    fill_tokens: &[u32],
    start_pos: usize,
    cache_hit: bool,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Err("Qwen4 native MTP prefill requires at least one prompt token".to_string());
    }
    if !cache_hit {
        if start_pos != 0 {
            return Err(format!(
                "Qwen4 native MTP cold prefill requires position zero, got {start_pos}"
            ));
        }
        if fill_tokens != prompt_tokens {
            return Err("Qwen4 native MTP prefill requires a complete prompt fill".to_string());
        }
        return Ok(());
    }
    if start_pos == 0 || start_pos >= prompt_tokens.len() {
        return Err(format!(
            "Qwen4 native MTP cache hit at {start_pos} needs a non-empty suffix of a {}-token prompt",
            prompt_tokens.len()
        ));
    }
    if fill_tokens != &prompt_tokens[start_pos..] {
        return Err("Qwen4 native MTP cache hit must fill exactly the prompt suffix".to_string());
    }
    Ok(())
}

/// Absolute target positions occupied by the committed verify prefix.
///
/// `position` is the seed's position and `num_accepted_tokens` includes that
/// seed, so the returned range has exactly `num_accepted_tokens` rows.  The
/// bonus is predicted, not consumed, and is therefore intentionally absent.
pub fn committed_target_positions(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<Vec<usize>, String> {
    let end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP target position overflow".to_string())?;
    Ok((position..end).collect())
}

#[cfg(any(test, feature = "reference-parity"))]
/// Compact target-aligned MTP QSA rows after a partial acceptance.
///
/// The caller supplies the seed position and the native count.  Rows at or
/// after the committed end are rejected-tail state and are discarded from the
/// sparse selection.  Full main K/V is restored/replayed by the target
/// transaction; this helper only handles the MTP side index row.
pub fn compact_native_qsa_selection(
    state: &mut Qwen4MtpState,
    position: usize,
    num_accepted_tokens: usize,
) -> Result<(), String> {
    let retained_end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP QSA position overflow".to_string())?;
    if retained_end > state.qsa.position {
        return Err(format!(
            "Qwen4 native MTP QSA commit end {retained_end} exceeds active position {}",
            state.qsa.position
        ));
    }
    let retained = state
        .qsa
        .selected_indices
        .iter()
        .copied()
        .filter(|&row| row < retained_end)
        .collect::<Vec<_>>();
    state
        .qsa
        .compact_selection(&retained)
        .map_err(|error| format!("Qwen4 native MTP QSA compaction: {error}"))
}

#[cfg(any(test, feature = "reference-parity"))]
/// Compute the target's post-commit position without mutating either owner.
pub fn native_commit_position(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<usize, String> {
    position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP commit position overflow".to_string())
}

/// Apply the runtime's one shared greedy acceptance rule to native MTP picks,
/// refusing (rather than debug-asserting) a verifier that returned no bonus
/// slot.  The seed is not an argument: it is already represented by the
/// target block and is excluded from `committed` by construction.
pub fn accept_native_greedy(
    drafts: &[u32],
    target_picks: &[u32],
    eos: Option<u32>,
) -> Result<GreedyAccept, String> {
    if target_picks.len() < drafts.len().saturating_add(1) {
        return Err(format!(
            "Qwen4 native MTP verifier returned {} picks for {} drafts; one bonus pick is required",
            target_picks.len(),
            drafts.len()
        ));
    }
    Ok(accept_greedy_prefix(drafts, target_picks, eos))
}

/// Lower an MTP window directly to the runtime's pending-seed result.
/// `MtpWindow::committed` already excludes the seed, so this is a 1:1 emit
/// mapping and never performs a DFlash-style seed re-echo transformation.
pub fn window_to_spec_step(window: MtpWindow) -> Result<SpecStep, String> {
    let next_seed = *window
        .committed
        .last()
        .ok_or_else(|| "Qwen4 native MTP committed no token (would stall decode)".to_string())?;
    if window.accepted > window.drafts_generated {
        return Err(format!(
            "Qwen4 native MTP accepted {} drafts out of {}",
            window.accepted, window.drafts_generated
        ));
    }
    Ok(SpecStep::new(
        window.committed,
        next_seed,
        window.drafts_generated,
        window.accepted,
    ))
}
/// Reusable Qwen4 target-side verify scratch.  GPU output buffers and the
/// captured wide hidden rows belong to the bundle; this object owns only the
/// checked-out rollback ticket and its fixed block capacity.
pub struct Qwen4SpecScratch {
    block_size: usize,
    target_snapshot: Option<Qwen4StateSnapshot>,
}

impl SpecScratch for Qwen4SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // A live ticket is consumed by verify/commit before the speculator is
        // released.  Bundle teardown also owns the arena, so there is no
        // independent GPU allocation to free here.
        debug_assert!(
            self.target_snapshot.is_none(),
            "Qwen4 verify scratch dropped with an active target snapshot"
        );
    }
}

impl SpecTarget for Qwen4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)
            .map_err(|error| format!("Qwen4 reset_recurrent: {error}"))
    }

    /// Native MTP retains no pre-window snapshot to repair from, and the
    /// prompt-cache checkpoint references the live QSA K/V rows a reset would
    /// zero: rewind to it so the next turn can still restore it.
    fn reset_after_unrepaired_terminal(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.rewind_to_prefix(gpu)
            .map_err(|error| format!("Qwen4 terminal rewind: {error}"))
    }

    fn retry_reset_eligible(&self) -> bool {
        true
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(1);
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 spec scratch requires attached forward resources".to_string())?
            .scratch
            .max_chunk;
        if block_size > max_chunk {
            return Err(format!(
                "Qwen4 spec block size {block_size} exceeds forward capacity {max_chunk}"
            ));
        }
        self.ensure_spec_hidden(gpu, block_size)
            .map_err(|error| error.to_string())?;
        Ok(Box::new(Qwen4SpecScratch {
            block_size,
            target_snapshot: None,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)?;
        }
        if tokens.is_empty() {
            return Err("Qwen4 spec_advance cannot process an empty token slice".to_string());
        }
        if self.state.position != start_pos {
            return Err(format!(
                "Qwen4 spec_advance position mismatch: expected {}, got {start_pos}",
                self.state.position
            ));
        }
        let end_pos = start_pos
            .checked_add(tokens.len())
            .ok_or_else(|| "Qwen4 spec position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 spec advance end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?
            .scratch
            .max_chunk;
        let mut offset = 0usize;
        let mut last_argmax = None;
        while offset < tokens.len() {
            if abort() {
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            let end = (offset + max_chunk).min(tokens.len());
            self.set_ple_lookahead(&tokens[end..]);
            // Only the final row's argmax is returned, so each chunk writes
            // one logit row instead of one per chunk row.
            let pick = self
                .spec_prefill_rows(gpu, &tokens[offset..end], false)
                .map_err(|error| error.to_string())?;
            last_argmax = Some(pick);
            offset = end;
        }
        if self.state.position != end_pos {
            return Err(format!(
                "Qwen4 spec advance ended at {}, expected {end_pos}",
                self.state.position
            ));
        }
        Ok(SpecAdvance::Ready {
            last_argmax: last_argmax.expect("non-empty spec advance produced no argmax"),
            last_logits: None,
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let block_size = {
            let s = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?;
            if s.target_snapshot.is_some() {
                return Err("Qwen4 verify_block: target snapshot is already active".to_string());
            }
            s.block_size
        };
        if block.is_empty() || block.len() > block_size {
            return Err(format!(
                "Qwen4 verify block length {} is outside scratch capacity {block_size}",
                block.len()
            ));
        }
        if self.state.position != position {
            return Err(format!(
                "Qwen4 verify_block position mismatch: expected {}, got {position}",
                self.state.position
            ));
        }
        let end_pos = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 verify position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 verify end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        // The verify leaves per-row rollback points, so a rejected suffix is
        // dropped without re-running the accepted rows (armed before the
        // snapshot: the GDN states then need no copy).
        self.state.row_capture_armed =
            block.len() >= 2 && self.state.row_capture_rows() >= block.len();
        let snapshot = match self.snapshot(gpu) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.state.row_capture_armed = false;
                return Err(error.to_string());
            }
        };
        scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?
            .target_snapshot = Some(snapshot);
        let result = self
            .spec_forward_rows(gpu, block, true)
            .map_err(|error| error.to_string());
        self.state.row_capture_armed = false;
        if let Err(error) = &result {
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{error}; target restore failed: {restore}"))?;
            }
            return Err(error.clone());
        }
        if self.state.position != end_pos {
            let mismatch = format!(
                "Qwen4 verify ended at {}, expected {end_pos}",
                self.state.position
            );
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{mismatch}; target restore failed: {restore}"))?;
            }
            return Err(mismatch);
        }
        result
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        if block.is_empty() {
            return Err("Qwen4 commit_prefix cannot commit an empty block".to_string());
        }
        let draft_len = block.len() - 1;
        if accept_len > draft_len {
            return Err(format!(
                "Qwen4 commit_prefix accepts {accept_len} drafts out of {draft_len}"
            ));
        }
        let verified_end = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        if self.state.position != verified_end {
            return Err(format!(
                "Qwen4 commit_prefix target position mismatch: expected {verified_end}, got {}",
                self.state.position
            ));
        }
        let committed_end = position
            .checked_add(accept_len + 1)
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        let snapshot = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 commit_prefix: scratch is not Qwen4SpecScratch")?
            .target_snapshot
            .take()
            .ok_or("Qwen4 commit_prefix: target snapshot is not active")?;
        if accept_len == draft_len {
            return self
                .commit(gpu, snapshot)
                .map_err(|error| error.to_string());
        }
        self.restore(gpu, snapshot)
            .map_err(|error| error.to_string())?;
        self.spec_forward_rows(gpu, &block[..accept_len + 1], true)
            .map(|_| ())
            .map_err(|error| error.to_string())?;
        if self.state.position != committed_end {
            return Err(format!(
                "Qwen4 commit_prefix replay ended at {}, expected {committed_end}",
                self.state.position
            ));
        }
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq_len
    }
}

/// Draft-step hidden conditioning mode; see [`Qwen4MtpDrafter::draft_pairing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftPairing {
    HeadState,
    AlignedHead,
    AlignedTarget,
}

/// Cost of a batched window that drafts K tokens (index K: K draft steps,
/// a (K+1)-row verify, rollback), in interleaved-route emitted tokens (one
/// single-row forward plus one draft step). gfx1151, Qwen3.8-Flash-Next,
/// measured: interleaved ~33.5 ms/token, K=2 window ~62.5 ms, K=3 ~75.4 ms;
/// the others from the few-row forward's growth (2 rows ~50 ms, 5 ~69,
/// 8 ~98) plus ~2.7 ms per draft step.
const MTP_WINDOW_COST: [f32; 8] = [1.0, 1.7, 1.87, 2.25, 2.55, 2.75, 3.0, 3.6];
/// Per-window decay of the per-depth agreement counts.
const MTP_AGREEMENT_DECAY: f32 = 0.875;
/// Per-depth (accepted, compared) counts a request starts from (0.8).
const MTP_AGREEMENT_PRIOR: (f32, f32) = (1.6, 2.0);
/// Draft depths tracked (the drafter's K is clamped to this).
const MTP_MAX_DEPTH: usize = 10;
/// Smallest probability that a verify row's whole draft prefix is accepted
/// for the row (plus its draft step) to pay: ~6 ms of a ~47 ms window
/// emitting ~2-3.5 tokens.
const MTP_ROW_WORTH: f32 = 0.2;
/// Best per-draft acceptance [`draft_accept_estimate`] gives.
const DRAFT_ACCEPT_MAX: f32 = 0.98;

/// Acceptance of a draft whose exact logit leads its runner-up (among the
/// re-scored candidates) by `margin`; measured on the committed sweep prompts
/// (gfx1151, Qwen3.8-Flash-Next, all depths pooled).
fn draft_accept_estimate(margin: f32) -> f32 {
    match margin {
        m if m < 0.25 => 0.3,
        m if m < 1.5 => 0.5,
        m if m < 3.0 => 0.7,
        m if m < 6.0 => 0.82,
        _ => DRAFT_ACCEPT_MAX,
    }
}

/// Speculative windows (native or external, `k > 0`) the AR floor observes
/// before its first measured decision.  Together with the one calibration
/// token this bounds the speculation work that precedes the first decision:
/// at most one head append plus this many windows.
const MTP_FLOOR_PROBE_WINDOWS: usize = 3;

type Agreement = [(f32, f32); MTP_MAX_DEPTH];

/// Route the floor picks for one native window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloorRoute {
    /// Ordinary target-only token, no head work (after sticky retirement).
    Ar,
    Interleaved,
    Batched(usize),
}

/// Which native route a measured window ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeWindow {
    Interleaved,
    Batched,
}

fn valid_us(us: f32) -> bool {
    us.is_finite() && us > 0.0
}

/// Add one observation to a decayed `(sum, weight)` pair; only the entry
/// being observed decays.
fn decay_add(entry: &mut (f32, f32), sum: f32, weight: f32) {
    entry.0 = entry.0 * MTP_AGREEMENT_DECAY + sum;
    entry.1 = entry.1 * MTP_AGREEMENT_DECAY + weight;
}

/// Expected tokens a `depth`-draft window emits (the seed's bonus plus the
/// accepted-prefix probabilities); the arithmetic of `batched_depth`.
fn expected_emitted(agreement: &Agreement, depth: usize) -> f32 {
    let mut prefix = 1.0f32;
    let mut expected = 1.0f32;
    for &(accepted, total) in agreement.iter().take(depth) {
        prefix *= accepted / total.max(f32::MIN_POSITIVE);
        expected += prefix;
    }
    expected
}

/// Same-request AR floor of native MTP.
///
/// The price of AR is `ar_us`, measured on this request, on this GPU, by one
/// calibration token (the ordinary target-only forward, including a sampled
/// draw).  Speculation is priced from this request's own windows: a batched
/// window by its decayed mean wall time at the depth it actually drafted,
/// divided by the agreement-predicted tokens it emits; the interleaved route
/// by its decayed wall time per emitted token.  The Halo cost table only
/// orders the options while fewer than [`MTP_FLOOR_PROBE_WINDOWS`] windows
/// have been observed.  After that budget only measured options are chosen,
/// and the request retires, stickily, to head-free AR as soon as no measured
/// option costs strictly less per emitted token than AR (a tie retires).  An
/// invalid AR measurement retires at the budget too: without a price, a win
/// cannot be asserted.
///
/// External (n-gram) windows are tracked separately and never feed the
/// native agreement; they count toward the shared probe budget.  A request
/// stays alive on a losing (or unmeasured) native route only while external
/// windows are measured and unblocked (cost per token below AR) AND the
/// realized per-token cost of all its windows, native and external, decayed
/// (`combined`), is strictly below AR (a tie loses).  Such a request routes
/// the cheapest measured native option within `k`, else interleaved; it
/// retires, stickily, at the first window where that stops holding.  A
/// measured native option that beats AR needs no external help.
#[derive(Clone, Debug)]
struct MtpFloor {
    calibrated: bool,
    /// Microseconds of one ordinary AR token; 0 when the measurement was
    /// invalid.
    ar_us: f32,
    probes: usize,
    retired: bool,
    /// Per actually drafted depth: decayed (wall us, windows).
    batched: [(f32, f32); MTP_WINDOW_COST.len()],
    /// Decayed (wall us, emitted tokens).
    interleaved: (f32, f32),
    /// Decayed (wall us, emitted tokens) of external windows.
    external: (f32, f32),
    external_blocked: bool,
    /// Decayed realized (wall us, emitted tokens) over every valid window
    /// observed after calibration, native and external alike.
    combined: (f32, f32),
}

impl MtpFloor {
    fn new() -> Self {
        Self {
            calibrated: false,
            ar_us: 0.0,
            probes: 0,
            retired: false,
            batched: [(0.0, 0.0); MTP_WINDOW_COST.len()],
            interleaved: (0.0, 0.0),
            external: (0.0, 0.0),
            external_blocked: false,
            combined: (0.0, 0.0),
        }
    }

    fn probing(&self) -> bool {
        self.probes < MTP_FLOOR_PROBE_WINDOWS
    }

    fn count_probe(&mut self) {
        self.probes = (self.probes + 1).min(MTP_FLOOR_PROBE_WINDOWS);
    }

    fn observe_calibration(&mut self, us: f32, agreement: &Agreement) {
        self.calibrated = true;
        self.ar_us = if valid_us(us) { us } else { 0.0 };
        self.decide(agreement);
    }

    /// One finished native window of `k > 0` budget with wall time `us`.
    fn observe_native(
        &mut self,
        route: NativeWindow,
        window: &MtpWindow,
        us: f32,
        k: usize,
        agreement: &Agreement,
    ) {
        if k == 0 || self.retired || !self.calibrated {
            return;
        }
        // An unusable timing records no cost but still spends probe budget,
        // so the number of unmeasured windows stays bounded.
        if valid_us(us) {
            decay_add(&mut self.combined, us, window.committed.len() as f32);
            match route {
                NativeWindow::Interleaved => {
                    decay_add(&mut self.interleaved, us, window.committed.len() as f32);
                }
                NativeWindow::Batched => {
                    if let Some(entry) = self
                        .batched
                        .get_mut(window.drafts_generated)
                        .filter(|_| window.drafts_generated >= 1)
                    {
                        decay_add(entry, us, 1.0);
                    }
                }
            }
        }
        self.count_probe();
        self.decide(agreement);
    }

    /// One finished external window (`drafts > 0` offered, `emitted` tokens
    /// committed, `wall_us` including its verify).  Returns `true` when the
    /// caller must not take over again this request: the floor retired, or
    /// external windows cost no less per token than AR.
    fn observe_external(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
        agreement: &Agreement,
    ) -> bool {
        if !self.retired && drafts > 0 {
            if valid_us(wall_us) {
                if self.calibrated {
                    decay_add(&mut self.combined, wall_us, emitted as f32);
                }
                if emitted > 0 {
                    decay_add(&mut self.external, wall_us, emitted as f32);
                }
            }
            self.count_probe();
            self.decide(agreement);
        }
        self.retired || self.external_blocked
    }

    /// Cheapest measured native option with at most `max_depth` drafts
    /// (interleaved is always eligible): `(route, us per emitted token)`.
    /// Ties keep the earlier (shallower) option.
    fn best_native(&self, agreement: &Agreement, max_depth: usize) -> Option<(FloorRoute, f32)> {
        let mut best = (self.interleaved.1 > 0.0)
            .then(|| (FloorRoute::Interleaved, self.interleaved.0 / self.interleaved.1));
        for depth in 1..self.batched.len().min(max_depth.saturating_add(1)) {
            let (sum, windows) = self.batched[depth];
            if windows <= 0.0 {
                continue;
            }
            let cost = sum / windows / expected_emitted(agreement, depth);
            if best.map_or(true, |(_, cheapest)| cost < cheapest) {
                best = Some((FloorRoute::Batched(depth), cost));
            }
        }
        best
    }

    /// External windows are measured and their price is below AR.
    fn external_alive(&self) -> bool {
        self.external.1 > 0.0 && !self.external_blocked
    }

    /// The decayed realized per-token cost of every window (native and
    /// external) is strictly below AR.
    fn combined_wins(&self) -> bool {
        self.combined.1 > 0.0 && self.combined.0 / self.combined.1 < self.ar_us
    }

    /// Retire when, after the probe budget, nothing measured beats AR, and
    /// external windows are not alive on a mixture that beats AR.
    fn decide(&mut self, agreement: &Agreement) {
        if self.retired || !self.calibrated || self.probing() {
            return;
        }
        if self.ar_us <= 0.0 {
            self.retired = true;
            return;
        }
        let external = (self.external.1 > 0.0).then(|| self.external.0 / self.external.1);
        self.external_blocked = external.is_some_and(|cost| cost >= self.ar_us);
        let native_wins = self
            .best_native(agreement, usize::MAX)
            .is_some_and(|(_, cost)| cost < self.ar_us);
        if !native_wins && !(self.external_alive() && self.combined_wins()) {
            self.retired = true;
        }
    }

    /// Route for the next native window with `k` drafts of budget.
    /// `provisional` is the unchanged Halo-table `batched_depth(k)`, used
    /// only while the probe budget lasts.  Afterwards only a measured option
    /// within `k` that costs strictly less per emitted token than AR may run;
    /// when there is none (an external-only budget, `k` below every winning
    /// measured depth, a losing measured option, no valid AR price) the
    /// request retires, stickily, and runs AR: no unmeasured or losing route
    /// is ever run after the budget.
    fn route(&mut self, agreement: &Agreement, k: usize, provisional: usize) -> FloorRoute {
        if self.retired {
            return FloorRoute::Ar;
        }
        if self.probing() || !self.calibrated {
            return match provisional {
                0 => FloorRoute::Interleaved,
                depth => FloorRoute::Batched(depth),
            };
        }
        let best = self.best_native(agreement, k);
        if let Some((route, cost)) = best {
            if cost < self.ar_us {
                return route;
            }
        }
        if self.external_alive() && self.combined_wins() {
            // Losing native route kept alive for the external windows: the
            // cheapest measured option within `k`, else interleaved.
            return best.map_or(FloorRoute::Interleaved, |(route, _)| route);
        }
        self.retired = true;
        FloorRoute::Ar
    }
}

fn elapsed_us(start: Instant) -> f32 {
    (start.elapsed().as_secs_f64() * 1e6) as f32
}

/// Wall time since `start`, after the device drained: a window's cost
/// includes its whole asynchronous GPU tail, so no tail is billed to the next
/// window (or to the AR calibration).
fn synced_elapsed_us(gpu: &Gpu, start: Instant) -> Result<f32, String> {
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("Qwen4 MTP floor window sync: {error}"))?;
    Ok(elapsed_us(start))
}

/// Sampled verification algorithm (`HIPFIRE_MTP_SAMPLED_MODE`, resolved once
/// per drafter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampledMode {
    /// Speculative rejection sampling (Leviathan et al. 2023, Chen et al.
    /// 2023): drafts drawn from `q`, emitted stream equals AR in
    /// distribution. The default.
    Leviathan,
    /// SpecInfer naive sampling: argmax drafts, one AR-sampler draw per verify
    /// row, accept iff equal. The emitted stream is AR's seeded stream.
    Naive,
}

impl SampledMode {
    /// `naive` selects [`SampledMode::Naive`]; unset or `leviathan` the
    /// default. Any other value is refused rather than silently defaulted.
    fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var("HIPFIRE_MTP_SAMPLED_MODE").as_deref() {
            Ok("naive") => Ok(Self::Naive),
            Ok("leviathan") | Err(_) => Ok(Self::Leviathan),
            Ok(other) => Err(format!(
                "HIPFIRE_MTP_SAMPLED_MODE={other:?}: expected `leviathan` or `naive`"
            )),
        }
    }
}

/// Per-request sampled verification (`speculation.mtp_sampled`, temperature
/// above zero).
///
/// [`SampledMode::Leviathan`]:
/// - `p`, a verify row's target distribution: the full logit row truncated by
///   [`SampleSpec::cpu_ar`] with the request's temperature, top_p, top_k and
///   min_p — exactly what the Qwen4 AR producer samples (`sampler::sample_cpu`
///   → `llama::sample_top_k_p`: a 20- or 64-wide pool, the top_k cap with
///   absent = 20 and 0 = 64, min_p, nucleus).
/// - `q`, a draft step's distribution: the same truncation applied to the
///   draft head's 8 re-scored candidates and their exact logits (the whole
///   draft row when the head does not re-score). The draft token is drawn
///   from this `q`, and the verdict reads this `q`. Tokens of `p` outside
///   the 8 candidates are reached through the residual or the bonus draw.
///
/// Accept with probability `min(1, p/q)`, otherwise emit a draw from
/// `(p - q)+`; a window whose drafts are all accepted emits its bonus from
/// `p`. One request-seeded stream supplies every draw.
///
/// [`SampledMode::Naive`] (`naive` set): drafts stay the head's argmax; each
/// verify row the verdict reads is drawn with `sampler::sample_cpu` and the
/// AR producer's sampler config, from the process-wide AR sampler RNG, and a
/// draft is accepted iff it equals its row's draw
/// ([`accept_naive_prefix`]). Every emitted token is one draw, in AR's order.
struct SampledVerify {
    spec: SampleSpec,
    rng: SpecRng,
    /// The AR producer's sampler in naive mode; `None` in Leviathan mode.
    naive: Option<SamplerConfig>,
    /// Host copy of one logit row.
    host: Vec<f32>,
    scratch: Vec<(u32, f32)>,
    target: SparseDist,
    /// `q` of each draft step in the current window (Leviathan only).
    drafts: Vec<SparseDist>,
}

impl SampledVerify {
    fn new(cfg: SpecRequestConfig, max_k: usize, mode: SampledMode) -> Self {
        let naive = (mode == SampledMode::Naive).then(|| {
            // The AR producer's per-request seeding of the shared sampler
            // RNG (`generate` already did it with this seed on the daemon
            // route; repeating it keeps a direct caller replayable).
            hipfire_runtime::llama::reset_cpu_sampler_rng(cfg.rng_seed as u32);
            naive_target_sampler(&cfg)
        });
        Self {
            spec: SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p),
            rng: SpecRng::new(cfg.rng_seed),
            drafts: if naive.is_some() {
                Vec::new()
            } else {
                vec![SparseDist::default(); max_k]
            },
            naive,
            host: Vec::new(),
            scratch: Vec::with_capacity(SampleSpec::MAX_POOL),
            target: SparseDist::default(),
        }
    }

    /// Leviathan mode draws the drafts from `q`; naive keeps the argmax.
    fn draws_drafts(&self) -> bool {
        self.naive.is_none()
    }

    /// Naive mode: verify row `row`'s target token, one AR-sampler draw.
    fn naive_draw(&mut self, gpu: &Gpu, bundle: &Qwen4Bundle, row: usize) -> Result<u32, String> {
        let sampler = self
            .naive
            .as_ref()
            .ok_or("Qwen4 sampled MTP: naive draw outside naive mode")?;
        bundle
            .spec_row_logits(gpu, row, &mut self.host)
            .map_err(|error| error.to_string())?;
        Ok(sample_cpu(&mut self.host, &[], sampler))
    }

    /// Draw the draft token from the last MTP prediction's `q`, kept as the
    /// window's draft distribution `index`.
    fn sample_draft(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        index: usize,
    ) -> Result<u32, String> {
        let q = self
            .drafts
            .get_mut(index)
            .ok_or_else(|| format!("Qwen4 sampled MTP draft index {index} exceeds K"))?;
        bundle
            .mtp_draft_dist(gpu, self.spec, &mut self.host, &mut self.scratch, q)
            .map_err(|error| error.to_string())?;
        Ok(q.sample(self.rng.next_f32()))
    }

    /// Load verify row `row`'s `p` into `self.target`.
    fn load_target(&mut self, gpu: &Gpu, bundle: &Qwen4Bundle, row: usize) -> Result<(), String> {
        bundle
            .spec_row_dist(
                gpu,
                row,
                self.spec,
                &mut self.host,
                &mut self.scratch,
                &mut self.target,
            )
            .map_err(|error| error.to_string())
    }
}

/// Native GPU MTP drafter.  The MTP operator/state stay model-owned by the
/// target bundle; this adapter owns only the reusable verifier scratch and one
/// pending target-hidden row needed to seed each MTP window.
pub struct Qwen4MtpDrafter {
    max_k: usize,
    ctx_capacity: usize,
    request: SpecRequestConfig,
    scratch: Option<Box<dyn SpecScratch>>,
    pending_hidden: Option<GpuTensor>,
    row_hidden: Option<GpuTensor>,
    /// Prompt rows one chunked prefill call may capture; mirrors the attached
    /// forward's chunk capacity and sizes the spec hidden capture buffer.
    prefill_rows: usize,
    /// Recent agreement per draft depth as decayed (accepted, compared)
    /// counts (see `observe_agreement`); picks the verify route and draft
    /// depth when `HIPFIRE_MTP_INCREMENTAL` is unset.
    agreement: [(f32, f32); MTP_MAX_DEPTH],
    /// `speculation.mtp_sampled`, resolved at construction.
    sampled_enabled: bool,
    /// `HIPFIRE_MTP_SAMPLED_MODE`, resolved at construction; an invalid
    /// value fails sampled requests with its message.
    sampled_mode: Result<SampledMode, String>,
    /// The current request's sampled verification; `None` for greedy.
    sampled: Option<SampledVerify>,
    /// Row-batched operator buffers of the batched prompt-fill Append pass
    /// (`HIPFIRE_QWEN4_MTP_BATCHED_FILL`); `None` where the pass cannot run
    /// (see `Qwen4MtpGpu::append_rows_supported`) or has not been enabled yet.
    append_scratch: Option<MtpAppendScratch>,
    /// Chat end-of-turn token (`<|im_end|>`). Windows stop at it instead of
    /// the config EOS so an accepted end-of-turn stays pending for the
    /// terminal flush: committing the draft tail past it leaves a
    /// strict-prefix terminal this drafter cannot repair (the spec terminal
    /// reset path).
    end_of_turn: Option<u32>,
    /// Same-request AR floor model; reset only by `configure_request`
    /// (sticky retirement survives `mtp_reset`, prefill realignment and
    /// head refills).
    floor: MtpFloor,
    /// `HIPFIRE_MTP_AR_FLOOR` (`0` opts out), read once per request by
    /// `configure_request`; false restores the pre-floor chooser.
    floor_enabled: bool,
    /// Request counters surfaced by `request_stats`; reset only by
    /// `configure_request`.
    stats: MtpRequestStats,
}

impl Qwen4MtpDrafter {
    pub fn new(max_k: usize, ctx_capacity: usize, end_of_turn: Option<u32>) -> Self {
        Self {
            max_k: max_k.clamp(1, 10),
            ctx_capacity,
            request: SpecRequestConfig::default(),
            scratch: None,
            pending_hidden: None,
            row_hidden: None,
            prefill_rows: 0,
            agreement: [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH],
            sampled_enabled: hipfire_config::mtp_sampled_enabled(),
            sampled_mode: SampledMode::from_env(),
            sampled: None,
            append_scratch: None,
            end_of_turn,
            floor: MtpFloor::new(),
            floor_enabled: true,
            stats: MtpRequestStats::default(),
        }
    }

    /// A sampled request needs sampled verification enabled (and a valid
    /// `HIPFIRE_MTP_SAMPLED_MODE`).
    fn require_supported_request(&self) -> Result<(), String> {
        if self.sampled.is_some() {
            Ok(())
        } else if let (true, Err(error)) = (self.sampled_enabled, &self.sampled_mode) {
            require_native_greedy(self.request.temp).map_err(|_| error.clone())
        } else {
            require_native_greedy(self.request.temp)
        }
    }

    fn bundle<'a>(target: &'a mut dyn SpecTarget) -> Result<&'a mut Qwen4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Qwen4Bundle>()
            .ok_or_else(|| "Qwen4MtpDrafter: target is not a Qwen4Bundle".to_string())
    }

    fn ensure_resources(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        let width = {
            let bundle = Self::bundle(target)?;
            bundle.mtp_position().map_err(|error| error.to_string())?;
            bundle
                .config
                .hc_count
                .checked_mul(bundle.config.hidden_size)
                .ok_or_else(|| "Qwen4 MTP hidden width overflow".to_string())?
        };
        let prefill_rows = {
            let bundle = Self::bundle(target)?;
            let config = &bundle.config;
            if native_mtp_row_capture(gpu, config) {
                bundle
                    .state
                    .ensure_row_capture(gpu, self.max_k + 1, config.linear_conv_kernel_dim - 1)
                    .map_err(|error| error.to_string())?;
            }
            bundle.spec_chunk_rows().unwrap_or(1).max(1)
        };
        if self.scratch.is_none() {
            let scratch = target.new_spec_scratch(gpu, (self.max_k + 1).max(prefill_rows))?;
            self.scratch = Some(scratch);
        }
        self.prefill_rows = prefill_rows;
        if self.pending_hidden.is_none() {
            self.pending_hidden = Some(
                gpu.zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP pending hidden allocation: {error}"))?,
            );
        }
        if self.row_hidden.is_none() {
            self.row_hidden = Some(
                gpu.zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP row hidden allocation: {error}"))?,
            );
        }
        if self.append_scratch.is_none() && mtp_batched_fill_enabled() {
            let bundle = Self::bundle(target)?;
            if bundle.mtp_append_rows_supported(gpu) {
                let rows = MTP_FILL_ROWS.min(prefill_rows);
                let scratch = MtpAppendScratch::new(gpu, &bundle.config, rows)
                    .map_err(|error| format!("Qwen4 MTP append scratch allocation: {error}"))?;
                self.append_scratch = Some(scratch);
            }
        }
        Ok(())
    }

    fn row_hidden(&self) -> Result<&GpuTensor, String> {
        self.row_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())
    }

    /// Native MTP draft-step conditioning.  `HeadState` is the historical
    /// chain: row 0 is fed the window's pending hidden and every later row is
    /// fed the head's own previous-step hidden (`forward_token` with no target hidden).
    /// `AlignedHead` feeds the target's hidden of the token being processed at
    /// row 0 and the head's own hidden afterwards; `AlignedTarget` feeds the
    /// target's same-row hidden at every row.  Selectable so the pairing can be
    /// measured rather than argued: `HIPFIRE_MTP_PAIRING=aligned-head|aligned`.
    fn draft_pairing() -> DraftPairing {
        match hipfire_config::developer_var("HIPFIRE_MTP_PAIRING").as_deref() {
            Ok("aligned-head") => DraftPairing::AlignedHead,
            Ok("aligned") => DraftPairing::AlignedTarget,
            _ => DraftPairing::HeadState,
        }
    }

    /// Incremental verify: interleave one target row with one draft step and
    /// stop at the first rejection.
    ///
    /// Every row is committed as it is produced, so the target state advances
    /// by exactly the rows that were verified and the head state advances with
    /// it: no snapshot, no restore, no re-forward of the accepted prefix.  A
    /// cycle costs `accepted + 1` single-row forwards instead of a `k+1`-row
    /// batch plus a replay of the same prefix, and no draft step is wasted past
    /// the rejection point. This is the default; `HIPFIRE_MTP_INCREMENTAL=0`
    /// explicitly selects batching.
    /// QSA reselects on row 0 of each window, including consecutive windows
    /// at nonzero request positions; later rows reuse that window's selection.
    /// With `sampled`, each row's draft is drawn from its `q` and verified
    /// against the row's `p` instead of matched against the argmax.
    #[allow(clippy::too_many_arguments)]
    fn mtp_step_incremental(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
    ) -> Result<MtpWindow, String> {
        let pairing = Self::draft_pairing();
        let mut timers = MtpPhaseTimers::new();
        let mut committed: Vec<u32> = Vec::with_capacity(k + 1);
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut picks: Vec<u32> = Vec::with_capacity(k + 1);
        let mut token = seed;
        let mut accepted = 0usize;
        let mut row = 0usize;
        loop {
            let token_position = position
                .checked_add(row)
                .ok_or_else(|| "Qwen4 MTP incremental position overflow".to_string())?;
            timers.mark(gpu, "target_row");
            let mut pick = {
                let bundle = Self::bundle(target)?;
                bundle
                    .spec_capture_token(gpu, token)
                    .map_err(|error| error.to_string())?
            };
            match sampled.as_deref_mut() {
                // Naive: the row's target token is its AR-sampler draw.
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(gpu, Self::bundle(target)?, 0)?
                }
                Some(s) => s.load_target(gpu, Self::bundle(target)?, 0)?,
                None => {}
            }
            picks.push(pick);
            let row_hidden = self.row_hidden()?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(gpu, 0, row_hidden)
                    .map_err(|error| error.to_string())?;
            }
            if row == k {
                // Every draft matched: this row's pick is the bonus.  The head
                // must still consume the row's token so its own position lands on
                // the committed end, exactly as the batched path's full-accept
                // branch does with one extra `mtp_forward_token`.
                timers.mark(gpu, "draft_step");
                let hidden = match pairing {
                    DraftPairing::HeadState => None,
                    DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(row_hidden),
                };
                {
                    let bundle = Self::bundle(target)?;
                    bundle
                        .mtp_append_token(gpu, token, hidden, token_position)
                        .map_err(|error| error.to_string())?;
                }
                committed.push(match sampled.as_deref_mut() {
                    Some(s) if s.draws_drafts() => s.target.sample(s.rng.next_f32()),
                    _ => pick,
                });
                break;
            }
            timers.mark(gpu, "draft_step");
            let hidden = match pairing {
                DraftPairing::HeadState if row > 0 => None,
                DraftPairing::AlignedHead if row > 0 => None,
                DraftPairing::HeadState => Some(self.pending_hidden()?),
                DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(row_hidden),
            };
            let mut draft = {
                let bundle = Self::bundle(target)?;
                bundle
                    .mtp_forward_token(gpu, token, hidden, token_position, row == 0)
                    .map_err(|error| error.to_string())?
            };
            let verdict = match sampled.as_deref_mut() {
                Some(s) if s.draws_drafts() => {
                    draft = s.sample_draft(gpu, Self::bundle(target)?, 0)?;
                    verify_sampled_draft(&s.target, &s.drafts[0], draft, &mut s.rng)
                }
                _ if draft == pick => DraftVerdict::Accept,
                _ => DraftVerdict::Reject(pick),
            };
            drafts.push(draft);
            match verdict {
                DraftVerdict::Accept => {
                    accepted += 1;
                    committed.push(draft);
                    if draft == eos {
                        break;
                    }
                    token = draft;
                    row += 1;
                }
                DraftVerdict::Reject(replacement) => {
                    committed.push(replacement);
                    break;
                }
            }
        }
        if committed.is_empty() {
            return Err("Qwen4 MTP incremental window committed no token".to_string());
        }
        // The next window's step-0 conditioning hidden is the hidden of the last
        // committed token, which is the row just captured: the baseline path
        // copies row `target_commit_accept_len` (= the last processed row) of its verify
        // capture, and this is the same row of the same forward shape.
        {
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 MTP incremental pending hidden copy: {error}"))?;
        }
        let committed_end = position
            .checked_add(committed.len())
            .ok_or_else(|| "Qwen4 MTP incremental commit overflow".to_string())?;
        let mtp_end = Self::bundle(target)?
            .mtp_position()
            .map_err(|error| error.to_string())?;
        if mtp_end != committed_end {
            return Err(format!(
                "Qwen4 MTP incremental transaction ended at mtp={mtp_end}, expected {committed_end}"
            ));
        }
        if trace || timers.enabled() {
            let rows = drafts
                .iter()
                .zip(picks.iter())
                .map(|(draft, pick)| {
                    format!(
                        "{{\"draft\":{draft},\"pick\":{pick},\"match\":{}}}",
                        draft == pick
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"rows\":[{rows}],\"committed\":{committed:?}}}"
            );
        }
        if timers.enabled() {
            let fields = format!(
                "\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"pairing\":\"{}\",\"t_end\":{}",
                match pairing {
                    DraftPairing::HeadState => "head-state",
                    DraftPairing::AlignedHead => "aligned-head",
                    DraftPairing::AlignedTarget => "aligned-target",
                },
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated: drafts.len(),
        })
    }

    /// Only the drafts compared before the first rejection count, so each
    /// depth measures the agreement conditional on its prefix being accepted
    /// (the same quantity on both routes).
    fn observe_agreement(&mut self, window: &MtpWindow) {
        let compared = (window.accepted + 1).min(window.drafts_generated);
        for (depth, (accepted, total)) in self.agreement.iter_mut().enumerate() {
            *accepted *= MTP_AGREEMENT_DECAY;
            *total *= MTP_AGREEMENT_DECAY;
            if depth < compared {
                *total += 1.0;
                *accepted += f32::from(u8::from(depth < window.accepted));
            }
        }
    }

    /// Draft depth for the next window: the K maximizing expected emitted
    /// tokens (1 + sum over depths of the accepted-prefix probability) per
    /// window cost, or 0 (interleaved) when no batched window beats one
    /// token per interleaved step.
    fn batched_depth(&self, k: usize) -> usize {
        let mut best = (0, 1.0f32);
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        let depths = self.agreement.iter().zip(&MTP_WINDOW_COST[1..]).take(k);
        for (i, (&(accepted, total), &cost)) in depths.enumerate() {
            let depth = i + 1;
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            expected += prefix;
            let rate = expected / cost;
            if rate > best.1 {
                best = (depth, rate);
            }
        }
        best.0
    }

    /// Route with the floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`): the
    /// pre-floor per-window chooser, `0` = interleaved, else batched at that
    /// depth.
    fn floor_off_depth(&self, k: usize) -> usize {
        self.batched_depth(k)
    }

    /// One native window of the AR floor: the ordinary target-only token
    /// `seed` at `position`, with a sampled draw when the request is sampled.
    ///
    /// `calibrate` (the request's first window) times the token including
    /// that draw, then, outside the timing, keeps the head synced exactly as
    /// a `k = 0` interleaved window does (the ordinary forward's wide hidden,
    /// head append with the same `DraftPairing` conditioning, pending hidden)
    /// before recording the AR price.  Otherwise (retired) it does no head
    /// work at all and never checks the head position.
    fn mtp_ar_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        calibrate: bool,
    ) -> Result<MtpWindow, String> {
        let end = position
            .checked_add(1)
            .ok_or_else(|| "Qwen4 MTP AR step position overflow".to_string())?;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = if calibrate {
                bundle.mtp_position().map_err(|error| error.to_string())?
            } else {
                position
            };
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP AR step position mismatch: target={target_position}, mtp={mtp_position}, position={position}"
                ));
            }
        }
        if calibrate {
            // Prefill's trailing head work must not be billed to AR.
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("Qwen4 MTP AR calibration sync: {error}"))?;
        }
        let started = Instant::now();
        let mut sampled = self.sampled.take();
        let drawn = (|| -> Result<u32, String> {
            let mut pick = Self::bundle(target)?
                .spec_ar_token(gpu, seed)
                .map_err(|error| error.to_string())?;
            match sampled.as_mut() {
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(gpu, Self::bundle(target)?, 0)?
                }
                Some(s) => {
                    s.load_target(gpu, Self::bundle(target)?, 0)?;
                    pick = s.target.sample(s.rng.next_f32());
                }
                None => {}
            }
            Ok(pick)
        })();
        self.sampled = sampled;
        let pick = drawn?;
        let us = elapsed_us(started);
        let target_end = Self::bundle(target)?.state.position;
        if target_end != end {
            return Err(format!(
                "Qwen4 MTP AR step ended at target={target_end}, expected {end}"
            ));
        }
        if calibrate {
            let pairing = Self::draft_pairing();
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_ar_hidden_to(gpu, row_hidden)
                    .map_err(|error| error.to_string())?;
                let hidden = match pairing {
                    DraftPairing::HeadState => None,
                    DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(row_hidden),
                };
                bundle
                    .mtp_append_token(gpu, seed, hidden, position)
                    .map_err(|error| error.to_string())?;
                let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
                if mtp_end != end {
                    return Err(format!(
                        "Qwen4 MTP AR calibration ended at mtp={mtp_end}, expected {end}"
                    ));
                }
            }
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 MTP AR calibration pending hidden copy: {error}"))?;
            // Drain the calibration's head work so the first probe window
            // does not bill it.
            gpu.hip
                .device_synchronize()
                .map_err(|error| format!("Qwen4 MTP AR calibration tail sync: {error}"))?;
            self.floor.observe_calibration(us, &self.agreement);
            self.stats.mtp_windows += 1;
            self.stats.mtp_retired = self.floor.retired;
        } else {
            self.stats.ar_windows += 1;
        }
        Ok(MtpWindow {
            committed: vec![pick],
            accepted: 0,
            drafts_generated: 0,
        })
    }

    /// Count one finished native window; observe it for the floor unless
    /// `bypass` (a developer-forced route, `HIPFIRE_MTP_INCREMENTAL=0|1`, or
    /// `HIPFIRE_MTP_AR_FLOOR=0`) skips the floor.
    fn floor_observe_native(
        &mut self,
        bypass: bool,
        route: NativeWindow,
        window: &MtpWindow,
        us: f32,
        k: usize,
    ) {
        self.stats.mtp_windows += 1;
        if !bypass {
            self.floor
                .observe_native(route, window, us, k, &self.agreement);
            self.stats.mtp_retired = self.floor.retired;
        }
    }

    /// Whether the floor retired this request to head-free AR.  A caller
    /// that composes an external drafter must check this before every
    /// takeover: after retirement the head lags the target and must not be
    /// extended.
    #[allow(dead_code)]
    pub(crate) fn floor_retired(&self) -> bool {
        self.floor.retired
    }

    /// Whether the request's first window, the AR calibration token, has yet
    /// to run.  False for a developer-forced route (`HIPFIRE_MTP_INCREMENTAL`
    /// `0|1`) and with the floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`), which
    /// bypass the floor.  A caller that composes an external drafter must not
    /// take over while this is true: the calibration window must be the
    /// request's first.
    #[allow(dead_code)]
    pub(crate) fn floor_needs_calibration(&self) -> bool {
        self.floor_enabled && !self.floor.calibrated && !Self::floor_forced()
    }

    /// `floor_retired() || floor_needs_calibration()`: the exact guard an
    /// external takeover must pass before it runs.
    #[allow(dead_code)]
    pub(crate) fn floor_takeover_blocked(&self) -> bool {
        self.floor.retired || self.floor_needs_calibration()
    }

    /// Developer-forced route that bypasses the floor.
    fn floor_forced() -> bool {
        matches!(
            hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref(),
            Ok("0" | "1")
        )
    }

    /// Report one finished external window: `drafts` offered, `emitted`
    /// tokens committed, `wall_us` of its whole window (drafting, verify and
    /// commit).  Counts toward the shared probe budget, prices external
    /// windows separately from the native agreement, and feeds the realized
    /// mixture of all windows.  A request stays alive on a losing native
    /// route only while external windows are measured and unblocked and the
    /// decayed realized per-token cost of all its windows is strictly below
    /// AR; it retires at the first window where that stops holding.
    /// Returns `true` when
    /// external takeover must stop for the rest of the request; never with
    /// the floor opted out (`HIPFIRE_MTP_AR_FLOOR=0`).
    #[allow(dead_code)]
    pub(crate) fn floor_observe_external_window(
        &mut self,
        emitted: usize,
        wall_us: f32,
        drafts: usize,
    ) -> bool {
        if !self.floor_enabled {
            return false;
        }
        let stop = self
            .floor
            .observe_external(emitted, wall_us, drafts, &self.agreement);
        self.stats.mtp_retired = self.floor.retired;
        stop
    }

    fn pending_hidden(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())
    }

    /// The pending target-hidden row, for the parity harnesses.
    pub(crate) fn pending_hidden_for_parity(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden()
    }
}

impl MtpDrafter for Qwen4MtpDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<u32, String> {
        self.require_supported_request()?;
        validate_native_mtp_prefill_request(prompt_tokens, fill_tokens, start_pos, cache_hit)?;
        self.agreement = [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH];
        // A miss resets target, head and draft policy; a hit restores the
        // bundle's canonical checkpoint at `start_pos` into all three.
        let plan = Qwen4PrefixPlan {
            start_pos: if cache_hit { start_pos } else { 0 },
        };
        Self::bundle(target)?
            .begin_prefix(gpu, prompt_tokens, plan, Qwen4PrefixMode::NativeMtp)
            .map_err(|error| error.to_string())?;
        self.ensure_resources(gpu, target)?;
        let capture_at = {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != start_pos || mtp_position != start_pos {
                return Err(format!(
                    "Qwen4 MTP prefill position mismatch: target={}, mtp={}, start={start_pos}",
                    target_position, mtp_position
                ));
            }
            bundle.prefix_capture_at()
        };
        // Field borrows (not `pending_hidden()`): the batched fill below also
        // borrows `append_scratch` mutably.
        let pending = self
            .pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
        let mut batched_scratch = if mtp_batched_fill_enabled() {
            self.append_scratch.as_mut()
        } else {
            None
        };
        let mut first_token = None;
        // One chunked target forward per chunk instead of one single-row forward
        // per prompt token: the shared forward already captures the whole
        // chunk's wide hidden, and the head then consumes each row in order.
        // Cost goes from ~1 single-row forward per prompt token to the ordinary
        // chunked prefill rate plus one head step per token.
        // Each prompt token selects with its own query and the pooled keys
        // visible at that position, regardless of target prefill chunking.
        let chunk_rows = self.prefill_rows.max(1);
        for (chunk_index, chunk) in fill_tokens.chunks(chunk_rows).enumerate() {
            if abort() {
                target.reset_recurrent(gpu)?;
                return Err("Qwen4 native MTP prefill aborted".to_string());
            }
            let base = chunk_index * chunk_rows;
            Self::bundle(target)?.set_ple_lookahead(&fill_tokens[base + chunk.len()..]);
            let pick = Self::bundle(target)?
                .spec_prefill_rows(gpu, chunk, true)
                .map_err(|error| error.to_string())?;
            if let Some(scratch) = batched_scratch.as_deref_mut() {
                // One batched Append pass per sub-chunk of at most `scratch.rows()`
                // rows; row `i` is still paired with spec hidden row `i`.
                let mut off = 0;
                while off < chunk.len() {
                    if abort() {
                        target.reset_recurrent(gpu)?;
                        return Err("Qwen4 native MTP prefill aborted".to_string());
                    }
                    let n = (chunk.len() - off).min(scratch.rows());
                    let position = start_pos
                        .checked_add(base)
                        .and_then(|value| value.checked_add(off))
                        .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
                    if n == 1 {
                        // The multirow projections need two rows or more; a
                        // lone row is exactly the per-row Append.
                        let bundle = Self::bundle(target)?;
                        bundle
                            .copy_spec_hidden_row_to(gpu, off, pending)
                            .map_err(|error| error.to_string())?;
                        bundle
                            .mtp_append_token(gpu, chunk[off], Some(pending), position)
                            .map_err(|error| error.to_string())?;
                    } else {
                        Self::bundle(target)?
                            .mtp_append_rows(gpu, scratch, &chunk[off..off + n], off, position)
                            .map_err(|error| error.to_string())?;
                    }
                    off += n;
                }
                // The per-row loop leaves `pending` on the chunk's last row: the
                // next draft step's row-0 hidden.
                Self::bundle(target)?
                    .copy_spec_hidden_row_to(gpu, chunk.len() - 1, pending)
                    .map_err(|error| error.to_string())?;
            } else {
                for (index, &token) in chunk.iter().enumerate() {
                    if abort() {
                        target.reset_recurrent(gpu)?;
                        return Err("Qwen4 native MTP prefill aborted".to_string());
                    }
                    let position = start_pos
                        .checked_add(base)
                        .and_then(|value| value.checked_add(index))
                        .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
                    let bundle = Self::bundle(target)?;
                    bundle
                        .copy_spec_hidden_row_to(gpu, index, pending)
                        .map_err(|error| error.to_string())?;
                    bundle
                        .mtp_append_token(gpu, token, Some(pending), position)
                        .map_err(|error| error.to_string())?;
                }
            }
            // The head has caught up with the target: the only point a
            // whole-chunk checkpoint of both owners is canonical.
            let end = start_pos + base + chunk.len();
            if capture_at == Some(end) {
                Self::bundle(target)?
                    .stage_prefix(gpu, &prompt_tokens[..end])
                    .map_err(|error| error.to_string())?;
            }
            first_token = Some(pick);
        }
        let pick = first_token.expect("non-empty MTP prefill produced no seed");
        // The seed is the first emitted token: sampled, it is drawn from the
        // last prompt row's `p` (naive: with the AR sampler itself), exactly
        // as AR draws its first token.
        match self.sampled.as_mut() {
            Some(s) if !s.draws_drafts() => s.naive_draw(gpu, Self::bundle(target)?, 0),
            Some(s) => {
                s.load_target(gpu, Self::bundle(target)?, 0)?;
                Ok(s.target.sample(s.rng.next_f32()))
            }
            None => Ok(pick),
        }
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        let eos = self.end_of_turn.unwrap_or(eos);
        self.require_supported_request()?;
        if k > self.max_k {
            return Err(format!(
                "Qwen4 native MTP draft budget {k} exceeds configured K {}",
                self.max_k
            ));
        }
        let incremental = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").ok();
        // A developer-forced route is a diagnostic override: the floor
        // neither calibrates nor retires, so the pinned route runs as before.
        // `HIPFIRE_MTP_AR_FLOOR=0` (`floor_enabled == false`) bypasses the
        // floor the same way and restores the pre-floor chooser.
        let forced = matches!(incremental.as_deref(), Some("0" | "1"));
        let floor_active = self.floor_enabled && !forced;
        if floor_active && self.floor.retired {
            return self.mtp_ar_step(gpu, target, position, seed, false);
        }
        self.ensure_resources(gpu, target)?;
        if floor_active && !self.floor.calibrated {
            return self.mtp_ar_step(gpu, target, position, seed, true);
        }
        let trace =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        // Route: a (k+1)-row batched verify costs about 2.5 single-row
        // forwards, so it pays only while drafts keep being accepted; the
        // interleaved verify costs one forward per emitted token and wastes no
        // draft. `HIPFIRE_MTP_INCREMENTAL=0|1` forces a route.
        let depth = match incremental.as_deref() {
            Some("0") => k,
            Some("1") => 0,
            _ if !self.floor_enabled => self.floor_off_depth(k),
            _ => match {
                let provisional = self.batched_depth(k);
                let route = self.floor.route(&self.agreement, k, provisional);
                self.stats.mtp_retired = self.floor.retired;
                route
            } {
                FloorRoute::Ar => return self.mtp_ar_step(gpu, target, position, seed, false),
                FloorRoute::Interleaved => 0,
                FloorRoute::Batched(depth) => depth,
            },
        };
        if depth == 0 {
            let window_start = Instant::now();
            let mut sampled = self.sampled.take();
            let window = self.mtp_step_incremental(
                gpu,
                target,
                position,
                seed,
                k,
                eos,
                trace,
                sampled.as_mut(),
            );
            self.sampled = sampled;
            let window = window?;
            self.observe_agreement(&window);
            self.floor_observe_native(
                !floor_active,
                NativeWindow::Interleaved,
                &window,
                if floor_active {
                    synced_elapsed_us(gpu, window_start)?
                } else {
                    0.0
                },
                k,
            );
            return Ok(window);
        }
        // History picks the route and the most drafts; the drafts' own
        // margins stop early.
        let k_max = depth;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP step position mismatch: target={}, mtp={}, position={position}",
                    target_position, mtp_position
                ));
            }
        }
        let floor_start = Instant::now();
        let mut snapshot = {
            let bundle = Self::bundle(target)?;
            Some(
                bundle
                    .mtp_snapshot(gpu)
                    .map_err(|error| error.to_string())?,
            )
        };
        let mut timers = MtpPhaseTimers::new();
        let mut accepted_drafts = 0usize;
        let mut sampled = self.sampled.take();
        let result = (|| -> Result<MtpWindow, String> {
            timers.mark(gpu, "draft");
            let mut drafts = Vec::with_capacity(k_max);
            let mut margins: Vec<f32> = Vec::with_capacity(k_max);
            let mut input = seed;
            // The verify's PLE rows are SSD-resident: start reading each
            // token's rows as soon as it is known, while the drafts run.
            let mut known = Vec::with_capacity(k_max + 1);
            known.push(seed);
            Self::bundle(target)?.warm_ple_rows(&known);
            // Probability the whole draft prefix is accepted, from each
            // draft's exact logit margin over its runner-up.
            let mut prefix = 1.0f32;
            let mut steps = 0usize;
            // Every proposal starts with a fresh QSA selection; only later
            // draft rows within this window reuse it.
            for index in 0..k_max {
                let hidden = if index == 0 {
                    Some(self.pending_hidden()?)
                } else {
                    None
                };
                let token_position = position
                    .checked_add(index)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                input = Self::bundle(target)?
                    .mtp_forward_token(gpu, input, hidden, token_position, index == 0)
                    .map_err(|error| error.to_string())?;
                if let Some(s) = sampled.as_mut().filter(|s| s.draws_drafts()) {
                    input = s.sample_draft(gpu, Self::bundle(target)?, index)?;
                }
                steps += 1;
                let margin = Self::bundle(target)?.mtp_draft_margin();
                prefix *= draft_accept_estimate(margin);
                // A verify row pays only while its whole prefix is likely
                // accepted; the first draft always rides.
                if index > 0 && prefix < MTP_ROW_WORTH {
                    break;
                }
                margins.push(margin);
                drafts.push(input);
                if prefix * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH {
                    break;
                }
                if index + 1 < k_max {
                    known.push(input);
                    Self::bundle(target)?.warm_ple_rows(&known);
                }
            }
            let k = drafts.len();
            let mut block = Vec::with_capacity(k + 1);
            block.push(seed);
            block.extend_from_slice(&drafts);
            timers.mark(gpu, "verify");
            let picks = Self::bundle(target)?;
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 native MTP pending hidden is not allocated".to_string())?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 native MTP verify scratch is not allocated".to_string())?;
            let mut target_picks = picks
                .verify_block(gpu, &block, position, scratch.as_mut(), None)
                .map_err(|error| error.to_string())?;
            if sampled.is_some() && target_picks.len() < k + 1 {
                return Err(format!(
                    "Qwen4 sampled MTP verifier returned {} rows for {k} drafts",
                    target_picks.len()
                ));
            }
            let acceptance = match sampled.as_mut() {
                // Naive: each row the verdict reads is replaced by its draw.
                Some(s) if !s.draws_drafts() => accept_naive_prefix(&drafts, Some(eos), |row| {
                    let draw = s.naive_draw(gpu, picks, row)?;
                    target_picks[row] = draw;
                    Ok(draw)
                })?,
                Some(s) => {
                    let SampledVerify {
                        spec,
                        rng,
                        host,
                        scratch: candidates,
                        target: target_dist,
                        drafts: draft_dists,
                        ..
                    } = s;
                    accept_sampled_prefix(
                        &drafts,
                        &draft_dists[..k],
                        Some(eos),
                        rng,
                        target_dist,
                        |row, out| {
                            picks
                                .spec_row_dist(gpu, row, *spec, host, candidates, out)
                                .map_err(|error| error.to_string())
                        },
                    )?
                }
                None => accept_native_greedy(&drafts, &target_picks, Some(eos))?,
            };
            accepted_drafts = acceptance.accepted;
            if trace {
                let rows = drafts
                    .iter()
                    .zip(target_picks.iter())
                    .zip(margins.iter())
                    .map(|((draft, pick), margin)| {
                        format!(
                            "{{\"draft\":{draft},\"pick\":{pick},\"match\":{},\"margin\":{margin:.4}}}",
                            draft == pick
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"rows\":[{rows}],\"committed\":{:?},\"baseline\":true}}",
                    acceptance.committed
                );
            }
            let target_accept_len = target_commit_accept_len(&acceptance);
            let full_accept = target_accept_len == k;
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 native MTP target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 native MTP target snapshot disappeared")?;

            // Keep both pre-window tickets active until every replay and hidden
            // copy succeeds. A retained restore lets the outer rollback repair
            // both owners if either side's GPU work fails.
            if !full_accept && picks.state.row_capture_rows() >= block.len() {
                timers.mark(gpu, "target_rollback");
                picks
                    .rollback_verify_rows_retain(
                        gpu,
                        target_snapshot,
                        target_accept_len + 1,
                        &block,
                    )
                    .map_err(|error| error.to_string())?;
            } else if !full_accept {
                timers.mark(gpu, "target_replay");
                picks
                    .restore_retain(gpu, target_snapshot)
                    .map_err(|error| error.to_string())?;
                picks
                    .spec_forward_rows(gpu, &block[..target_accept_len + 1], true)
                    .map(|_| ())
                    .map_err(|error| error.to_string())?;
            }
            timers.mark(gpu, "mtp_commit");
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            if full_accept && steps > k {
                // The step that drafted the dropped token already consumed
                // the last kept draft at its position.
            } else if full_accept && k > 0 {
                let last_draft = *drafts
                    .last()
                    .ok_or_else(|| "Qwen4 native MTP full accept has no final draft".to_string())?;
                let last_position = position
                    .checked_add(k)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                picks
                    .mtp_append_token(gpu, last_draft, None, last_position)
                    .map_err(|error| error.to_string())?;
            } else if target_accept_len < drafts.len() {
                // The draft steps already consumed the kept prefix with the
                // replay's exact inputs: keep them, drop the rejected tail.
                picks
                    .mtp_truncate_retain(mtp_ticket, target_accept_len + 1)
                    .map_err(|error| error.to_string())?;
            } else {
                picks
                    .mtp_restore_retain(gpu, mtp_ticket)
                    .map_err(|error| error.to_string())?;
                // Snapshot restore brings back the pre-window QSA selection;
                // replay must reselect on its first row.
                for (index, &token) in block[..target_accept_len + 1].iter().enumerate() {
                    let hidden = if index == 0 {
                        Some(pending_hidden)
                    } else {
                        None
                    };
                    let token_position = position
                        .checked_add(index)
                        .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                    let result = if index == target_accept_len {
                        picks.mtp_append_token(gpu, token, hidden, token_position)
                    } else {
                        picks.mtp_advance_token(gpu, token, hidden, token_position, index == 0)
                    };
                    result.map_err(|error| error.to_string())?;
                }
            }
            timers.mark(gpu, "commit");
            let committed_end = position
                .checked_add(target_accept_len + 1)
                .ok_or_else(|| "Qwen4 MTP commit position overflow".to_string())?;
            if picks.state.position != committed_end
                || picks.mtp_position().map_err(|error| error.to_string())? != committed_end
            {
                return Err(format!(
                    "Qwen4 native MTP transaction ended at target={} mtp={}, expected {committed_end}",
                    picks.state.position,
                    picks.mtp_position().map_err(|error| error.to_string())?
                ));
            }
            picks
                .copy_spec_hidden_row_to(gpu, target_accept_len, pending_hidden)
                .map_err(|error| error.to_string())?;

            let window = MtpWindow {
                committed: acceptance.committed,
                accepted: acceptance.accepted,
                drafts_generated: drafts.len(),
            };
            // All fallible GPU operations are complete. Validate both tickets
            // before invalidating either arena, then perform the no-copy commit
            // boundary and clear the target scratch ticket.
            picks
                .validate_commit(target_snapshot)
                .map_err(|error| error.to_string())?;
            picks
                .mtp_validate_commit(mtp_ticket)
                .map_err(|error| error.to_string())?;
            picks.commit_validated(target_snapshot);
            picks.mtp_commit_validated(mtp_ticket);
            target_scratch.target_snapshot = None;
            snapshot = None;
            Ok(window)
        })();
        self.sampled = sampled;
        if let Ok(window) = &result {
            self.observe_agreement(window);
            let us = if !floor_active && !timers.enabled() {
                0.0
            } else {
                synced_elapsed_us(gpu, floor_start)?
            };
            self.floor_observe_native(!floor_active, NativeWindow::Batched, window, us, k);
        }
        if timers.enabled() {
            let fields = format!(
                "\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"window_us\":{:.1},\"t_end\":{}",
                floor_start.elapsed().as_secs_f64() * 1e6,
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
            let mut rollback_errors = Vec::new();
            let target_ticket = match self.scratch.as_mut() {
                Some(scratch) => match scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                    Some(scratch) => scratch.target_snapshot.take(),
                    None => {
                        rollback_errors.push("target rollback scratch type changed".to_string());
                        None
                    }
                },
                None => None,
            };
            if let Some(ticket) = target_ticket {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("target rollback failed: {rollback}"));
                }
            }
            if let Some(ticket) = snapshot.take() {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .mtp_restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("MTP rollback failed: {rollback}"));
                }
            }
            if !rollback_errors.is_empty() {
                return Err(format!(
                    "{error}; rollback failed: {}",
                    rollback_errors.join("; ")
                ));
            }
        }
        result
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        if abort() {
            return Ok(true);
        }
        // Retired: the head lags the target for good, so the caller advances
        // the target alone (no capture, no head append).
        if self.floor.retired {
            return Ok(false);
        }
        self.ensure_resources(gpu, target)?;
        let pending = self.pending_hidden()?;
        for (index, &token) in tokens.iter().enumerate() {
            if abort() {
                return Ok(true);
            }
            let position = start_pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP forced position overflow".to_string())?;
            let bundle = Self::bundle(target)?;
            bundle
                .spec_capture_token(gpu, token)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(gpu, 0, pending)
                .map_err(|error| error.to_string())?;
            bundle
                .mtp_append_token(gpu, token, Some(pending), position)
                .map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(scratch) = self.scratch.as_mut() {
            if let Some(scratch) = scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                scratch.target_snapshot = None;
            }
        }
        if let Some(hidden) = self.pending_hidden.as_ref() {
            gpu.hip
                .memset(&hidden.buf, 0, hidden.buf.size())
                .map_err(|error| format!("Qwen4 native MTP pending reset: {error}"))?;
        }
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        let Self {
            scratch,
            pending_hidden,
            append_scratch,
            ..
        } = *self;
        if let Some(scratch) = scratch {
            scratch.free(gpu);
        }
        if let Some(hidden) = pending_hidden {
            let _ = gpu.free_tensor(hidden);
        }
        if let Some(scratch) = append_scratch {
            let _ = scratch.free_gpu(gpu);
        }
    }

    fn k(&self) -> usize {
        self.max_k
    }

    fn proposal_capacity(&self) -> usize {
        self.max_k
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        !self.sampled_enabled
    }

    /// With `speculation.mtp_sampled`, a sampled request verifies by
    /// speculative rejection sampling, or naive sampling under
    /// `HIPFIRE_MTP_SAMPLED_MODE=naive`; greedy requests keep the argmax match.
    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        self.floor = MtpFloor::new();
        self.floor_enabled =
            hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() != Ok("0");
        self.stats = MtpRequestStats::default();
        let sampled = self.sampled_enabled && cfg.temp.is_finite() && cfg.temp > 1.0e-6;
        self.sampled = match (sampled, &self.sampled_mode) {
            (true, Ok(mode)) => Some(SampledVerify::new(cfg, self.max_k, *mode)),
            _ => None,
        };
    }

    fn supports_temp_verify(&self) -> bool {
        self.sampled_enabled
    }

    fn request_stats(&self) -> MtpRequestStats {
        self.stats
    }
}

/// Build the generic runtime adapter around the native Qwen4 GPU MTP core.
/// `end_of_turn` is the tokenizer's `<|im_end|>` id, when it has one.
pub fn build_qwen4_mtp_speculator(
    max_k: usize,
    ctx_capacity: usize,
    end_of_turn: Option<u32>,
) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Qwen4MtpDrafter::new(
        max_k,
        ctx_capacity,
        end_of_turn,
    )))
}

/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL`: the batched prompt-fill Append pass is on
/// unless set to `0`; read at every `mtp_prefill`.
fn mtp_batched_fill_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_BATCHED_FILL", true)
}

/// Whether this GPU's GDN route captures verify rows: row capture rides the
/// few-row persistent GDN recurrence.
pub fn native_mtp_row_capture(gpu: &Gpu, config: &crate::Qwen4Config) -> bool {
    gpu.arch_caps.has_gfx11_plus_simt()
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.linear_conv_kernel_dim == 4
}

/// Whether a native MTP request on this GPU allocates the batched
/// prompt-fill scratch (`MtpAppendScratch`): exact gfx1151 with
/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL` on.
pub fn native_mtp_batched_fill(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1151() && mtp_batched_fill_enabled()
}

/// Device bytes native MTP adds to a load whose QSA context storage is
/// `context` (its admitted `max_seq` and committed part) with a
/// `chunk_rows` prefill chunk, drafts of up to `max_k` tokens and the
/// language head stored as `head_dtype`: what the attached head
/// (`Qwen4MtpGpu`) commits, plus the larger of its build scratch (released
/// at attach) and what the first speculative request allocates after it —
/// the verify hidden rows (`max_k + 1` rows or one forward chunk, whichever
/// is larger), the pending and row hidden carries, with `row_capture`
/// (the GDN state format, where [`native_mtp_row_capture`]) the
/// `max_k + 1`-row GDN capture in that format, and with `batched_fill`
/// (where [`native_mtp_batched_fill`]) the batched prompt-fill scratch.
pub fn native_mtp_device_bytes(
    config: &crate::Qwen4Config,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    max_k: usize,
    head_dtype: rdna_compute::DType,
    row_capture: Option<crate::GdnStateFormat>,
    batched_fill: bool,
) -> Option<u64> {
    let rows = max_k.clamp(1, 10) + 1;
    let hidden_row = config
        .hc_count
        .checked_mul(config.hidden_size)?
        .checked_mul(std::mem::size_of::<f32>())?;
    let verify_rows = rows.max(context.max_seq.min(chunk_rows));
    let capture = match row_capture {
        Some(gdn) => crate::state::Qwen4State::row_capture_bytes(config, gdn, rows)?,
        None => 0,
    };
    let (resident, scratch) =
        crate::mtp_gpu::Qwen4MtpGpu::device_bytes(config, context, head_dtype)?;
    let fill = if batched_fill {
        MtpAppendScratch::device_bytes(config, MTP_FILL_ROWS.min(chunk_rows).max(1))?
    } else {
        0
    };
    let request = verify_rows
        .checked_add(2)?
        .checked_mul(hidden_row)?
        .checked_add(capture)?
        .checked_add(fill)?;
    u64::try_from(resident.checked_add(scratch.max(request))?).ok()
}

#[cfg(any(test, feature = "reference-parity"))]
/// Convert an MTP request-state error into the erased runtime error type.
pub fn mtp_error(error: MtpError) -> String {
    format!("Qwen4 native MTP: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_mtp::MtpQsaGeometry;

    #[test]
    fn native_acceptance_lowers_to_the_runtime_window_and_step() {
        let result = accept_native_greedy(&[10, 11], &[10, 99, 100], None).unwrap();
        assert_eq!(result.committed, vec![10, 99]);
        assert_eq!(result.accepted, 1);
        assert!(!result.hit_eos);
        assert_eq!(target_commit_accept_len(&result), 1);

        let step = window_to_spec_step(MtpWindow {
            committed: result.committed,
            accepted: result.accepted,
            drafts_generated: 2,
        })
        .unwrap();
        assert_eq!(step.emit.as_slice(), &[10, 99]);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 2);
        assert_eq!(step.accepted, 1);
    }

    #[test]
    fn native_zero_partial_full_and_eos_counts_are_explicit() {
        let zero = accept_native_greedy(&[], &[42], None).unwrap();
        assert_eq!(zero.committed, vec![42]);
        assert_eq!(target_commit_accept_len(&zero), 0);

        let partial = accept_native_greedy(&[10, 11], &[10, 12, 100], None).unwrap();
        assert_eq!(target_commit_accept_len(&partial), 1);

        let full = accept_native_greedy(&[10, 11], &[10, 11, 12], None).unwrap();
        assert_eq!(full.committed, vec![10, 11, 12]);
        assert_eq!(target_commit_accept_len(&full), 2);

        let eos = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert_eq!(eos.committed, vec![10, 99]);
        assert_eq!(eos.accepted, 2);
        assert!(eos.hit_eos);

        assert!(require_native_greedy(-0.0).is_ok());
        assert!(require_native_greedy(1.0e-6).is_ok());
        assert!(require_native_greedy(1.0e-5).is_err());
        assert!(require_native_greedy(f32::INFINITY).is_err());
        assert!(require_native_greedy(f32::NEG_INFINITY).is_err());
        assert!(require_native_greedy(f32::NAN).is_err());
    }

    #[test]
    fn accepted_eos_stays_pending_for_terminal_flush() {
        let accepted = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert_eq!(target_commit_accept_len(&accepted), 1);
        assert_eq!(
            committed_target_positions(7, target_commit_accept_len(&accepted)).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, target_commit_accept_len(&accepted)).unwrap(),
            8
        );

        let bonus = accept_native_greedy(&[10, 11], &[10, 99, 12], Some(99)).unwrap();
        assert!(bonus.hit_eos);
        assert_eq!(target_commit_accept_len(&bonus), 1);
    }

    #[test]
    fn zero_draft_replays_seed_before_terminal_flush() {
        let zero = accept_native_greedy(&[], &[42], Some(99)).unwrap();
        assert_eq!(target_commit_accept_len(&zero), 0);
        assert_eq!(
            committed_target_positions(7, target_commit_accept_len(&zero) + 1).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, target_commit_accept_len(&zero) + 1).unwrap(),
            8
        );
    }

    #[test]
    fn native_zero_one_all_acceptance_advances_mtp_state_in_lockstep() {
        let cases = [
            (vec![], vec![42], 0usize),
            (vec![10], vec![10, 42], 1usize),
            (vec![10, 11], vec![10, 11, 42], 2usize),
        ];
        for (drafts, target_picks, expected_accept_len) in cases {
            let acceptance = accept_native_greedy(&drafts, &target_picks, None).unwrap();
            assert_eq!(target_commit_accept_len(&acceptance), expected_accept_len);
            let consumed = target_commit_accept_len(&acceptance) + 1;
            let expected_position = native_commit_position(7, consumed).unwrap();
            let mut state = Qwen4MtpState::new(MtpQsaGeometry {
                q_heads: 2,
                kv_heads: 1,
                head_dim: 2,
                index_heads: 1,
                index_dim: 2,
                compress_ratio: 2,
                budget: 4,
                max_seq_len: 16,
                rotary_dim: 2,
                rope_theta: 10_000,
            })
            .unwrap();
            state.position = 7;
            state.qsa.position = 7;
            let snapshot = state.begin_transaction();
            state.forced_advance(consumed).unwrap();
            state.commit(snapshot).unwrap();
            assert_eq!(state.position, expected_position);
            assert_eq!(state.qsa.position, expected_position);
            assert_eq!(state.step_index, consumed);
        }
    }

    #[test]
    fn compact_native_selection_keeps_only_target_aligned_prefix() {
        let mut state = Qwen4MtpState::new(MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 16,
            rotary_dim: 2,
            rope_theta: 10_000,
        })
        .unwrap();
        state.qsa.position = 6;
        state.qsa.selected_indices = vec![0, 1, 3, 5];
        compact_native_qsa_selection(&mut state, 1, 3).unwrap();
        assert_eq!(state.qsa.selected_indices, vec![0, 1, 3]);
        assert_eq!(committed_target_positions(1, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(native_commit_position(1, 3).unwrap(), 4);
    }

    #[test]
    fn malformed_target_picks_are_rejected_before_acceptance() {
        assert!(accept_native_greedy(&[2], &[], None).is_err());
        assert!(accept_native_greedy(&[2, 3], &[2, 3], None).is_err());
    }

    fn agreement(accepted: f32, total: f32) -> Agreement {
        [(accepted, total); MTP_MAX_DEPTH]
    }

    fn window(accepted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![0; accepted + 1],
            accepted,
            drafts_generated: drafts,
        }
    }

    fn calibrated(ar_us: f32, agreement: &Agreement) -> MtpFloor {
        let mut floor = MtpFloor::new();
        floor.observe_calibration(ar_us, agreement);
        floor
    }

    fn batched_windows(
        floor: &mut MtpFloor,
        agreement: &Agreement,
        depth: usize,
        us: f32,
        n: usize,
    ) {
        for _ in 0..n {
            floor.observe_native(NativeWindow::Batched, &window(depth, depth), us, 7, agreement);
        }
    }

    /// `n` batched windows that drafted `depth` tokens but realized three
    /// committed tokens in 60 us (20 us per token), whatever the agreement
    /// predicts.
    fn batched_windows_realized(floor: &mut MtpFloor, agreement: &Agreement, depth: usize, n: usize) {
        for _ in 0..n {
            floor.observe_native(NativeWindow::Batched, &window(2, depth), 60.0, 7, agreement);
        }
    }

    #[test]
    fn floor_low_acceptance_retires_within_the_probe_budget() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        for _ in 0..MTP_FLOOR_PROBE_WINDOWS {
            assert!(!floor.retired);
            assert_eq!(floor.route(&bad, 3, 2), FloorRoute::Batched(2));
            floor.observe_native(NativeWindow::Batched, &window(0, 2), 90.0, 3, &bad);
        }
        assert!(floor.retired);
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.route(&bad, 3, 2), FloorRoute::Ar);
    }

    #[test]
    fn floor_winning_cost_depth_is_unchanged() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(33.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, MTP_FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);
        // The provisional (Halo-table) depth is ignored once measured.
        assert_eq!(floor.route(&good, 7, 5), FloorRoute::Batched(3));
        // `k` below the measured winning depth: no eligible measured option,
        // so the request retires to AR rather than run an unmeasured route.
        assert_eq!(floor.route(&good, 2, 5), FloorRoute::Ar);
        assert!(floor.retired);
    }

    #[test]
    fn floor_tie_with_ar_retires() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        for depth in 1..=MTP_FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Batched, &window(0, depth), 30.0, 3, &bad);
        }
        assert!(floor.retired);
    }

    #[test]
    fn floor_true_ar_price_changes_choices_across_cards() {
        // Identical measured windows (75 us for 3.44 predicted tokens, 21.8
        // us per token); only the card's measured AR price differs.
        let good = agreement(1.8, 2.0);
        for (ar_us, retires) in [(33.0, false), (22.0, false), (21.0, true), (10.0, true)] {
            let mut floor = calibrated(ar_us, &good);
            batched_windows(&mut floor, &good, 3, 75.0, MTP_FLOOR_PROBE_WINDOWS);
            assert_eq!(floor.retired, retires, "ar_us {ar_us}");
        }
    }

    #[test]
    fn floor_interleaved_is_priced_per_emitted_token() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        for _ in 0..MTP_FLOOR_PROBE_WINDOWS {
            floor.observe_native(NativeWindow::Interleaved, &window(1, 1), 50.0, 3, &bad);
        }
        assert!(!floor.retired);
        assert_eq!(floor.route(&bad, 3, 2), FloorRoute::Interleaved);
    }

    #[test]
    fn floor_probe_budget_is_bounded_even_for_unusable_timings() {
        let good = agreement(1.8, 2.0);
        let mut floor = MtpFloor::new();
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(floor.probes, 0, "uncalibrated windows are not observed");
        floor.observe_calibration(33.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(0, 0), 75.0, 0, &good);
        assert_eq!(floor.probes, 0, "k == 0 is not a probe");
        // Unusable timings record nothing but spend the budget, and with
        // nothing measured the request retires at the budget.
        for us in [f32::NAN, f32::INFINITY, 0.0, -1.0, f32::NAN] {
            floor.observe_native(NativeWindow::Batched, &window(2, 2), us, 3, &good);
            assert!(floor.probes <= MTP_FLOOR_PROBE_WINDOWS);
        }
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.batched[2], (0.0, 0.0));
        assert!(floor.retired);

        let mut floor = calibrated(33.0, &good);
        for _ in 0..MTP_FLOOR_PROBE_WINDOWS {
            floor.observe_external(0, f32::NAN, 4, &good);
        }
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        assert!(floor.retired);

        let mut floor = calibrated(33.0, &good);
        batched_windows(&mut floor, &good, 2, 60.0, 2);
        floor.observe_external(4, 60.0, 4, &good);
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        batched_windows(&mut floor, &good, 2, 60.0, 10);
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
    }

    #[test]
    fn floor_probe_route_is_the_provisional_depth() {
        let good = agreement(1.8, 2.0);
        let mut floor = MtpFloor::new();
        for calibration in [false, true] {
            if calibration {
                floor.observe_calibration(33.0, &good);
            }
            for provisional in 0..=7 {
                let expected = match provisional {
                    0 => FloorRoute::Interleaved,
                    depth => FloorRoute::Batched(depth),
                };
                assert_eq!(floor.route(&good, 7, provisional), expected);
            }
        }
    }

    #[test]
    fn floor_retirement_is_sticky() {
        let bad = agreement(0.0, 2.0);
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 2, 90.0, MTP_FLOOR_PROBE_WINDOWS);
        assert!(floor.retired);
        batched_windows(&mut floor, &good, 3, 5.0, 10);
        assert!(floor.observe_external(8, 10.0, 8, &good));
        floor.observe_calibration(1000.0, &good);
        assert!(floor.retired);
        assert_eq!(floor.route(&good, 7, 3), FloorRoute::Ar);
    }

    #[test]
    fn floor_external_costs_are_separate_from_native_agreement() {
        let good = agreement(1.8, 2.0);
        let before = good;
        let mut floor = calibrated(30.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, 2);
        // 15 us per external token beats AR; native wins too.
        assert!(!floor.observe_external(4, 60.0, 4, &good));
        assert!(!floor.retired);
        assert_eq!(good, before);
        assert_eq!(floor.interleaved, (0.0, 0.0));
        // Once external windows lose, takeover stops; native keeps running.
        let mut stopped = false;
        for _ in 0..30 {
            if floor.observe_external(1, 200.0, 4, &good) {
                stopped = true;
                break;
            }
        }
        assert!(stopped && !floor.retired);
        assert_eq!(floor.route(&good, 7, 5), FloorRoute::Batched(3));
    }

    #[test]
    fn floor_good_external_keeps_a_losing_native_route_alive_while_the_mixture_beats_ar() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // 10 us per external token beats AR.
        assert!(!floor.observe_external(4, 40.0, 4, &bad));
        assert!(!floor.observe_external(4, 40.0, 4, &bad));
        // One losing native window (90 us for one token) spends the budget;
        // the realized mixture (~20.6 us per token) still beats AR.
        floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        assert!(!floor.retired);
        // The measured native route loses to AR, yet it keeps running while
        // the mixture wins.
        assert_eq!(floor.route(&bad, 7, 5), FloorRoute::Batched(1));
        assert!(!floor.retired);
        // Losing native windows drag the mixture up to AR; it then retires at
        // the first window where the mixture stops beating AR, stickily.
        let mut kept_alive = 0;
        while !floor.retired {
            assert_eq!(floor.route(&bad, 7, 5), FloorRoute::Batched(1));
            floor.observe_native(NativeWindow::Batched, &window(0, 1), 90.0, 7, &bad);
            kept_alive += 1;
            assert!(kept_alive < 50, "never retired");
        }
        assert!(kept_alive > 0);
        assert!(floor.combined.0 / floor.combined.1 >= 30.0);
        assert_eq!(floor.route(&bad, 7, 5), FloorRoute::Ar);
        assert!(floor.retired);
        assert!(floor.observe_external(4, 40.0, 4, &bad));
        assert_eq!(floor.route(&bad, 7, 5), FloorRoute::Ar);
    }

    #[test]
    fn floor_three_good_external_hits_do_not_retire_and_route_interleaved() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        for _ in 0..MTP_FLOOR_PROBE_WINDOWS {
            assert!(!floor.observe_external(4, 60.0, 4, &good));
            assert!(!floor.retired);
        }
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        // More good hits with zero native observations still keep it alive.
        assert!(!floor.observe_external(4, 60.0, 4, &good));
        assert!(!floor.retired);
        // No native option is measured: the next native window runs the
        // interleaved route rather than retiring.
        assert_eq!(floor.route(&good, 7, 5), FloorRoute::Interleaved);
        assert!(!floor.retired);
    }

    #[test]
    fn floor_good_external_with_no_native_measured_within_k_routes_interleaved() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        assert!(!floor.observe_external(4, 40.0, 4, &good));
        assert!(!floor.observe_external(4, 40.0, 4, &good));
        // Depth 3: 150 us over 3.44 predicted tokens loses to AR, but the
        // realized mixture (~20 us per token) wins.
        floor.observe_native(NativeWindow::Batched, &window(3, 3), 150.0, 7, &good);
        assert!(!floor.retired);
        assert_eq!(floor.route(&good, 7, 5), FloorRoute::Batched(3));
        // `k` below the only measured depth: nothing measured within k.
        assert_eq!(floor.route(&good, 2, 5), FloorRoute::Interleaved);
        assert!(!floor.retired);
    }

    #[test]
    fn floor_blocked_external_does_not_keep_a_losing_native_route_alive() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        // Native windows lose by prediction (one token per window) while
        // their realized cost (20 us per token) would win the mixture.
        batched_windows_realized(&mut floor, &bad, 1, 2);
        // One external window at exactly AR's price is blocked (a tie loses).
        assert!(floor.observe_external(1, 30.0, 4, &bad));
        assert!(floor.combined.0 / floor.combined.1 < 30.0);
        assert!(floor.retired);
        assert_eq!(floor.route(&bad, 7, 5), FloorRoute::Ar);
    }

    #[test]
    fn floor_combined_mixture_prices_every_valid_window_after_calibration() {
        let good = agreement(1.8, 2.0);
        let mut floor = MtpFloor::new();
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(floor.combined, (0.0, 0.0), "uncalibrated windows are not priced");
        floor.observe_calibration(30.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 75.0, 3, &good);
        assert_eq!(floor.combined, (75.0, 3.0));
        floor.observe_native(NativeWindow::Interleaved, &window(1, 1), 40.0, 3, &good);
        assert_eq!(
            floor.combined,
            (75.0 * MTP_AGREEMENT_DECAY + 40.0, 3.0 * MTP_AGREEMENT_DECAY + 2.0)
        );
        let before = floor.combined;
        floor.observe_native(NativeWindow::Batched, &window(1, 1), f32::NAN, 3, &good);
        assert_eq!(floor.combined, before, "invalid timings are not priced");
        floor.observe_external(4, 40.0, 4, &good);
        assert_eq!(
            floor.combined,
            (before.0 * MTP_AGREEMENT_DECAY + 40.0, before.1 * MTP_AGREEMENT_DECAY + 4.0)
        );
        assert_eq!(MtpFloor::new().combined, (0.0, 0.0));
    }

    #[test]
    fn floor_bad_external_retires_when_native_does_not_win() {
        let bad = agreement(0.0, 2.0);
        let mut floor = calibrated(30.0, &bad);
        batched_windows(&mut floor, &bad, 1, 90.0, 2);
        assert!(floor.observe_external(1, 200.0, 4, &bad));
        assert!(floor.retired);
    }

    #[test]
    fn floor_blocked_external_leaves_winning_native_running() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(30.0, &good);
        batched_windows(&mut floor, &good, 3, 75.0, 2);
        assert!(floor.observe_external(1, 90.0, 2, &good));
        assert!(!floor.retired);
        assert_eq!(floor.route(&good, 7, 5), FloorRoute::Batched(3));
    }

    #[test]
    fn floor_invalid_ar_price_retires_at_the_budget_not_before() {
        let good = agreement(1.8, 2.0);
        for bad_us in [f32::NAN, 0.0, -5.0] {
            let mut floor = calibrated(bad_us, &good);
            batched_windows(&mut floor, &good, 3, 75.0, MTP_FLOOR_PROBE_WINDOWS - 1);
            assert!(!floor.retired);
            batched_windows(&mut floor, &good, 3, 75.0, 1);
            assert!(floor.retired);
        }
    }

    #[test]
    fn floor_depth_is_keyed_by_drafts_actually_generated() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(33.0, &good);
        floor.observe_native(NativeWindow::Batched, &window(1, 1), 50.0, 3, &good);
        assert!(floor.batched[1].1 > 0.0);
        assert_eq!(floor.batched[3], (0.0, 0.0));
        let depth_one = floor.batched[1];
        floor.observe_native(NativeWindow::Batched, &window(2, 2), 60.0, 3, &good);
        assert_eq!(floor.batched[1], depth_one, "decay is entry-local");
    }

    #[test]
    fn floor_max_k_beyond_the_cost_table_is_clamped() {
        let good = agreement(1.8, 2.0);
        let mut floor = calibrated(100.0, &good);
        let top = MTP_WINDOW_COST.len() - 1;
        assert!(top < MTP_MAX_DEPTH);
        floor.observe_native(NativeWindow::Batched, &window(top, top), 100.0, 10, &good);
        floor.observe_native(NativeWindow::Batched, &window(top, top), 100.0, 10, &good);
        // A window longer than the table has no cost entry; it must not
        // panic and does not create one.
        floor.observe_native(
            NativeWindow::Batched,
            &window(top + 1, top + 1),
            100.0,
            10,
            &good,
        );
        assert_eq!(floor.probes, MTP_FLOOR_PROBE_WINDOWS);
        assert_eq!(floor.batched.len(), MTP_WINDOW_COST.len());
        assert!(!floor.retired);
        assert_eq!(floor.route(&good, 10, 5), FloorRoute::Batched(top));
    }

    fn floor_drafter(floor_enabled: bool) -> Qwen4MtpDrafter {
        let mut drafter = Qwen4MtpDrafter::new(7, 4096, None);
        drafter.floor_enabled = floor_enabled;
        drafter
    }

    #[test]
    fn floor_optout_never_calibrates_or_blocks_takeover() {
        let drafter = floor_drafter(false);
        assert!(!drafter.floor.calibrated);
        assert!(!drafter.floor_needs_calibration());
        assert!(!drafter.floor_takeover_blocked());
        assert!(!drafter.floor_retired());
    }

    #[test]
    fn floor_optout_external_windows_never_stop_takeover_or_retire() {
        let mut drafter = floor_drafter(false);
        for _ in 0..(4 * MTP_FLOOR_PROBE_WINDOWS) {
            assert!(!drafter.floor_observe_external_window(0, f32::INFINITY, 4));
            assert!(!drafter.floor_observe_external_window(1, 1.0e9, 4));
        }
        assert!(!drafter.floor.retired);
        assert_eq!(drafter.floor.probes, 0);
        assert!(!drafter.request_stats().mtp_retired);
        assert_eq!(drafter.request_stats().ar_windows, 0);
        assert!(!drafter.floor_takeover_blocked());
    }

    #[test]
    fn floor_optout_native_windows_count_without_observing_the_floor() {
        let mut drafter = floor_drafter(false);
        let good = window(2, 2);
        for _ in 0..(2 * MTP_FLOOR_PROBE_WINDOWS) {
            drafter.floor_observe_native(true, NativeWindow::Batched, &good, 1.0e9, 3);
            drafter.floor_observe_native(true, NativeWindow::Interleaved, &good, 1.0e9, 3);
        }
        let stats = drafter.request_stats();
        assert_eq!(stats.mtp_windows, 4 * MTP_FLOOR_PROBE_WINDOWS);
        assert!(!stats.mtp_retired);
        assert_eq!(stats.ar_windows, 0);
        assert_eq!(drafter.floor.probes, 0);
    }

    #[test]
    fn floor_optout_control_floor_on_still_retires_on_terrible_external_windows() {
        let mut drafter = floor_drafter(true);
        let bad = agreement(0.0, 2.0);
        drafter.floor.observe_calibration(30.0, &bad);
        let mut stopped = false;
        for _ in 0..30 {
            if drafter.floor_observe_external_window(1, 1.0e9, 4) {
                stopped = true;
                break;
            }
        }
        assert!(stopped);
    }

    #[test]
    fn floor_optout_depth_is_the_pre_floor_batched_depth() {
        let mut drafter = floor_drafter(false);
        for (accepted, total) in [(0.0, 2.0), (1.0, 2.0), (1.8, 2.0), (2.0, 2.0)] {
            drafter.agreement = agreement(accepted, total);
            for k in 0..=drafter.max_k {
                assert_eq!(drafter.floor_off_depth(k), drafter.batched_depth(k));
            }
        }
        // Perfect agreement pays for batching; none falls back to interleaved.
        drafter.agreement = agreement(2.0, 2.0);
        assert!(drafter.floor_off_depth(drafter.max_k) > 0);
        drafter.agreement = agreement(0.0, 2.0);
        assert_eq!(drafter.floor_off_depth(drafter.max_k), 0);
    }
}
