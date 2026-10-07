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
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuStateSnapshot, MTP_FILL_ROWS};
#[cfg(any(test, feature = "reference-parity"))]
use crate::reference_mtp::{MtpError, Qwen4MtpState};
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::ngram_mod::{MtpNgramContext, NgramModConfig};
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::spec::{
    accept_greedy_prefix, GreedyAccept, MtpDrafter, MtpRequestStats, MtpSpeculator, MtpWindow,
    SpecAdvance, SpecGrammar, SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use hipfire_runtime::spec_sampling::{
    accept_naive_prefix, accept_sampled_prefix, naive_target_sampler, verify_sampled_draft,
    DraftVerdict, PenaltyHistory, SampleSpec, SparseDist, SpecRng,
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
/// Emitted tokens a request's first takeover is assumed to yield (one
/// pseudo-window): the offline 5/3/3 TC/Hermes accepted-prefix-plus-bonus on
/// a pool hit was 2.9-3.2 (`release-0.4.1/mtp-ngram-plan.md` §1).
const NGRAM_YIELD_PRIOR: f32 = 3.0;
/// Per declined pool hit, the takeover yield history decays by this, so a
/// source that lost to native MTP is re-tried once the evidence ages out.
const NGRAM_YIELD_RECOVERY: f32 = 0.95;
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

/// `QWEN4_MTP_TRACE` per-row `{draft, pick, match}` objects.
fn trace_rows(drafts: &[u32], picks: &[u32]) -> String {
    drafts
        .iter()
        .zip(picks)
        .map(|(draft, pick)| {
            format!(
                "{{\"draft\":{draft},\"pick\":{pick},\"match\":{}}}",
                draft == pick
            )
        })
        .collect::<Vec<_>>()
        .join(",")
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
/// Target policy: the Qwen4 AR producer is host `sampler::sample_cpu` over
/// the FULL rendered prompt plus the generated tokens, with the request's
/// repeat / presence / frequency penalties over the trailing `repeat_window`
/// tokens, then temperature, top_p, top_k and min_p. [`Self::policy`]
/// ([`naive_target_sampler`]) is that sampler and is the target policy of
/// BOTH modes; [`Self::history`] supplies its token history. The history
/// penalized at verify row `i` (the row that scores the token after `i`
/// accepted drafts) is `H_i = suffix_W(P ‖ E ‖ drafts[..i])`, with `P` the
/// prompt, `E` the tokens emitted before the window (pending seed included),
/// and `W` the penalty window. Rebuilt from `E` every window
/// ([`Self::begin_window`]): a rejected draft or a pruned proposal never
/// enters it. A neutral request has window 0 and does no history work.
///
/// [`SampledMode::Leviathan`]:
/// - `p`, a verify row's target distribution: the logit row penalized by
///   `H_i`, then truncated by [`SampleSpec::cpu_ar`] with the request's
///   temperature, top_p, top_k and min_p — exactly what the Qwen4 AR
///   producer samples (`sampler::sample_cpu` → `llama::sample_top_k_p`: a
///   20- or 64-wide pool, the top_k cap with absent = 20 and 0 = 64, min_p,
///   nucleus).
/// - `q`, a draft step's distribution: the same policy and truncation
///   applied to the draft head's 8 re-scored candidates and their exact
///   logits (the whole draft row when the head does not re-score), penalized
///   by the same `H_i` as the verify row the draft is proposed for. The
///   draft token is drawn from this `q`, and the verdict reads this `q`.
///   Tokens of `p` outside the 8 candidates are reached through the residual
///   or the bonus draw.
///
/// Accept with probability `min(1, p/q)`, otherwise emit a draw from
/// `(p - q)+`; a window whose drafts are all accepted emits its bonus from
/// `p`. One request-seeded stream supplies every draw.
///
/// [`SampledMode::Naive`] (`naive` set): drafts stay the head's argmax; each
/// verify row the verdict reads is drawn with `sampler::sample_cpu`, the
/// penalty history `H_i` and the AR producer's sampler config, from the
/// process-wide AR sampler RNG, and a draft is accepted iff it equals its
/// row's draw ([`accept_naive_prefix`]). Every emitted token is one draw, in
/// AR's order.
struct SampledVerify {
    spec: SampleSpec,
    rng: SpecRng,
    /// [`SampledMode::Naive`]: one AR-sampler draw per verify row, argmax
    /// drafts. `false`: Leviathan.
    naive: bool,
    /// The AR producer's sampler ([`naive_target_sampler`]): the target
    /// policy of both modes, penalties included.
    policy: SamplerConfig,
    /// Penalty history: prompt tail, tokens emitted before the window and
    /// the window's kept drafts; window 0 when no penalty is active.
    history: PenaltyHistory,
    /// The request's prompt is in `history`. Set by the request's first
    /// prefill; a later prefill of the same request is a strict-prefix
    /// realign whose `prompt_tokens` replay the prompt AND emitted tokens,
    /// which `begin_window(emitted)` already supplies.
    prompt_set: bool,
    /// Host copy of one logit row.
    host: Vec<f32>,
    scratch: Vec<(u32, f32)>,
    target: SparseDist,
    /// `q` of each draft step in the current window (Leviathan only).
    drafts: Vec<SparseDist>,
    /// Point-mass `q`s of drafts that were not drawn from a distribution
    /// (`accept_leviathan` with `point_mass`), reused across windows.
    points: Vec<SparseDist>,
}

impl SampledVerify {
    fn new(cfg: SpecRequestConfig, max_k: usize, mode: SampledMode) -> Self {
        let naive = mode == SampledMode::Naive;
        if naive {
            // The AR producer's per-request seeding of the shared sampler
            // RNG (`generate` already did it with this seed on the daemon
            // route; repeating it keeps a direct caller replayable).
            hipfire_runtime::llama::reset_cpu_sampler_rng(cfg.rng_seed as u32);
        }
        Self {
            spec: SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p),
            rng: SpecRng::new(cfg.rng_seed),
            drafts: if naive {
                Vec::new()
            } else {
                vec![SparseDist::default(); max_k]
            },
            points: Vec::new(),
            naive,
            policy: naive_target_sampler(&cfg),
            history: PenaltyHistory::new(cfg.penalty_window()),
            prompt_set: false,
            host: Vec::new(),
            scratch: Vec::with_capacity(SampleSpec::MAX_POOL),
            target: SparseDist::default(),
        }
    }

    /// Leviathan mode draws the drafts from `q`; naive keeps the argmax.
    fn draws_drafts(&self) -> bool {
        !self.naive
    }

    /// Start a window: rebuild the penalty history from the emitted tokens
    /// (the pending seed included); returns the history length before drafts.
    fn begin_window(&mut self, emitted: &[u32]) -> usize {
        self.history.begin_window(emitted)
    }

    /// Naive mode: physical verify row `row`'s target token, one AR-sampler
    /// draw penalized by history row `hist_row`.
    fn naive_draw(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        row: usize,
        hist_row: usize,
    ) -> Result<u32, String> {
        if !self.naive {
            return Err("Qwen4 sampled MTP: naive draw outside naive mode".to_string());
        }
        bundle
            .spec_row_logits(gpu, row, &mut self.host)
            .map_err(|error| error.to_string())?;
        Ok(sample_cpu(
            &mut self.host,
            self.history.row(hist_row),
            &self.policy,
        ))
    }

    /// Draw the draft token from the last MTP prediction's `q` (penalized by
    /// history row `hist_row`), kept as the window's draft distribution
    /// `index`.
    fn sample_draft(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        index: usize,
        hist_row: usize,
    ) -> Result<u32, String> {
        let q = self
            .drafts
            .get_mut(index)
            .ok_or_else(|| format!("Qwen4 sampled MTP draft index {index} exceeds K"))?;
        bundle
            .mtp_draft_dist(
                gpu,
                self.spec,
                self.history.row(hist_row),
                &self.policy,
                &mut self.host,
                &mut self.scratch,
                q,
            )
            .map_err(|error| error.to_string())?;
        Ok(q.sample(self.rng.next_f32()))
    }

    /// Load physical verify row `row`'s `p` (penalized by history row
    /// `hist_row`) into `self.target`.
    fn load_target(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        row: usize,
        hist_row: usize,
    ) -> Result<(), String> {
        bundle
            .spec_row_dist(
                gpu,
                row,
                self.spec,
                self.history.row(hist_row),
                &self.policy,
                &mut self.host,
                &mut self.scratch,
                &mut self.target,
            )
            .map_err(|error| error.to_string())
    }

    /// Leviathan acceptance over a batched verify: `drafts[i]` is verified
    /// against `p_i` penalized by `H_i` (physical verify row == history
    /// row). `q_i` is the distribution the draft was drawn from
    /// (`self.drafts[i]`), or with `point_mass` the point mass on the draft.
    fn accept_leviathan(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        drafts: &[u32],
        point_mass: bool,
        eos: u32,
    ) -> Result<GreedyAccept, String> {
        let SampledVerify {
            spec,
            rng,
            policy,
            history,
            host,
            scratch,
            target,
            drafts: draft_dists,
            points,
            ..
        } = self;
        history.rewind_drafts();
        for &draft in drafts {
            history.push_draft(draft);
        }
        let qs: &[SparseDist] = if point_mass {
            if points.len() < drafts.len() {
                points.resize_with(drafts.len(), SparseDist::default);
            }
            for (point, &draft) in points.iter_mut().zip(drafts) {
                point.set_point_mass(draft);
            }
            &points[..drafts.len()]
        } else {
            draft_dists
                .get(..drafts.len())
                .ok_or_else(|| "Qwen4 sampled MTP: more drafts than K".to_string())?
        };
        let history = &*history;
        let spec = *spec;
        accept_sampled_prefix(drafts, qs, Some(eos), rng, target, |row, out| {
            bundle
                .spec_row_dist(gpu, row, spec, history.row(row), policy, host, scratch, out)
                .map_err(|error| error.to_string())
        })
    }

    /// Naive acceptance over a batched verify: row `i`'s draw is penalized
    /// by `H_i` and recorded in `picks[i]`.
    fn accept_naive(
        &mut self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        drafts: &[u32],
        eos: u32,
        picks: &mut [u32],
    ) -> Result<GreedyAccept, String> {
        self.history.rewind_drafts();
        for &draft in drafts {
            self.history.push_draft(draft);
        }
        accept_naive_prefix(drafts, Some(eos), |row| {
            let draw = self.naive_draw(gpu, bundle, row, row)?;
            picks[row] = draw;
            Ok(draw)
        })
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
    /// N-gram pool configuration resolved at load for this GPU
    /// (`hipfire_config::ngram_mod_triple_for_arch`); `None` disables the
    /// n-gram takeover for every request.
    ngram_config: Option<NgramModConfig>,
    /// Request-local n-gram proposal owner, allocated on the first armed
    /// request and reused (cleared and reseeded) by every later one.
    ngram: Option<MtpNgramContext>,
    /// This request tries an n-gram takeover before each native window
    /// (`SpecRequestConfig::allow_ngram_modifier`).
    ngram_active: bool,
    /// The current window's candidates, copied out of the pool (reused).
    ngram_candidates: Vec<u32>,
    /// Decayed (emitted tokens, windows) of this request's takeover windows:
    /// the n-gram source's own yield, kept apart from the native per-depth
    /// agreement (see `ngram_wins`).
    ngram_yield: (f32, f32),
    /// `[seed, candidates..]` of the current takeover window (reused).
    takeover_block: Vec<u32>,
    /// Request-local wire counters (reset by `configure_request`).
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
            ngram_config: None,
            ngram: None,
            ngram_active: false,
            ngram_candidates: Vec::new(),
            ngram_yield: (0.0, 0.0),
            takeover_block: Vec::new(),
            stats: MtpRequestStats::default(),
        }
    }

    /// Enable n-gram takeover windows with `config` for requests that arm
    /// `allow_ngram_modifier`; `None` keeps every window native.
    pub fn with_ngram(mut self, config: Option<NgramModConfig>) -> Self {
        self.ngram_config = config;
        self
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
                    pick = s.naive_draw(gpu, Self::bundle(target)?, 0, row)?
                }
                Some(s) => s.load_target(gpu, Self::bundle(target)?, 0, row)?,
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
                    draft = s.sample_draft(gpu, Self::bundle(target)?, 0, row)?;
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
                    // The accepted draft is history for the next row (and
                    // the bonus); a rejected one never enters it.
                    if let Some(s) = sampled.as_deref_mut() {
                        s.history.push_draft(draft);
                    }
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
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"interleaved\",\"position\":{position},\"k\":{k},\"accepted\":{accepted},\"rows\":[{rows}],\"committed\":{committed:?}}}"
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

    /// Best expected emitted tokens per unit of window cost the native route
    /// offers at the current agreement (`batched_depth`'s objective; the
    /// interleaved route is 1.0).
    fn native_rate(&self, k: usize) -> f32 {
        let mut best = 1.0f32;
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        let depths = self.agreement.iter().zip(&MTP_WINDOW_COST[1..]).take(k);
        for (&(accepted, total), &cost) in depths {
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            expected += prefix;
            best = best.max(expected / cost);
        }
        best
    }

    /// Whether an `n`-candidate takeover is expected to emit at least as much
    /// per unit cost as the native window it displaces. N-gram hits and native
    /// drafts can succeed on the same spans (a pool hit on a span the model
    /// rewrites displaces a better native window), so the source is chosen by
    /// each source's own measured yield: the takeover's decayed emitted per
    /// window (prior `NGRAM_YIELD_PRIOR` tokens, one pseudo-window) against
    /// the native agreement's best rate. Declined hits let the takeover
    /// history decay back toward the prior, so the n-gram source is re-tried.
    fn ngram_wins(&self, n: usize, k: usize) -> bool {
        let (emitted, windows) = self.ngram_yield;
        let expected = (emitted + NGRAM_YIELD_PRIOR) / (windows + 1.0);
        let cost = MTP_WINDOW_COST[n.min(MTP_WINDOW_COST.len() - 1)];
        expected / cost >= self.native_rate(k.min(self.max_k))
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

    /// One n-gram takeover window: the pool's `candidates` replace the MTP
    /// head's drafts for this window.
    ///
    /// `[seed, candidates..]` is verified through the same target executor
    /// as a native window: the batched `verify_block` with row-capture
    /// rollback, or under `HIPFIRE_MTP_INCREMENTAL=1` the interleaved
    /// one-row route that stops at the first rejection. Acceptance is the
    /// shared rule of the request: greedy prefix match, or sampled with each
    /// candidate's draft distribution the point mass δ(candidate) (accept with
    /// probability p(candidate), otherwise draw from p without it), or the
    /// naive AR-sampler draw per row.
    ///
    /// The head never proposed these rows, so after the target keeps its
    /// consumed rows (`seed` plus the accepted candidates; an accepted
    /// end-of-turn stays pending) the MTP head appends exactly those rows,
    /// teacher-forced: token `p` paired with the target hidden of row `p`,
    /// the pairing of the prompt fill and `mtp_forced_advance`, through the
    /// batched append where it runs. The pending hidden becomes the last
    /// consumed row's, so the next native window drafts as it would after a
    /// prompt fill ending there. A rejected tail never reaches either owner.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_takeover_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        candidates: &[u32],
        eos: u32,
    ) -> Result<MtpWindow, String> {
        let eos = self.end_of_turn.unwrap_or(eos);
        self.require_supported_request()?;
        // The window's penalty history restarts from `emitted` (idempotent
        // after `mtp_step`'s own `begin_window`).
        if let Some(s) = self.sampled.as_mut() {
            s.begin_window(emitted);
        }
        if candidates.is_empty() {
            return Err("Qwen4 MTP takeover has no candidates".to_string());
        }
        self.ensure_resources(gpu, target)?;
        {
            let bundle = Self::bundle(target)?;
            let vocab = bundle.config.vocab_size;
            if let Some(&token) = candidates.iter().find(|&&token| token as usize >= vocab) {
                return Err(format!(
                    "Qwen4 MTP takeover candidate {token} is outside the {vocab}-token vocabulary"
                ));
            }
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP takeover position mismatch: target={target_position}, mtp={mtp_position}, position={position}"
                ));
            }
        }
        let trace =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        let interleaved = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL")
            .is_ok_and(|value| value == "1");
        let mut sampled = self.sampled.take();
        let window = if interleaved {
            self.takeover_interleaved(
                gpu,
                target,
                position,
                seed,
                candidates,
                eos,
                trace,
                sampled.as_mut(),
            )
        } else {
            self.takeover_batched(
                gpu,
                target,
                position,
                seed,
                candidates,
                eos,
                trace,
                sampled.as_mut(),
            )
        };
        self.sampled = sampled;
        window
    }

    /// With the request armed, ask the pool for a candidate run over the
    /// authoritative history (prompt plus `emitted`, which holds the seed);
    /// a hit runs one takeover window, a miss (`None`) leaves the window to
    /// native MTP. Takeover results never feed the native agreement table.
    #[allow(clippy::too_many_arguments)]
    fn try_ngram_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        k: usize,
        eos: u32,
    ) -> Result<Option<MtpWindow>, String> {
        if !self.ngram_active {
            return Ok(None);
        }
        let Some(ctx) = self.ngram.as_mut() else {
            return Ok(None);
        };
        let Some(candidates) = ctx.propose(emitted, k) else {
            return Ok(None);
        };
        self.ngram_candidates.clear();
        self.ngram_candidates.extend_from_slice(candidates);
        if !self.ngram_wins(self.ngram_candidates.len(), k) {
            self.ngram_yield.0 *= NGRAM_YIELD_RECOVERY;
            self.ngram_yield.1 *= NGRAM_YIELD_RECOVERY;
            return Ok(None);
        }
        let candidates = std::mem::take(&mut self.ngram_candidates);
        let window =
            self.mtp_takeover_step(gpu, target, position, seed, emitted, &candidates, eos);
        self.ngram_candidates = candidates;
        let window = window?;
        self.ngram_yield.0 = self.ngram_yield.0 * MTP_AGREEMENT_DECAY + window.committed.len() as f32;
        self.ngram_yield.1 = self.ngram_yield.1 * MTP_AGREEMENT_DECAY + 1.0;
        self.stats.ngram_mod_windows += 1;
        self.stats.ngram_mod_drafts += window.drafts_generated;
        self.stats.ngram_mod_accepted += window.accepted;
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.observe_result(window.drafts_generated, window.accepted);
        }
        Ok(Some(window))
    }

    /// Batched takeover: one `[seed, candidates..]` verify, rollback to the
    /// consumed rows, teacher-forced head append. Two-owner transaction as in
    /// the native batched window: any failure restores target and head.
    #[allow(clippy::too_many_arguments)]
    fn takeover_batched(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        candidates: &[u32],
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
    ) -> Result<MtpWindow, String> {
        let n = candidates.len();
        let mut block = std::mem::take(&mut self.takeover_block);
        block.clear();
        block.push(seed);
        block.extend_from_slice(candidates);
        let mut timers = MtpPhaseTimers::new();
        let window_start = Instant::now();
        let mut accepted_drafts = 0usize;
        let mut snapshot = match Self::bundle(target).and_then(|bundle| {
            bundle.mtp_snapshot(gpu).map_err(|error| error.to_string())
        }) {
            Ok(ticket) => Some(ticket),
            Err(error) => {
                self.takeover_block = block;
                return Err(error);
            }
        };
        let result = (|| -> Result<MtpWindow, String> {
            // The verify's PLE rows are SSD-resident: start reading them now.
            Self::bundle(target)?.warm_ple_rows(&block);
            timers.mark(gpu, "verify");
            let bundle = Self::bundle(target)?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 MTP takeover verify scratch is not allocated".to_string())?;
            let mut target_picks = bundle.verify_block(gpu, &block, position, scratch.as_mut(), None)?;
            if target_picks.len() < block.len() {
                return Err(format!(
                    "Qwen4 MTP takeover verifier returned {} rows for {n} candidates",
                    target_picks.len()
                ));
            }
            // The request's shared acceptance: each candidate row's `p` is
            // penalized with the window history plus the candidates before it,
            // and a sampled candidate's `q` is the point mass on it.
            let acceptance = match sampled.as_deref_mut() {
                Some(s) if !s.draws_drafts() => {
                    s.accept_naive(gpu, bundle, candidates, eos, &mut target_picks)?
                }
                Some(s) => s.accept_leviathan(gpu, bundle, candidates, true, eos)?,
                None => accept_native_greedy(candidates, &target_picks, Some(eos))?,
            };
            accepted_drafts = acceptance.accepted;
            let target_accept_len = target_commit_accept_len(&acceptance);
            let consumed = target_accept_len + 1;
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 MTP takeover target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 MTP takeover target snapshot disappeared")?;
            if consumed < block.len() && bundle.state.row_capture_rows() >= block.len() {
                timers.mark(gpu, "target_rollback");
                bundle
                    .rollback_verify_rows_retain(gpu, target_snapshot, consumed, &block)
                    .map_err(|error| error.to_string())?;
            } else if consumed < block.len() {
                timers.mark(gpu, "target_replay");
                bundle
                    .restore_retain(gpu, target_snapshot)
                    .map_err(|error| error.to_string())?;
                bundle
                    .spec_forward_rows(gpu, &block[..consumed], true)
                    .map_err(|error| error.to_string())?;
            }
            // The kept rows' captured hiddens are spec hidden rows
            // `0..consumed` (the verify's, or the replay's).
            timers.mark(gpu, "head_fill");
            let fill = self
                .append_scratch
                .as_mut()
                .filter(|fill| consumed >= 2 && consumed <= fill.rows() && mtp_batched_fill_enabled());
            match fill {
                Some(fill) => bundle
                    .mtp_append_rows(gpu, fill, &block[..consumed], 0, position)
                    .map_err(|error| error.to_string())?,
                None => {
                    let row_hidden = self
                        .row_hidden
                        .as_ref()
                        .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
                    for (row, &token) in block[..consumed].iter().enumerate() {
                        bundle
                            .copy_spec_hidden_row_to(gpu, row, row_hidden)
                            .map_err(|error| error.to_string())?;
                        bundle
                            .mtp_append_token(gpu, token, Some(row_hidden), position + row)
                            .map_err(|error| error.to_string())?;
                    }
                }
            }
            timers.mark(gpu, "commit");
            let committed_end = position + consumed;
            let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
            if bundle.state.position != committed_end || mtp_end != committed_end {
                return Err(format!(
                    "Qwen4 MTP takeover ended at target={} mtp={mtp_end}, expected {committed_end}",
                    bundle.state.position
                ));
            }
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())?;
            bundle
                .copy_spec_hidden_row_to(gpu, target_accept_len, pending_hidden)
                .map_err(|error| error.to_string())?;
            if trace {
                eprintln!(
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"ngram\",\"verify_route\":\"batched\",\"position\":{position},\"k\":{n},\"accepted\":{},\"rows\":[{}],\"committed\":{:?},\"baseline\":true}}",
                    acceptance.accepted,
                    trace_rows(candidates, &target_picks),
                    acceptance.committed
                );
            }
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            bundle
                .validate_commit(target_snapshot)
                .map_err(|error| error.to_string())?;
            bundle
                .mtp_validate_commit(mtp_ticket)
                .map_err(|error| error.to_string())?;
            bundle.commit_validated(target_snapshot);
            bundle.mtp_commit_validated(mtp_ticket);
            target_scratch.target_snapshot = None;
            snapshot = None;
            Ok(MtpWindow {
                committed: acceptance.committed,
                accepted: acceptance.accepted,
                drafts_generated: n,
            })
        })();
        self.takeover_block = block;
        if timers.enabled() {
            let fields = format!(
                "\"source\":\"ngram\",\"verify_route\":\"batched\",\"position\":{position},\"k\":{n},\"accepted\":{accepted_drafts},\"window_us\":{:.1},\"t_end\":{}",
                window_start.elapsed().as_secs_f64() * 1e6,
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
            if let Err(rollback) = self.rollback_failed_window(gpu, target, snapshot.take()) {
                return Err(format!("{error}; rollback failed: {rollback}"));
            }
        }
        result
    }

    /// Restore both owners after a failed batched window: the target from
    /// the verify scratch's active ticket, the head from `mtp_ticket`. Errors
    /// name every restore that failed.
    fn rollback_failed_window(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        mtp_ticket: Option<MtpGpuStateSnapshot>,
    ) -> Result<(), String> {
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
        if let Some(ticket) = mtp_ticket {
            if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                bundle
                    .mtp_restore(gpu, ticket)
                    .map_err(|restore| restore.to_string())
            }) {
                rollback_errors.push(format!("MTP rollback failed: {rollback}"));
            }
        }
        if rollback_errors.is_empty() {
            Ok(())
        } else {
            Err(rollback_errors.join("; "))
        }
    }

    /// Interleaved takeover (`HIPFIRE_MTP_INCREMENTAL=1`): the native
    /// one-row target route with the candidates in place of head drafts.
    /// Each row commits as it is produced and the head appends it
    /// teacher-forced; the window stops at the first rejection.
    #[allow(clippy::too_many_arguments)]
    fn takeover_interleaved(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        candidates: &[u32],
        eos: u32,
        trace: bool,
        mut sampled: Option<&mut SampledVerify>,
    ) -> Result<MtpWindow, String> {
        let n = candidates.len();
        let mut timers = MtpPhaseTimers::new();
        let mut committed: Vec<u32> = Vec::with_capacity(n + 1);
        let mut picks: Vec<u32> = Vec::with_capacity(n + 1);
        // Row `r`'s penalty history is the window base plus the candidates
        // accepted before it.
        if let Some(s) = sampled.as_deref_mut() {
            s.history.rewind_drafts();
        }
        let mut token = seed;
        let mut accepted = 0usize;
        let mut row = 0usize;
        loop {
            let token_position = position + row;
            timers.mark(gpu, "target_row");
            let mut pick = Self::bundle(target)?
                .spec_capture_token(gpu, token)
                .map_err(|error| error.to_string())?;
            match sampled.as_deref_mut() {
                Some(s) if !s.draws_drafts() => {
                    pick = s.naive_draw(gpu, Self::bundle(target)?, 0, row)?
                }
                Some(s) => s.load_target(gpu, Self::bundle(target)?, 0, row)?,
                None => {}
            }
            picks.push(pick);
            timers.mark(gpu, "head_fill");
            let row_hidden = self
                .row_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 MTP row hidden is not allocated".to_string())?;
            {
                let bundle = Self::bundle(target)?;
                bundle
                    .copy_spec_hidden_row_to(gpu, 0, row_hidden)
                    .map_err(|error| error.to_string())?;
                bundle
                    .mtp_append_token(gpu, token, Some(row_hidden), token_position)
                    .map_err(|error| error.to_string())?;
            }
            if row == n {
                committed.push(match sampled.as_deref_mut() {
                    Some(s) if s.draws_drafts() => s.target.sample(s.rng.next_f32()),
                    _ => pick,
                });
                break;
            }
            let candidate = candidates[row];
            let verdict = match sampled.as_deref_mut() {
                Some(s) if s.draws_drafts() => {
                    if s.points.is_empty() {
                        s.points.push(SparseDist::default());
                    }
                    s.points[0].set_point_mass(candidate);
                    verify_sampled_draft(&s.target, &s.points[0], candidate, &mut s.rng)
                }
                _ if candidate == pick => DraftVerdict::Accept,
                _ => DraftVerdict::Reject(pick),
            };
            match verdict {
                DraftVerdict::Accept => {
                    accepted += 1;
                    committed.push(candidate);
                    if let Some(s) = sampled.as_deref_mut() {
                        s.history.push_draft(candidate);
                    }
                    if candidate == eos {
                        break;
                    }
                    token = candidate;
                    row += 1;
                }
                DraftVerdict::Reject(replacement) => {
                    committed.push(replacement);
                    break;
                }
            }
        }
        // The next window's row-0 hidden is the last consumed row's.
        {
            let row_hidden = self.row_hidden()?;
            let pending = self.pending_hidden()?;
            gpu.copy_d2d(row_hidden, pending, pending.byte_size())
                .map_err(|error| format!("Qwen4 MTP takeover pending hidden copy: {error}"))?;
        }
        let committed_end = position + row + 1;
        let bundle = Self::bundle(target)?;
        let mtp_end = bundle.mtp_position().map_err(|error| error.to_string())?;
        if bundle.state.position != committed_end || mtp_end != committed_end {
            return Err(format!(
                "Qwen4 MTP interleaved takeover ended at target={} mtp={mtp_end}, expected {committed_end}",
                bundle.state.position
            ));
        }
        if trace {
            eprintln!(
                "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"ngram\",\"verify_route\":\"interleaved\",\"position\":{position},\"k\":{n},\"accepted\":{accepted},\"rows\":[{}],\"committed\":{committed:?}}}",
                trace_rows(candidates, &picks)
            );
        }
        if timers.enabled() {
            let fields = format!(
                "\"source\":\"ngram\",\"verify_route\":\"interleaved\",\"position\":{position},\"k\":{n},\"accepted\":{accepted},\"t_end\":{}",
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        Ok(MtpWindow {
            committed,
            accepted,
            drafts_generated: n,
        })
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
        // The request's n-gram history starts from the full canonical prompt
        // (cache hit or miss), once per request: a realignment re-prefill
        // keeps the history it already holds.
        if self.ngram_active {
            if let Some(ctx) = self.ngram.as_mut().filter(|ctx| !ctx.is_begun()) {
                ctx.begin_request(prompt_tokens);
            }
        }
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
        // as AR draws its first token. Its penalty history is the full prompt
        // (cold and cache-hit alike); `fill_tokens` is only the uncached tail.
        match self.sampled.as_mut() {
            Some(s) => {
                if !s.prompt_set {
                    s.history.set_prompt(prompt_tokens);
                    s.prompt_set = true;
                }
                if s.draws_drafts() {
                    s.load_target(gpu, Self::bundle(target)?, 0, 0)?;
                    Ok(s.target.sample(s.rng.next_f32()))
                } else {
                    s.naive_draw(gpu, Self::bundle(target)?, 0, 0)
                }
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
        emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        let eos = self.end_of_turn.unwrap_or(eos);
        self.require_supported_request()?;
        if k > self.proposal_capacity() {
            return Err(format!(
                "Qwen4 native MTP draft budget {k} exceeds the proposal capacity {}",
                self.proposal_capacity()
            ));
        }
        // The window's penalty history is rebuilt from the tokens emitted so
        // far (the pending seed included); rejected drafts of earlier windows
        // never enter it.
        if let Some(s) = self.sampled.as_mut() {
            s.begin_window(emitted);
        }

        self.ensure_resources(gpu, target)?;
        if let Some(window) = self.try_ngram_window(gpu, target, position, seed, emitted, k, eos)? {
            return Ok(window);
        }
        // A pool miss drafts natively, within the head's own K.
        let k = k.min(self.max_k);
        self.stats.mtp_windows += 1;
        let trace =
            hipfire_config::developer_var("HIPFIRE_MTP_TRACE").is_ok_and(|value| value == "1");
        // Route: a (k+1)-row batched verify costs about 2.5 single-row
        // forwards, so it pays only while drafts keep being accepted; the
        // interleaved verify costs one forward per emitted token and wastes no
        // draft. `HIPFIRE_MTP_INCREMENTAL=0|1` forces a route.
        let depth = match hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref() {
            Ok("0") => k,
            Ok("1") => 0,
            _ => self.batched_depth(k),
        };
        if depth == 0 {
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
        let window_start = Instant::now();
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
                    input = s.sample_draft(gpu, Self::bundle(target)?, index, index)?;
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
                // Only a kept draft enters the penalty history: a pruned
                // proposal broke out above.
                if let Some(s) = sampled.as_mut() {
                    s.history.push_draft(input);
                }
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
                Some(s) if !s.draws_drafts() => {
                    s.accept_naive(gpu, picks, &drafts, eos, &mut target_picks)?
                }
                Some(s) => s.accept_leviathan(gpu, picks, &drafts, false, eos)?,
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
                    "QWEN4_MTP_TRACE {{\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"batched\",\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"rows\":[{rows}],\"committed\":{:?},\"baseline\":true}}",
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
        }
        if timers.enabled() {
            let fields = format!(
                "\"position\":{position},\"k\":{k},\"accepted\":{accepted_drafts},\"window_us\":{:.1},\"t_end\":{}",
                window_start.elapsed().as_secs_f64() * 1e6,
                unix_micros()
            );
            timers.finish(gpu, "QWEN4_MTP_PHASE", &fields);
        }
        if let Err(error) = &result {
            if let Err(rollback) = self.rollback_failed_window(gpu, target, snapshot.take()) {
                return Err(format!("{error}; rollback failed: {rollback}"));
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

    /// An armed n-gram pool may offer up to its `n_max` candidates; native
    /// windows still draft at most K.
    fn proposal_capacity(&self) -> usize {
        match (self.ngram_active, self.ngram_config) {
            (true, Some(config)) => self.max_k.max(config.n_max),
            _ => self.max_k,
        }
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
        let sampled = self.sampled_enabled && cfg.temp.is_finite() && cfg.temp > 1.0e-6;
        self.sampled = match (sampled, &self.sampled_mode) {
            (true, Ok(mode)) => Some(SampledVerify::new(cfg, self.max_k, *mode)),
            _ => None,
        };
        // N-gram takeovers: greedy and sampled, thinking on or off. The pool
        // allocation is reused; its history restarts with this request.
        self.stats = MtpRequestStats::default();
        self.ngram_active = false;
        self.ngram_yield = (0.0, 0.0);
        if let Some(ctx) = self.ngram.as_mut() {
            ctx.reset_request();
        }
        if let (true, Some(config)) = (cfg.allow_ngram_modifier, self.ngram_config) {
            if self.ngram.as_ref().is_none_or(|ctx| ctx.config() != &config) {
                self.ngram = MtpNgramContext::new(config).ok();
            }
            self.ngram_active = self.ngram.is_some();
        }
        self.stats.mtp_ngram = self.ngram_active;
    }

    fn supports_temp_verify(&self) -> bool {
        self.sampled_enabled
    }

    fn request_stats(&self) -> MtpRequestStats {
        let mut stats = self.stats;
        stats.ngram_mod_accept_rate = if stats.ngram_mod_drafts > 0 {
            let rate = stats.ngram_mod_accepted as f64 / stats.ngram_mod_drafts as f64;
            (rate * 1000.0).round() / 1000.0
        } else {
            0.0
        };
        stats
    }
}

/// Build the generic runtime adapter around the native Qwen4 GPU MTP core.
/// `end_of_turn` is the tokenizer's `<|im_end|>` id, when it has one;
/// `ngram` is the n-gram pool configuration requests may arm.
pub fn build_qwen4_mtp_speculator(
    max_k: usize,
    ctx_capacity: usize,
    end_of_turn: Option<u32>,
    ngram: Option<NgramModConfig>,
) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(
        Qwen4MtpDrafter::new(max_k, ctx_capacity, end_of_turn).with_ngram(ngram),
    ))
}

/// The n-gram pool configuration for a `(n_match, n_min, n_max)` triple
/// (`hipfire_config::ngram_mod_triple_for_arch`). A takeover verifies
/// `n + 1` rows, so `n_max` is capped one below the spec verify capacity
/// (and `n_min` with it); a triple the pool cannot run is an error, never a
/// silently different policy.
pub fn qwen4_ngram_mod_config(
    (n_match, n_min, n_max): (usize, usize, usize),
) -> Result<NgramModConfig, String> {
    if n_match == 0 || n_max == 0 || n_max > 64 || n_min > n_max {
        return Err(format!(
            "invalid n-gram triple n_match={n_match} n_min={n_min} n_max={n_max} (need 1 <= n_match, n_min <= n_max <= 64)"
        ));
    }
    let n_max = n_max.min(crate::gpu_forward::QWEN4_SPEC_VERIFY_ROWS - 1);
    Ok(NgramModConfig {
        n_match,
        n_min: n_min.min(n_max),
        n_max,
        ..NgramModConfig::default()
    })
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

    #[test]
    fn sampled_verify_carries_penalties_into_policy_and_history() {
        let cfg = SpecRequestConfig {
            temp: 0.7,
            repeat_penalty: 1.15,
            repeat_window: 64,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            ..SpecRequestConfig::default()
        };
        assert_eq!(cfg.penalty_window(), 64);
        for mode in [SampledMode::Leviathan, SampledMode::Naive] {
            let s = SampledVerify::new(cfg, 4, mode);
            assert_eq!(s.policy.repeat_penalty, 1.15);
            assert_eq!(s.policy.repeat_window, 64);
            assert_eq!(s.policy.presence_penalty, 0.5);
            assert_eq!(s.policy.frequency_penalty, 0.25);
            assert_eq!(s.history.window(), 64);
            assert_eq!(s.draws_drafts(), mode == SampledMode::Leviathan);
        }
    }

    #[test]
    fn sampled_verify_neutral_request_has_no_history_window() {
        let neutral = SpecRequestConfig {
            temp: 0.7,
            ..SpecRequestConfig::default()
        };
        assert_eq!(neutral.penalty_window(), 0);
        let s = SampledVerify::new(neutral, 4, SampledMode::Leviathan);
        assert_eq!(s.history.window(), 0);
        assert_eq!(s.policy.repeat_penalty, 1.0);
        assert_eq!(s.drafts.len(), 4);
        // A window with no penalty never buffers history.
        let mut s = s;
        s.history.set_prompt(&[1, 2, 3]);
        assert_eq!(s.begin_window(&[4, 5]), 0);
        s.history.push_draft(6);
        assert!(s.history.row(1).is_empty());

        // A repeat window with no active penalty is also window 0.
        let inert = SpecRequestConfig {
            temp: 0.7,
            repeat_window: 64,
            ..SpecRequestConfig::default()
        };
        assert_eq!(
            SampledVerify::new(inert, 4, SampledMode::Leviathan)
                .history
                .window(),
            0
        );
    }

    #[test]
    fn sampled_verify_history_rows_follow_prompt_emitted_and_drafts() {
        let cfg = SpecRequestConfig {
            temp: 0.7,
            repeat_penalty: 1.2,
            repeat_window: 4,
            ..SpecRequestConfig::default()
        };
        let mut s = SampledVerify::new(cfg, 4, SampledMode::Naive);
        s.history.set_prompt(&[1, 2, 3]);
        assert_eq!(s.history.row(0), &[1, 2, 3]);
        // `emitted` already holds the pending seed.
        s.begin_window(&[7]);
        assert_eq!(s.history.row(0), &[1, 2, 3, 7]);
        s.history.push_draft(8);
        s.history.push_draft(9);
        assert_eq!(s.history.row(2), &[3, 7, 8, 9]);
        s.history.rewind_drafts();
        assert_eq!(s.history.row(0), &[1, 2, 3, 7]);
    }

    #[test]
    fn ngram_source_runs_only_while_its_yield_matches_native() {
        let mut drafter = Qwen4MtpDrafter::new(3, 4096, None);
        assert!(drafter.ngram_wins(3, 3), "a fresh request tries the pool");
        // A pool emitting 2 tokens per 3-candidate window (the Halo smoke's
        // code_edit_rewrite_copy pattern) loses to native MTP agreeing at 0.95.
        drafter.agreement = [(0.95, 1.0); MTP_MAX_DEPTH];
        drafter.ngram_yield = (16.0, 8.0);
        assert!(!drafter.ngram_wins(3, 3));
        // A verbatim copy (4 tokens per window) beats it.
        drafter.ngram_yield = (32.0, 8.0);
        assert!(drafter.ngram_wins(3, 3));
    }

    #[test]
    fn ngram_config_caps_n_max_below_the_verify_rows() {
        let config = qwen4_ngram_mod_config((5, 3, 3)).expect("arch-16 default");
        assert_eq!((config.n_match, config.n_min, config.n_max), (5, 3, 3));
        assert_eq!(config.capacity, NgramModConfig::default().capacity);
        // [seed, 64 candidates] would be 65 verify rows: one too many.
        let config = qwen4_ngram_mod_config((24, 48, 64)).expect("inherited triple");
        assert_eq!((config.n_match, config.n_min, config.n_max), (24, 48, 63));
        assert!(qwen4_ngram_mod_config((0, 1, 3)).is_err());
        assert!(qwen4_ngram_mod_config((5, 4, 3)).is_err());
        assert!(qwen4_ngram_mod_config((5, 3, 0)).is_err());
        assert!(qwen4_ngram_mod_config((5, 3, 65)).is_err());
    }
}
