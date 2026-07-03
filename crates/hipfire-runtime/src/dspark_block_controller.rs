//! Pure (no-GPU) draft-block-size + spec↔AR controller for DSpark spec-decode.
//!
//! Scores each choice by throughput = committed tokens ÷ window wall-time:
//!   • spec block N (≥1): τ(N) / (t_ar + N·Δt), where τ(N)=1+Σ_{k=1..N} S(k) is the
//!     expected committed tokens/window (S(k)=P(accept_len≥k), the acceptance-depth
//!     survival), Δt is the marginal per-position window cost and t_ar the n_verify=1
//!     intercept — a block-N window verifies N+1 positions so it costs t_ar + N·Δt.
//!   • AR (block 0): 1 committed token per plain target forward (no drafter), at the
//!     measured AR-window cost — the spec→AR fallback for when speculation isn't repaid.
//! Δt and t_ar come from the live window-cost calibration: the FULL per-window wall-time
//! (draft+heads+verify) measured vs block, NOT verify alone (verify-only omits the fixed
//! drafter/launch overhead and over-charges large blocks).
//!
//! This auto-adapts across architectures without per-arch tuning:
//!   • DeepSeek4 (expensive MoE verify, spec ≫ AR): Δt large AND survival saturates
//!     early (S(3+)≈0) → the best spec block is small (2) and AR loses → stays on spec.
//!   • Qwen3 (cheap verify): a spec window barely out-commits a plain AR forward but
//!     costs the drafter overhead, so at temp>0 AR beats every spec block → falls to AR.
//!
//! Stability: the choice is held with SWITCH-HYSTERESIS (a candidate must beat the
//! CURRENT choice's throughput by SWITCH_MARGIN to displace it), because spec block
//! scores can sit within a few % of each other and the AR/spec estimates are noisy — a
//! bare argmax would chatter and the churn (an AR window clears the drafter context)
//! degrades acceptance. See `observe`.
//!
//! Ramp phase (breaks the calibration deadlock): after WARMUP, each request sweeps
//! min_block→max_block (RAMP_HOLD windows/step, starting at an AR window when min_block=0)
//! to record window timing at ≥2 distinct n_verify (calibration) and seed the survival
//! counts at every depth. After the ramp the hysteretic decision runs.

/// Cost-model draft-block controller for the DSpark drafter.
pub(crate) struct BlockController {
    block: usize,
    default_block: usize,
    min_block: usize,
    max_block: usize,
    /// Deepest block ever drafted; upper bound of the argmax search. Grows during
    /// the ramp phase to max_block, then stays fixed — S(k) above it is unobservable
    /// (but after the ramp all depths have been tried).
    max_tried: usize,
    /// Per-depth acceptance-survival COUNTS (request-specific; reset each request).
    /// `s_hit[k]` = #windows with accept_len ≥ k; `s_tot[k]` = #windows that drafted
    /// ≥ k (so depth k was observable). The survival estimate is S(k)=s_hit[k]/s_tot[k],
    /// k ∈ 1..=8. Counts grow only for depths actually drafted, so once the block
    /// settles below k, s_tot[k] stops growing and S(k) retains its last value — the
    /// cap-trap fix (the old decaying P(accept==k) histogram forgot the ramp's
    /// deep-survival samples, so a low-settled block could never re-discover depth).
    /// Counts converge fast (a deterministic full-accept depth reads 2/2=1 after just
    /// the ramp), where a slow EMA would still read ≈0.1.
    s_hit: [f32; 9],
    s_tot: [f32; 9],
    windows_seen: u32,
    // ── live window-cost calibration (hardware cost; stable across requests) ──
    /// Total timing samples observed; gated to ≥TIMING_WARMUP before calibrating.
    timing_samples: u32,
    /// Per-n EMA of full-window wall-time in ms (draft+heads+verify), indexed by
    /// n_verify (= 1 + drafted block), slots 2..9.
    t_window_by_n: [f32; 10],
    /// True once the window-cost curve has been fit; preserved across reset().
    calibrated: bool,
    /// Marginal per-position window cost (ms) = slope of the window-cost curve,
    /// clamped to ≥0 (a flat curve lets the block climb to the acceptance depth).
    dt: f32,
    /// Single-block window cost (ms) = intercept of the window-cost curve.
    t_ar: f32,
    /// True once dt/t_ar are usable (live-calibrated or test-seeded). Until then
    /// the cost decision is disabled and the block stays at default_block.
    cost_ready: bool,
}

/// Skip the first few windows so the block doesn't react to bootstrap noise.
const WARMUP_WINDOWS: u32 = 6;
/// Minimum timing samples before attempting window-curve calibration.
const TIMING_WARMUP: u32 = 16;
/// Windows each ramp block is held so the histogram can collect survival samples
/// at that depth; after ramp_end the argmax takes over.
const RAMP_HOLD: u32 = 2;
/// Switch-hysteresis margin: the controller only MOVES off its current block/AR choice
/// when a candidate beats the CURRENT choice's throughput by more than this. The
/// current choice is sticky, so the controller commits to one decision per genre
/// instead of chattering between close-scoring candidates (DS4 blocks 1/2/3 are within
/// ~5%; the AR/spec boundary is noisy from the 2-sample ramp AR-cost estimate and an
/// AR window's context-clear). This is what keeps DS4 pinned to its optimum and qwen3
/// pinned to AR once each is reached, instead of flip-flopping on estimate noise.
const SWITCH_MARGIN: f32 = 0.10;

impl BlockController {
    pub(crate) fn new(
        default_block: usize,
        min_block: usize,
        max_block: usize,
        p_star: f32,
    ) -> Self {
        let default_block = default_block.clamp(min_block, max_block);
        Self {
            block: default_block,
            default_block,
            min_block,
            max_block,
            max_tried: default_block,
            s_hit: [0.0f32; 9],
            s_tot: [0.0f32; 9],
            windows_seen: 0,
            timing_samples: 0,
            t_window_by_n: [0.0f32; 10],
            calibrated: false,
            // Dormant cost prior: only the dt/t_ar RATIO drives the argmax, and it
            // stays disabled (cost_ready=false) until live window timing refines
            // these into real milliseconds. Seeded from the caller's p* prior so the
            // ratio is sane if ever consulted before calibration (assume ~100ms AR).
            t_ar: 100.0,
            dt: p_star * 100.0,
            cost_ready: false,
        }
    }

    pub(crate) fn block(&self) -> usize {
        self.block
    }

    pub(crate) fn reset(&mut self) {
        // Reset only request-specific state. Calibration (dt, t_ar, cost_ready,
        // timing_samples, t_window_by_n, calibrated) is a thermal-invariant hardware
        // cost — calibrate once, reuse across requests.
        self.block = self.default_block;
        self.max_tried = self.default_block;
        self.windows_seen = 0;
        self.s_hit = [0.0f32; 9];
        self.s_tot = [0.0f32; 9];
    }

    /// Observe one window's full wall-time (draft+heads+verify). Accumulates per-n
    /// EMAs and fits the line `t_window(n) ≈ t_ar + (n−1)·Δt` once TIMING_WARMUP
    /// samples span ≥2 distinct n. The slope is clamped to ≥0: a flat or slightly
    /// negative measured slope means the per-window wall time barely grows with the
    /// block (a cheap-verify arch whose fixed drafter/launch overhead dominates), in
    /// which case the argmax should be free to climb toward the acceptance depth
    /// rather than be blocked by phantom marginal cost. Only a clearly-too-steep fit
    /// (Δt > t_ar/2, i.e. a thermal spike) is rejected. Preserved across reset().
    pub(crate) fn observe_timing(&mut self, t_window_ms: f32, n_verify: usize) {
        // Record n_verify=1 too — that's the AR-window cost (block 0, no draft), which
        // the argmax needs as the spec→AR fallback candidate. The linear SPEC-line fit
        // below still spans only (2..10): AR is off that line (no drafter overhead), so
        // including it would flatten the slope wrongly.
        if (1..10).contains(&n_verify) && t_window_ms > 0.0 {
            let slot = &mut self.t_window_by_n[n_verify];
            *slot = if *slot == 0.0 {
                t_window_ms
            } else {
                0.7 * *slot + 0.3 * t_window_ms
            };
        }
        self.timing_samples = self.timing_samples.saturating_add(1);
        if self.calibrated || self.timing_samples < TIMING_WARMUP {
            return;
        }
        let lo = (2..10).find(|&n| self.t_window_by_n[n] > 0.0);
        let hi = (2..10).rev().find(|&n| self.t_window_by_n[n] > 0.0);
        if let (Some(n_lo), Some(n_hi)) = (lo, hi) {
            // If the controller never visits ≥2 distinct n_verify (block pinned),
            // n_hi > n_lo never holds and cost_ready stays false — the block then
            // safely stays at default_block for the process lifetime.
            if n_hi > n_lo {
                let slope =
                    (self.t_window_by_n[n_hi] - self.t_window_by_n[n_lo]) / (n_hi - n_lo) as f32;
                // Clamp negative/flat slope to 0 → "cost is flat in block", so the
                // argmax climbs to the survival-supported depth (a shallow-acceptance
                // arch still settles low because τ saturates). Anchor t_ar at the
                // cheapest measured point so the clamp keeps a sane intercept.
                let dt = slope.max(0.0);
                let t_ar = self.t_window_by_n[n_lo] - dt * (n_lo as f32 - 1.0);
                if t_ar > 0.0 && dt / t_ar <= 0.5 {
                    self.dt = dt;
                    self.t_ar = t_ar;
                    self.cost_ready = true;
                    self.calibrated = true;
                    eprintln!(
                        "[dspark] cost calibrated: dt={:.2}ms t_ar={:.1}ms (ratio={:.3}, n{}={:.1}ms n{}={:.1}ms)",
                        dt, t_ar, dt / t_ar, n_lo, self.t_window_by_n[n_lo], n_hi, self.t_window_by_n[n_hi]
                    );
                }
            }
        }
    }

    /// Observe one spec window's acceptance depth and re-decide the block (or AR) via the
    /// hysteretic cost comparison below. During the ramp phase (post-warmup, pre-ramp_end)
    /// the block sweeps min→max to seed the window-cost calibration and the survival
    /// counts; after ramp_end the switch-hysteresis decision drives the choice.
    pub(crate) fn observe(&mut self, accept_len: usize, n_proposed: usize) {
        // Accumulate survival counts ONLY for depths we actually drafted (k ≤
        // n_proposed): for those k, `accept_len ≥ k` is a real observation. Depths above
        // n_proposed are unobservable this window, so their counts don't grow — S(k)
        // retains its last value (the cap-trap fix; the old decaying P(accept==k)
        // histogram forgot the ramp's deep-survival samples so a low-settled block
        // could never re-discover that drafting deeper pays).
        let depth = n_proposed.min(8);
        for k in 1..=depth {
            self.s_tot[k] += 1.0;
            if accept_len >= k {
                self.s_hit[k] += 1.0;
            }
        }
        self.windows_seen += 1;
        if self.windows_seen < WARMUP_WINDOWS {
            return;
        }
        let ramp_end = WARMUP_WINDOWS + RAMP_HOLD * (self.max_block - self.min_block + 1) as u32;
        if self.windows_seen < ramp_end {
            // Sweep block min→max to seed calibration (≥2 distinct n_verify) AND survival
            // at every depth. With min_block=0 the sweep starts at an AR window (block 0),
            // measuring the AR-window cost so the argmax can compare spec vs AR. Without
            // the sweep the block never varies and calibration deadlocks.
            let step = (self.windows_seen - WARMUP_WINDOWS) / RAMP_HOLD;
            self.block = (self.min_block + step as usize).min(self.max_block);
            self.max_tried = self.block.max(self.max_tried);
            return;
        }
        // Post-ramp decision with switch-hysteresis. The ramp left the block at max_block,
        // so restart from the default (a good spec block) on the first post-ramp window.
        // Then, each window, two sticky comparisons keep the controller committed to one
        // choice per genre instead of chattering on estimate noise:
        //   • AR↔spec has a DEADBAND: enter AR only if it clearly (by SWITCH_MARGIN) beats
        //     the best spec block; once on AR, leave only if the best spec block clearly
        //     beats AR. (Same-threshold enter/leave would flip-flop at the boundary.)
        //   • spec block size is sticky: adopt a different spec block only if it clearly
        //     beats the CURRENT one (DS4's 1/2/3 scores are within ~5%, so a bare argmax
        //     would chatter; the deadband pins it to the true optimum).
        if !self.cost_ready {
            self.block = self.default_block; // fallback if calibration never completed
            return;
        }
        if self.windows_seen == ramp_end {
            self.block = self.default_block;
            return;
        }
        let (best_spec, best_spec_score) = self.best_spec();
        let ar_score = if self.min_block == 0 {
            self.score_of(0)
        } else {
            f32::MIN
        };
        if self.block == 0 {
            // Currently AR: return to spec only if the best spec block clearly beats AR.
            if best_spec_score > ar_score * (1.0 + SWITCH_MARGIN) {
                self.block = best_spec;
            }
        } else if ar_score > best_spec_score * (1.0 + SWITCH_MARGIN) {
            // Currently spec: fall to AR only if it clearly beats the best spec block.
            self.block = 0;
        } else if best_spec != self.block
            && best_spec_score > self.score_of(self.block) * (1.0 + SWITCH_MARGIN)
        {
            // Stay in spec, but move to a different block only on a clear win.
            self.block = best_spec;
        }
    }

    /// Throughput score (committed tokens ÷ window-ms) of one choice. Block 0 is the AR
    /// fallback: 1 committed token at the MEASURED AR-window cost t_window_by_n[1] (a
    /// single target forward, no drafter). Block n≥1 is a spec block: τ(n)=1+Σ_{k=1..n}
    /// S[k] over the fitted window t_ar + n·Δt — block n verifies n+1 positions, i.e. the
    /// point n+1 on the line whose n_verify=1 intercept is t_ar (using (n-1)·Δt would
    /// price block-1 spec at the bare intercept, i.e. as cheap as an AR window, breaking
    /// the AR-vs-spec comparison).
    fn score_of(&self, n: usize) -> f32 {
        if n == 0 {
            let ar_cost = self.t_window_by_n[1];
            return if ar_cost > 0.0 {
                1.0 / ar_cost
            } else {
                f32::MIN
            };
        }
        let mut tau = 1.0f32;
        for k in 1..=n.min(8) {
            // S(k)=P(accept_len≥k) from the counts; 0 for a never-drafted depth.
            tau += if self.s_tot[k] > 0.0 {
                self.s_hit[k] / self.s_tot[k]
            } else {
                0.0
            };
        }
        let window_ms = self.t_ar + n as f32 * self.dt;
        if window_ms > 0.0 {
            tau / window_ms
        } else {
            f32::MIN
        }
    }

    /// The highest-scoring SPEC block (≥1) over 1..=max_tried, as (block, score). AR
    /// (block 0) is compared against this separately in `observe`, with a deadband.
    fn best_spec(&self) -> (usize, f32) {
        if self.t_ar <= 0.0 || self.dt < 0.0 {
            return (self.default_block.max(1), f32::MIN);
        }
        let mut best = self.default_block.max(1);
        let mut best_score = f32::MIN;
        for n in 1..=self.max_tried.min(8) {
            let s = self.score_of(n);
            if s > best_score {
                best_score = s;
                best = n;
            }
        }
        (best, best_score)
    }

    #[cfg(test)]
    fn set_cost_for_test(&mut self, dt: f32, t_ar: f32) {
        self.dt = dt;
        self.t_ar = t_ar;
        self.cost_ready = true;
    }

    #[cfg(test)]
    fn cost_for_test(&self) -> (f32, f32, bool) {
        (self.dt, self.t_ar, self.cost_ready)
    }

    #[cfg(test)]
    fn set_ar_cost_for_test(&mut self, ar_cost_ms: f32) {
        self.t_window_by_n[1] = ar_cost_ms; // n_verify=1 = AR-window cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // DS4-like: expensive verify (dt large) + survival that SATURATES at 2
    // (accept_len ∈ {0,1,2}). τ stops growing past 2, so the argmax settles low.
    // Run well past ramp_end (ramp sweeps min→max in 2*max_block=14 post-warmup
    // windows; we run 200 total so the argmax has long settled).
    #[test]
    fn settles_low_when_survival_saturates_and_verify_expensive() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(15.0, 85.0); // DS4-like
        for i in 0..200 {
            c.observe([0, 1, 2, 2][i % 4], 5); // depth ~1.5, never > 2
        }
        assert!((1..=2).contains(&c.block()), "got {}", c.block());
    }

    // qwen3-like: cheap verify (dt small) + a drafter that accepts the whole drafted
    // block (survival deep, cap always hit). The ramp seeds survival at all depths;
    // after ramp_end the cost model rewards over-drafting and settles high.
    #[test]
    fn climbs_high_when_survival_deep_and_verify_cheap() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(4.0, 33.0); // qwen3-like
                                        // Always accept the whole drafted block, so all depths are rewarded.
        for _ in 0..200 {
            let b = c.block();
            c.observe(b.min(7), b);
        }
        assert!(
            c.block() >= 5,
            "cheap verify + deep accept should climb high, got {}",
            c.block()
        );
    }

    // Cost sensitivity: SAME (decreasing) survival, cheap vs expensive verify -> the
    // cheaper verify justifies a larger block. accept depth cycles 1..4 out of 7
    // drafted, so S saturates (S(5+)=0) → τ tops out ~4 and the marginal cost alone
    // decides how deep to go (cheap → 4, expensive → 1).
    #[test]
    fn cheaper_verify_picks_larger_block() {
        let mk = |dt: f32| {
            let mut c = BlockController::new(2, 1, 7, 0.0);
            c.set_cost_for_test(dt, 50.0);
            for i in 0..400 {
                c.observe([1, 2, 3, 4][i % 4], 7);
            }
            c.block()
        };
        let cheap = mk(2.0);
        let expensive = mk(20.0);
        assert!(
            cheap > expensive,
            "cheap {} should exceed expensive {}",
            cheap,
            expensive
        );
    }

    // Fit dt/t_ar from the window-cost curve: t_window(n) ≈ t_ar + (n-1)·Δt.
    // n=2 → 90ms, n=6 → 150ms: dt=(150-90)/4=15, t_ar=90-15=75; flips cost_ready.
    #[test]
    fn calibrates_dt_t_ar_from_window_curve() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        let (dt, t_ar, ready) = c.cost_for_test();
        assert!(ready, "window-curve fit should flip cost_ready");
        assert!((dt - 15.0).abs() < 0.5, "dt={}", dt);
        assert!((t_ar - 75.0).abs() < 1.0, "t_ar={}", t_ar);
    }

    // A too-STEEP fit (Δt > t_ar/2, ratio > 0.5 — a thermal spike) must be REJECTED:
    // cost_ready stays false (the block then safely stays at default).
    #[test]
    fn rejects_out_of_range_fit() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        // t_w[2]=100, t_w[6]=300 → dt=50, t_ar=50, ratio=1.0 > 0.5 → reject.
        for _ in 0..30 {
            c.observe_timing(100.0, 2);
            c.observe_timing(300.0, 6);
        }
        assert!(
            !c.cost_for_test().2,
            "out-of-range fit must not flip cost_ready"
        );
    }

    // Cheap-verify arch (qwen3): window cost is ~flat/slightly-decreasing in block.
    // The slope clamps to 0 (NOT rejected by an old lower ratio floor), so cost_ready
    // flips with dt=0 and the argmax is free to climb. This is the fix for the qwen3
    // "stuck at min block" regression — a flat window curve must calibrate, not fall
    // back to default. t_w[2]=80, t_w[8]=68 → slope=−2 → dt clamped 0, t_ar=80.
    #[test]
    fn flat_window_cost_calibrates_dt_zero() {
        let mut c = BlockController::new(2, 1, 7, 0.05);
        for _ in 0..30 {
            c.observe_timing(80.0, 2);
            c.observe_timing(68.0, 8);
        }
        let (dt, t_ar, ready) = c.cost_for_test();
        assert!(ready, "flat window curve must calibrate (not fall back)");
        assert!(
            dt.abs() < 1e-6,
            "flat/negative slope must clamp to 0, got dt={dt}"
        );
        assert!((t_ar - 80.0).abs() < 1.0, "t_ar={t_ar}");
    }

    // Spec→AR fallback: cheap AR window (27ms → 37 tok/s) + a drafter that IS accepted
    // but only shallowly (S(1)=0.25, S(2+)=0) so the best spec block tops out ~1.25
    // committed / 40ms = 31 tok/s < AR. The argmax must pick block 0 (AR). This is the
    // qwen3-prose-at-temp>0 case: speculation works but doesn't beat plain AR.
    #[test]
    fn falls_back_to_ar_when_spec_not_repaid() {
        let mut c = BlockController::new(2, 0, 7, 0.18);
        c.set_cost_for_test(1.0, 40.0); // cheap dense spec line
        c.set_ar_cost_for_test(27.0); // AR forward faster than any spec window here
        for i in 0..200 {
            let b = c.block();
            c.observe([0, 0, 0, 1][i % 4], b); // ~25% shallow accept, never ≥2
        }
        assert_eq!(
            c.block(),
            0,
            "cheap AR must beat weak spec, got {}",
            c.block()
        );
    }

    // The opposite: expensive AR (85ms → 12 tok/s) and a drafter that pays (S(1)=1,
    // S(2)=0.75 → τ(2)=2.75 / 100ms = 27 tok/s). Spec dominates, so even with AR
    // available (min_block=0) the argmax stays on a spec block. DS4-at-greedy case.
    #[test]
    fn keeps_spec_when_it_beats_ar() {
        let mut c = BlockController::new(2, 0, 7, 0.18);
        c.set_cost_for_test(15.0, 85.0); // expensive MoE spec line
        c.set_ar_cost_for_test(85.0); // AR forward ~ one target pass, no draft
        for i in 0..200 {
            let b = c.block().max(1);
            c.observe([1, 2, 2, 2][i % 4], b); // deep accept, spec pays
        }
        assert!(c.block() >= 1, "spec must beat AR here, got {}", c.block());
    }

    // Hysteresis: AR is only ~5% faster than the best spec block — inside AR_MARGIN
    // (10%) — so the controller must NOT flip to AR (avoids the churn that regressed
    // DS4 code, where noisy borderline AR windows kept clearing the drafter context).
    // Deterministic full-accept-at-1 (S(1)=1, S(2+)=0) so the spec score is robust:
    // best spec = τ(1)=2 / 40ms = 0.0500; AR = 1/19ms = 0.0526 (5.3% > spec, < margin).
    #[test]
    fn keeps_spec_when_ar_within_margin() {
        let mut c = BlockController::new(2, 0, 7, 0.18);
        c.set_cost_for_test(1.0, 40.0);
        c.set_ar_cost_for_test(19.0); // only ~5% faster than best spec → within margin
        for _ in 0..200 {
            let b = c.block();
            c.observe(1, b); // full-accept at depth 1: S(1)=1, S(2+)=0
        }
        assert!(
            c.block() >= 1,
            "AR within margin must not displace spec, got {}",
            c.block()
        );
    }

    // reset() restores request state (block, histogram) but PRESERVES the calibrated
    // window cost (thermal-invariant hardware ratio).
    #[test]
    fn reset_preserves_calibration() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        let cost = c.cost_for_test();
        assert!(cost.2, "should be calibrated");
        // Pollute request-specific state.
        for _ in 0..80 {
            c.observe(0, 4);
        }
        c.reset();
        assert_eq!(
            c.cost_for_test(),
            cost,
            "reset() must preserve dt/t_ar/cost_ready"
        );
        assert_eq!(c.block(), 3, "block returns to default after reset");
    }

    // n_proposed == 0 (degenerate window) must not panic, and all-reject settles the
    // block to the minimum without indexing OOB.
    #[test]
    fn zero_proposed_is_safe() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        c.set_cost_for_test(10.0, 50.0);
        for _ in 0..50 {
            c.observe(0, 0); // n_proposed=0; all-reject
        }
        assert!((1..=7).contains(&c.block()), "got {}", c.block());
    }
}
