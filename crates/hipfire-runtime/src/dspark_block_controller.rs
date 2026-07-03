//! Pure (no-GPU) draft-block-size controller for DSpark spec-decode.
//!
//! Picks the block that maximizes decode throughput via a cost-model argmax:
//!   block = argmax_N  τ(N) / (t_ar + (N-1)·Δt)
//! where τ(N) = 1 + Σ_{k=1..N} S(k) is the expected committed tokens per window
//! (S(k) = P(accept_len ≥ k), the acceptance-depth survival), Δt is the marginal
//! per-position verify cost (ms) and t_ar the single-forward (AR) cost (ms). Both
//! costs come from the live verify-curve calibration (slope = Δt, intercept = t_ar),
//! so the argmax directly maximizes committed-tokens ÷ window-wall-time = tok/s.
//!
//! This auto-adapts across architectures without per-arch tuning:
//!   • DeepSeek4 (expensive MoE verify): survival saturates early (S(3+)≈0) so τ
//!     stops growing → argmax settles at 2 despite the large Δt.
//!   • Qwen3 (cheap dense verify): survival stays deep and Δt is tiny → argmax
//!     climbs toward the drafter's true acceptance depth (≈7).
//!
//! Cap-trap exploration: accept_len is capped at the current draft block, so S(k)
//! is unobservable for k > block. When the argmax pins at `max_tried` (the optimum
//! wants to go deeper than we've ever drafted) we probe one block deeper and HOLD
//! it for PROBE_HOLD windows to collect survival samples at the new depth, then let
//! the argmax keep or revert it. This climbs 2→7 on qwen3 but stops at 2 on DS4
//! (where a probe to 3 reveals S(3)≈0 and the argmax immediately reverts).

/// Cost-model draft-block controller for the DSpark drafter.
pub(crate) struct BlockController {
    block: usize,
    default_block: usize,
    min_block: usize,
    max_block: usize,
    /// Deepest block ever drafted; upper bound of the argmax search. Grows only
    /// via cap-trap probes, never shrinks — S(k) above it is unobservable.
    max_tried: usize,
    /// EMA of the accept_len distribution: hist[i] ≈ P(accept_len == i), i ∈ 0..=8.
    hist: [f32; 9],
    windows_seen: u32,
    /// Post-warmup, post-hold windows since the last probe; reset to 0 on each probe.
    explore_timer: u32,
    /// Remaining windows in the current probe hold; 0 = not holding.
    probe_hold: u32,
    // ── live verify-cost calibration (hardware cost; stable across requests) ──
    /// Total timing samples observed; gated to ≥TIMING_WARMUP before calibrating.
    timing_samples: u32,
    /// Per-n EMA of verify wall-time in ms (indexed by n_verify, slots 2..8).
    t_verify_by_n: [f32; 8],
    /// True once the verify curve has been fit; preserved across reset().
    calibrated: bool,
    /// Marginal per-position verify cost (ms) = slope of the verify curve.
    dt: f32,
    /// Single-forward (AR) verify cost (ms) = intercept of the verify curve.
    t_ar: f32,
    /// True once dt/t_ar are usable (live-calibrated or test-seeded). Until then
    /// the argmax is disabled and the block stays at default_block.
    cost_ready: bool,
}

/// EMA rate for the accept_len histogram. Slow so the survival estimate tracks the
/// true distribution rather than per-window bursts (~1/α ≈ 20 windows to halve).
const HIST_ALPHA: f32 = 0.05;
/// Skip the first few windows so the block doesn't react to bootstrap noise.
const WARMUP_WINDOWS: u32 = 6;
/// Minimum timing samples before attempting verify-curve calibration.
const TIMING_WARMUP: u32 = 16;
/// Post-warmup, post-hold windows between cap-trap probe-up attempts.
const EXPLORE_INTERVAL: u32 = 40;
/// Windows to hold a probed (deeper) block so the histogram can collect survival
/// samples at the new depth before the argmax keeps or reverts it.
const PROBE_HOLD: u32 = 15;

impl BlockController {
    pub(crate) fn new(
        default_block: usize,
        min_block: usize,
        max_block: usize,
        p_star: f32,
    ) -> Self {
        let default_block = default_block.clamp(min_block, max_block);
        let mut hist = [0.0f32; 9];
        // Seed the histogram at the default depth so the first argmax (once cost is
        // ready) picks ≈default_block.
        hist[default_block.min(8)] = 1.0;
        Self {
            block: default_block,
            default_block,
            min_block,
            max_block,
            max_tried: default_block,
            hist,
            windows_seen: 0,
            explore_timer: 0,
            probe_hold: 0,
            timing_samples: 0,
            t_verify_by_n: [0.0f32; 8],
            calibrated: false,
            // Dormant cost prior: only the dt/t_ar RATIO drives the argmax, and it
            // stays disabled (cost_ready=false) until live verify timing refines
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
        // timing_samples, t_verify_by_n, calibrated) is a thermal-invariant hardware
        // cost — calibrate once, reuse across requests.
        self.block = self.default_block;
        self.max_tried = self.default_block;
        self.windows_seen = 0;
        self.explore_timer = 0;
        self.probe_hold = 0;
        self.hist = [0.0f32; 9];
        self.hist[self.default_block.min(8)] = 1.0;
    }

    /// Observe verify timing. Accumulates per-n EMAs and fits the line
    /// `t_verify(n) ≈ t_ar + (n−1)·Δt` once TIMING_WARMUP samples have been
    /// collected across ≥2 distinct n. On a sane fit (ratio Δt/t_ar in [0.05,0.5])
    /// stores Δt and t_ar and flips `cost_ready`. Preserved across reset().
    pub(crate) fn observe_timing(&mut self, t_verify_ms: f32, n_verify: usize) {
        if (2..8).contains(&n_verify) && t_verify_ms > 0.0 {
            let slot = &mut self.t_verify_by_n[n_verify];
            *slot = if *slot == 0.0 {
                t_verify_ms
            } else {
                0.7 * *slot + 0.3 * t_verify_ms
            };
        }
        self.timing_samples = self.timing_samples.saturating_add(1);
        if self.calibrated || self.timing_samples < TIMING_WARMUP {
            return;
        }
        let lo = (2..8).find(|&n| self.t_verify_by_n[n] > 0.0);
        let hi = (2..8).rev().find(|&n| self.t_verify_by_n[n] > 0.0);
        if let (Some(n_lo), Some(n_hi)) = (lo, hi) {
            // If the controller never visits ≥2 distinct n_verify (block pinned),
            // n_hi > n_lo never holds and cost_ready stays false — the block then
            // safely stays at default_block for the process lifetime.
            if n_hi > n_lo {
                let dt =
                    (self.t_verify_by_n[n_hi] - self.t_verify_by_n[n_lo]) / (n_hi - n_lo) as f32;
                let t_ar = self.t_verify_by_n[n_lo] - dt * (n_lo as f32 - 1.0);
                if dt > 0.0 && t_ar > 0.0 {
                    let ratio = dt / t_ar;
                    if (0.05..=0.5).contains(&ratio) {
                        self.dt = dt;
                        self.t_ar = t_ar;
                        self.cost_ready = true;
                        self.calibrated = true;
                        eprintln!(
                            "[dspark] cost calibrated: dt={:.2}ms t_ar={:.1}ms (ratio={:.3}, n{}={:.1}ms n{}={:.1}ms)",
                            dt, t_ar, ratio, n_lo, self.t_verify_by_n[n_lo], n_hi, self.t_verify_by_n[n_hi]
                        );
                    }
                }
            }
        }
    }

    /// Observe one spec window's acceptance depth and re-decide the block via the
    /// cost-model argmax (+ cap-trap probing). `_n_proposed` is unused; the block is
    /// chosen from the survival histogram and the calibrated verify cost.
    pub(crate) fn observe(&mut self, accept_len: usize, _n_proposed: usize) {
        let a = accept_len.min(8);
        for slot in self.hist.iter_mut() {
            *slot *= 1.0 - HIST_ALPHA; // decay all buckets
        }
        self.hist[a] += HIST_ALPHA; // bump the observed depth
        self.windows_seen += 1;
        if self.windows_seen < WARMUP_WINDOWS || !self.cost_ready {
            return;
        }
        // Hold a freshly probed (deeper) block so the histogram can measure survival
        // at that depth before the argmax judges it.
        if self.probe_hold > 0 {
            self.probe_hold -= 1;
            return;
        }
        let n_star = self.argmax_block();
        self.explore_timer += 1;
        // Cap-trap probe: the argmax wants the deepest depth we've drafted, but S(k)
        // beyond it is unobservable. Draft one deeper and hold to reveal it.
        if n_star == self.max_tried
            && self.max_tried < self.max_block
            && self.explore_timer >= EXPLORE_INTERVAL
        {
            self.max_tried += 1;
            self.block = self.max_tried;
            self.explore_timer = 0;
            self.probe_hold = PROBE_HOLD;
            return;
        }
        self.block = n_star;
    }

    /// argmax over N ∈ [1, max_tried] of τ(N)/(t_ar + (N-1)·Δt), clamped to
    /// [min_block, max_block]. τ(N) = 1 + Σ_{k=1..N} S(k), S(k)=P(accept_len≥k).
    fn argmax_block(&self) -> usize {
        let total: f32 = self.hist.iter().sum();
        if total <= 0.0 || self.t_ar <= 0.0 || self.dt < 0.0 {
            return self.default_block;
        }
        let mut tau = 1.0f32;
        let mut best_n = 1usize;
        let mut best_score = f32::MIN;
        for n in 1..=self.max_tried.min(8) {
            // S(n) = P(accept_len ≥ n) = (Σ_{j≥n} hist[j]) / total.
            let survival: f32 = self.hist[n..].iter().sum::<f32>() / total;
            tau += survival;
            let window_ms = self.t_ar + (n as f32 - 1.0) * self.dt; // > 0 by the guard above
            let score = tau / window_ms;
            if score > best_score {
                best_score = score;
                best_n = n;
            }
        }
        best_n.clamp(self.min_block, self.max_block)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // DS4-like: expensive verify (dt large) + survival that SATURATES at 2
    // (accept_len ∈ {0,1,2}). τ stops growing past 2, so the argmax settles low and
    // a probe to 3 (S(3)=0) is immediately reverted.
    #[test]
    fn settles_low_when_survival_saturates_and_verify_expensive() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(15.0, 85.0); // DS4-like
        for i in 0..600 {
            c.observe([0, 1, 2, 2][i % 4], 5); // depth ~1.5, never > 2
        }
        assert!((1..=2).contains(&c.block()), "got {}", c.block());
    }

    // qwen3-like: cheap verify (dt small) + a drafter that accepts the whole drafted
    // block (survival deep, cap always hit). The cost model rewards over-drafting and
    // cap-trap exploration must climb the block upward.
    #[test]
    fn climbs_high_when_survival_deep_and_verify_cheap() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(4.0, 33.0); // qwen3-like
                                        // Always accept the whole drafted block, so the cap is always hit -> climb.
        for _ in 0..2000 {
            let b = c.block();
            c.observe(b.min(7), b);
        }
        assert!(
            c.block() >= 5,
            "cheap verify + deep accept should climb high, got {}",
            c.block()
        );
    }

    // Cost sensitivity: SAME (deep) survival, cheap vs expensive verify -> the
    // cheaper verify justifies a larger block. Expensive verify reverts the first
    // probe (half-populated deeper bucket doesn't pay for its cost).
    #[test]
    fn cheaper_verify_picks_larger_block() {
        let mk = |dt: f32| {
            let mut c = BlockController::new(2, 1, 7, 0.0);
            c.set_cost_for_test(dt, 50.0);
            for _ in 0..2000 {
                let b = c.block();
                c.observe(b.min(7), b);
            }
            c.block()
        };
        assert!(
            mk(2.0) > mk(20.0),
            "cheap {} should exceed expensive {}",
            mk(2.0),
            mk(20.0)
        );
    }

    // Fit dt/t_ar from the verify curve: t_verify(n) ≈ t_ar + (n-1)·Δt.
    // n=2 → 90ms, n=6 → 150ms: dt=(150-90)/4=15, t_ar=90-15=75; flips cost_ready.
    #[test]
    fn calibrates_dt_t_ar_from_verify_curve() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        let (dt, t_ar, ready) = c.cost_for_test();
        assert!(ready, "verify-curve fit should flip cost_ready");
        assert!((dt - 15.0).abs() < 0.5, "dt={}", dt);
        assert!((t_ar - 75.0).abs() < 1.0, "t_ar={}", t_ar);
    }

    // A fit whose ratio Δt/t_ar is out of [0.05,0.5] must be REJECTED: cost_ready
    // stays false (the block then safely stays at default).
    #[test]
    fn rejects_out_of_range_fit() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        // t_v[2]=100, t_v[6]=300 → dt=50, t_ar=50, ratio=1.0 > 0.5 → reject.
        for _ in 0..30 {
            c.observe_timing(100.0, 2);
            c.observe_timing(300.0, 6);
        }
        assert!(
            !c.cost_for_test().2,
            "out-of-range fit must not flip cost_ready"
        );
    }

    // reset() restores request state (block, histogram) but PRESERVES the calibrated
    // verify cost (thermal-invariant hardware ratio).
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
