//! Pure (no-GPU) draft-block-size controller for DSpark spec-decode.
//! Direct target from smoothed mean accept-length: block = ceil(EMA(accept_len)),
//! plus periodic probe-up exploration to escape the block-1 cap-trap: at block=1
//! accept_len ∈ {0,1} so the EMA can never reveal that a higher block is profitable.
//! A probe bumps block by +1 every EXPLORE_INTERVAL windows, then holds for PROBE_HOLD
//! windows so accept_ema has time to measure the new depth before reverting/keeping.

/// Accept-length EMA controller for the DSpark draft block cap.
pub(crate) struct BlockController {
    block: usize,
    default_block: usize,
    min_block: usize,
    max_block: usize,
    /// EMA of accept_len (acceptance depth); direct signal for block sizing.
    accept_ema: f32,
    windows_seen: u32,
    /// Counts post-warmup, post-hold windows since the last probe. Resets to 0 after each probe.
    explore_timer: u32,
    /// Remaining windows in the current probe hold; 0 = not holding.
    probe_hold: u32,
    // ── live p* calibration (hardware cost ratio; stable across requests) ──
    /// Total timing samples observed; gated to ≥TIMING_WARMUP before calibrating.
    timing_samples: u32,
    /// Per-n EMA of verify wall-time in ms (indexed by n_verify, slots 2..8).
    t_verify_by_n: [f32; 8],
    /// True once p* has been measured from live timing; preserved across reset().
    calibrated: bool,
    /// Cost-ratio break-even; reserved: cost-ratio margin, see accept-length redesign.
    #[allow(dead_code)]
    p_star: f32,
}

// accept_len is a real-valued depth signal (not binary). Alpha is kept slow
// so the EMA tracks the true mean depth rather than per-window bursts — at
// temp>0 code, frequent zero-accept windows otherwise yank EMA below 1.0
// and flip ceil back to block=1. ~4% move per window → ~25 windows to halve.
const ACCEPT_ALPHA: f32 = 0.04;
/// Skip the first few windows so the block doesn't react to early bootstrap
/// noise before the EMA has seen enough real samples.
const WARMUP_WINDOWS: u32 = 6;
/// Minimum timing samples before attempting verify-curve calibration.
/// Gives the GPU time to warm up and collects samples at ≥2 distinct n values.
const TIMING_WARMUP: u32 = 16;
/// How many post-warmup, post-hold windows between probe-up attempts.
const EXPLORE_INTERVAL: u32 = 50;
/// How many windows to hold the probed (higher) block so accept_ema can
/// measure acceptance depth there before the settle branch reverts or keeps it.
/// Matched to the slower alpha: 1/ACCEPT_ALPHA ≈ 25 windows to halve; hold
/// ~20 so the EMA has time to reveal the true depth at the probed block.
const PROBE_HOLD: u32 = 20;

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
            // Seed so ceil(accept_ema) == default_block: e.g. default=3 → seed 2.5 → ceil 3.
            accept_ema: default_block as f32 - 0.5,
            windows_seen: 0,
            explore_timer: 0,
            probe_hold: 0,
            timing_samples: 0,
            t_verify_by_n: [0.0f32; 8],
            calibrated: false,
            p_star,
        }
    }

    pub(crate) fn block(&self) -> usize {
        self.block
    }

    pub(crate) fn set_p_star(&mut self, p_star: f32) {
        self.p_star = p_star;
    }

    pub(crate) fn reset(&mut self) {
        // Reset only request-specific state. Calibration fields (timing_samples,
        // t_verify_by_n, calibrated, p_star) are thermal-invariant hardware
        // cost ratios — calibrate once, reuse across requests.
        self.block = self.default_block;
        self.accept_ema = self.default_block as f32 - 0.5;
        self.windows_seen = 0;
        self.explore_timer = 0;
        self.probe_hold = 0;
    }

    /// Observe verify timing. Accumulates per-n EMAs and fits a line
    /// `t_verify(n) ≈ t_AR + (n−1)·Δt` once TIMING_WARMUP samples have been
    /// collected, deriving `p* = Δt / t_AR` from two distinct n points on the
    /// verify curve. Both terms come from the same kernel at the same thermal
    /// phase, so the ratio is stable under DPM scaling. Calibration is preserved
    /// across reset() calls.
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
            // Note: if the controller never visits ≥2 distinct n_verify values
            // (e.g. block pinned at a fixed cap), n_hi > n_lo never holds and
            // calibration silently stays on the 0.18 prior for the process
            // lifetime — acceptable, since the prior is a safe gfx1151 default.
            if n_hi > n_lo {
                let dt =
                    (self.t_verify_by_n[n_hi] - self.t_verify_by_n[n_lo]) / (n_hi - n_lo) as f32;
                let t_ar = self.t_verify_by_n[n_lo] - dt * (n_lo as f32 - 1.0);
                if dt > 0.0 && t_ar > 0.0 {
                    let raw = dt / t_ar;
                    if (0.05..=0.5).contains(&raw) {
                        self.p_star = raw;
                        self.calibrated = true;
                        eprintln!(
                            "[dspark] live p*={:.3} (verify-curve fit: n{}={:.1}ms n{}={:.1}ms dt={:.2} t_ar={:.1})",
                            raw, n_lo, self.t_verify_by_n[n_lo], n_hi, self.t_verify_by_n[n_hi], dt, t_ar
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn p_star_for_test(&self) -> f32 {
        self.p_star
    }

    pub(crate) fn observe(&mut self, accept_len: usize, _n_proposed: usize) {
        self.accept_ema = (1.0 - ACCEPT_ALPHA) * self.accept_ema + ACCEPT_ALPHA * accept_len as f32;
        self.windows_seen += 1;
        if self.windows_seen < WARMUP_WINDOWS {
            return;
        }
        // Hold the probed (higher) block so accept_ema can actually MEASURE acceptance
        // depth there before ceil(accept_ema) decides to keep or revert it.
        if self.probe_hold > 0 {
            self.probe_hold -= 1;
            return;
        }
        // Periodically probe one block deeper to escape the low-block cap-trap: at a small
        // block accept_len is capped and can't reveal deeper depth. After the hold,
        // ceil(accept_ema) keeps the probe (code discovers a deeper block is sustainable)
        // or reverts it (prose falls back to 1).
        self.explore_timer += 1;
        if self.explore_timer >= EXPLORE_INTERVAL && self.block < self.max_block {
            self.block += 1;
            self.explore_timer = 0;
            self.probe_hold = PROBE_HOLD;
            return;
        }
        // Settle at the observed acceptance depth.
        self.block = (self.accept_ema.ceil() as usize).clamp(self.min_block, self.max_block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deep acceptance (mean ~4) drives block to ≥4; allow periodic +1 probe.
    #[test]
    fn tracks_high_accept_depth() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..300 {
            c.observe(4, 5); // mean depth 4 -> ceil = 4
        }
        assert!(c.block() >= 4, "expected block ≥ 4, got {}", c.block());
    }

    // Zero acceptance (mean → 0) clamps block to 1; allow periodic +1 probe (≤2).
    #[test]
    fn tracks_low_accept_depth() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..300 {
            c.observe(0, 5); // depth 0 -> ceil(0)=0 -> clamp 1
        }
        assert!(c.block() <= 2, "expected block ≤ 2, got {}", c.block());
    }

    // Mean depth ~1.75 (the measured code case) settles in region 2..=3.
    #[test]
    fn tracks_mid_accept_depth() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        // 25% chance of 4, 75% chance of 1: mean = 0.25*4 + 0.75*1 = 1.75
        for i in 0..300 {
            c.observe(if i % 4 == 0 { 4 } else { 1 }, 5);
        }
        assert!(
            (2..=3).contains(&c.block()),
            "expected block in 2..=3, got {}",
            c.block()
        );
    }

    // reset() restores the default block and clears history.
    #[test]
    fn reset_restores_default() {
        let mut c = BlockController::new(2, 1, 5, 0.18);
        for _ in 0..50 {
            c.observe(0, 4);
        }
        assert_eq!(c.block(), 1);
        c.reset();
        assert_eq!(c.block(), 2);
    }

    // n_proposed == 0 (degenerate window) must not panic.
    #[test]
    fn zero_proposed_is_safe() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..50 {
            c.observe(0, 0);
        }
        assert!(c.block() >= 1 && c.block() <= 5);
    }

    // Fit p* from the verify curve: t_verify(n) ≈ t_AR + (n-1)·Δt.
    // n=2 → 90ms, n=6 → 150ms: dt=(150-90)/4=15, t_ar=90-15=75, p*=15/75=0.2.
    #[test]
    fn calibrates_from_verify_curve() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        // dt=(150-90)/4=15, t_ar=90-15*(2-1)=75, p*=15/75=0.2
        assert!(
            (c.p_star_for_test() - 0.2).abs() < 0.01,
            "p*={}",
            c.p_star_for_test()
        );
    }

    // A fit that computes p* > 0.5 must be REJECTED (not clamped): keep the prior.
    #[test]
    fn rejects_high_fit_keeps_prior() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        // t_v[2]=100, t_v[6]=300 → dt=50, t_ar=100-50=50, raw=1.0 > 0.5 → reject
        for _ in 0..30 {
            c.observe_timing(100.0, 2);
            c.observe_timing(300.0, 6);
        }
        assert!(
            (c.p_star_for_test() - 0.18).abs() < 1e-6,
            "should keep prior on out-of-range fit, got {}",
            c.p_star_for_test()
        );
    }

    // A fit that computes p* < 0.05 must be REJECTED: keep the prior.
    #[test]
    fn rejects_low_fit_keeps_prior() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        // t_v[2]=100, t_v[6]=101 → dt=0.25, t_ar=99.75, raw≈0.0025 < 0.05 → reject
        for _ in 0..30 {
            c.observe_timing(100.0, 2);
            c.observe_timing(101.0, 6);
        }
        assert!(
            (c.p_star_for_test() - 0.18).abs() < 1e-6,
            "should keep prior on tiny-ratio fit, got {}",
            c.p_star_for_test()
        );
    }

    // Exploration escapes the block-1 cap-trap: always accepting exactly what's drafted
    // keeps EMA capped at 1, but periodic probes let accept_ema discover higher depth.
    #[test]
    fn exploration_escapes_cap_trap() {
        let mut c = BlockController::new(1, 1, 5, 0.18); // start pinned at min
                                                         // Always accept exactly what's drafted (capped) — true depth is really deep.
        for _ in 0..600 {
            let b = c.block();
            c.observe(b, b);
        }
        assert!(
            c.block() >= 3,
            "exploration should climb out of the block-1 trap, got {}",
            c.block()
        );
    }

    // reset() preserves the calibrated p* (thermal-invariant hardware ratio).
    // gfx1151 shape: n=2→97ms, n=6→149ms: dt=13, t_ar=84, p*≈0.155 < 0.175.
    #[test]
    fn reset_preserves_calibrated_p_star() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..30 {
            c.observe_timing(97.0, 2);
            c.observe_timing(149.0, 6);
        }
        let calibrated_p = c.p_star_for_test();
        assert!(
            calibrated_p < 0.18 - 0.005,
            "p* should have moved from prior"
        );
        c.reset();
        assert_eq!(
            c.p_star_for_test(),
            calibrated_p,
            "reset() must preserve calibrated p*"
        );
        assert_eq!(c.block(), 3, "block returns to default after reset");
    }
}
