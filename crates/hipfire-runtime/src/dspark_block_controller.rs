//! Pure (no-GPU) draft-block-size controller for DSpark spec-decode.
//! Marginal-accept hill-climb: grow the block while the whole drafted block keeps
//! being accepted (headroom), shrink when it doesn't (over-drafting). The decision
//! threshold `p*` is the break-even full-accept rate; it is a live-measured,
//! thermal-invariant cost ratio fitted from the verify-curve slope, seeded with a prior.

/// Marginal-accept hill-climb over the DSpark draft block cap.
pub(crate) struct BlockController {
    block: usize,
    default_block: usize,
    min_block: usize,
    max_block: usize,
    /// EMA of the per-window full-accept indicator (accept_len == n_proposed).
    full_accept_ema: f32,
    /// Break-even full-accept rate; grow above p*+H, shrink below p*-H.
    p_star: f32,
    windows_seen: u32,
    // ── live p* calibration (hardware cost ratio; stable across requests) ──
    /// Total timing samples observed; gated to ≥TIMING_WARMUP before calibrating.
    timing_samples: u32,
    /// Per-n EMA of verify wall-time in ms (indexed by n_verify, slots 2..8).
    t_verify_by_n: [f32; 8],
    /// True once p* has been measured from live timing; preserved across reset().
    calibrated: bool,
}

// Small alpha: the full-accept signal is binary {0,1}, so one window bumps the
// EMA by alpha. Keep it well below the hysteresis band (0.06) so a single impulse
// can't cross it — a STABLE acceptance rate near p* must hold; only a sustained
// shift should move the block. Provisional; tuned empirically in a later GPU task.
const EMA_ALPHA: f32 = 0.05;
const HYSTERESIS: f32 = 0.06;
/// Skip the first few windows so the block doesn't react to early bootstrap
/// noise. The EMA is seeded at `p*` (neutral), so a handful of real observations
/// is enough to establish a trend — this is a short guard, not the full EMA
/// settling time.
const WARMUP_WINDOWS: u32 = 6;
/// Minimum timing samples before attempting verify-curve calibration.
/// Gives the GPU time to warm up and collects samples at ≥2 distinct n values.
const TIMING_WARMUP: u32 = 16;

impl BlockController {
    pub(crate) fn new(
        default_block: usize,
        min_block: usize,
        max_block: usize,
        p_star: f32,
    ) -> Self {
        Self {
            block: default_block.clamp(min_block, max_block),
            default_block: default_block.clamp(min_block, max_block),
            min_block,
            max_block,
            full_accept_ema: p_star, // neutral start: no initial bias to grow/shrink
            p_star,
            windows_seen: 0,
            timing_samples: 0,
            t_verify_by_n: [0.0f32; 8],
            calibrated: false,
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
        self.full_accept_ema = self.p_star;
        self.windows_seen = 0;
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

    pub(crate) fn observe(&mut self, accept_len: usize, n_proposed: usize) {
        let full_accept = if n_proposed > 0 && accept_len >= n_proposed {
            1.0
        } else {
            0.0
        };
        self.full_accept_ema = (1.0 - EMA_ALPHA) * self.full_accept_ema + EMA_ALPHA * full_accept;
        self.windows_seen += 1;
        if self.windows_seen < WARMUP_WINDOWS {
            return;
        }
        if self.full_accept_ema > self.p_star + HYSTERESIS {
            self.block = (self.block + 1).min(self.max_block);
        } else if self.full_accept_ema < self.p_star - HYSTERESIS {
            self.block = self.block.saturating_sub(1).max(self.min_block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A run of fully-accepted windows must push the block up to max.
    #[test]
    fn grows_to_max_on_full_accept() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..50 {
            c.observe(4, 4); // accept_len == n_proposed => full accept
        }
        assert_eq!(c.block(), 5);
    }

    // A run of zero-accept windows must push the block down to min.
    #[test]
    fn shrinks_to_min_on_no_accept() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..50 {
            c.observe(0, 4); // accepted nothing we drafted
        }
        assert_eq!(c.block(), 1);
    }

    // At a STATIONARY full-accept rate ≈ p* (evenly spread, not lumpy), the block
    // must not drift to an extreme. An EMA tracks the *local* rate, so a stationary
    // near-p* input keeps it inside the hysteresis band. (A lumpy input — long full
    // runs then long zero runs — legitimately moves the block: that is the
    // controller working, not a bug.)
    #[test]
    fn holds_near_default_at_stationary_p_star() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for i in 0..600 {
            if i % 6 == 0 {
                c.observe(4, 4); // rate 1/6 ≈ 0.167 ≈ p*
            } else {
                c.observe(2, 4);
            }
        }
        assert!(
            (2..=4).contains(&c.block()),
            "block={} drifted to an extreme at a stationary p* rate",
            c.block()
        );
    }

    // reset() restores the default block and clears history.
    #[test]
    fn reset_restores_default() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        for _ in 0..50 {
            c.observe(0, 4);
        }
        assert_eq!(c.block(), 1);
        c.reset();
        assert_eq!(c.block(), 3);
    }

    // n_proposed == 0 (degenerate window) must not panic or count as full-accept.
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
