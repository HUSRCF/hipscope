//! Pure (no-GPU) draft-block-size controller for DSpark spec-decode.
//! Marginal-accept hill-climb: grow the block while the whole drafted block keeps
//! being accepted (headroom), shrink when it doesn't (over-drafting). The decision
//! threshold `p*` is the break-even full-accept rate; it is a live-measured,
//! thermal-invariant cost ratio (see `calibrate_p_star`), seeded with a prior.

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
    /// EMA of single-position (bootstrap) forward time in ms.
    t_ar_ema: f32,
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
            t_ar_ema: 0.0,
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
        // Reset only request-specific state. Calibration fields (t_ar_ema,
        // t_verify_by_n, calibrated, p_star) are thermal-invariant hardware
        // cost ratios — calibrate once, reuse across requests.
        self.block = self.default_block;
        self.full_accept_ema = self.p_star;
        self.windows_seen = 0;
    }

    /// Observe wall-clock timing for a spec window. `t_ar_ms` is the
    /// single-position bootstrap forward time (Some only on bootstrap windows);
    /// `t_verify_ms` is the verify wall time; `n_verify` is the number of
    /// positions verified. Fires `calibrate_p_star` once we have both an AR
    /// sample and a high-n verify sample, then leaves calibration in place
    /// across subsequent reset() calls.
    pub(crate) fn observe_timing(
        &mut self,
        t_ar_ms: Option<f32>,
        t_verify_ms: f32,
        n_verify: usize,
    ) {
        if let Some(ar) = t_ar_ms {
            self.t_ar_ema = if self.t_ar_ema == 0.0 {
                ar
            } else {
                0.7 * self.t_ar_ema + 0.3 * ar
            };
        }
        if (2..8).contains(&n_verify) {
            let slot = &mut self.t_verify_by_n[n_verify];
            *slot = if *slot == 0.0 {
                t_verify_ms
            } else {
                0.7 * (*slot) + 0.3 * t_verify_ms
            };
        }
        if !self.calibrated && self.t_ar_ema > 0.0 {
            if let Some(n) = (2..8).rev().find(|&n| self.t_verify_by_n[n] > 0.0) {
                let p = calibrate_p_star(self.t_ar_ema, self.t_verify_by_n[n], n);
                self.p_star = p;
                self.calibrated = true;
                eprintln!(
                    "[dspark] block-controller live p*={:.3} (t_ar={:.1}ms, t_verify[n={}]={:.1}ms)",
                    p, self.t_ar_ema, n, self.t_verify_by_n[n]
                );
            }
        }
    }

    #[cfg(test)]
    fn p_star(&self) -> f32 {
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

/// Break-even full-accept rate from a live-measured, thermal-invariant timing ratio.
/// `t_ar_ms` = one single-position forward; `t_verify_ms` = a batched verify over
/// `n_verify` positions. Both measured in the same thermal state, so the ratio is
/// stable under DPM/thermal scaling. Returns the 0.18 prior for degenerate input.
pub(crate) fn calibrate_p_star(t_ar_ms: f32, t_verify_ms: f32, n_verify: usize) -> f32 {
    if n_verify < 2 || t_ar_ms <= 0.0 {
        return 0.18;
    }
    let dt_position = (t_verify_ms - t_ar_ms) / (n_verify as f32 - 1.0);
    (dt_position / t_ar_ms).clamp(0.05, 0.5)
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

    // gfx1151-mq2lloyd shape: t_AR≈84ms, verify of 6 positions≈149ms => Δt≈13ms/pos,
    // p* ≈ 13/84 ≈ 0.155. Must land near the offline 0.18 and inside the clamp.
    #[test]
    fn calibrate_matches_gfx1151_prior() {
        let p = calibrate_p_star(84.0, 149.0, 6);
        assert!((0.10..0.25).contains(&p), "p*={p} not near the 0.18 prior");
    }

    // Cheap-marginal arch (verify barely grows with n) => small p* (favor big blocks),
    // clamped at the floor.
    #[test]
    fn calibrate_clamps_low() {
        let p = calibrate_p_star(84.0, 88.0, 6); // Δt≈0.8ms/pos => ~0.01, clamp 0.05
        assert_eq!(p, 0.05);
    }

    // Expensive-marginal arch (verify grows steeply) => large p* (favor small blocks),
    // clamped at the ceiling.
    #[test]
    fn calibrate_clamps_high() {
        let p = calibrate_p_star(84.0, 84.0 + 84.0 * 5.0, 6); // Δt≈t_ar/pos => 1.0, clamp 0.5
        assert_eq!(p, 0.5);
    }

    // Degenerate n_verify<2 falls back to the prior (no division by zero).
    #[test]
    fn calibrate_degenerate_returns_prior() {
        assert_eq!(calibrate_p_star(84.0, 84.0, 1), 0.18);
    }

    // observe_timing fires calibrate_p_star on the first window that has both
    // t_ar and a valid t_verify. gfx1151-mq2lloyd shape: t_ar≈84ms, verify of
    // 6 positions≈149ms => Δt≈13ms/pos, p*≈0.155. The prior (0.18) is replaced.
    #[test]
    fn observe_timing_calibrates_live_p_star() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        // Bootstrap window: t_ar=84ms, verify n=6 at 149ms.
        c.observe_timing(Some(84.0), 149.0, 6);
        let p = c.p_star();
        assert!(
            (0.10..0.25).contains(&p),
            "live p*={p} not near the expected ≈0.155"
        );
        // Calibrated p* should be close to calibrate_p_star(84,149,6).
        let expected = calibrate_p_star(84.0, 149.0, 6);
        assert!(
            (p - expected).abs() < 1e-5,
            "p*={p} != calibrate_p_star result {expected}"
        );
    }

    // reset() preserves the calibrated p* (thermal-invariant hardware ratio).
    #[test]
    fn reset_preserves_calibrated_p_star() {
        let mut c = BlockController::new(3, 1, 5, 0.18);
        c.observe_timing(Some(84.0), 149.0, 6);
        let calibrated_p = c.p_star();
        assert!(
            calibrated_p < 0.18 - 0.005,
            "p* should have moved from prior"
        );
        c.reset();
        assert_eq!(
            c.p_star(),
            calibrated_p,
            "reset() must preserve calibrated p*"
        );
        assert_eq!(c.block(), 3, "block returns to default after reset");
    }
}
