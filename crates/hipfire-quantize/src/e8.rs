// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — E8 lattice codec for mfp4-E8 (DType MFP4G32E8).
//
// E8 = D8 ∪ (D8 + 1/2·1),  D8 = { x ∈ Z^8 : sum(x) even }.
// Operates on block-NORMALIZED weight vectors v[0..8] in ≈[-6,6].
// QUANT_STEP = 0.88 (constant, tuned to the FWHT block-normalized domain).
//
// 32-bit index bijection (bit-exact CPU↔kernel):
//   bits[4i .. 4i+4) = e[i]  for i=0..6   (28 bits)
//   bits[28..31)     = e[7]>>1 & 0x7       (high 3 bits, LSB dropped)
//   bit[31]          = coset                (0=D8, 1=D8+1/2)
// Total = 32 bits.  Decode recovers e[7] LSB from parity of sum(e[0..7]).

pub const QUANT_STEP: f32 = 0.88;
/// E8_SIGMA: std-dev of the FWHT-normalized weight distribution (used in tests).
pub const E8_SIGMA: f32 = 1.7;

const COORD_BIAS: i32 = 7;   // biased range [0,15] → integer range [-7,8]
const COORD_BITS: u32 = 4;

// --------------------------------------------------------------------------
// Round-to-nearest, ties away from zero (integer-deterministic, matches kernel).
// --------------------------------------------------------------------------
#[inline(always)]
fn round_tie_away(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }
}

// --------------------------------------------------------------------------
// Nearest point in D8 (Conway-Sloane).
// --------------------------------------------------------------------------
fn closest_d8(u: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0f32; 8];
    let mut s: i64 = 0;
    let mut worst_idx = 0usize;
    let mut worst_abs = -1.0f32;
    let mut worst_dir = 0.0f32;
    for i in 0..8 {
        let ri = round_tie_away(u[i]);
        r[i] = ri;
        s += ri as i64;
        let e = u[i] - ri;
        let a = e.abs();
        if a > worst_abs {
            worst_abs = a;
            worst_idx = i;
            // e >= 0 means ri was rounded DOWN; flip UP (+1).
            worst_dir = if e >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    // Fix parity: if sum is odd, flip the coord with largest fractional distance.
    if (s & 1) != 0 {
        r[worst_idx] += worst_dir;
    }
    r
}

// --------------------------------------------------------------------------
// Nearest point in E8 = min(closest in D8, closest in D8+1/2).
// --------------------------------------------------------------------------
fn closest_e8(u: &[f32; 8]) -> [f32; 8] {
    let a = closest_d8(u);
    // Shift for the D8+1/2 coset.
    let mut ush = [0.0f32; 8];
    for i in 0..8 { ush[i] = u[i] - 0.5; }
    let bsh = closest_d8(&ush);
    let mut b = [0.0f32; 8];
    for i in 0..8 { b[i] = bsh[i] + 0.5; }

    let da: f32 = (0..8).map(|i| { let e = u[i] - a[i]; e * e }).sum();
    let db: f32 = (0..8).map(|i| { let e = u[i] - b[i]; e * e }).sum();
    if da <= db { a } else { b }
}

// --------------------------------------------------------------------------
// Encode: E8 point → u32 index.
// --------------------------------------------------------------------------
pub fn encode_index(p: &[f32; 8]) -> u32 {
    // Determine coset: half-integer coords → coset=1.
    let coset = if (p[0].fract().abs() - 0.5).abs() < 0.1 { 1u32 } else { 0u32 };

    // Integer coords of the underlying D8 point.
    let mut w = [0i32; 8];
    for i in 0..8 {
        w[i] = if coset == 1 { (p[i] - 0.5).round() as i32 } else { p[i].round() as i32 };
    }

    // Bias and clamp to [0,15].
    let mut e = [0u32; 8];
    for i in 0..8 {
        e[i] = (w[i] + COORD_BIAS).clamp(0, 15) as u32;
    }

    // Re-establish even sum (may have been broken by clamp).
    let sl: u32 = e.iter().sum();
    if (sl & 1) != 0 {
        // Nudge e[7] by ±1 within [0,15].
        if e[7] < 15 { e[7] += 1; } else { e[7] -= 1; }
    }

    // Pack: bits[4i..4i+4) = e[i] for i=0..6, bits[28..31) = e[7]>>1, bit[31]=coset.
    let mut idx: u32 = 0;
    for i in 0..7 {
        idx |= (e[i] & 0xF) << (i as u32 * COORD_BITS);
    }
    idx |= ((e[7] >> 1) & 0x7) << 28;
    idx |= coset << 31;
    idx
}

// --------------------------------------------------------------------------
// Decode: u32 index → E8 point (f32[8]).
// --------------------------------------------------------------------------
pub fn decode_index(idx: u32) -> [f32; 8] {
    let coset = (idx >> 31) & 1;
    let mut e = [0u32; 8];
    let mut sl: u32 = 0;
    for i in 0..7 {
        e[i] = (idx >> (i as u32 * COORD_BITS)) & 0xF;
        sl += e[i];
    }
    let e7_high = (idx >> 28) & 0x7;
    let p7 = e7_high << 1;
    // Recover LSB: parity of (sl + p7) must be even (sum of all 8 biased coords even).
    let lsb = (sl + p7) & 1;
    e[7] = p7 | lsb;

    let mut p = [0.0f32; 8];
    for i in 0..8 {
        let c = (e[i] as i32 - COORD_BIAS) as f32;
        p[i] = if coset == 1 { c + 0.5 } else { c };
    }
    p
}

// --------------------------------------------------------------------------
// Public API: quantize 8 weights → u32, dequantize u32 → 8 f32.
// --------------------------------------------------------------------------

/// Quantize 8 block-normalized weights (range ≈[-6,6]) to a 32-bit E8 codeword.
/// `q` = QUANT_STEP (0.88); pass `QUANT_STEP` constant.
pub fn quantize8(v: &[f32; 8], q: f32) -> u32 {
    let mut u = [0.0f32; 8];
    for i in 0..8 { u[i] = v[i] / q; }
    let p = closest_e8(&u);
    encode_index(&p)
}

/// Dequantize a 32-bit E8 codeword back to 8 f32 values (block-normalized domain).
/// `q` = QUANT_STEP (0.88).
pub fn dequantize8(idx: u32, q: f32) -> [f32; 8] {
    let p = decode_index(idx);
    let mut v = [0.0f32; 8];
    for i in 0..8 { v[i] = p[i] * q; }
    v
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic LCG for test randomness (no external dep).
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let hi = (*state >> 32) as u32;
        (hi as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    }

    fn box_muller(state: &mut u64, sigma: f32) -> (f32, f32) {
        loop {
            let u1 = (lcg_next(state) + 1.0) * 0.5; // uniform (0,1]
            let u2 = (lcg_next(state) + 1.0) * 0.5;
            if u1 <= 0.0 { continue; }
            let r = (-2.0 * u1.ln()).sqrt() * sigma;
            let theta = 2.0 * std::f32::consts::PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }

    #[test]
    fn round_tie_basic() {
        assert_eq!(round_tie_away(0.5), 1.0);
        assert_eq!(round_tie_away(-0.5), -1.0);
        assert_eq!(round_tie_away(1.4), 1.0);
        assert_eq!(round_tie_away(1.6), 2.0);
        assert_eq!(round_tie_away(-1.4), -1.0);
        assert_eq!(round_tie_away(-1.6), -2.0);
    }

    #[test]
    fn d8_even_sum() {
        let mut state = 0x12345678u64;
        for _ in 0..2000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 { v[i] = lcg_next(&mut state) * 6.0; }
            let p = closest_d8(&v);
            let s: i32 = p.iter().map(|&x| x as i32).sum();
            assert_eq!(s & 1, 0, "D8 sum not even: {:?}", p);
        }
    }

    #[test]
    fn index_roundtrip() {
        let mut state = 0xdeadbeef_cafebabeu64;
        let mut failures = 0usize;
        for _ in 0..500_000 {
            let mut v = [0.0f32; 8];
            for i in 0..8 { v[i] = lcg_next(&mut state) * 6.0; }
            let idx = quantize8(&v, QUANT_STEP);
            // Re-encode the decoded point — must give the same index.
            let p = decode_index(idx);
            let idx2 = encode_index(&p);
            if idx != idx2 {
                failures += 1;
            }
            // dequantize8 must equal decode_index * QUANT_STEP
            let dq = dequantize8(idx, QUANT_STEP);
            for i in 0..8 {
                let expected = p[i] * QUANT_STEP;
                assert!((dq[i] - expected).abs() < 1e-6,
                    "dequantize8 mismatch at i={i}: {:.6} vs {:.6}", dq[i], expected);
            }
        }
        assert_eq!(failures, 0, "bijection failures: {failures}");
    }

    /// Packing-gain gate: E8 MSE must beat E2M1 scalar MSE at the same block normalization.
    /// At q=0.88 on FWHT-normalized Gaussian, ratio should be ~1.298.
    #[test]
    fn packing_gain_beats_e2m1() {
        const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        fn e2m1_nearest(x: f32) -> f32 {
            let mut best = 0.0f32;
            let mut best_d = f32::MAX;
            for &m in &E2M1_MAG {
                for &s in &[1.0f32, -1.0f32] {
                    let v = m * s;
                    let d = (x - v).abs();
                    if d < best_d { best_d = d; best = v; }
                }
            }
            best
        }

        let mut state = 0xabcdef0123456789u64;
        let sigmas = [1.0f32, 1.5, E8_SIGMA, 2.0, 2.5];

        for &sigma in &sigmas {
            let n_blocks = 62500usize; // 500k weights / 8
            let mut e8_mse = 0.0f64;
            let mut e2_mse = 0.0f64;
            let mut count = 0usize;

            for _ in 0..n_blocks {
                let mut block = [0.0f32; 8];
                for i in 0..4 {
                    let (a, b) = box_muller(&mut state, sigma);
                    block[2*i] = a;
                    block[2*i+1] = b;
                }
                // Per-block normalization: max/6 (same as mfp4+P).
                let bmax = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
                if bmax == 0.0 { continue; }
                let sc = bmax / 6.0;
                let mut vn = [0.0f32; 8];
                for i in 0..8 { vn[i] = block[i] / sc; }

                // E8 MSE in original domain.
                let dq = dequantize8(quantize8(&vn, QUANT_STEP), QUANT_STEP);
                for i in 0..8 {
                    let err = (vn[i] - dq[i]) * sc;
                    e8_mse += (err * err) as f64;
                }

                // E2M1 MSE in original domain.
                for i in 0..8 {
                    let nearest = e2m1_nearest(vn[i]);
                    let err = (vn[i] - nearest) * sc;
                    e2_mse += (err * err) as f64;
                }
                count += 8;
            }

            let e8_avg = e8_mse / count as f64;
            let e2_avg = e2_mse / count as f64;
            let ratio = e2_avg / e8_avg;
            eprintln!("sigma={sigma:.1}: E8 {e8_avg:.6} E2M1 {e2_avg:.6} ratio {ratio:.3}");
            assert!(
                e8_avg < e2_avg,
                "E8 packing gain FAILED at sigma={sigma}: e8_mse={e8_avg:.6} >= e2m1_mse={e2_avg:.6}"
            );
            assert!(
                ratio > 1.1 && ratio < 1.6,
                "E8/E2M1 MSE ratio out of expected range at sigma={sigma}: {ratio:.3}"
            );
        }
    }
}
