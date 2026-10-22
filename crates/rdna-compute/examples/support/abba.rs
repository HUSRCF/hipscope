// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! ABBA hipEvent timing core shared by `pm_decode_twins time` and `pm_loop`:
//! the allocation-free ABBA schedule, sample statistics and the separability
//! rule. `Incumbent` is arm A (the shipped/baseline kernel), `Candidate` is
//! arm B. Included with `#[path = "support/abba.rs"] mod abba;`.
#![allow(dead_code)]

use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn positive_count(s: &str) -> Result<usize> {
    let n = s.parse::<usize>().map_err(err)?;
    if n == 0 {
        return Err("counts must be positive".into());
    }
    Ok(n)
}

pub fn timing_counts(n: usize, rounds: usize) -> Result<(usize, usize)> {
    if n == 0 || rounds == 0 {
        return Err("counts must be positive".into());
    }
    let per_arm = n
        .checked_mul(rounds)
        .and_then(|v| v.checked_mul(2))
        .ok_or("timing call count overflow")?;
    let total = per_arm.checked_mul(2).ok_or("timing call count overflow")?;
    per_arm
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("sample storage overflow")?;
    Ok((per_arm, total))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingArm {
    Incumbent,
    Candidate,
}

/// Pure, allocation-free ABBA iterator; construction checks all count arithmetic.
pub fn timing_schedule(n: usize, rounds: usize) -> Result<impl Iterator<Item = TimingArm>> {
    let (_, total) = timing_counts(n, rounds)?;
    Ok((0..total).map(move |i| match (i / n) % 4 {
        0 | 3 => TimingArm::Incumbent,
        _ => TimingArm::Candidate,
    }))
}

#[derive(Debug)]
pub struct TimingStats {
    pub count: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p90: f64,
    pub mean: f64,
}

impl TimingStats {
    pub fn spread(&self) -> f64 {
        self.max - self.min
    }
    pub fn json(&self) -> Value {
        json!({"call_count": self.count, "median_ns": self.median, "min_ns": self.min,
            "max_ns": self.max, "p90_ns": self.p90, "mean_ns": self.mean,
            "spread_ns": self.spread()})
    }
}

pub fn timing_stats(samples: &[f64]) -> Result<TimingStats> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("timing samples must be nonempty, positive and finite".into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n.is_multiple_of(2) {
        sorted[n / 2 - 1] / 2.0 + sorted[n / 2] / 2.0
    } else {
        sorted[n / 2]
    };
    // ceil(0.9*n), without overflowing integer multiplication.
    let rank = n - n / 10;
    let mean = sorted.iter().map(|v| v / n as f64).sum::<f64>();
    if !mean.is_finite() {
        return Err("nonfinite timing mean".into());
    }
    Ok(TimingStats {
        count: n,
        median,
        min: sorted[0],
        max: sorted[n - 1],
        p90: sorted[rank - 1],
        mean,
    })
}

pub fn timing_relation(a: &TimingStats, b: &TimingStats) -> (f64, f64, f64, bool) {
    let spread = a.spread().max(b.spread());
    let delta = b.median - a.median;
    (
        b.median / a.median,
        spread,
        delta,
        delta.abs() > spread && (a.max < b.min || b.max < a.min),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_abba_two_rounds() {
        use TimingArm::{Candidate as B, Incumbent as A};
        let got: Vec<_> = timing_schedule(2, 2).unwrap().collect();
        assert_eq!(got, vec![A, A, B, B, B, B, A, A, A, A, B, B, B, B, A, A]);
        assert_eq!(got.iter().filter(|&&v| v == A).count(), 8);
        assert_eq!(got.iter().filter(|&&v| v == B).count(), 8);
    }
    #[test]
    fn identical_not_separable() {
        let s = timing_stats(&[3.0, 3.0]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&s, &s);
        assert_eq!(ratio, 1.0);
        assert!(!separable);
    }
    #[test]
    fn constant_double_separable() {
        let a = timing_stats(&[2.0; 4]).unwrap();
        let b = timing_stats(&[4.0; 4]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&a, &b);
        assert_eq!(ratio, 2.0);
        assert!(separable);
    }
    #[test]
    fn noisy_overlap_not_separable() {
        let a = timing_stats(&[1.0, 5.0, 9.0]).unwrap();
        let b = timing_stats(&[8.0, 10.0, 12.0]).unwrap();
        assert!(!timing_relation(&a, &b).3);
    }
    #[test]
    fn statistics_centers_and_rank() {
        let s = timing_stats(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(
            (s.count, s.median, s.min, s.max, s.p90, s.mean, s.spread()),
            (4, 2.5, 1.0, 4.0, 4.0, 2.5, 3.0)
        );
        assert_eq!(timing_stats(&[3.0, 1.0, 2.0]).unwrap().median, 2.0);
        assert_eq!(
            timing_stats(&(1..=10).map(f64::from).collect::<Vec<_>>())
                .unwrap()
                .p90,
            9.0
        );
    }
    #[test]
    fn bad_samples_and_counts() {
        for values in [&[][..], &[0.0], &[-1.0], &[f64::NAN], &[f64::INFINITY]] {
            assert!(timing_stats(values).is_err());
        }
        assert!(positive_count("0").is_err());
        assert!(positive_count("-1").is_err());
        assert!(timing_counts(usize::MAX, 1).is_err());
        assert!(timing_counts(1, usize::MAX).is_err());
        assert!(timing_schedule(0, 1).is_err());
    }
}
