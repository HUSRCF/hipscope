// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lossless sampled speculative verification over sparse truncated
//! distributions (Leviathan et al. 2023, Chen et al. 2023).
//!
//! A draft token `x` drawn from the draft distribution `q` is accepted with
//! probability `min(1, p(x) / q(x))`; a rejection emits a draw from the
//! normalized residual `(p - q)+`, and a window whose drafts were all accepted
//! emits its bonus from `p`. Every emitted token is then distributed exactly
//! as `p`, whatever `q` is, as long as `x` really was drawn from `q`.
//!
//! Both sides are [`SparseDist`]s built by the same truncation
//! ([`SampleSpec`]): each side keeps its own nucleus (the DFlash convention,
//! `docs/plans/mtp-sampled-tighten-design-2026-06-23.md` §(a)), and the
//! residual is taken across the two supports. `p` is the distribution the
//! autoregressive producer samples, so the speculative stream is the AR
//! stream in distribution.

use crate::spec::{request_rng_state, GreedyAccept};

/// Truncation of a temperature-scaled softmax: top-k by logit, then min-p
/// relative to the most probable kept token, then the smallest probability
/// prefix whose mass reaches `top_p` of what is left (the boundary token is
/// kept). `top_k == 0` keeps every finite logit; `min_p == 0` and
/// `top_p >= 1` disable those cuts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSpec {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
}

impl SampleSpec {
    /// Candidates `llama::sample_top_p` keeps before its nucleus cut.
    pub const CPU_AR_TOP_K: usize = 20;

    /// The distribution the host AR sampler (`sampler::sample_cpu` →
    /// `llama::sample_top_p`) draws from: the top 20 raw logits, softmax at
    /// `temperature`, nucleus at `top_p`. That sampler ignores a request's
    /// `top_k` and `min_p`, so this does too.
    pub fn cpu_ar(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            top_k: Self::CPU_AR_TOP_K,
            min_p: 0.0,
        }
    }
}

/// A truncated distribution: `(token, probability)` in descending
/// probability order, probabilities summing to one. Rebuilt in place so a
/// caller can reuse its buffers across rows.
#[derive(Clone, Debug, Default)]
pub struct SparseDist {
    entries: Vec<(u32, f32)>,
}

impl SparseDist {
    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    /// Probability of `token` (zero outside the support).
    pub fn prob(&self, token: u32) -> f32 {
        self.entries
            .iter()
            .find(|&&(t, _)| t == token)
            .map_or(0.0, |&(_, p)| p)
    }

    /// Inverse-CDF draw for `u` in `[0, 1)`.
    pub fn sample(&self, u: f32) -> u32 {
        let mut acc = 0.0f32;
        for &(token, p) in &self.entries {
            acc += p;
            if u < acc {
                return token;
            }
        }
        self.entries.last().map_or(0, |&(token, _)| token)
    }

    /// Build from a full logit row (row index = token id). `scratch` holds
    /// the kept candidates between calls. Fails when no finite mass is left.
    pub fn build_from_logits(
        &mut self,
        logits: &[f32],
        spec: SampleSpec,
        scratch: &mut Vec<(u32, f32)>,
    ) -> Result<(), String> {
        scratch.clear();
        let k = spec.top_k;
        if k == 0 || k >= logits.len() {
            scratch.extend(
                logits
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.is_finite())
                    .map(|(i, &l)| (i as u32, l)),
            );
        } else {
            // Kept sorted by (logit desc, id asc): a later id enters only
            // with a strictly larger logit, so ties keep the lower id.
            for (i, &l) in logits.iter().enumerate() {
                if !l.is_finite() || (scratch.len() == k && l <= scratch[k - 1].1) {
                    continue;
                }
                if scratch.len() == k {
                    scratch.pop();
                }
                let at = scratch.partition_point(|&(_, v)| v >= l);
                scratch.insert(at, (i as u32, l));
            }
        }
        self.build_from_candidates(scratch, spec)
    }

    /// Build from `(token, logit)` candidates (any order; non-finite logits
    /// are dropped). `candidates` is reordered.
    pub fn build_from_candidates(
        &mut self,
        candidates: &mut Vec<(u32, f32)>,
        spec: SampleSpec,
    ) -> Result<(), String> {
        if !(spec.temperature > 0.0) || !spec.temperature.is_finite() {
            return Err(format!(
                "sampled verification needs a positive temperature, got {}",
                spec.temperature
            ));
        }
        candidates.retain(|(_, l)| l.is_finite());
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if spec.top_k > 0 {
            candidates.truncate(spec.top_k);
        }
        let Some(&(_, max)) = candidates.first() else {
            return Err("sampled verification row has no finite logit".to_string());
        };
        let inv_temp = 1.0 / spec.temperature;
        self.entries.clear();
        // Same f32 arithmetic as `llama::sample_top_p`.
        self.entries.extend(
            candidates
                .iter()
                .map(|&(token, l)| (token, ((l - max) * inv_temp).exp())),
        );
        if spec.min_p > 0.0 {
            let floor = spec.min_p * self.entries[0].1;
            self.entries.retain(|&(_, p)| p >= floor);
        }
        let sum: f32 = self.entries.iter().map(|&(_, p)| p).sum();
        if !(sum > 0.0) || !sum.is_finite() {
            return Err("sampled verification row has no finite mass".to_string());
        }
        let threshold = spec.top_p.clamp(0.0, 1.0) * sum;
        let mut kept = 0.0f32;
        let mut len = self.entries.len();
        for (i, &(_, p)) in self.entries.iter().enumerate() {
            kept += p;
            if kept >= threshold {
                len = i + 1;
                break;
            }
        }
        self.entries.truncate(len);
        let mass: f32 = self.entries.iter().map(|&(_, p)| p).sum();
        for entry in &mut self.entries {
            entry.1 /= mass;
        }
        Ok(())
    }
}

/// Draw from the normalized residual `(p - q)+`. When `p <= q` everywhere
/// (equal distributions up to rounding) the residual is empty and the draw
/// falls back to `p`, which a rejection then had probability zero to reach.
pub fn sample_residual(p: &SparseDist, q: &SparseDist, u: f32) -> u32 {
    let residual = |&(token, pt): &(u32, f32)| (token, (pt - q.prob(token)).max(0.0));
    let mass: f32 = p.entries.iter().map(|e| residual(e).1).sum();
    if !(mass > 0.0) {
        return p.sample(u);
    }
    let target = u * mass;
    let mut acc = 0.0f32;
    let mut last = None;
    for (token, r) in p.entries.iter().map(residual) {
        if r <= 0.0 {
            continue;
        }
        acc += r;
        if target < acc {
            return token;
        }
        last = Some(token);
    }
    last.unwrap_or_else(|| p.sample(u))
}

/// Outcome of verifying one sampled draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftVerdict {
    Accept,
    /// Rejected; the token is the residual draw that replaces the draft.
    Reject(u32),
}

/// Accept `draft` (drawn from `q`) with probability `min(1, p/q)`, else draw
/// its replacement from the residual. Consumes one uniform on accept, two
/// on reject.
pub fn verify_sampled_draft(
    p: &SparseDist,
    q: &SparseDist,
    draft: u32,
    rng: &mut SpecRng,
) -> DraftVerdict {
    let qx = q.prob(draft);
    let px = p.prob(draft);
    let u = rng.next_f32();
    // Strict: a token with p(x) = 0 is never accepted, and p >= q accepts
    // for every u in [0, 1).
    if qx > 0.0 && u * qx < px {
        DraftVerdict::Accept
    } else {
        DraftVerdict::Reject(sample_residual(p, q, rng.next_f32()))
    }
}

/// The sampled counterpart of [`crate::spec::accept_greedy_prefix`], with the
/// same result shape and EOS rule: accepted drafts in order, stopping (no
/// bonus) at an accepted EOS draft, then either the residual replacement of
/// the first rejected draft or the bonus drawn from `p` at row
/// `drafts.len()`. `target(row, out)` fills the target distribution of a
/// verify row; it is called only for rows the verdict reads, in order.
pub fn accept_sampled_prefix<F>(
    drafts: &[u32],
    draft_dists: &[SparseDist],
    eos: Option<u32>,
    rng: &mut SpecRng,
    target: &mut SparseDist,
    mut fill_target: F,
) -> Result<GreedyAccept, String>
where
    F: FnMut(usize, &mut SparseDist) -> Result<(), String>,
{
    if draft_dists.len() != drafts.len() {
        return Err(format!(
            "sampled verify has {} draft distributions for {} drafts",
            draft_dists.len(),
            drafts.len()
        ));
    }
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    for (row, (&draft, q)) in drafts.iter().zip(draft_dists).enumerate() {
        fill_target(row, target)?;
        match verify_sampled_draft(target, q, draft, rng) {
            DraftVerdict::Accept => {
                committed.push(draft);
                accepted += 1;
                if eos == Some(draft) {
                    return Ok(GreedyAccept {
                        committed,
                        accepted,
                        hit_eos: true,
                    });
                }
            }
            DraftVerdict::Reject(token) => {
                committed.push(token);
                return Ok(GreedyAccept {
                    committed,
                    accepted,
                    hit_eos: eos == Some(token),
                });
            }
        }
    }
    fill_target(drafts.len(), target)?;
    let bonus = target.sample(rng.next_f32());
    committed.push(bonus);
    Ok(GreedyAccept {
        committed,
        accepted,
        hit_eos: eos == Some(bonus),
    })
}

/// Request-seeded xorshift64* stream for draft draws and verdicts. The same
/// seed replays the same draws.
#[derive(Clone, Copy, Debug)]
pub struct SpecRng(u64);

impl SpecRng {
    pub fn new(rng_seed: u64) -> Self {
        Self(
            request_rng_state(rng_seed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .max(1),
        )
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on the 2^-24 grid of `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pearson chi-square of `counts` (total `n`) against `expected`
    /// probabilities, cells with expected count under 5 pooled into one.
    /// Returns `(statistic, degrees of freedom)`.
    fn chi_square(counts: &[u64], expected: &[f64], n: u64) -> (f64, usize) {
        let mut stat = 0.0;
        let mut cells = 0usize;
        let (mut pooled_obs, mut pooled_exp) = (0.0, 0.0);
        for (&c, &e) in counts.iter().zip(expected) {
            let e = e * n as f64;
            if e < 5.0 {
                pooled_obs += c as f64;
                pooled_exp += e;
                assert!(
                    e > 0.0 || c == 0,
                    "token outside the target support was emitted"
                );
                continue;
            }
            stat += (c as f64 - e).powi(2) / e;
            cells += 1;
        }
        if pooled_exp >= 5.0 {
            stat += (pooled_obs - pooled_exp).powi(2) / pooled_exp;
            cells += 1;
        }
        (stat, cells.saturating_sub(1).max(1))
    }

    /// Wilson–Hilferty chi-square quantile at standard-normal `z`; z = 4.75
    /// is the 1 - 1e-6 quantile.
    fn chi_square_critical(df: usize, z: f64) -> f64 {
        let df = df as f64;
        let a = 2.0 / (9.0 * df);
        df * (1.0 - a + z * a.sqrt()).powi(3)
    }

    fn total_variation(counts: &[u64], expected: &[f64], n: u64) -> f64 {
        counts
            .iter()
            .zip(expected)
            .map(|(&c, &e)| (c as f64 / n as f64 - e).abs())
            .sum::<f64>()
            / 2.0
    }

    /// Deterministic pseudo-logits.
    fn logits(vocab: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = SpecRng::new(seed);
        (0..vocab)
            .map(|_| {
                // Sum of uniforms: a bell-ish spread with a few strong heads.
                let g: f32 = (0..4).map(|_| rng.next_f32()).sum::<f32>() - 2.0;
                g * scale
            })
            .collect()
    }

    /// The draft side the Qwen4 MTP drafter builds: a noisy ranking of the
    /// target's logits picks 8 candidates, which are then "re-scored" with
    /// their exact draft logits (themselves a perturbation of the target).
    fn rescored_draft(target: &[f32], seed: u64, spec: SampleSpec) -> SparseDist {
        let noise = logits(target.len(), seed, 1.5);
        let exact = logits(target.len(), seed ^ 0xABCD, 0.7);
        let mut ranked: Vec<(u32, f32)> = target
            .iter()
            .zip(&noise)
            .enumerate()
            .map(|(i, (t, n))| (i as u32, t + n))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut candidates: Vec<(u32, f32)> = ranked[..8]
            .iter()
            .map(|&(i, _)| (i, target[i as usize] + exact[i as usize]))
            .collect();
        let mut q = SparseDist::default();
        q.build_from_candidates(&mut candidates, spec).unwrap();
        q
    }

    fn dense(dist: &SparseDist, vocab: usize) -> Vec<f64> {
        let mut out = vec![0.0; vocab];
        for &(t, p) in dist.entries() {
            out[t as usize] = p as f64;
        }
        out
    }

    const TRIALS: u64 = 200_000;

    /// One-position exactness: draft from the 8-candidate re-scored `q`,
    /// verify against `p`, and the emitted token must follow `p` (chi-square
    /// at alpha = 1e-6 and a total-variation bound) across the truncations
    /// Qwen4 serves and the ones it does not (top-k < 20, min-p).
    #[test]
    fn single_position_emits_target_distribution() {
        let vocab = 64;
        let specs = [
            SampleSpec::cpu_ar(1.0, 0.95),
            SampleSpec::cpu_ar(0.7, 0.8),
            SampleSpec::cpu_ar(1.0, 1.0),
            SampleSpec {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 5,
                min_p: 0.0,
            },
            SampleSpec {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 0,
                min_p: 0.1,
            },
        ];
        for (case, &spec) in specs.iter().enumerate() {
            let target_logits = logits(vocab, 11 + case as u64, 2.5);
            let mut p = SparseDist::default();
            p.build_from_logits(&target_logits, spec, &mut Vec::new())
                .unwrap();
            let q = rescored_draft(&target_logits, 100 + case as u64, spec);
            let mut rng = SpecRng::new(7 + case as u64);
            let mut counts = vec![0u64; vocab];
            let mut accepted = 0u64;
            for _ in 0..TRIALS {
                let x = q.sample(rng.next_f32());
                let token = match verify_sampled_draft(&p, &q, x, &mut rng) {
                    DraftVerdict::Accept => {
                        accepted += 1;
                        x
                    }
                    DraftVerdict::Reject(t) => t,
                };
                counts[token as usize] += 1;
            }
            let expected = dense(&p, vocab);
            let (stat, df) = chi_square(&counts, &expected, TRIALS);
            let crit = chi_square_critical(df, 4.75);
            let tv = total_variation(&counts, &expected, TRIALS);
            // Acceptance must equal sum_x min(p, q) (its exact expectation).
            let overlap: f64 = q
                .entries()
                .iter()
                .map(|&(t, qt)| (qt as f64).min(p.prob(t) as f64))
                .sum();
            let rate = accepted as f64 / TRIALS as f64;
            eprintln!(
                "case {case} {spec:?}: chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4} accept={rate:.4} overlap={overlap:.4}"
            );
            assert!(stat < crit, "case {case}: chi2 {stat} >= {crit} (df {df})");
            assert!(tv < 0.01, "case {case}: total variation {tv}");
            assert!(
                (rate - overlap).abs() < 0.01,
                "case {case}: accept {rate} vs {overlap}"
            );
        }
    }

    /// Two chained positions through `accept_sampled_prefix`, with the
    /// second target row conditioned on the first token and an EOS token in
    /// the vocabulary: the emitted (t1, t2) pairs must follow
    /// p1(t1) p2(t2 | t1), with an emitted EOS ending the window.
    #[test]
    fn chained_window_emits_target_joint_distribution() {
        let vocab = 8usize;
        let eos = 3u32;
        let spec = SampleSpec::cpu_ar(1.0, 0.95);
        let p1_logits = logits(vocab, 21, 1.5);
        let p2_logits: Vec<Vec<f32>> = (0..vocab)
            .map(|t| logits(vocab, 40 + t as u64, 1.5))
            .collect();
        let mut p1 = SparseDist::default();
        p1.build_from_logits(&p1_logits, spec, &mut Vec::new())
            .unwrap();
        let p2: Vec<SparseDist> = p2_logits
            .iter()
            .map(|l| {
                let mut d = SparseDist::default();
                d.build_from_logits(l, spec, &mut Vec::new()).unwrap();
                d
            })
            .collect();
        // Category index: t1 * (vocab + 1) + t2, t2 == vocab = "window ended".
        let cells = vocab * (vocab + 1);
        let mut expected = vec![0.0f64; cells];
        for &(t1, a) in p1.entries() {
            if t1 == eos {
                expected[t1 as usize * (vocab + 1) + vocab] += a as f64;
                continue;
            }
            for &(t2, b) in p2[t1 as usize].entries() {
                expected[t1 as usize * (vocab + 1) + t2 as usize] += a as f64 * b as f64;
            }
        }
        let mut rng = SpecRng::new(99);
        let mut counts = vec![0u64; cells];
        let mut target = SparseDist::default();
        for trial in 0..TRIALS {
            // Draft depth 1 or 2 so both the bonus and the residual reach t2.
            let depth = 1 + (trial % 2) as usize;
            let q1 = rescored_draft(&p1_logits, 500, spec);
            let x1 = q1.sample(rng.next_f32());
            let mut drafts = vec![x1];
            let mut qs = vec![q1];
            if depth == 2 {
                let q2 = rescored_draft(&p2_logits[x1 as usize], 600 + x1 as u64, spec);
                drafts.push(q2.sample(rng.next_f32()));
                qs.push(q2);
            }
            let window = accept_sampled_prefix(
                &drafts,
                &qs,
                Some(eos),
                &mut rng,
                &mut target,
                |row, out| {
                    let src = if row == 0 {
                        &p1
                    } else {
                        &p2[drafts[row - 1] as usize]
                    };
                    out.clone_from(src);
                    Ok(())
                },
            )
            .unwrap();
            let t1 = window.committed[0] as usize;
            let t2 = if window.committed[0] == eos {
                assert!(window.hit_eos);
                vocab
            } else if let Some(&t2) = window.committed.get(1) {
                t2 as usize
            } else {
                // Rejected at row 0: the second token comes from the next
                // window, whose seed row is p2(. | t1).
                let mut next = SparseDist::default();
                next.clone_from(&p2[t1]);
                next.sample(rng.next_f32()) as usize
            };
            counts[t1 * (vocab + 1) + t2] += 1;
        }
        let (stat, df) = chi_square(&counts, &expected, TRIALS);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(&counts, &expected, TRIALS);
        eprintln!("joint: chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4}");
        assert!(stat < crit, "joint chi2 {stat} >= {crit} (df {df})");
        assert!(tv < 0.01, "joint total variation {tv}");
    }

    /// `SampleSpec::cpu_ar` is the distribution the host AR sampler draws
    /// from: 200k `llama::sample_top_p` draws match it.
    #[test]
    fn cpu_ar_spec_matches_host_ar_sampler() {
        let vocab = 300;
        for (case, (temp, top_p)) in [(1.0f32, 0.95f32), (0.7, 0.8)].into_iter().enumerate() {
            let row = logits(vocab, 77 + case as u64, 3.0);
            let mut p = SparseDist::default();
            p.build_from_logits(&row, SampleSpec::cpu_ar(temp, top_p), &mut Vec::new())
                .unwrap();
            crate::llama::reset_cpu_sampler_rng(1234 + case as u32);
            let mut counts = vec![0u64; vocab];
            for _ in 0..TRIALS {
                counts[crate::llama::sample_top_p(&row, temp, top_p) as usize] += 1;
            }
            let expected = dense(&p, vocab);
            let (stat, df) = chi_square(&counts, &expected, TRIALS);
            let crit = chi_square_critical(df, 4.75);
            let tv = total_variation(&counts, &expected, TRIALS);
            eprintln!("cpu-ar case {case}: chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4}");
            assert!(stat < crit, "case {case}: chi2 {stat} >= {crit} (df {df})");
            assert!(tv < 0.01, "case {case}: total variation {tv}");
        }
    }

    #[test]
    fn truncation_keeps_boundary_and_order() {
        let logits = [0.0f32, 3.0, f32::NAN, 2.0, 1.0, f32::NEG_INFINITY];
        let mut d = SparseDist::default();
        let spec = SampleSpec {
            temperature: 1.0,
            top_p: 0.7,
            top_k: 3,
            min_p: 0.0,
        };
        d.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        // softmax over {1: 3, 3: 2, 4: 1} = .665/.245/.090; top_p 0.7 keeps
        // token 1 and the boundary token 3.
        let tokens: Vec<u32> = d.entries().iter().map(|e| e.0).collect();
        assert_eq!(tokens, [1, 3]);
        let sum: f32 = d.entries().iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let mut d2 = SparseDist::default();
        let min_p = SampleSpec {
            min_p: 0.3,
            top_p: 1.0,
            ..spec
        };
        d2.build_from_logits(&logits, min_p, &mut Vec::new())
            .unwrap();
        // e^-1 = .37 >= .3 keeps token 3; e^-2 = .135 < .3 drops token 4.
        assert_eq!(d2.entries().iter().map(|e| e.0).collect::<Vec<_>>(), [1, 3]);
        assert!(d
            .build_from_logits(&[f32::NAN; 4], spec, &mut Vec::new())
            .is_err());
    }

    #[test]
    fn same_seed_replays_the_same_stream() {
        let mut a = SpecRng::new(42);
        let mut b = SpecRng::new(42);
        let mut c = SpecRng::new(43);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        assert_eq!(xs, (0..8).map(|_| b.next_u64()).collect::<Vec<_>>());
        assert_ne!(xs, (0..8).map(|_| c.next_u64()).collect::<Vec<_>>());
        // Seed 0 maps onto the request sentinel instead of the stuck state.
        assert_eq!(
            SpecRng::new(0).next_u64(),
            SpecRng::new(0x1357_9BDF).next_u64()
        );
    }
}
