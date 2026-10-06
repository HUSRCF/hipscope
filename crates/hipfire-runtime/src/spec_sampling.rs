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
//! Both sides are [`SparseDist`]s truncated by one [`SampleSpec`]: `p` from a
//! full target logit row with exactly the arithmetic of the host AR sampler
//! ([`crate::llama::sample_top_k_p`]), `q` from the draft's candidates. Each
//! side keeps its own nucleus (the DFlash convention,
//! `docs/plans/mtp-sampled-tighten-design-2026-06-23.md` §(a)), and the
//! residual is taken across the two supports. `p` is the distribution the
//! autoregressive producer samples, so the speculative stream is the AR
//! stream in distribution.
//!
//! [`accept_naive_prefix`] is the alternative verifier (SpecInfer naive
//! sampling): drafts stay the draft head's argmax, each verify row's target
//! token is drawn with the host AR sampler itself ([`naive_target_sampler`],
//! shared request-seeded RNG), and a draft is accepted iff it equals its
//! row's draw. Every emitted token is one AR draw in AR's order, so a seeded
//! request emits AR's exact tokens wherever the verify logits equal AR's.

use crate::llama::{CPU_SAMPLE_LEGACY_POOL, CPU_SAMPLE_WIDE_POOL};
use crate::sampler::SamplerConfig;
use crate::spec::{request_rng_state, GreedyAccept, SpecRequestConfig};

/// A request's truncation, resolved as the host AR sampler
/// (`sampler::sample_cpu` → [`crate::llama::sample_top_k_p`]) resolves it:
///
/// 1. `pool`: the highest finite logits gathered first — 20 when `top_k` is
///    absent or `1..=20`, 64 otherwise.
/// 2. `cap`: candidates kept — 20 when `top_k` is absent, `k` for `1..=64`,
///    64 for `0` or above 64 (`top_k = 0` is the 64-wide pool, not the whole
///    vocabulary).
/// 3. `min_p` (`0` disables): the cap shrinks to the first rank whose
///    probability is below `min_p` times the most probable token's.
/// 4. Nucleus at `top_p` over the capped candidates, boundary token kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSpec {
    pub temperature: f32,
    /// Clamped to `[0, 1]`.
    pub top_p: f32,
    pub pool: usize,
    /// In `1..=pool`.
    pub cap: usize,
    /// In `[0, 1]`.
    pub min_p: f32,
}

impl SampleSpec {
    /// Widest candidate pool any request gathers.
    pub const MAX_POOL: usize = CPU_SAMPLE_WIDE_POOL;

    /// The distribution the host AR sampler draws from for these request
    /// controls. `top_k` is the request's value as sent: `None` (20
    /// candidates) and `Some(0)` (64) differ. A non-finite or non-positive
    /// `min_p` disables the cut, as in the AR sampler.
    pub fn cpu_ar(temperature: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> Self {
        let (pool, cap) = match top_k {
            None => (CPU_SAMPLE_LEGACY_POOL, CPU_SAMPLE_LEGACY_POOL),
            Some(k) if (1..=CPU_SAMPLE_LEGACY_POOL as u32).contains(&k) => {
                (CPU_SAMPLE_LEGACY_POOL, k as usize)
            }
            Some(0) => (CPU_SAMPLE_WIDE_POOL, CPU_SAMPLE_WIDE_POOL),
            Some(k) => (CPU_SAMPLE_WIDE_POOL, (k as usize).min(CPU_SAMPLE_WIDE_POOL)),
        };
        Self {
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            pool,
            cap,
            min_p: if min_p.is_finite() && min_p > 0.0 {
                min_p.min(1.0)
            } else {
                0.0
            },
        }
    }

    fn check_temperature(self) -> Result<(), String> {
        if self.temperature > 0.0 && self.temperature.is_finite() {
            Ok(())
        } else {
            Err(format!(
                "sampled verification needs a positive temperature, got {}",
                self.temperature
            ))
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

    /// Build the target distribution `p` from a full logit row (row index =
    /// token id) with exactly the arithmetic of `llama::sample_top_k_p`: the
    /// same pool gather (and tie order), f32 softmax against the row maximum,
    /// stable descending sort, cap and min-p cut, summation order and nucleus
    /// boundary. Its two-pass draw (over the cut mass, then again within the
    /// nucleus) picks nucleus token `k` with probability `p_k / mass`, which
    /// is what this stores. A row with no finite mass is the AR sampler's
    /// argmax fallback, a point mass. `scratch` holds the pool between calls.
    pub fn build_from_logits(
        &mut self,
        logits: &[f32],
        spec: SampleSpec,
        scratch: &mut Vec<(u32, f32)>,
    ) -> Result<(), String> {
        spec.check_temperature()?;
        let pool = spec.pool.max(1);
        // `sample_pool`'s fixed slots: each finite logit above the current
        // minimum replaces it, then the minimum is rescanned.
        scratch.clear();
        scratch.resize(pool, (0, f32::NEG_INFINITY));
        let mut min_pos = 0usize;
        let mut min_val = f32::NEG_INFINITY;
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if !l.is_finite() {
                continue;
            }
            if l > max_logit {
                max_logit = l;
            }
            if l > min_val {
                scratch[min_pos] = (i as u32, l);
                min_val = f32::INFINITY;
                for (j, &(_, v)) in scratch.iter().enumerate() {
                    if v < min_val {
                        min_val = v;
                        min_pos = j;
                    }
                }
            }
        }
        let inv_temp = 1.0 / spec.temperature;
        let entries = &mut self.entries;
        entries.clear();
        let mut sum = 0.0f32;
        for &(token, l) in scratch.iter() {
            let p = if l.is_finite() {
                let p = ((l - max_logit) * inv_temp).exp();
                if p.is_finite() {
                    p
                } else {
                    0.0
                }
            } else {
                0.0
            };
            entries.push((token, p));
            sum += p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            entries.clear();
            entries.push((crate::llama::argmax(logits), 1.0));
            return Ok(());
        }
        // Stable insertion sort, descending: equal probabilities keep slot
        // order, as in the AR sampler.
        for i in 1..pool {
            let mut j = i;
            while j > 0 && entries[j].1 > entries[j - 1].1 {
                entries.swap(j, j - 1);
                j -= 1;
            }
        }
        let mut cap = spec.cap.clamp(1, pool);
        if spec.min_p > 0.0 {
            let floor = spec.min_p * entries[0].1;
            if let Some(cut) = (1..cap).find(|&i| entries[i].1 < floor) {
                cap = cut;
            }
        }
        // The uncut pool keeps the slot-order sum; a cut sums the kept prefix.
        if cap < pool {
            sum = entries[..cap].iter().map(|&(_, p)| p).sum();
        }
        let threshold = spec.top_p * sum;
        let mut cumulative = 0.0f32;
        let mut boundary = None;
        for (i, &(_, p)) in entries[..cap].iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                boundary = Some(i + 1);
                break;
            }
        }
        match boundary {
            Some(len) => {
                entries.truncate(len);
                for entry in entries.iter_mut() {
                    entry.1 /= cumulative;
                }
            }
            None => {
                // Rounding left the prefix short of the threshold: the AR
                // draw is `p_k / sum`, and its fall-through returns the top
                // token.
                entries.truncate(cap);
                for entry in entries.iter_mut() {
                    entry.1 /= sum;
                }
                entries[0].1 += (1.0 - cumulative / sum).max(0.0);
            }
        }
        entries.retain(|&(_, p)| p > 0.0);
        Ok(())
    }

    /// Build a draft distribution `q` from `(token, logit)` candidates (any
    /// order; non-finite logits are dropped; `candidates` is reordered) by
    /// the same steps over the candidates alone: sort by logit, keep
    /// `spec.cap`, softmax at the temperature, min-p, nucleus. `q` only has
    /// to be the distribution the draft is drawn from; exactness rests on `p`.
    /// Fails when no candidate is finite.
    pub fn build_from_candidates(
        &mut self,
        candidates: &mut Vec<(u32, f32)>,
        spec: SampleSpec,
    ) -> Result<(), String> {
        spec.check_temperature()?;
        candidates.retain(|(_, l)| l.is_finite());
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(spec.cap.max(1));
        let Some(&(_, max)) = candidates.first() else {
            return Err("sampled draft has no finite candidate logit".to_string());
        };
        let inv_temp = 1.0 / spec.temperature;
        self.entries.clear();
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
            return Err("sampled draft has no finite mass".to_string());
        }
        let threshold = spec.top_p * sum;
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

/// The host AR producer's sampler for a sampled request with neutral
/// penalties: what `sampler::sample_cpu` draws a Qwen4 AR token with.
/// `top_k` passes through as sent (absent and `Some(0)` differ); a
/// non-positive `min_p` is the AR sampler's absent.
pub fn naive_target_sampler(cfg: &SpecRequestConfig) -> SamplerConfig {
    SamplerConfig {
        temperature: cfg.temp,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        min_p: (cfg.min_p > 0.0).then_some(cfg.min_p),
        ..SamplerConfig::greedy()
    }
}

/// SpecInfer naive sampled verification, with the result shape and EOS rule
/// of [`crate::spec::accept_greedy_prefix`]. `draw(row)` returns verify row
/// `row`'s target draw; it is called in row order and only up to the row the
/// verdict ends on: the first draft that differs from its draw (the draw
/// replaces it), an accepted EOS draft (no bonus), or the bonus row
/// `drafts.len()`. Each emitted token is therefore exactly one draw, in
/// emission order.
pub fn accept_naive_prefix<F>(
    drafts: &[u32],
    eos: Option<u32>,
    mut draw: F,
) -> Result<GreedyAccept, String>
where
    F: FnMut(usize) -> Result<u32, String>,
{
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    for (row, &draft) in drafts.iter().enumerate() {
        let token = draw(row)?;
        committed.push(token);
        if token != draft || eos == Some(token) {
            return Ok(GreedyAccept {
                committed,
                accepted: row + usize::from(token == draft),
                hit_eos: eos == Some(token),
            });
        }
    }
    let bonus = draw(drafts.len())?;
    committed.push(bonus);
    Ok(GreedyAccept {
        committed,
        accepted: drafts.len(),
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

    /// Request `top_k`: absent, inside the legacy 20-wide pool, its edge,
    /// the 64-wide pool, its edge, and 0 (the 64-wide pool, not the vocab).
    const TOP_KS: [Option<u32>; 6] = [None, Some(5), Some(20), Some(40), Some(64), Some(0)];
    const MIN_PS: [f32; 3] = [0.0, 0.05, 0.1];

    struct Case {
        name: String,
        row: Vec<f32>,
        temp: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: f32,
    }

    impl Case {
        fn new(row: Vec<f32>, temp: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> Self {
            Self {
                name: format!("T{temp} top_p {top_p} top_k {top_k:?} min_p {min_p}"),
                row,
                temp,
                top_p,
                top_k,
                min_p,
            }
        }

        fn spec(&self) -> SampleSpec {
            SampleSpec::cpu_ar(self.temp, self.top_p, self.top_k, self.min_p)
        }

        fn target(&self) -> SparseDist {
            let mut p = SparseDist::default();
            p.build_from_logits(&self.row, self.spec(), &mut Vec::new())
                .unwrap();
            p
        }
    }

    /// The full `TOP_KS` x `MIN_PS` grid at T1.0 / top_p 0.95, other
    /// temperatures and nuclei, and a tied row on which the AR pool evicts a
    /// non-lowest id (20 equal logits fill the legacy pool, then a larger one
    /// replaces slot 0, so id 0 drops out while ids 20.. never enter).
    fn cases() -> Vec<Case> {
        let vocab = 300;
        let mut out = Vec::new();
        for (i, &top_k) in TOP_KS.iter().enumerate() {
            for (j, &min_p) in MIN_PS.iter().enumerate() {
                let row = logits(vocab, 1000 + (i * MIN_PS.len() + j) as u64, 2.5);
                out.push(Case::new(row, 1.0, 0.95, top_k, min_p));
            }
        }
        for (n, (temp, top_p, top_k, min_p)) in [
            (0.7, 0.8, None, 0.0),
            (1.0, 1.0, None, 0.0),
            (1.0, 1.0, Some(0), 0.0),
            (0.7, 0.9, Some(40), 0.05),
            (1.3, 1.0, Some(64), 0.1),
        ]
        .into_iter()
        .enumerate()
        {
            out.push(Case::new(
                logits(vocab, 2000 + n as u64, 2.5),
                temp,
                top_p,
                top_k,
                min_p,
            ));
        }
        let mut tied = vec![-4.0f32; vocab];
        tied[..30].fill(1.0);
        tied[25] = 2.0;
        for top_k in [None, Some(5)] {
            let mut case = Case::new(tied.clone(), 1.0, 1.0, top_k, 0.0);
            case.name.push_str(" (tied row)");
            out.push(case);
        }
        out
    }

    /// Chi-square (alpha 1e-6) and total-variation check of `counts`
    /// against `p`.
    fn assert_follows(name: &str, counts: &[u64], p: &SparseDist) {
        let expected = dense(p, counts.len());
        let (stat, df) = chi_square(counts, &expected, TRIALS);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(counts, &expected, TRIALS);
        eprintln!(
            "{name}: support={} chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4}",
            p.entries().len()
        );
        assert!(stat < crit, "{name}: chi2 {stat} >= {crit} (df {df})");
        assert!(tv < 0.01, "{name}: total variation {tv}");
    }

    /// `p` is the host AR sampler's distribution: 200k
    /// `llama::sample_top_k_p` draws per case follow it, over every request
    /// `top_k` (absent and 0 included) and `min_p`.
    #[test]
    fn target_is_the_host_ar_sampler_distribution() {
        // Draws from the process-global AR RNG; hold it for the whole run.
        let _rng = crate::llama::sampler_rng_test_guard();
        for (n, case) in cases().iter().enumerate() {
            let p = case.target();
            // Well-spread seeds: the AR xorshift32 maps close seeds to close
            // first draws.
            let seed = (SpecRng::new(n as u64 + 1).next_u64() >> 32) as u32;
            crate::llama::reset_cpu_sampler_rng(seed);
            let min_p = (case.min_p > 0.0).then_some(case.min_p);
            let mut counts = vec![0u64; case.row.len()];
            for _ in 0..TRIALS {
                let token = crate::llama::sample_top_k_p(
                    &case.row, case.temp, case.top_p, case.top_k, min_p,
                );
                counts[token as usize] += 1;
            }
            assert_follows(&format!("ar {}", case.name), &counts, &p);
        }
    }

    /// One-position exactness: draft from the 8-candidate re-scored `q`
    /// (built by the same truncation), verify against `p`, and the emitted
    /// token must follow `p` for every case; the acceptance rate must be
    /// its expectation `sum_x min(p, q)`.
    #[test]
    fn single_position_emits_target_distribution() {
        for (n, case) in cases().iter().enumerate() {
            let p = case.target();
            let q = rescored_draft(&case.row, 100 + n as u64, case.spec());
            let mut rng = SpecRng::new(7 + n as u64);
            let mut counts = vec![0u64; case.row.len()];
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
            assert_follows(&format!("spec {}", case.name), &counts, &p);
            let overlap: f64 = q
                .entries()
                .iter()
                .map(|&(t, qt)| (qt as f64).min(p.prob(t) as f64))
                .sum();
            let rate = accepted as f64 / TRIALS as f64;
            eprintln!("spec {}: accept={rate:.4} overlap={overlap:.4}", case.name);
            assert!(
                (rate - overlap).abs() < 0.01,
                "{}: accept {rate} vs {overlap}",
                case.name
            );
        }
    }

    /// Negative control: the verify of `single_position_emits_target_distribution`
    /// with the rejection replacement drawn from `p` instead of the residual
    /// `(p - q)+` must fail the same chi-square.
    #[test]
    fn residual_drawn_from_target_is_detected() {
        for (n, case) in cases().iter().enumerate().step_by(4) {
            let p = case.target();
            let q = rescored_draft(&case.row, 100 + n as u64, case.spec());
            let mut rng = SpecRng::new(7 + n as u64);
            let mut counts = vec![0u64; case.row.len()];
            for _ in 0..TRIALS {
                let x = q.sample(rng.next_f32());
                let qx = q.prob(x);
                let token = if qx > 0.0 && rng.next_f32() * qx < p.prob(x) {
                    x
                } else {
                    p.sample(rng.next_f32())
                };
                counts[token as usize] += 1;
            }
            let expected = dense(&p, counts.len());
            let (stat, df) = chi_square(&counts, &expected, TRIALS);
            let crit = chi_square_critical(df, 4.75);
            eprintln!(
                "control {}: chi2={stat:.1} df={df} crit(1e-6)={crit:.1}",
                case.name
            );
            assert!(
                stat > crit,
                "{}: a residual drawn from p went undetected (chi2 {stat} < {crit})",
                case.name
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
        let spec = SampleSpec::cpu_ar(1.0, 0.95, None, 0.0);
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

    #[test]
    fn truncation_keeps_boundary_order_and_request_top_k() {
        let logits = [0.0f32, 3.0, f32::NAN, 2.0, 1.0, f32::NEG_INFINITY];
        let mut d = SparseDist::default();
        let spec = SampleSpec::cpu_ar(1.0, 0.7, Some(3), 0.0);
        d.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        // softmax over {1: 3, 3: 2, 4: 1} = .665/.245/.090; top_p 0.7 keeps
        // token 1 and the boundary token 3.
        let tokens: Vec<u32> = d.entries().iter().map(|e| e.0).collect();
        assert_eq!(tokens, [1, 3]);
        let sum: f32 = d.entries().iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let mut d2 = SparseDist::default();
        d2.build_from_logits(
            &logits,
            SampleSpec::cpu_ar(1.0, 1.0, Some(3), 0.3),
            &mut Vec::new(),
        )
        .unwrap();
        // e^-1 = .37 >= .3 keeps token 3; e^-2 = .135 < .3 drops token 4.
        assert_eq!(d2.entries().iter().map(|e| e.0).collect::<Vec<_>>(), [1, 3]);
        // No finite logit: the AR sampler's argmax fallback, a point mass.
        let nan = [f32::NAN; 4];
        d.build_from_logits(&nan, spec, &mut Vec::new()).unwrap();
        assert_eq!(d.entries(), [(crate::llama::argmax(&nan), 1.0)]);
        // A flat 100-token row at top_p 1 keeps exactly the request's cap:
        // absent is 20, 0 is the 64-wide pool, larger values clamp to it.
        let row: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.01).collect();
        for (top_k, kept) in [
            (None, 20),
            (Some(7), 7),
            (Some(20), 20),
            (Some(21), 21),
            (Some(40), 40),
            (Some(64), 64),
            (Some(0), 64),
            (Some(1000), 64),
        ] {
            let spec = SampleSpec::cpu_ar(1.0, 1.0, top_k, 0.0);
            d.build_from_logits(&row, spec, &mut Vec::new()).unwrap();
            assert_eq!(d.entries().len(), kept, "top_k {top_k:?}");
        }
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

    /// Toy target for the naive-verify stream test: each context's logit
    /// row is a hash of its last two tokens, and EOS gains mass late so
    /// some streams end on it (as an accepted draft, a replacement or a
    /// bonus).
    const TOY_VOCAB: usize = 300;
    const TOY_EOS: u32 = 5;

    fn toy_row(ctx: &[u32]) -> Vec<f32> {
        let n = ctx.len();
        let key = ((ctx[n - 1] as u64) << 20) | ctx[n - 2] as u64;
        let mut row = logits(TOY_VOCAB, key ^ 0x70E5, 2.5);
        if n >= 40 {
            row[TOY_EOS as usize] += 4.0;
        }
        row
    }

    /// The draft head's argmax, wrong on about a quarter of the contexts
    /// (it picks the runner-up there).
    fn toy_draft(ctx: &[u32]) -> u32 {
        let row = toy_row(ctx);
        let mut ranked: Vec<u32> = (0..TOY_VOCAB as u32).collect();
        ranked.sort_unstable_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        let salt = ctx.len() as u64 ^ ((ctx[ctx.len() - 1] as u64) << 8);
        let miss = SpecRng::new(salt).next_u64() % 4 == 0;
        ranked[usize::from(miss)]
    }

    fn toy_draw(ctx: &[u32], sampler: &SamplerConfig) -> u32 {
        crate::sampler::sample_cpu(&mut toy_row(ctx), &[], sampler)
    }

    /// Naive verification emits the AR producer's exact stream at the same
    /// seed, over windows of every depth 1..=4 (the interleaved route is the
    /// one-draft case), under each request truncation the AR sampler honours.
    #[test]
    fn naive_verify_emits_the_seeded_ar_stream() {
        const TOKENS: usize = 64;
        let _rng = crate::llama::sampler_rng_test_guard();
        let prompt = [11u32, 42, 7];
        let (mut accepted, mut rejected, mut eos_ends) = (0usize, 0usize, 0usize);
        for (temp, top_p, top_k, min_p) in [
            (0.7, 0.8, Some(20), 0.0),
            (1.0, 0.95, None, 0.0),
            (0.8, 0.9, Some(0), 0.0),
            (0.7, 0.9, Some(40), 0.05),
        ] {
            let sampler = naive_target_sampler(&SpecRequestConfig {
                temp,
                top_p,
                top_k,
                min_p,
                ..SpecRequestConfig::default()
            });
            for seed in 1..=24u32 {
                let seed = (SpecRng::new(seed as u64).next_u64() >> 32) as u32;
                crate::llama::reset_cpu_sampler_rng(seed);
                let mut ar = prompt.to_vec();
                while ar.len() - prompt.len() < TOKENS {
                    let token = toy_draw(&ar, &sampler);
                    ar.push(token);
                    if token == TOY_EOS {
                        break;
                    }
                }
                let ar = &ar[prompt.len()..];
                eos_ends += usize::from(ar.last() == Some(&TOY_EOS));

                for depths in [[1usize, 1, 1, 1], [1, 3, 2, 4], [4, 4, 4, 4]] {
                    crate::llama::reset_cpu_sampler_rng(seed);
                    let mut ctx = prompt.to_vec();
                    // The prefill seed: the last prompt row's draw.
                    ctx.push(toy_draw(&ctx, &sampler));
                    let mut window = 0usize;
                    while ctx.len() - prompt.len() < TOKENS && ctx.last() != Some(&TOY_EOS) {
                        let mut block = ctx.clone();
                        let mut drafts = Vec::new();
                        for _ in 0..depths[window % depths.len()] {
                            let draft = toy_draft(&block);
                            drafts.push(draft);
                            block.push(draft);
                        }
                        window += 1;
                        let verdict = accept_naive_prefix(&drafts, Some(TOY_EOS), |row| {
                            let mut rows = ctx.clone();
                            rows.extend_from_slice(&drafts[..row]);
                            Ok(toy_draw(&rows, &sampler))
                        })
                        .unwrap();
                        accepted += verdict.accepted;
                        rejected += usize::from(
                            verdict.accepted < drafts.len()
                                && !(verdict.hit_eos
                                    && verdict.committed.len() == verdict.accepted),
                        );
                        ctx.extend_from_slice(&verdict.committed);
                    }
                    let mut spec = ctx[prompt.len()..].to_vec();
                    spec.truncate(TOKENS);
                    assert_eq!(
                        spec, ar,
                        "T{temp} top_p {top_p} top_k {top_k:?} min_p {min_p} seed {seed} depths {depths:?}"
                    );
                }
            }
        }
        assert!(
            accepted > 0 && rejected > 0,
            "accepted {accepted}, rejected {rejected}"
        );
        assert!(eos_ends > 0, "no stream reached EOS");
    }
}
