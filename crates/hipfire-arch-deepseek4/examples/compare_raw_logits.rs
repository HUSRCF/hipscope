// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare two raw little-endian F32 logit vectors.

use std::cmp::Ordering;
use std::fs;

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{path}: {} bytes is not an F32 vector",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn logsumexp(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + values
        .iter()
        .map(|&v| ((v as f64) - max).exp())
        .sum::<f64>()
        .ln()
}

fn top_ids(values: &[f32], n: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    ids.truncate(n.min(ids.len()));
    ids
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("usage: {} REF.f32 CAND.f32", args[0]));
    }
    let reference = read_f32(&args[1])?;
    let candidate = read_f32(&args[2])?;
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(format!(
            "shape mismatch: ref={} candidate={}",
            reference.len(),
            candidate.len()
        ));
    }
    if reference.iter().chain(&candidate).any(|x| !x.is_finite()) {
        return Err("non-finite logits".to_string());
    }

    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&a, &b) in reference.iter().zip(&candidate) {
        let d = ((a as f64) - (b as f64)).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += d * d;
    }
    let n = reference.len() as f64;
    let lse_ref = logsumexp(&reference);
    let lse_cand = logsumexp(&candidate);
    let mut kld = 0.0f64;
    for (&a, &b) in reference.iter().zip(&candidate) {
        let log_p = a as f64 - lse_ref;
        let log_q = b as f64 - lse_cand;
        kld += log_p.exp() * (log_p - log_q);
    }
    if kld < -1e-10 {
        return Err(format!("negative KLD beyond roundoff: {kld}"));
    }
    let top_ref = top_ids(&reference, 8);
    let top_cand = top_ids(&candidate, 8);
    let same_set = top_ref.iter().all(|id| top_cand.contains(id));
    println!("n={}", reference.len());
    println!(
        "max_abs={max_abs:.9e} mean_abs={:.9e} rms={:.9e} kld_ref_to_cand={:.9e}",
        sum_abs / n,
        (sum_sq / n).sqrt(),
        kld.max(0.0)
    );
    println!(
        "argmax_ref={} argmax_cand={} same_argmax={} top8_same_set={} top8_same_order={}",
        top_ref[0],
        top_cand[0],
        top_ref[0] == top_cand[0],
        same_set,
        top_ref == top_cand
    );
    println!("top8_ref={top_ref:?}");
    println!("top8_cand={top_cand:?}");
    Ok(())
}
