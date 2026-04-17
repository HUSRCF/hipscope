//! CASK: core-aware selective KV compression (arXiv:2604.10900).
//!
//! Sits atop TriAttention. TriAttention scores tokens and keeps the top-B;
//! CASK reinterprets the bottom tail as *mergeable scratch* rather than
//! discard, then folds groups of `m` scratch tokens into single
//! representative slots via weighted-average K and V.
//!
//! Paper §2.1: split the cache into protected **core** (top-α·B tokens,
//! never merged) and **mergeable scratch** (next m·(1-α)·B tokens by score,
//! folded into (1-α)·B groups). Output size equals `budget`, same as
//! TriAttention, but effective coverage is α·B + m·(1-α)·B tokens.
//!
//! v1 is Q8-only, L2-grouped, softmax-weighted merge on CPU. asym2/3/4
//! and GPU-side merge are follow-ups — at the cadence we evict, per-layer
//! CPU fold is <1 ms and not a wall-clock factor.
//!
//! Weight choice: `a_i = exp(z_i)` where z_i is the same z-score +
//! max-GQA aggregation TriAttention uses for top-B. The paper offers
//! "score mass, similarity, or position-aware" as options; softmax over
//! score is the score-mass variant with a numerically stable normalizer.

use crate::llama::{f16_to_f32, f32_to_f16, KvCache};
use crate::triattn::EvictionCtx;
use hip_bridge::HipResult;
use rdna_compute::Gpu;

/// Core-Aware Selective KV Compression policy.
///
/// Wraps a TriAttention `EvictionCtx` — scoring, kernels, scratch all
/// reused. Adds only the CASK post-processing step: split retained
/// tokens into core vs scratch, greedy-group scratch by L2-K similarity,
/// fold groups via weighted avg, re-quantize back into the cache.
pub struct CaskCtx {
    pub base: EvictionCtx,
    /// Fraction of budget reserved for singleton core tokens. `core_frac = 1.0`
    /// degenerates to plain TriAttention; `core_frac = 0.0` folds every kept
    /// slot into an m-group.
    pub core_frac: f32,
    /// Merge group size. 2 is the conservative default (1.5× coverage at
    /// α=0.5); 4 gives 2.5× coverage but relies more on within-group
    /// similarity holding up.
    pub fold_m: usize,
}

impl CaskCtx {
    pub fn new(base: EvictionCtx, core_frac: f32, fold_m: usize) -> Self {
        assert!((0.0..=1.0).contains(&core_frac), "core_frac must be in [0, 1]");
        assert!(fold_m >= 2, "fold_m must be >= 2 (use plain TriAttention for m=1)");
        Self { base, core_frac, fold_m }
    }

    pub fn eviction_count(&self) -> usize {
        self.base.eviction_count.get()
    }

    /// Same trigger + return convention as `EvictionCtx::maybe_evict`.
    /// Falls back to plain TriAttention for non-Q8 modes in v1.
    pub fn maybe_evict(
        &self,
        gpu: &mut Gpu,
        kv: &mut KvCache,
        current_physical: usize,
    ) -> HipResult<Option<usize>> {
        if current_physical < self.base.budget + self.base.beta {
            return Ok(None);
        }

        // v1: CASK merge only for Q8 (block layout: 34 B per 32 elements,
        // trivial dequant). For asym2/3/4 modes fall back to plain top-B
        // eviction — still benefits from TriAttention scoring.
        if !kv.quant_q8 {
            return self.base.maybe_evict(gpu, kv, current_physical);
        }

        let absolute_pos = current_physical + kv.compact_offset;
        let p_q = absolute_pos as f32;

        // Budget math: output always has exactly `budget` slots (c core + s merged,
        // c + s = budget). Input constraint: c + m*s ≤ physical.
        // Solve for max merge_slots s = min(target_merge, (physical-budget)/(m-1)).
        // When physical == budget + beta (threshold), s = beta/(m-1) at m=2 → s=beta.
        let budget = self.base.budget;
        let target_core = (budget as f32 * self.core_frac).floor() as usize;
        let target_merge = budget - target_core;
        let merge_slots = if self.fold_m > 1 && current_physical > budget {
            let max_by_input = (current_physical - budget) / (self.fold_m - 1);
            target_merge.min(max_by_input)
        } else {
            0
        };
        let core_slots = budget - merge_slots;
        let merge_pool = merge_slots * self.fold_m;
        // Sanity: we consume exactly core_slots + merge_pool = budget + merge_slots*(m-1) tokens.
        debug_assert!(core_slots + merge_pool <= current_physical);

        let n_kv = self.base.n_kv_heads;
        let d = self.base.head_dim;
        let n_blocks = d / 32;
        let row_bytes = n_kv * n_blocks * 34;

        for (fa_i, &layer_idx) in self.base.fa_layer_ids.iter().enumerate() {
            // 1. TriAttention scoring.
            let offset = fa_i * self.base.centers_per_layer;
            let centers_layer = self.base.centers_dev
                .sub_offset(offset, self.base.centers_per_layer);
            gpu.triattn_score_q8(
                &kv.k_gpu[layer_idx], &centers_layer, &self.base.scores_buf,
                self.base.n_heads, self.base.n_kv_heads, self.base.head_dim,
                self.base.n_rot, self.base.rope_theta, p_q, current_physical,
            )?;
            gpu.hip.device_synchronize()?;
            let scores = gpu.download_f32(&self.base.scores_buf)?;

            // 2. Aggregate: per-head z-score, then max across heads.
            let agg = aggregate_scores(
                &scores[..self.base.n_heads * current_physical],
                self.base.n_heads, current_physical,
            );

            // 3. Rank tokens; top `core_slots` = core, next `merge_pool` = scratch.
            let mut ranked: Vec<(f32, usize)> =
                agg.iter().copied().enumerate().map(|(i, s)| (s, i)).collect();
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let core_ranked = &ranked[..core_slots];
            let scratch_ranked = &ranked[core_slots..core_slots + merge_pool];

            let core_idx: Vec<u32> = core_ranked.iter().map(|(_, i)| *i as u32).collect();
            let scratch_idx: Vec<u32> = scratch_ranked.iter().map(|(_, i)| *i as u32).collect();
            let scratch_scores: Vec<f32> = scratch_ranked.iter().map(|(s, _)| *s).collect();

            // 4. Download full layer K and V — small (<1 MB per layer) and
            //    simpler than a gather kernel for v1.
            let mut k_all = vec![0u8; current_physical * row_bytes];
            let mut v_all = vec![0u8; current_physical * row_bytes];
            gpu.hip.memcpy_dtoh(&mut k_all, &kv.k_gpu[layer_idx].buf)?;
            gpu.hip.memcpy_dtoh(&mut v_all, &kv.v_gpu[layer_idx].buf)?;

            // 5. Group scratch tokens by L2-K similarity.
            let mut merged_k = vec![0u8; merge_slots * row_bytes];
            let mut merged_v = vec![0u8; merge_slots * row_bytes];
            let mut merged_pos = Vec::with_capacity(merge_slots);

            if merge_slots > 0 {
                let groups = greedy_group_by_l2(
                    &k_all, &scratch_idx, n_kv, d, self.fold_m,
                );
                // 6. Fold each group: softmax over z-scores weights a weighted
                //    avg of dequantized K and V, then requant block-wise.
                for (slot, group) in groups.iter().enumerate() {
                    let abs_positions: Vec<u32> = group.iter().map(|&gi| scratch_idx[gi]).collect();
                    let raw_scores: Vec<f32> = group.iter().map(|&gi| scratch_scores[gi]).collect();
                    let weights = softmax(&raw_scores);

                    let k_row = weighted_avg_q8(&k_all, &abs_positions, &weights, n_kv, d);
                    let v_row = weighted_avg_q8(&v_all, &abs_positions, &weights, n_kv, d);
                    merged_k[slot * row_bytes..(slot + 1) * row_bytes].copy_from_slice(&k_row);
                    merged_v[slot * row_bytes..(slot + 1) * row_bytes].copy_from_slice(&v_row);

                    // Centroid position: weighted mean for temporal ordering.
                    let centroid: f32 = abs_positions.iter().zip(weights.iter())
                        .map(|(&p, &w)| p as f32 * w).sum();
                    merged_pos.push(centroid as u32);
                }
            }

            // 7. Assemble final cache in temporal order. Each slot is either a
            //    verbatim core row or a merged row; positional order matters
            //    only for later-token RoPE relative-phase consistency (keys are
            //    already post-RoPE so their own phases are frozen).
            let output_len = core_slots + merge_slots;
            let mut order: Vec<(u32, SlotSrc)> = Vec::with_capacity(output_len);
            for (i, &pos) in core_idx.iter().enumerate() {
                order.push((pos, SlotSrc::Core(i)));
            }
            for (i, &pos) in merged_pos.iter().enumerate() {
                order.push((pos, SlotSrc::Merged(i)));
            }
            order.sort_by_key(|&(p, _)| p);

            let mut final_k = vec![0u8; output_len * row_bytes];
            let mut final_v = vec![0u8; output_len * row_bytes];
            for (new_slot, &(_, src)) in order.iter().enumerate() {
                let dst_rng = new_slot * row_bytes..(new_slot + 1) * row_bytes;
                match src {
                    SlotSrc::Core(i) => {
                        let pos = core_idx[i] as usize;
                        final_k[dst_rng.clone()].copy_from_slice(&k_all[pos * row_bytes..(pos + 1) * row_bytes]);
                        final_v[dst_rng].copy_from_slice(&v_all[pos * row_bytes..(pos + 1) * row_bytes]);
                    }
                    SlotSrc::Merged(i) => {
                        final_k[dst_rng.clone()].copy_from_slice(&merged_k[i * row_bytes..(i + 1) * row_bytes]);
                        final_v[dst_rng].copy_from_slice(&merged_v[i * row_bytes..(i + 1) * row_bytes]);
                    }
                }
            }

            gpu.hip.memcpy_htod(&kv.k_gpu[layer_idx].buf, &final_k)?;
            gpu.hip.memcpy_htod(&kv.v_gpu[layer_idx].buf, &final_v)?;
        }

        // Output size is always `budget` slots (core_slots + merge_slots = budget).
        kv.compact_offset += current_physical - budget;
        self.base.eviction_count.set(self.base.eviction_count.get() + 1);
        Ok(Some(budget))
    }
}

#[derive(Copy, Clone)]
enum SlotSrc {
    Core(usize),
    Merged(usize),
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Per-head z-score, then max across heads. Same aggregation TriAttention
/// uses for top-B but returned directly so we can rank-by-score.
pub fn aggregate_scores(scores: &[f32], n_heads: usize, seq_len: usize) -> Vec<f32> {
    let mut agg = vec![f32::NEG_INFINITY; seq_len];
    for h in 0..n_heads {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let mean: f32 = row.iter().sum::<f32>() / seq_len as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
            / seq_len as f32;
        let std = var.sqrt().max(1e-6);
        for p in 0..seq_len {
            let z = (row[p] - mean) / std;
            if z > agg[p] { agg[p] = z; }
        }
    }
    agg
}

/// Stable softmax over a small vector.
pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Greedy L2-NN grouping: process scratch tokens in input order (which is
/// arbitrary — caller passes scratch sorted by score desc). For each
/// ungrouped anchor, pick the (m-1) nearest-L2 ungrouped neighbors in
/// dequantized K space. Returns groups as indices into the input array.
pub fn greedy_group_by_l2(
    k_all: &[u8],
    scratch_idx: &[u32],
    n_kv: usize,
    head_dim: usize,
    m: usize,
) -> Vec<Vec<usize>> {
    let n = scratch_idx.len();
    let n_blocks = head_dim / 32;
    let row_bytes = n_kv * n_blocks * 34;
    let feat_dim = n_kv * head_dim;

    let mut feats = vec![0f32; n * feat_dim];
    for (i, &pos) in scratch_idx.iter().enumerate() {
        let row = &k_all[pos as usize * row_bytes..(pos as usize + 1) * row_bytes];
        dequant_q8_row(row, &mut feats[i * feat_dim..(i + 1) * feat_dim], n_kv, head_dim);
    }

    let mut used = vec![false; n];
    let n_groups = n / m;
    let mut groups = Vec::with_capacity(n_groups);

    for anchor in 0..n {
        if used[anchor] || groups.len() == n_groups { continue; }
        used[anchor] = true;
        let mut group = vec![anchor];
        let anchor_feat = &feats[anchor * feat_dim..(anchor + 1) * feat_dim];

        // Collect distances to all unused candidates.
        let mut cands: Vec<(f32, usize)> = (0..n)
            .filter(|&j| !used[j])
            .map(|j| {
                let fj = &feats[j * feat_dim..(j + 1) * feat_dim];
                let d2: f32 = anchor_feat.iter().zip(fj)
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                (d2, j)
            })
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(_, j) in cands.iter().take(m - 1) {
            used[j] = true;
            group.push(j);
        }
        if group.len() == m {
            groups.push(group);
        }
    }
    groups
}

/// Dequantize a single Q8_0 row to flat f32. `out.len() == n_kv * head_dim`.
pub fn dequant_q8_row(row: &[u8], out: &mut [f32], n_kv: usize, head_dim: usize) {
    let n_blocks = head_dim / 32;
    for h in 0..n_kv {
        for b in 0..n_blocks {
            let block_off = (h * n_blocks + b) * 34;
            let scale = f16_to_f32(u16::from_le_bytes([row[block_off], row[block_off + 1]]));
            for q in 0..32 {
                let v = row[block_off + 2 + q] as i8;
                out[h * head_dim + b * 32 + q] = scale * (v as f32);
            }
        }
    }
}

/// Weighted average of multiple Q8_0 rows, requantized per-block.
/// Output layout matches a single Q8_0 row: `n_kv * n_blocks * 34` bytes.
pub fn weighted_avg_q8(
    all_rows: &[u8],
    positions: &[u32],
    weights: &[f32],
    n_kv: usize,
    head_dim: usize,
) -> Vec<u8> {
    assert_eq!(positions.len(), weights.len());
    let n_blocks = head_dim / 32;
    let row_bytes = n_kv * n_blocks * 34;
    let feat_dim = n_kv * head_dim;

    let mut acc = vec![0f32; feat_dim];
    let mut deq = vec![0f32; feat_dim];
    for (i, &pos) in positions.iter().enumerate() {
        let w = weights[i];
        let row = &all_rows[pos as usize * row_bytes..(pos as usize + 1) * row_bytes];
        dequant_q8_row(row, &mut deq, n_kv, head_dim);
        for j in 0..feat_dim {
            acc[j] += w * deq[j];
        }
    }

    let mut out = vec![0u8; row_bytes];
    for h in 0..n_kv {
        for b in 0..n_blocks {
            let block_off = (h * n_blocks + b) * 34;
            let slice = &acc[h * head_dim + b * 32..h * head_dim + b * 32 + 32];
            let max_abs = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
            let s16 = f32_to_f16(scale);
            out[block_off] = (s16 & 0xFF) as u8;
            out[block_off + 1] = (s16 >> 8) as u8;
            for q in 0..32 {
                let qv = (slice[q] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                out[block_off + 2 + q] = qv as u8;
            }
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_q8_row(vals: &[f32], n_kv: usize, head_dim: usize) -> Vec<u8> {
        let n_blocks = head_dim / 32;
        let row_bytes = n_kv * n_blocks * 34;
        let mut out = vec![0u8; row_bytes];
        for h in 0..n_kv {
            for b in 0..n_blocks {
                let block_off = (h * n_blocks + b) * 34;
                let slice = &vals[h * head_dim + b * 32..h * head_dim + b * 32 + 32];
                let max_abs = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 127.0;
                let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
                let s16 = f32_to_f16(scale);
                out[block_off] = (s16 & 0xFF) as u8;
                out[block_off + 1] = (s16 >> 8) as u8;
                for q in 0..32 {
                    let qv = (slice[q] * inv).round().clamp(-127.0, 127.0) as i8;
                    out[block_off + 2 + q] = qv as u8;
                }
            }
        }
        out
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn softmax_numerical_stability_large_inputs() {
        let s = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax overflowed on large z");
    }

    #[test]
    fn aggregate_z_score_max_gqa() {
        // 2 heads, 4 positions. Head 0: [1,2,3,4], Head 1: [4,3,2,1].
        // Per-head z-score then max-across-heads should give position 0 and 3
        // the highest aggregate (extremes on one head each).
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let agg = aggregate_scores(&scores, 2, 4);
        assert!(agg[0] > agg[1]);
        assert!(agg[3] > agg[2]);
        assert!((agg[0] - agg[3]).abs() < 1e-5, "symmetric case, endpoints should tie");
    }

    #[test]
    fn dequant_requant_q8_near_exact() {
        let n_kv = 2;
        let head_dim = 32;
        let vals: Vec<f32> = (0..n_kv * head_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let packed = pack_q8_row(&vals, n_kv, head_dim);
        let mut back = vec![0f32; n_kv * head_dim];
        dequant_q8_row(&packed, &mut back, n_kv, head_dim);
        for i in 0..vals.len() {
            assert!((vals[i] - back[i]).abs() < 0.02, "dequant drift at {}: {} vs {}", i, vals[i], back[i]);
        }
    }

    #[test]
    fn weighted_avg_of_identical_rows_is_identity() {
        let n_kv = 2;
        let head_dim = 32;
        let vals: Vec<f32> = (0..n_kv * head_dim).map(|i| (i as f32) * 0.05).collect();
        let row = pack_q8_row(&vals, n_kv, head_dim);
        let mut all = Vec::new();
        for _ in 0..3 { all.extend_from_slice(&row); }

        let merged = weighted_avg_q8(&all, &[0, 1, 2], &[0.2, 0.3, 0.5], n_kv, head_dim);
        let mut back = vec![0f32; n_kv * head_dim];
        dequant_q8_row(&merged, &mut back, n_kv, head_dim);
        for i in 0..vals.len() {
            assert!((vals[i] - back[i]).abs() < 0.02, "identity merge drift at {}", i);
        }
    }

    #[test]
    fn weighted_avg_of_two_orthogonal_is_blend() {
        let n_kv = 1;
        let head_dim = 32;
        let a: Vec<f32> = (0..head_dim).map(|i| if i < 16 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..head_dim).map(|i| if i >= 16 { 1.0 } else { 0.0 }).collect();
        let mut all = pack_q8_row(&a, n_kv, head_dim);
        all.extend_from_slice(&pack_q8_row(&b, n_kv, head_dim));

        // Equal weights: merged ≈ [0.5, 0.5, …]
        let merged = weighted_avg_q8(&all, &[0, 1], &[0.5, 0.5], n_kv, head_dim);
        let mut back = vec![0f32; head_dim];
        dequant_q8_row(&merged, &mut back, n_kv, head_dim);
        for i in 0..head_dim {
            assert!((back[i] - 0.5).abs() < 0.05, "blend drift at {}: {}", i, back[i]);
        }
    }

    #[test]
    fn greedy_group_pairs_nearby_tokens() {
        // Two clusters: tokens 0,1 similar; tokens 2,3 similar (different cluster).
        // Expect groups {0,1} and {2,3} (or swap), never {0,2} or {0,3}.
        let n_kv = 1;
        let head_dim = 32;
        let t0: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin() * 0.5).collect();
        let mut t1 = t0.clone();
        t1[0] += 0.01; // near-duplicate
        let t2: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 1.3).cos() * 0.5).collect();
        let mut t3 = t2.clone();
        t3[0] += 0.01;

        let mut all = Vec::new();
        for v in [&t0, &t1, &t2, &t3] {
            all.extend_from_slice(&pack_q8_row(v, n_kv, head_dim));
        }

        let scratch_idx: Vec<u32> = vec![0, 1, 2, 3];
        let groups = greedy_group_by_l2(&all, &scratch_idx, n_kv, head_dim, 2);
        assert_eq!(groups.len(), 2);
        // Each group should have indices both from cluster A (0,1) OR both from B (2,3).
        for g in &groups {
            let all_low = g.iter().all(|&i| i < 2);
            let all_high = g.iter().all(|&i| i >= 2);
            assert!(all_low || all_high, "group crossed clusters: {:?}", g);
        }
    }
}
