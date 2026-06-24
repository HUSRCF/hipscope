// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched temperature-scaled softmax, out-of-place.
//
// For each row r (one block per row), computes
//
//     s_i   = logits[r*n + i] * inv_t
//     m     = max_i s_i
//     e_i   = expf(s_i - m)
//     probs[r*n + i] = e_i / sum_i e_i
//
// This MIRRORS the host `softmax_temp_into` in
// crates/hipfire-arch-qwen35/src/speculative.rs: temperature is applied to
// each logit FIRST (v * inv_t), the max is taken over the scaled logits, and
// exp is taken of (scaled - max). The arithmetic ORDER of the reduction is a
// tree reduction here vs. a sequential sum on the host, so the produced
// probabilities are DISTRIBUTION-parity, NOT byte-identical, to the host
// softmax (see HIPFIRE_DFLASH_FAST_SAMPLE gating in speculative.rs).
//
// Out-of-place: the input `logits` buffer is left untouched (the caller still
// needs the raw logits for argmax on the same rows).

#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_temp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int n,
    float inv_t
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // max over scaled logits
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // exp(scaled - max), accumulate sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// Variant that additionally emits the per-row nucleus (top_p) threshold and
// kept mass, WITHOUT modifying `probs` (host applies the truncation). For each
// row r it writes:
//
//   tau_cut[r] = the inclusive nucleus probability threshold: the value of the
//                crossing token's softmax probability. Host keeps tokens with
//                p_i >= tau_cut[r].
//   Z[r]       = the kept mass = sum of p_i for p_i >= tau_cut[r]. Host renorms
//                kept tokens by 1/Z[r].
//
// When top_p >= 1.0 the nucleus phase is skipped and tau_cut=0, Z=1 (host
// truncation collapses to identity, byte-path unchanged vs the no-top_p path).
//
// tau_cut is found by bisection over [0, p_max] on the monotone predicate
// mass(tau) = sum_{p_i >= tau} p_i, which increases as tau decreases. The AR
// nucleus rule (sort desc, accumulate, cut at the FIRST prefix whose cumulative
// >= top_p, INCLUSIVE crossing token, renorm by kept mass) is reproduced in
// threshold form: find the largest tau whose mass(tau) >= top_p. Because the
// full softmax already sums to 1, the kept-set sum used by AR is 1, so the
// cumulative target is exactly top_p. No sort: each bisection iteration is one
// strided pass over row_out + a block tree-reduce. A final exact pass recomputes
// Z = mass(tau_cut) from the converged threshold rather than trusting the
// interpolated mass. Comparator is `>=` (inclusive crossing token); exact float
// ties at the boundary are kept together (measure-zero divergence from AR's
// sort-position tie-break — see speculative.rs apply_topp_trunc doc).
extern "C" __global__ void softmax_temp_topp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    float* __restrict__ tau_cut,   // [rows] out
    float* __restrict__ Z,         // [rows] out
    int n,
    float inv_t,
    float top_p
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // ── Phase 1: max over scaled logits ────────────────────────────────────
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // ── Phase 2: exp(scaled - max), accumulate sum ─────────────────────────
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // ── Phase 3: normalize → row_out holds the full softmax p_i, sum == 1 ───
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
    __syncthreads();

    // ── top_p >= 1.0 guard: nucleus disabled → identity truncation ─────────
    if (top_p >= 1.0f) {
        if (threadIdx.x == 0) {
            tau_cut[row] = 0.0f;
            Z[row] = 1.0f;
        }
        return;
    }

    // ── Phase 4: p_max over the normalized row (bisection upper bound) ──────
    float pmax = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pmax = fmaxf(pmax, row_out[i]);
    }
    sdata[threadIdx.x] = pmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    pmax = sdata[0];
    __syncthreads();

    // ── Phase 5: bisect tau in [lo, hi] = [0, pmax] for the crossing value ──
    // Predicate: mass(tau) = sum_{p_i >= tau} p_i. mass decreases as tau rises.
    // We want the largest tau with mass(tau) >= top_p (the crossing token's
    // probability). Invariant: mass(lo) >= top_p, mass(hi) < top_p (when
    // top_p<1, mass(0)==1>=top_p and mass(pmax) is just the top token(s) mass
    // which may or may not satisfy — handled by the predicate directly).
    float lo = 0.0f;
    float hi = pmax;
    // ~30 iterations on the f32 range drives the interval below 1e-7 relative.
    for (int it = 0; it < 30; ++it) {
        float mid = 0.5f * (lo + hi);
        float m = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float p = row_out[i];
            if (p >= mid) m += p;
        }
        sdata[threadIdx.x] = m;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        float mass = sdata[0];
        __syncthreads();
        // mass(mid) >= top_p → the nucleus reaches at least mid; we can raise
        // the bar (keep fewer tokens). Else lower it (keep more).
        if (mass >= top_p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // lo is the largest tau with mass(lo) >= top_p → the inclusive crossing
    // threshold. Host keeps p_i >= lo.
    float tau = lo;

    // ── Phase 6: exact Z = mass(tau) from the converged threshold ──────────
    float kept = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row_out[i];
        if (p >= tau) kept += p;
    }
    sdata[threadIdx.x] = kept;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    kept = sdata[0];
    __syncthreads();

    if (threadIdx.x == 0) {
        tau_cut[row] = tau;
        Z[row] = kept;
    }
}
