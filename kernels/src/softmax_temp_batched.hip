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
