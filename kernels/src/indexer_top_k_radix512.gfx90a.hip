// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact stable top-512 selection for DeepSeek V4 eager Indexer decode on
// gfx90a. A sortable 64-bit composite key preserves the reference order:
// score descending, then index ascending. Radix selection finds the threshold
// score in 32 linear passes, a tie scan resolves the stable index cutoff, and
// the selected composite keys are bitonic-sorted in LDS. This replaces the
// O(N^2) per-candidate rank kernel at long context.

#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ uint32_t indexer_score_key(float score) {
    uint32_t bits = __float_as_uint(score);
    // The reference comparator treats +0 and -0 as equal. Canonicalize them
    // so the lower-index tie break remains authoritative.
    if ((bits << 1) == 0) bits = 0;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

static __device__ __forceinline__ uint64_t indexer_stable_key(float score, uint32_t index) {
    return (uint64_t(indexer_score_key(score)) << 32) | uint64_t(0xffffffffu - index);
}

extern "C" __global__ __launch_bounds__(256, 1)
void indexer_top_k_radix512_gfx90a(
    const float* __restrict__ scores,
    int* __restrict__ top_indices,
    int n_stride,
    int n_iter
) {
    constexpr int K = 512;
    const int tid = int(threadIdx.x);

    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;
    __shared__ uint64_t selected[K];
    __shared__ uint32_t remaining;
    __shared__ uint32_t bit_ones;
    __shared__ uint32_t selected_count;
    __shared__ uint32_t tie_count;
    __shared__ uint32_t tie_limit;

    if (tid == 0) {
        prefix = 0;
        prefix_mask = 0;
        remaining = K;
    }
    __syncthreads();

    // Select the threshold SCORE in 32 radix passes. The old implementation
    // included the 32-bit index tiebreak in this loop, doubling the global
    // score scans. Once the score threshold is known, one linear scan can
    // resolve the stable lower-index tie break exactly.
    for (int bit = 31; bit >= 0; --bit) {
        if (tid == 0) bit_ones = 0;
        __syncthreads();

        uint32_t local = 0;
        const uint32_t bit_mask = uint32_t(1) << bit;
        for (int i = tid; i < n_iter; i += int(blockDim.x)) {
            const uint32_t key = indexer_score_key(scores[i]);
            local += ((key & prefix_mask) == prefix) && ((key & bit_mask) != 0);
        }
        if (local) atomicAdd(&bit_ones, local);
        __syncthreads();

        if (tid == 0) {
            if (bit_ones >= remaining) {
                prefix |= bit_mask;
            } else {
                remaining -= bit_ones;
            }
            prefix_mask |= bit_mask;
        }
        __syncthreads();
    }

    if (tid == 0) {
        tie_count = 0;
        tie_limit = 0xffffffffu;
    }
    __syncthreads();
    for (int i = tid; i < n_iter; i += int(blockDim.x)) {
        if (indexer_score_key(scores[i]) == prefix) {
            atomicAdd(&tie_count, 1u);
            atomicMin(&tie_limit, uint32_t(i));
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (remaining == tie_count) {
            tie_limit = 0xffffffffu;
        } else if (remaining > 1) {
            uint32_t needed = remaining;
            for (int i = 0; i < n_iter; ++i) {
                if (indexer_score_key(scores[i]) == prefix && --needed == 0) {
                    tie_limit = uint32_t(i);
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) selected_count = 0;
    __syncthreads();
    for (int i = tid; i < n_iter; i += int(blockDim.x)) {
        const uint32_t score_key = indexer_score_key(scores[i]);
        const uint64_t key = indexer_stable_key(scores[i], uint32_t(i));
        if (score_key > prefix || (score_key == prefix && uint32_t(i) <= tie_limit)) {
            const uint32_t slot = atomicAdd(&selected_count, 1u);
            if (slot < K) selected[slot] = key;
        }
    }
    __syncthreads();

    for (int width = 2; width <= K; width <<= 1) {
        for (int distance = width >> 1; distance > 0; distance >>= 1) {
            for (int i = tid; i < K; i += int(blockDim.x)) {
                const int peer = i ^ distance;
                if (peer > i) {
                    const uint64_t a = selected[i];
                    const uint64_t b = selected[peer];
                    const bool ascending = (i & width) == 0;
                    if ((a > b) == ascending) {
                        selected[i] = b;
                        selected[peer] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int out = tid; out < K; out += int(blockDim.x)) {
        const uint32_t low = uint32_t(selected[K - 1 - out]);
        top_indices[out] = int(~low);
    }
    (void)n_stride;
}
