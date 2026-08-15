// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact stable top-512 selection for DeepSeek V4 eager Indexer decode on
// gfx90a. A sortable 64-bit composite key preserves the reference order:
// score descending, then index ascending. Radix selection finds the 512th
// key in 64 linear passes, after which the selected keys are bitonic-sorted
// in LDS. This replaces the O(N^2) per-candidate rank kernel at long context.

#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ uint64_t indexer_stable_key(float score, uint32_t index) {
    uint32_t bits = __float_as_uint(score);
    // The reference comparator treats +0 and -0 as equal. Canonicalize them
    // so the lower-index tie break remains authoritative.
    if ((bits << 1) == 0) bits = 0;
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (uint64_t(ordered) << 32) | uint64_t(0xffffffffu - index);
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

    __shared__ uint64_t prefix;
    __shared__ uint64_t prefix_mask;
    __shared__ uint64_t selected[K];
    __shared__ uint32_t remaining;
    __shared__ uint32_t bit_ones;
    __shared__ uint32_t selected_count;

    if (tid == 0) {
        prefix = 0;
        prefix_mask = 0;
        remaining = K;
    }
    __syncthreads();

    for (int bit = 63; bit >= 0; --bit) {
        if (tid == 0) bit_ones = 0;
        __syncthreads();

        uint32_t local = 0;
        const uint64_t bit_mask = uint64_t(1) << bit;
        for (int i = tid; i < n_iter; i += int(blockDim.x)) {
            const uint64_t key = indexer_stable_key(scores[i], uint32_t(i));
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

    if (tid == 0) selected_count = 0;
    __syncthreads();
    for (int i = tid; i < n_iter; i += int(blockDim.x)) {
        const uint64_t key = indexer_stable_key(scores[i], uint32_t(i));
        if (key >= prefix) {
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
