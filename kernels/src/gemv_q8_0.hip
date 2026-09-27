// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q8_0 GEMV v3: 32 threads, 8 blocks/iteration (256 elements), warp shuffle.
// Unrolled 8x to amortize loop overhead. Each thread processes 8 elements per iteration.
// One row's dot (valid in lane 0) from `row_data` (global or LDS).
static __device__ __forceinline__ float gemv_q8_0_row(
    const unsigned char* row_data,
    const float* __restrict__ x,
    int K,
    int tid
) {
    const int blocks_per_row = K / 32;
    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        // Unrolled: 8 blocks, each: load scale (f16→f32), load byte, FMA
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks (if K is not multiple of 256)
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    return sum;
}

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const float sum = gemv_q8_0_row(A_q8 + (size_t)row * (K / 32) * 34, x, K, threadIdx.x);
    if (threadIdx.x == 0) y[row] = sum;
}

// gemv_q8_0 for K = 2560 (2720-byte rows): the wave first copies its row into
// LDS with dword loads, every load before the first store (32 x 4 B per load
// instead of the row loop's 32 x 1 B quant loads and broadcast scale loads),
// then runs the same row loop on the copy.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0_k2560_staged(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
) {
    __shared__ __align__(16) unsigned int staged_row[680];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 2720);
    unsigned int v[22];
#pragma unroll
    for (int u = 0; u < 21; ++u) v[u] = src[tid + u * 32];
    v[21] = tid < 8 ? src[tid + 672] : 0u;
#pragma unroll
    for (int u = 0; u < 21; ++u) staged_row[tid + u * 32] = v[u];
    if (tid < 8) staged_row[tid + 672] = v[21];
    // One wave: its LDS accesses complete in order, so no workgroup fence.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const float sum = gemv_q8_0_row(
        reinterpret_cast<const unsigned char*>(staged_row), x, 2560, tid);
    if (tid == 0) y[row] = sum;
}
