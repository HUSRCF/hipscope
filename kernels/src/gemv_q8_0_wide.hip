// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q8_0 GEMV, wide variant: 2 rows per block, one row per warp. Used on
// all archs for large-M Q8_0 GEMVs in Qwen3.5's attention Q8 path.
//
// Per-row math uses 4 accumulators over 8-wide interleaved blocks, with
// a final pairwise combine `acc = (acc0 + acc1) + (acc2 + acc3)`. This
// matches the interleaved summation pattern used by the HFQ4 kernels
// (see gemv_hfq4g256.gfx1100.hip for the full writeup on why FP
// associativity makes that specific ordering required for byte-exact
// cross-arch output against the MQ4 quality-gate baselines).
//
// Before the port, the kernel used a single-accumulator sequential sum
// with an 8-block unroll hint:
//
//     float sum = 0.0f;
//     for (bi = 0; bi + 7 < blocks_per_row; bi += 8) {
//         #pragma unroll
//         for (u = 0; u < 8; u++) { sum += d * qval * x; }
//     }
//
// The 8-way unroll was a compiler ILP hint, not an algorithmic decision,
// so the compiler was free to schedule the additions in any order. On
// gfx1010 / gfx1013 (BC-250 Cyan Skillfish) this produced a different
// reduction order than the gfx1100 baseline runs.

extern "C" __global__ void gemv_q8_0_wide(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const unsigned char* bp0 = row_data + (bi + 0) * 34;
        const unsigned char* bp1 = row_data + (bi + 1) * 34;
        const unsigned char* bp2 = row_data + (bi + 2) * 34;
        const unsigned char* bp3 = row_data + (bi + 3) * 34;

        float d0 = (float)*((const _Float16*)bp0);
        float d1 = (float)*((const _Float16*)bp1);
        float d2 = (float)*((const _Float16*)bp2);
        float d3 = (float)*((const _Float16*)bp3);

        signed char q0 = (signed char)bp0[2 + lane];
        signed char q1 = (signed char)bp1[2 + lane];
        signed char q2 = (signed char)bp2[2 + lane];
        signed char q3 = (signed char)bp3[2 + lane];

        acc0 += d0 * (float)q0 * x[(bi + 0) * 32 + lane];
        acc1 += d1 * (float)q1 * x[(bi + 1) * 32 + lane];
        acc2 += d2 * (float)q2 * x[(bi + 2) * 32 + lane];
        acc3 += d3 * (float)q3 * x[(bi + 3) * 32 + lane];
    }

    // Tail blocks accumulate into acc[bi % 4].
    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const unsigned char* bp = row_data + bi * 34;
        float d = (float)*((const _Float16*)bp);
        signed char qv = (signed char)bp[2 + lane];
        float contrib = d * (float)qv * x[bi * 32 + lane];
        if (t == 0) acc0 += contrib;
        else if (t == 1) acc1 += contrib;
        else if (t == 2) acc2 += contrib;
    }

    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// gfx1151, K = 640 (680-byte rows, Qwen4 decode shared-expert down): the same
// four-accumulator math on a row staged through LDS with dword loads issued
// before the x loads, x (20 values per lane) held in registers, four rows
// (waves) per block. Bitwise gemv_q8_0_wide; 10.5 -> 9.5 us per call.
__launch_bounds__(128)
extern "C" __global__ void gemv_q8_0_wide_k640_staged(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
) {
    __shared__ __align__(16) unsigned int staged[4][170];
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + wave;
    if (row >= M) return;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 680);
    unsigned int v[6];
#pragma unroll
    for (int u = 0; u < 5; ++u) v[u] = src[lane + u * 32];
    v[5] = lane < 10 ? src[lane + 160] : 0u;
    float xr[20];
#pragma unroll
    for (int b = 0; b < 20; ++b) xr[b] = x[b * 32 + lane];
#pragma unroll
    for (int u = 0; u < 5; ++u) staged[wave][lane + u * 32] = v[u];
    if (lane < 10) staged[wave][lane + 160] = v[5];
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const unsigned char* data = reinterpret_cast<const unsigned char*>(staged[wave]);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        const unsigned char* bp = data + q * 4 * 34;
        acc0 += (float)*((const _Float16*)bp) * (float)(signed char)bp[2 + lane] * xr[q * 4 + 0];
        acc1 += (float)*((const _Float16*)(bp + 34)) * (float)(signed char)bp[36 + lane] * xr[q * 4 + 1];
        acc2 += (float)*((const _Float16*)(bp + 68)) * (float)(signed char)bp[70 + lane] * xr[q * 4 + 2];
        acc3 += (float)*((const _Float16*)(bp + 102)) * (float)(signed char)bp[104 + lane] * xr[q * 4 + 3];
    }
    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// gemv_q8_0_wide over R activation rows (x row stride K, y row stride M):
// each weight row's blocks are read once and folded into every row's four
// accumulators in the single-row kernel's order, so each output is bitwise
// gemv_q8_0_wide's (speculative verify must match single-token decode).
template <int R>
static __device__ __forceinline__ void gemv_q8_0_wide_rows_body(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row, int lane) {
    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;
    float acc0[R], acc1[R], acc2[R], acc3[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;
    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const unsigned char* bp0 = row_data + (bi + 0) * 34;
        const unsigned char* bp1 = row_data + (bi + 1) * 34;
        const unsigned char* bp2 = row_data + (bi + 2) * 34;
        const unsigned char* bp3 = row_data + (bi + 3) * 34;
        const float d0 = (float)*((const _Float16*)bp0);
        const float d1 = (float)*((const _Float16*)bp1);
        const float d2 = (float)*((const _Float16*)bp2);
        const float d3 = (float)*((const _Float16*)bp3);
        const signed char q0 = (signed char)bp0[2 + lane];
        const signed char q1 = (signed char)bp1[2 + lane];
        const signed char q2 = (signed char)bp2[2 + lane];
        const signed char q3 = (signed char)bp3[2 + lane];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float* xr = x + (size_t)r * K;
            acc0[r] += d0 * (float)q0 * xr[(bi + 0) * 32 + lane];
            acc1[r] += d1 * (float)q1 * xr[(bi + 1) * 32 + lane];
            acc2[r] += d2 * (float)q2 * xr[(bi + 2) * 32 + lane];
            acc3[r] += d3 * (float)q3 * xr[(bi + 3) * 32 + lane];
        }
    }
    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float contrib = d * (float)qv * x[(size_t)r * K + bi * 32 + lane];
            if (t == 0) acc0[r] += contrib;
            else if (t == 1) acc1[r] += contrib;
            else if (t == 2) acc2[r] += contrib;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float sum = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
        if (lane == 0) y[(size_t)r * M + row] = sum;
    }
}

extern "C" __global__ void gemv_q8_0_wide_rows(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int rows
) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + (threadIdx.x >> 5);
    if (row >= M) return;
    switch (rows) {
#define WIDE_ROWS(R) \
    case R: gemv_q8_0_wide_rows_body<R>(A_q8, x, y, M, K, row, lane); return;
    WIDE_ROWS(1) WIDE_ROWS(2) WIDE_ROWS(3) WIDE_ROWS(4) WIDE_ROWS(5) WIDE_ROWS(6) WIDE_ROWS(7) WIDE_ROWS(8)
#undef WIDE_ROWS
    }
}
