// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

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

// One K = 2560 row (2720 bytes, 4-byte aligned): the wave first copies it into
// `staged_row` (LDS) with dword loads, every load before the first store (32 x
// 4 B per load instead of the row loop's 32 x 1 B quant loads and broadcast
// scale loads), then runs gemv_q8_0_row on the copy.
static __device__ __forceinline__ float gemv_q8_0_row_2560_staged(
    const unsigned char* __restrict__ row_data,
    unsigned int* staged_row,
    const float* __restrict__ x,
    int tid
) {
    const unsigned int* src = reinterpret_cast<const unsigned int*>(row_data);
    unsigned int v[22];
#pragma unroll
    for (int u = 0; u < 21; ++u) v[u] = src[tid + u * 32];
    v[21] = tid < 8 ? src[tid + 672] : 0u;
#pragma unroll
    for (int u = 0; u < 21; ++u) staged_row[tid + u * 32] = v[u];
    if (tid < 8) staged_row[tid + 672] = v[21];
    // One wave: its LDS accesses complete in order, so no workgroup fence.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    return gemv_q8_0_row(reinterpret_cast<const unsigned char*>(staged_row), x, 2560, tid);
}

// gemv_q8_0 for K = 2560 (2720-byte rows), staged through LDS.
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
    const float sum = gemv_q8_0_row_2560_staged(A_q8 + (size_t)row * 2720, staged_row, x, tid);
    if (tid == 0) y[row] = sum;
}

// Long-K rows (Qwen4 decode HC read down, K = 4 * 2560), as gemv_bf16_xf32_k4:
// four waves take a quarter of each row (staged like gemv_q8_0_k2560_staged
// when a quarter is 2560), quarter sums fold (q0 + q1) + (q2 + q3), and
// `hc_act` applies the HC activation (BF16 round trip, `scale`, round trip,
// SiLU, round trip). Grid [M], block 128, K % 128 == 0.
__launch_bounds__(128)
extern "C" __global__ void gemv_q8_0_k4(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ __align__(16) unsigned int staged[4][680];
    __shared__ float quarter[4];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int part = K / 4;
    const unsigned char* src = A_q8 + ((size_t)row * 4 + wave) * (part / 32) * 34;
    const float s = part == 2560
        ? gemv_q8_0_row_2560_staged(src, staged[wave], x + wave * part, lane)
        : gemv_q8_0_row(src, x + wave * part, part, lane);
    if (lane == 0) quarter[wave] = s;
    __syncthreads();
    if (threadIdx.x != 0) return;
    float value = (quarter[0] + quarter[1]) + (quarter[2] + quarter[3]);
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[row] = value;
}

// Requantize BF16 to Q8_0, one thread per 32-block: scale = f16(amax / 127),
// q = round(w / scale) with the stored (f16) scale.
extern "C" __global__ void quantize_bf16_q8_0(
    const unsigned short* __restrict__ w,
    unsigned char* __restrict__ out,
    int blocks
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = __uint_as_float((unsigned int)w[(size_t)b * 32 + i] << 16);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const _Float16 d16 = (_Float16)(amax / 127.0f);
    const float d = (float)d16;
    unsigned char* o = out + (size_t)b * 34;
    *reinterpret_cast<_Float16*>(o) = d16;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = d == 0.0f ? 0 : max(-127, min(127, (int)rintf(v[i] / d)));
        o[2 + i] = (unsigned char)(signed char)q;
    }
}
