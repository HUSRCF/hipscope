// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HIPFIRE_MQ2_I8DOT_ROW_TILE
#define HIPFIRE_MQ2_I8DOT_ROW_TILE 8
#endif

#ifndef HIPFIRE_MQ2_I8DOT_KERNEL
#define HIPFIRE_MQ2_I8DOT_KERNEL \
    gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_row8_gfx90a
#endif

static_assert(HIPFIRE_MQ2_I8DOT_ROW_TILE % 8 == 0, "row tile must be a multiple of eight");

struct block_a8_g128 {
    float d;
    int sum;
    int8_t qs[128];
};

__device__ __forceinline__ unsigned int mq2_i8_selector_sg4(unsigned int indices) {
    const unsigned int t =
        (indices | (indices << 12)) & 0x000f000fu;
    return (t | (t << 6)) & 0x03030303u;
}

__device__ __forceinline__ int mq2_i8_lut4_sg4(
    unsigned int selector, unsigned int lut
) {
    return (int)__builtin_amdgcn_perm(lut, lut, selector);
}

__device__ __forceinline__ int mq2_i8_reduce_sum_4_dpp(int value) {
    value += __builtin_amdgcn_mov_dpp(value, 0x04e, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x0b1, 0xf, 0xf, true);
    return value;
}

// Sixteen 4-lane subgroups map to eight rows x two activation halves. Four
// independent integer accumulators hide dot4 dependencies; quad DPP completes
// each 4-lane reduction without LDS cross-lane traffic.
extern "C" __global__ __launch_bounds__(64)
void HIPFIRE_MQ2_I8DOT_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const block_a8_g128* __restrict__ xq,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int lane = threadIdx.x;
    const int lane4 = lane & 3;
    const int subgroup = lane >> 2;
    const int row_local = subgroup >> 1;
    const int half = subgroup & 1;
    const int row_start = blockIdx.x * HIPFIRE_MQ2_I8DOT_ROW_TILE;
    if (row_start >= M) return;

    const int krank = blockIdx.y;
    const char* weights = reinterpret_cast<const char*>(
        expert_ptrs[topk_indices[krank]]);
    const int groups = K / 256;
    const long long row_bytes = (long long)groups * 72;
    constexpr int ROW_BATCHES = HIPFIRE_MQ2_I8DOT_ROW_TILE / 8;
    float output[ROW_BATCHES] = {};

    __shared__ unsigned int selector_lut[256];
    #pragma unroll
    for (int i = lane; i < 256; i += 64) {
        selector_lut[i] = mq2_i8_selector_sg4((unsigned int)i);
    }
    __syncwarp();

    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        const block_a8_g128* xb = xq + 2 * g + half;

        #pragma unroll
        for (int batch = 0; batch < ROW_BATCHES; ++batch) {
            const int row = min(row_start + batch * 8 + row_local, M - 1);
            const char* gp = weights + (long long)row * row_bytes + g * 72;
            const unsigned int lut = *reinterpret_cast<const unsigned int*>(gp);
            int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;

            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
                const int offset = lane4 + chunk * 4;
                const int activation =
                    reinterpret_cast<const int*>(xb->qs)[offset];
                const unsigned int indices =
                    *(reinterpret_cast<const unsigned char*>(
                        gp + 8 + half * 32) + offset);
                const int weight =
                    mq2_i8_lut4_sg4(selector_lut[indices], lut);
                if ((chunk & 3) == 0) {
                    dot0 = __builtin_amdgcn_sdot4(activation, weight, dot0, false);
                } else if ((chunk & 3) == 1) {
                    dot1 = __builtin_amdgcn_sdot4(activation, weight, dot1, false);
                } else if ((chunk & 3) == 2) {
                    dot2 = __builtin_amdgcn_sdot4(activation, weight, dot2, false);
                } else {
                    dot3 = __builtin_amdgcn_sdot4(activation, weight, dot3, false);
                }
            }
            const int dot = mq2_i8_reduce_sum_4_dpp(
                (dot0 + dot1) + (dot2 + dot3));

            if (lane4 == 3) {
                const float2 scale_bias =
                    __half22float2(*reinterpret_cast<const __half2*>(gp + 4));
                output[batch] += xb->d * fmaf(
                    scale_bias.x, (float)dot, scale_bias.y * (float)xb->sum);
            }
        }
    }

    #pragma unroll
    for (int batch = 0; batch < ROW_BATCHES; ++batch) {
        output[batch] += __shfl_down(output[batch], 4, 8);
    }

    if (lane4 == 3 && half == 0) {
        const int split = M >> 1;
        #pragma unroll
        for (int batch = 0; batch < ROW_BATCHES; ++batch) {
            const int row = row_start + batch * 8 + row_local;
            if (row < M) {
                if (row < split) {
                    y_gate[(long long)krank * split + row] = output[batch];
                } else {
                    y_up[(long long)krank * split + row - split] = output[batch];
                }
            }
        }
    }
}
