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

static_assert(HIPFIRE_MQ2_I8DOT_ROW_TILE % 4 == 0, "row tile must be a multiple of four");

struct block_a8_g128 {
    float d;
    int sum;
    int8_t qs[128];
};
static_assert(sizeof(block_a8_g128) == 136, "bad block_a8_g128 size");

__device__ __forceinline__ unsigned int mq2_i8_selector_sg8(unsigned int indices) {
    const unsigned int t =
        (indices | (indices << 12)) & 0x000f000fu;
    return (t | (t << 6)) & 0x03030303u;
}

__device__ __forceinline__ unsigned int mq2_i8_selector_direct4(unsigned int indices) {
    return (indices | (indices << 6) | (indices << 12) | (indices << 18))
        & 0x03030303u;
}

#if defined(HIPFIRE_MQ2_I8DOT_SELECTOR_CONST)
__device__ __constant__ unsigned int mq2_i8_selector_const[256] = {
    0x00000000u, 0x00000001u, 0x00000002u, 0x00000003u, 0x00000100u, 0x00000101u, 0x00000102u, 0x00000103u,
    0x00000200u, 0x00000201u, 0x00000202u, 0x00000203u, 0x00000300u, 0x00000301u, 0x00000302u, 0x00000303u,
    0x00010000u, 0x00010001u, 0x00010002u, 0x00010003u, 0x00010100u, 0x00010101u, 0x00010102u, 0x00010103u,
    0x00010200u, 0x00010201u, 0x00010202u, 0x00010203u, 0x00010300u, 0x00010301u, 0x00010302u, 0x00010303u,
    0x00020000u, 0x00020001u, 0x00020002u, 0x00020003u, 0x00020100u, 0x00020101u, 0x00020102u, 0x00020103u,
    0x00020200u, 0x00020201u, 0x00020202u, 0x00020203u, 0x00020300u, 0x00020301u, 0x00020302u, 0x00020303u,
    0x00030000u, 0x00030001u, 0x00030002u, 0x00030003u, 0x00030100u, 0x00030101u, 0x00030102u, 0x00030103u,
    0x00030200u, 0x00030201u, 0x00030202u, 0x00030203u, 0x00030300u, 0x00030301u, 0x00030302u, 0x00030303u,
    0x01000000u, 0x01000001u, 0x01000002u, 0x01000003u, 0x01000100u, 0x01000101u, 0x01000102u, 0x01000103u,
    0x01000200u, 0x01000201u, 0x01000202u, 0x01000203u, 0x01000300u, 0x01000301u, 0x01000302u, 0x01000303u,
    0x01010000u, 0x01010001u, 0x01010002u, 0x01010003u, 0x01010100u, 0x01010101u, 0x01010102u, 0x01010103u,
    0x01010200u, 0x01010201u, 0x01010202u, 0x01010203u, 0x01010300u, 0x01010301u, 0x01010302u, 0x01010303u,
    0x01020000u, 0x01020001u, 0x01020002u, 0x01020003u, 0x01020100u, 0x01020101u, 0x01020102u, 0x01020103u,
    0x01020200u, 0x01020201u, 0x01020202u, 0x01020203u, 0x01020300u, 0x01020301u, 0x01020302u, 0x01020303u,
    0x01030000u, 0x01030001u, 0x01030002u, 0x01030003u, 0x01030100u, 0x01030101u, 0x01030102u, 0x01030103u,
    0x01030200u, 0x01030201u, 0x01030202u, 0x01030203u, 0x01030300u, 0x01030301u, 0x01030302u, 0x01030303u,
    0x02000000u, 0x02000001u, 0x02000002u, 0x02000003u, 0x02000100u, 0x02000101u, 0x02000102u, 0x02000103u,
    0x02000200u, 0x02000201u, 0x02000202u, 0x02000203u, 0x02000300u, 0x02000301u, 0x02000302u, 0x02000303u,
    0x02010000u, 0x02010001u, 0x02010002u, 0x02010003u, 0x02010100u, 0x02010101u, 0x02010102u, 0x02010103u,
    0x02010200u, 0x02010201u, 0x02010202u, 0x02010203u, 0x02010300u, 0x02010301u, 0x02010302u, 0x02010303u,
    0x02020000u, 0x02020001u, 0x02020002u, 0x02020003u, 0x02020100u, 0x02020101u, 0x02020102u, 0x02020103u,
    0x02020200u, 0x02020201u, 0x02020202u, 0x02020203u, 0x02020300u, 0x02020301u, 0x02020302u, 0x02020303u,
    0x02030000u, 0x02030001u, 0x02030002u, 0x02030003u, 0x02030100u, 0x02030101u, 0x02030102u, 0x02030103u,
    0x02030200u, 0x02030201u, 0x02030202u, 0x02030203u, 0x02030300u, 0x02030301u, 0x02030302u, 0x02030303u,
    0x03000000u, 0x03000001u, 0x03000002u, 0x03000003u, 0x03000100u, 0x03000101u, 0x03000102u, 0x03000103u,
    0x03000200u, 0x03000201u, 0x03000202u, 0x03000203u, 0x03000300u, 0x03000301u, 0x03000302u, 0x03000303u,
    0x03010000u, 0x03010001u, 0x03010002u, 0x03010003u, 0x03010100u, 0x03010101u, 0x03010102u, 0x03010103u,
    0x03010200u, 0x03010201u, 0x03010202u, 0x03010203u, 0x03010300u, 0x03010301u, 0x03010302u, 0x03010303u,
    0x03020000u, 0x03020001u, 0x03020002u, 0x03020003u, 0x03020100u, 0x03020101u, 0x03020102u, 0x03020103u,
    0x03020200u, 0x03020201u, 0x03020202u, 0x03020203u, 0x03020300u, 0x03020301u, 0x03020302u, 0x03020303u,
    0x03030000u, 0x03030001u, 0x03030002u, 0x03030003u, 0x03030100u, 0x03030101u, 0x03030102u, 0x03030103u,
    0x03030200u, 0x03030201u, 0x03030202u, 0x03030203u, 0x03030300u, 0x03030301u, 0x03030302u, 0x03030303u
};
#endif

__device__ __forceinline__ int mq2_i8_lut4_sg8(
    unsigned int selector, unsigned int lut
) {
    return (int)__builtin_amdgcn_perm(lut, lut, selector);
}

__device__ __forceinline__ int mq2_i8_reduce_sum_8_gfx9(int value) {
    value += __builtin_amdgcn_ds_swizzle(value, 0x101f);
    value += __builtin_amdgcn_mov_dpp(value, 0x04e, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x0b1, 0xf, 0xf, true);
    return value;
}

// Eight 8-lane subgroups map to four rows x two activation halves. Each lane
// consumes 16 weights through four dot4 instructions before the 8-lane tree.
// This computes four rows concurrently while keeping one native wave64.
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
    const int lane8 = lane & 7;
    const int subgroup = lane >> 3;
    const int row_local = subgroup >> 1;
    const int half = subgroup & 1;
    const int row_start = blockIdx.x * HIPFIRE_MQ2_I8DOT_ROW_TILE;
    if (row_start >= M) return;

    const int krank = blockIdx.y;
    const char* weights = reinterpret_cast<const char*>(
        expert_ptrs[topk_indices[krank]]);
    const int groups = K / 256;
#if !defined(HIPFIRE_MQ2_I8DOT_TILED8)
    const long long row_bytes = (long long)groups * 72;
#else
    static_assert(HIPFIRE_MQ2_I8DOT_ROW_TILE == 8, "tiled8 requires row tile 8");
    const long long tile_bytes = (long long)groups * 8 * 72;
#endif
    constexpr int ROW_BATCHES = HIPFIRE_MQ2_I8DOT_ROW_TILE / 4;
    float output[ROW_BATCHES] = {};

#if !defined(HIPFIRE_MQ2_I8DOT_ABLATE_SELECTOR) \
    && !defined(HIPFIRE_MQ2_I8DOT_SELECTOR_DIRECT4) \
    && !defined(HIPFIRE_MQ2_I8DOT_SELECTOR_CONST)
    __shared__ unsigned int selector_lut[256];
    #pragma unroll
    for (int i = lane; i < 256; i += 64) {
        selector_lut[i] = mq2_i8_selector_sg8((unsigned int)i);
    }
    __syncwarp();
#endif

    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        const block_a8_g128* xb = xq + 2 * g + half;
#if defined(HIPFIRE_MQ2_I8DOT_TILED_A8)
        const int4 activations = reinterpret_cast<const int4*>(xb->qs)[lane8];
#endif

        #pragma unroll
        for (int batch = 0; batch < ROW_BATCHES; ++batch) {
            const int row = min(row_start + batch * 4 + row_local, M - 1);
#if defined(HIPFIRE_MQ2_I8DOT_TILED8)
            const char* group_tile =
                weights + (long long)(row_start / 8) * tile_bytes + g * 8 * 72;
            const char* gp = group_tile + (batch * 4 + row_local) * 8;
#else
            const char* gp = weights + (long long)row * row_bytes + g * 72;
#endif
            const unsigned int lut = *reinterpret_cast<const unsigned int*>(gp);
#if defined(HIPFIRE_MQ2_I8DOT_TILED8)
            const unsigned int packed_indices =
                *reinterpret_cast<const unsigned int*>(
                    group_tile + 64 + batch * 256
                    + (subgroup * 8 + lane8) * 4);
#endif
            int dot0 = 0;
            int dot1 = 0;

            #pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
#if defined(HIPFIRE_MQ2_I8DOT_TILED_A8)
                const int activation = reinterpret_cast<const int*>(&activations)[chunk];
#else
                const int offset = lane8 + chunk * 8;
                const int activation =
                    reinterpret_cast<const int*>(xb->qs)[offset];
#endif
#if defined(HIPFIRE_MQ2_I8DOT_TILED8)
                const unsigned int indices =
                    (packed_indices >> (chunk * 8)) & 0xffu;
#else
                const unsigned int indices =
                    *(reinterpret_cast<const unsigned char*>(
                        gp + 8 + half * 32) + offset);
#endif
#if defined(HIPFIRE_MQ2_I8DOT_ABLATE_SELECTOR)
                const int weight = (int)lut;
#elif defined(HIPFIRE_MQ2_I8DOT_SELECTOR_DIRECT4)
                const int weight = mq2_i8_lut4_sg8(
                    mq2_i8_selector_direct4(indices), lut);
#elif defined(HIPFIRE_MQ2_I8DOT_SELECTOR_CONST)
                const int weight = mq2_i8_lut4_sg8(
                    mq2_i8_selector_const[indices], lut);
#else
                const int weight =
                    mq2_i8_lut4_sg8(selector_lut[indices], lut);
#endif
                if ((chunk & 1) == 0) {
                    dot0 = __builtin_amdgcn_sdot4(
                        activation, weight, dot0, false);
                } else {
                    dot1 = __builtin_amdgcn_sdot4(
                        activation, weight, dot1, false);
                }
            }
#if defined(HIPFIRE_MQ2_I8DOT_ABLATE_REDUCE)
            const int dot = dot0 + dot1;
#else
            const int dot = mq2_i8_reduce_sum_8_gfx9(dot0 + dot1);
#endif

            if (lane8 == 7) {
#if defined(HIPFIRE_MQ2_I8DOT_SCALE_ONLY)
                const float scale = __half2float(*reinterpret_cast<const __half*>(gp + 4));
                output[batch] += xb->d * scale * (float)dot;
#elif defined(HIPFIRE_MQ2_I8DOT_ABLATE_EPILOGUE)
                output[batch] += (float)dot;
#else
                const float2 scale_bias =
                    __half22float2(*reinterpret_cast<const __half2*>(gp + 4));
                output[batch] += xb->d * fmaf(
                    scale_bias.x, (float)dot, scale_bias.y * (float)xb->sum);
#endif
            }
        }
    }

    #pragma unroll
    for (int batch = 0; batch < ROW_BATCHES; ++batch) {
        output[batch] += __shfl_down(output[batch], 8, 16);
    }

    if (lane8 == 7 && half == 0) {
        const int split = M >> 1;
        #pragma unroll
        for (int batch = 0; batch < ROW_BATCHES; ++batch) {
            const int row = row_start + batch * 4 + row_local;
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
