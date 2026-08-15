// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HIPFIRE_MQ2_I8DOT_ROW_TILE
#define HIPFIRE_MQ2_I8DOT_ROW_TILE 2
#endif

#ifndef HIPFIRE_MQ2_I8DOT_KERNEL
#define HIPFIRE_MQ2_I8DOT_KERNEL \
    gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_row2_gfx90a
#endif

// One block stores one symmetric per-128 activation group. The FP32 scale and
// INT32 sum keep the first probe numerically strict; the transient footprint is
// negligible compared with routed-expert weights.
struct block_a8_g128 {
    float d;
    int sum;
    int8_t qs[128];
};
static_assert(sizeof(block_a8_g128) == 136, "bad block_a8_g128 size");

extern "C" __global__ __launch_bounds__(64)
void quantize_f32_a8_g128_wave64_gfx90a(
    const float* __restrict__ x,
    block_a8_g128* __restrict__ xq,
    int K
) {
    const int lane = threadIdx.x;
    const int group = blockIdx.x;
    const int base = group * 128 + lane * 2;
    if (base >= K) return;

    const float2 xv = reinterpret_cast<const float2*>(x + base)[0];
    float amax = fmaxf(fabsf(xv.x), fabsf(xv.y));
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down(amax, offset, 64));
    }
    amax = __shfl(amax, 0, 64);
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q0 = __float2int_rn(xv.x * inv_d);
    int q1 = __float2int_rn(xv.y * inv_d);
    q0 = max(-127, min(127, q0));
    q1 = max(-127, min(127, q1));

    int sum = q0 + q1;
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 64);
    }

    block_a8_g128* out = xq + group;
    const unsigned short packed =
        (unsigned short)((q0 & 0xff) | ((q1 & 0xff) << 8));
    reinterpret_cast<unsigned short*>(out->qs)[lane] = packed;
    if (lane == 0) {
        out->d = d;
        out->sum = sum;
    }
}

// Same values and block ABI, reordered so each SG8 lane owns four dwords.
extern "C" __global__ __launch_bounds__(64)
void quantize_f32_a8_g128_sg8_wave64_gfx90a(
    const float* __restrict__ x, block_a8_g128* __restrict__ xq, int K
) {
    const int lane = threadIdx.x;
    const int group = blockIdx.x;
    const int base = group * 128 + lane * 2;
    if (base >= K) return;
    const float2 xv = reinterpret_cast<const float2*>(x + base)[0];
    float amax = fmaxf(fabsf(xv.x), fabsf(xv.y));
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down(amax, offset, 64));
    amax = __shfl(amax, 0, 64);
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[2] = {
        max(-127, min(127, __float2int_rn(xv.x * inv_d))),
        max(-127, min(127, __float2int_rn(xv.y * inv_d))),
    };
    int sum = q[0] + q[1];
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 64);
    block_a8_g128* out = xq + group;
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int src_byte = lane * 2 + item;
        const int src_dword = src_byte >> 2;
        const int dst_dword = (src_dword & 7) * 4 + (src_dword >> 3);
        out->qs[dst_dword * 4 + (src_byte & 3)] = (int8_t)q[item];
    }
    if (lane == 0) { out->d = d; out->sum = sum; }
}

__device__ __forceinline__ unsigned int mq2_i8_selector(unsigned int indices) {
    const unsigned int t =
        (indices | (indices << 12)) & 0x000f000fu;
    return (t | (t << 6)) & 0x03030303u;
}

__device__ __forceinline__ int mq2_i8_lut4(unsigned int selector, unsigned int lut) {
    return (int)__builtin_amdgcn_perm(lut, lut, selector);
}

__device__ __forceinline__ int mq2_i8_reduce_sum_32_dpp(int value) {
    // All-reduce each 16-lane row, then row_bcast:15 folds row 0 into
    // row 1 and row 2 into row 3. Lanes 31 and 63 hold the two sums.
    value += __builtin_amdgcn_mov_dpp(value, 0x0b1, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x04e, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x124, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x128, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x142, 0xf, 0xf, true);
    return value;
}

// MQ2-I8DOT group layout remains 72 bytes:
//   [q0:q3 i8][scale f16][bias f16][64 B packed 2-bit indices].
// One native wave64 computes ROW_TILE rows. Its two 32-lane halves reduce the
// two per-128 activation subgroups independently and apply affine correction
// only at subgroup leaders.
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
    const int half_lane = lane & 31;
    const int half = lane >> 5;
    const int row_start = blockIdx.x * HIPFIRE_MQ2_I8DOT_ROW_TILE;
    if (row_start >= M) return;

    const int krank = blockIdx.y;
    const int expert_id = topk_indices[krank];
    if (expert_id < 0) {
        if (lane == 31) {
            const int split = M >> 1;
            #pragma unroll
            for (int r = 0; r < HIPFIRE_MQ2_I8DOT_ROW_TILE; ++r) {
                const int row = row_start + r;
                if (row < M) {
                    if (row < split) y_gate[(long long)krank * split + row] = 0.0f;
                    else y_up[(long long)krank * split + row - split] = 0.0f;
                }
            }
        }
        return;
    }
    const char* weights = reinterpret_cast<const char*>(
        expert_ptrs[expert_id]);
    const int groups = K / 256;
    const long long row_bytes = (long long)groups * 72;
    float output[HIPFIRE_MQ2_I8DOT_ROW_TILE] = {};

    // DPP keeps the integer reduction in VALU and converts to FP only at the
    // two 128-element subgroup leaders.
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        const block_a8_g128* xb = xq + 2 * g + half;
        const int activation = reinterpret_cast<const int*>(xb->qs)[half_lane];
        int dots[HIPFIRE_MQ2_I8DOT_ROW_TILE];

        #pragma unroll
        for (int r = 0; r < HIPFIRE_MQ2_I8DOT_ROW_TILE; ++r) {
            const int row = min(row_start + r, M - 1);
            const char* gp = weights + (long long)row * row_bytes + g * 72;
            const unsigned int lut = *reinterpret_cast<const unsigned int*>(gp);
            const unsigned int indices =
                *(reinterpret_cast<const unsigned char*>(gp + 8) + lane);
            const int weight = mq2_i8_lut4(mq2_i8_selector(indices), lut);
            dots[r] = mq2_i8_reduce_sum_32_dpp(
                __builtin_amdgcn_sdot4(activation, weight, 0, false));
        }

        if (half_lane == 31) {
            const float dx = xb->d;
            const float sum_a = (float)xb->sum;
            #pragma unroll
            for (int r = 0; r < HIPFIRE_MQ2_I8DOT_ROW_TILE; ++r) {
                const int row = min(row_start + r, M - 1);
                const char* gp = weights + (long long)row * row_bytes + g * 72;
                const float2 scale_bias =
                    __half22float2(*reinterpret_cast<const __half2*>(gp + 4));
                output[r] += dx * fmaf(
                    scale_bias.x, (float)dots[r], scale_bias.y * sum_a);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < HIPFIRE_MQ2_I8DOT_ROW_TILE; ++r) {
        output[r] += __shfl_down(output[r], 32, 64);
    }
    if (lane == 31) {
        const int split = M >> 1;
        #pragma unroll
        for (int r = 0; r < HIPFIRE_MQ2_I8DOT_ROW_TILE; ++r) {
            const int row = row_start + r;
            if (row < M) {
                if (row < split) {
                    y_gate[(long long)krank * split + row] = output[r];
                } else {
                    y_up[(long long)krank * split + row - split] = output[r];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(64)
void quantize_f32_a8_g128_sg8_batched_wave64_gfx90a(
    const float* __restrict__ x, block_a8_g128* __restrict__ xq, int K
) {
    const int lane = threadIdx.x;
    const int group = blockIdx.x;
    const int token = blockIdx.y;
    const int base = group * 128 + lane * 2;
    if (base >= K) return;
    x += (long long)token * K;
    xq += (long long)token * (K / 128);
    const float2 xv = reinterpret_cast<const float2*>(x + base)[0];
    float amax = fmaxf(fabsf(xv.x), fabsf(xv.y));
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down(amax, offset, 64));
    amax = __shfl(amax, 0, 64);
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[2] = {
        max(-127, min(127, __float2int_rn(xv.x * inv_d))),
        max(-127, min(127, __float2int_rn(xv.y * inv_d))),
    };
    int sum = q[0] + q[1];
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 64);
    block_a8_g128* out = xq + group;
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int src_byte = lane * 2 + item;
        const int src_dword = src_byte >> 2;
        const int dst_dword = (src_dword & 7) * 4 + (src_dword >> 3);
        out->qs[dst_dword * 4 + (src_byte & 3)] = (int8_t)q[item];
    }
    if (lane == 0) { out->d = d; out->sum = sum; }
}
