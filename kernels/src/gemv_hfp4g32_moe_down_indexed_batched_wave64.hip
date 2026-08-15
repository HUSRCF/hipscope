// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx90a/wave64 batched indexed MoE down GEMV for released DeepSeek-V4
// FP4 experts reframed as HFP4G32. The input is the plain post-SwiGLU
// activation; released weights are not FWHT-rotated.

extern "C" __global__ void gemv_hfp4g32_moe_down_residual_scaled_indexed_batched_wave64(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ gate_batch,
    float* __restrict__ x_residual,
    int M, int K, int K_TOP
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];

    if (expert_id < 0) return;

    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
    __syncthreads();

    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        gate_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = 16 + (K / 32) * 17;
    const char* hdr = A + (long long)row * row_bytes;
    const char* row_ptr = hdr + 16;
    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const float row_scale =
        (float)__builtin_bit_cast(_Float16, row_scale_bits);

    const int block_in_group = lane >> 2;
    const int nibble_offset = (lane & 3) * 4;
    const int thread_block_offset = block_in_group * 17;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define HFP4_DOG(pk, sc, b, a) \
        (a) += ((sc) * (float)lut[(pk) & 0xFu]) * x[(b)] \
             + ((sc) * (float)lut[((pk) >> 4) & 0xFu]) * x[(b) + 1] \
             + ((sc) * (float)lut[((pk) >> 8) & 0xFu]) * x[(b) + 2] \
             + ((sc) * (float)lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + ((sc) * (float)lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + ((sc) * (float)lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + ((sc) * (float)lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + ((sc) * (float)lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    #define HFP4_GROUP(g, a) do { \
        const char* gp = row_ptr + (long long)(g) * 136 + thread_block_offset; \
        const unsigned char e = *(const unsigned char*)gp; \
        const float sc = row_scale * \
            __builtin_bit_cast(float, ((unsigned int)e) << 23); \
        const unsigned int pk = \
            *(const unsigned int*)(gp + 1 + nibble_offset); \
        const int base = (g) * 256 + lane * 8; \
        HFP4_DOG(pk, sc, base, a); \
    } while (0)

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        HFP4_GROUP(g, acc0);
        HFP4_GROUP(g + 1, acc1);
        HFP4_GROUP(g + 2, acc2);
        HFP4_GROUP(g + 3, acc3);
    }
    const int tail_base = quads << 2;
    if (tail >= 1) HFP4_GROUP(tail_base, acc0);
    if (tail >= 2) HFP4_GROUP(tail_base + 1, acc1);
    if (tail >= 3) HFP4_GROUP(tail_base + 2, acc2);

    #undef HFP4_GROUP
    #undef HFP4_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, 32);

    if (lane == 0) {
        atomicAdd(
            &x_residual[(long long)bid * M + row],
            topk_weights[routing_base + krank] * acc
        );
    }
}
