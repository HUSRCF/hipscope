// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float mq3_dot4_gfx90a(
    const char* group, int lane, const float* codebook, const float* x
) {
    const unsigned char* indices =
        reinterpret_cast<const unsigned char*>(group + 16);
    const int bit = lane * 12;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    const unsigned int packed =
        (unsigned int)indices[byte] | ((unsigned int)indices[byte + 1] << 8);
    const unsigned int codes = packed >> shift;
    return codebook[codes & 7u] * x[0]
         + codebook[(codes >> 3) & 7u] * x[1]
         + codebook[(codes >> 6) & 7u] * x[2]
         + codebook[(codes >> 9) & 7u] * x[3];
}

// Native gfx90a wave64 MQ3-Lloyd GEMV. One complete wave computes one
// output row. Each lane decodes four of the 256 3-bit indices in a group.
extern "C" __global__ __launch_bounds__(64)
void gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_gfx90a(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int lane = threadIdx.x;
    const int expert_id = topk_indices[krank];
    if (expert_id < 0) {
        if (lane == 0) {
            const int mi = M >> 1;
            if (row < mi) y_gate[(long long)krank * mi + row] = 0.0f;
            else y_up[(long long)krank * mi + row - mi] = 0.0f;
        }
        return;
    }

    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups = K / 256;
    const char* row_ptr = A + (long long)row * groups * 112;
    __shared__ float codebooks[32];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups >> 2;

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 112;
        if (lane < 32) {
            const __half* cb =
                reinterpret_cast<const __half*>(gp0 + (lane >> 3) * 112);
            codebooks[lane] = __half2float(cb[lane & 7]);
        }
        __syncwarp();
        const int base = g * 256 + lane * 4;
        acc0 += mq3_dot4_gfx90a(gp0, lane, codebooks, x + base);
        acc1 += mq3_dot4_gfx90a(gp0 + 112, lane, codebooks + 8, x + base + 256);
        acc2 += mq3_dot4_gfx90a(gp0 + 224, lane, codebooks + 16, x + base + 512);
        acc3 += mq3_dot4_gfx90a(gp0 + 336, lane, codebooks + 24, x + base + 768);
        __syncwarp();
    }

    for (int g = quads << 2; g < groups; ++g) {
        const char* gp = row_ptr + g * 112;
        if (lane < 8)
            codebooks[lane] =
                __half2float(reinterpret_cast<const __half*>(gp)[lane]);
        __syncwarp();
        acc0 += mq3_dot4_gfx90a(
            gp, lane, codebooks, x + g * 256 + lane * 4);
        __syncwarp();
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 32; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[(long long)krank * mi + row] = acc;
        else y_up[(long long)krank * mi + row - mi] = acc;
    }
}
