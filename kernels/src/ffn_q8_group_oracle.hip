// SPDX-License-Identifier: Apache-2.0
// Diagnostic-only oracle for dynamic dense-FFN group reduction.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// One workgroup ranks the rotated, quantized 256-channel groups for one token
// by reconstructed L2 energy. Dropped groups are zeroed in place. This is an
// oracle quality probe: it does not reduce gate/up work or down-GEMM work.
extern "C" __launch_bounds__(256, 1)
__global__ void ffn_q8_group_oracle_topk(
    block_q8_1_mmq* __restrict__ xq,
    int K,
    int N,
    int keep_groups
) {
    constexpr int max_groups = 256;
    __shared__ float energies[max_groups];

    const int token = blockIdx.x;
    const int group = threadIdx.x;
    const int groups = K / 256;
    if (token >= N || groups > max_groups) return;

    float energy = 0.0f;
    if (group < groups) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const block_q8_1_mmq* block =
                xq + (long long)(2 * group + half) * N + token;
            #pragma unroll
            for (int subgroup = 0; subgroup < 4; ++subgroup) {
                int sum_q2 = 0;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int q = (int)block->qs[32 * subgroup + i];
                    sum_q2 += q * q;
                }
                const float d = __half22float2(block->ds4[subgroup]).x;
                energy += d * d * (float)sum_q2;
            }
        }
        energies[group] = energy;
    }
    __syncthreads();

    if (group >= groups) return;
    int rank = 0;
    #pragma unroll 1
    for (int other = 0; other < groups; ++other) {
        const float candidate = energies[other];
        rank += candidate > energy || (candidate == energy && other < group);
    }
    if (rank < keep_groups) return;

    const uint4 zero = make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint4* block = reinterpret_cast<uint4*>(
            xq + (long long)(2 * group + half) * N + token);
        #pragma unroll
        for (int word = 0; word < 9; ++word) block[word] = zero;
    }
}
