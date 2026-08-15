// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Native gfx90a wave64 F16 per-group O-LoRA projection. Each 32-lane
// half-wave computes one output row, preserving gemm_f16_tiled's exact
// eight-accumulator tree while using all 64 lanes of the CDNA2 wave.
//
// wo_a:  [G, M, K] F16
// x_in:  [B, G, K] F32
// y_out: [B, G, M] F32
__launch_bounds__(64, 8)
extern "C" __global__ void wo_per_group_batched_f16_wave64_row2_gfx90a(
    const _Float16* __restrict__ wo_a,
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int G,
    int M,
    int K,
    int batch_size
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 2 + (tid >> 5);
    const int b = (int)blockIdx.y;
    const int g = (int)blockIdx.z;
    if (r >= M || b >= batch_size || g >= G) return;

    const _Float16* w_row = wo_a + ((size_t)g * M + r) * K;
    const float* x_row = x_in + ((size_t)b * G + g) * K;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

    int k = lane;
    for (; k + 224 < K; k += 256) {
        sum0 += (float)w_row[k]       * x_row[k];
        sum1 += (float)w_row[k +  32] * x_row[k +  32];
        sum2 += (float)w_row[k +  64] * x_row[k +  64];
        sum3 += (float)w_row[k +  96] * x_row[k +  96];
        sum4 += (float)w_row[k + 128] * x_row[k + 128];
        sum5 += (float)w_row[k + 160] * x_row[k + 160];
        sum6 += (float)w_row[k + 192] * x_row[k + 192];
        sum7 += (float)w_row[k + 224] * x_row[k + 224];
    }
    for (; k < K; k += 32) {
        sum0 += (float)w_row[k] * x_row[k];
    }

    float sum = ((sum0 + sum1) + (sum2 + sum3))
              + ((sum4 + sum5) + (sum6 + sum7));
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }
    if (lane == 0) {
        y_out[((size_t)b * G + g) * M + r] = sum;
    }
}
