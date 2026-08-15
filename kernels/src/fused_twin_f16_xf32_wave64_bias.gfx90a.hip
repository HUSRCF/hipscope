// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx90a wave64 twin F16xF32 projection. Each 32-lane half-wave computes one
// output row with the exact accumulation tree used by gemm_f16_tiled. The
// second projection receives a per-row F32 bias at writeback.
__launch_bounds__(64, 8)
extern "C" __global__ void fused_twin_f16_xf32_wave64_bias_gfx90a(
    const _Float16* __restrict__ w0,
    const _Float16* __restrict__ w1,
    const float* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    const float* __restrict__ bias1,
    const int* __restrict__ pos_buf,
    int ratio,
    int m0,
    int m1,
    int K
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int row = (int)blockIdx.x * 2 + (tid >> 5);
    const int total_m = m0 + m1;
    if (row >= total_m) return;

    const bool second = row >= m0;
    const int out_row = second ? row - m0 : row;
    const _Float16* w_row =
        (second ? w1 : w0) + (size_t)out_row * (size_t)K;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

    int k = lane;
    for (; k + 224 < K; k += 256) {
        sum0 += (float)w_row[k]       * x[k];
        sum1 += (float)w_row[k +  32] * x[k +  32];
        sum2 += (float)w_row[k +  64] * x[k +  64];
        sum3 += (float)w_row[k +  96] * x[k +  96];
        sum4 += (float)w_row[k + 128] * x[k + 128];
        sum5 += (float)w_row[k + 160] * x[k + 160];
        sum6 += (float)w_row[k + 192] * x[k + 192];
        sum7 += (float)w_row[k + 224] * x[k + 224];
    }
    for (; k < K; k += 32) {
        sum0 += (float)w_row[k] * x[k];
    }

    float sum = ((sum0 + sum1) + (sum2 + sum3))
              + ((sum4 + sum5) + (sum6 + sum7));
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);

    if (lane == 0) {
        if (second) {
            int bias_row = pos_buf[0] % ratio;
            if (bias_row < 0) bias_row += ratio;
            y1[out_row] = sum + bias1[(size_t)bias_row * m1 + out_row];
        } else {
            y0[out_row] = sum;
        }
    }
}
