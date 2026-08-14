// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

// CDNA2 decode O-LoRA projection for Q8_0 wo_a.
//
// One native wave64 computes two output rows. Each lane loads one activation
// from a 64-element K tile and reuses it for both rows. This avoids the
// half-populated wave produced by the generic 32-thread kernel on gfx90a.
//
// Grid: [ceil(M / 2), B, G], block: [64, 1, 1].

#include <hip/hip_runtime.h>

__launch_bounds__(64, 8)
extern "C" __global__ void wo_per_group_batched_q8_0_wave64_row2_gfx90a(
    const unsigned char* __restrict__ wo_a,
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int G,
    int M,
    int K,
    int batch_size
) {
    const int row0 = (int)blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int b = (int)blockIdx.y;
    const int g = (int)blockIdx.z;
    if (row0 >= M || b >= batch_size || g >= G) return;

    const int lane = (int)threadIdx.x;
    const int q_lane = lane & 31;
    const int block_in_pair = lane >> 5;
    const int blocks_per_row = K / 32;
    const size_t row_stride = (size_t)blocks_per_row * 34;
    const size_t group_stride = (size_t)M * row_stride;
    const unsigned char* group_w = wo_a + (size_t)g * group_stride;
    const unsigned char* w0 = group_w + (size_t)row0 * row_stride;
    const unsigned char* w1 =
        group_w + (size_t)((row1 < M) ? row1 : row0) * row_stride;
    const float* x = x_in + ((size_t)b * G + g) * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int pairs = blocks_per_row / 2;
    for (int p = 0; p < pairs; ++p) {
        const int bi = p * 2 + block_in_pair;
        const unsigned char* q0 = w0 + (size_t)bi * 34;
        const unsigned char* q1 = w1 + (size_t)bi * 34;
        const float xv = x[(size_t)p * 64 + lane];
        const float d0 = (float)*((const _Float16*)q0);
        const float d1 = (float)*((const _Float16*)q1);
        const signed char v0 = (signed char)q0[2 + q_lane];
        const signed char v1 = (signed char)q1[2 + q_lane];
        acc0 += d0 * (float)v0 * xv;
        acc1 += d1 * (float)v1 * xv;
    }

    for (int offset = 32; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset, 64);
        acc1 += __shfl_down(acc1, offset, 64);
    }
    if (lane == 0) {
        const size_t out = ((size_t)b * G + g) * M;
        y_out[out + row0] = acc0;
        if (row1 < M) y_out[out + row1] = acc1;
    }
}
