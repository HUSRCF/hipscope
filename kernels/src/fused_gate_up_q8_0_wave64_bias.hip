// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// CDNA wave64 specialization of the twin Q8_0 decode projection.
// One wave computes two output rows: lanes 0..31 compute the first row and
// lanes 32..63 compute the second. The second projection optionally receives
// a per-row bias, used by DeepSeek V4's compressor score APE epilogue.
__launch_bounds__(64, 8)
extern "C" __global__ void fused_gate_up_q8_0_wave64_bias(
    const unsigned char* __restrict__ a_gate,
    const unsigned char* __restrict__ a_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    const float* __restrict__ up_bias,
    const int* __restrict__ pos_buf,
    int ratio, int gate_m, int up_m, int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = (int)blockIdx.x * 2 + (tid >> 5);
    const int total_m = gate_m + up_m;
    if (row >= total_m) return;

    const int nblocks = K / 32;
    const bool is_up = row >= gate_m;
    const int out_row = is_up ? row - gate_m : row;
    const unsigned char* row_data =
        (is_up ? a_up : a_gate) + (size_t)out_row * nblocks * 34;

    float sum = 0.0f;
    const int outer_iters = nblocks / 8;
    for (int oi = 0; oi < outer_iters; ++oi) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;
#pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            const unsigned char* block = base + sub * 34;
            const float d = (float)*((const _Float16*)block);
            const signed char qval = (signed char)block[2 + lane];
            sum += d * (float)qval * xb[sub * 32 + lane];
        }
    }
    for (int bi = outer_iters * 8; bi < nblocks; ++bi) {
        const unsigned char* block = row_data + bi * 34;
        const float d = (float)*((const _Float16*)block);
        const signed char qval = (signed char)block[2 + lane];
        sum += d * (float)qval * x[bi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);

    if (lane == 0) {
        if (is_up) {
            int bias_row = pos_buf[0] % ratio;
            if (bias_row < 0) bias_row += ratio;
            y_up[out_row] = sum + up_bias[(size_t)bias_row * up_m + out_row];
        } else {
            y_gate[out_row] = sum;
        }
    }
}
