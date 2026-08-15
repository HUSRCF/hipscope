// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Graph-safe DeepSeek V4 compressor APE add. The graph retains the base
// pointers while pos_buf[0] is refreshed before every replay.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256, 4)
__global__ void compressor_add_ape_pos_buf_f32(
    float*       __restrict__ score,
    const float* __restrict__ ape,
    const int*   __restrict__ pos_buf,
    int ratio,
    int proj_dim
) {
    const int d = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (d >= proj_dim) return;

    int row = pos_buf[0] % ratio;
    if (row < 0) row += ratio;
    score[d] += ape[(size_t)row * (size_t)proj_dim + (size_t)d];
}
