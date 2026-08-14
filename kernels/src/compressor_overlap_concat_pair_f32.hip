// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

// Pair form of compressor_overlap_concat_f32. grid.y selects KV or score,
// preserving the original per-element copy while eliminating one launch.
extern "C" __global__ void compressor_overlap_concat_pair_f32(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    int ratio,
    int head_dim
) {
    const int stream = (int)blockIdx.y;
    const int row = (int)blockIdx.x;
    const int d = (int)threadIdx.x;
    if (row >= 2 * ratio || d >= head_dim) return;

    const float* src = stream ? src1 : src0;
    float* dst = stream ? dst1 : dst0;
    const int src_col = row < ratio ? 0 : head_dim;
    dst[row * head_dim + d] =
        src[row * (2 * head_dim) + src_col + d];
}
