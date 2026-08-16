// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Scalar-decode mHC postprocess. The pre/post lanes and the 4x4 comb lanes
// touch disjoint controls, so they can share one launch without a barrier.
extern "C" __global__ __launch_bounds__(32, 1)
void hc_pre_post_sinkhorn_4x4_f32(
    float* __restrict__ hc_c,
    float eps,
    float post_scale,
    int iters
) {
    const int tid = (int)threadIdx.x;
    if (blockIdx.x != 0) return;

    if (tid < 4) {
        const float z = hc_c[tid];
        hc_c[tid] = 1.0f / (1.0f + __expf(-z)) + eps;
    } else if (tid < 8) {
        const float z = hc_c[tid];
        hc_c[tid] = post_scale * (1.0f / (1.0f + __expf(-z)));
    }

    if (tid >= 16) return;

    const int idx = tid + 8;
    float v = hc_c[idx];

    float rm = v;
    rm = fmaxf(rm, __shfl_xor(rm, 1));
    rm = fmaxf(rm, __shfl_xor(rm, 2));
    v = __expf(v - rm);
    float rs = v;
    rs += __shfl_xor(rs, 1);
    rs += __shfl_xor(rs, 2);
    v = v / rs + eps;

    float cs = v;
    cs += __shfl_xor(cs, 4);
    cs += __shfl_xor(cs, 8);
    cs += eps;
    v = v / cs;

    for (int it = 0; it < iters - 1; ++it) {
        rs = v;
        rs += __shfl_xor(rs, 1);
        rs += __shfl_xor(rs, 2);
        rs += eps;
        v = v / rs;

        cs = v;
        cs += __shfl_xor(cs, 4);
        cs += __shfl_xor(cs, 8);
        cs += eps;
        v = v / cs;
    }

    hc_c[idx] = v;
}
