// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx90a research variant: one 512-thread block evaluates two control rows.
// Each x element and x^2 is loaded once for the pair while the total launched
// thread count remains 12*512 == 24*256 for n_ctrl=24.
extern "C" __global__ __launch_bounds__(512, 1)
void hc_compute_control_f32weights_pair_gfx90a(
    const float* __restrict__ x_flat,
    const float* __restrict__ w_fn,
    const float* __restrict__ base,
    float* __restrict__ c,
    int n_ctrl,
    int x_dim
) {
    const int ctrl0 = (int)blockIdx.x * 2;
    const int ctrl1 = ctrl0 + 1;
    const int tid = (int)threadIdx.x;
    if (ctrl0 >= n_ctrl) return;

    float dot0 = 0.0f;
    float dot1 = 0.0f;
    float sq = 0.0f;
    for (int d = tid; d < x_dim; d += 512) {
        const float x = x_flat[d];
        dot0 += x * w_fn[(size_t)ctrl0 * x_dim + d];
        if (ctrl1 < n_ctrl) {
            dot1 += x * w_fn[(size_t)ctrl1 * x_dim + d];
        }
        sq += x * x;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        dot0 += __shfl_down(dot0, offset);
        dot1 += __shfl_down(dot1, offset);
        sq += __shfl_down(sq, offset);
    }

    __shared__ float partial0[16];
    __shared__ float partial1[16];
    __shared__ float partial_sq[16];
    if ((tid & 31) == 0) {
        const int group = tid >> 5;
        partial0[group] = dot0;
        partial1[group] = dot1;
        partial_sq[group] = sq;
    }
    __syncthreads();

    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        float total_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            total0 += partial0[i];
            total1 += partial1[i];
            total_sq += partial_sq[i];
        }
        const float scale = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[ctrl0] = total0 * scale + base[ctrl0];
        if (ctrl1 < n_ctrl) {
            c[ctrl1] = total1 * scale + base[ctrl1];
        }
    }
}
