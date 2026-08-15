// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC control-vector computation for the official source
// checkpoint, whose hc_*_fn and hc_*_base tensors are FP32.

#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control_f32weights(
    const float* __restrict__ x_flat,
    const float* __restrict__ w_fn,
    const float* __restrict__ base,
    float*       __restrict__ c,
    int n_ctrl,
    int x_dim
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    float acc_dot = 0.0f;
    float acc_sq = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        const float xv = x_flat[d];
        acc_dot += xv * w_fn[(size_t)ctrl * x_dim + d];
        acc_sq += xv * xv;
    }

    // Reduce independent logical 32-lane groups. This is valid on both
    // wave32 and wave64: only lanes whose id is a multiple of 32 publish.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_dot += __shfl_down(acc_dot, offset);
        acc_sq += __shfl_down(acc_sq, offset);
    }

    __shared__ float partial_dot[THREADS_PER_BLOCK / 32];
    __shared__ float partial_sq[THREADS_PER_BLOCK / 32];
    if (tid % 32 == 0) {
        partial_dot[tid / 32] = acc_dot;
        partial_sq[tid / 32] = acc_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total_dot = 0.0f;
        float total_sq = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) {
            total_dot += partial_dot[i];
            total_sq += partial_sq[i];
        }
        const float rsqrt = rsqrtf(total_sq / (float)x_dim + 1e-6f);
        c[ctrl] = total_dot * rsqrt + base[ctrl];
    }
}
