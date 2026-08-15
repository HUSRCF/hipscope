// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched DeepSeek V4 mHC control-vector computation for official FP32
// hc_*_fn and hc_*_base tensors.

#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control_batched_f32weights(
    const float* __restrict__ x_flat,
    const float* __restrict__ w_fn,
    const float* __restrict__ base,
    float*       __restrict__ c,
    int n_ctrl,
    int x_dim,
    int batch_size
) {
    const int ctrl = blockIdx.x;
    const int b = blockIdx.y;
    if (ctrl >= n_ctrl || b >= batch_size) return;
    const int tid = threadIdx.x;
    const float* x_b = x_flat + (size_t)b * x_dim;

    float acc_dot = 0.0f;
    float acc_sq = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        const float xv = x_b[d];
        acc_dot += xv * w_fn[(size_t)ctrl * x_dim + d];
        acc_sq += xv * xv;
    }
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
        c[(size_t)b * n_ctrl + ctrl] = total_dot * rsqrt + base[ctrl];
    }
}
