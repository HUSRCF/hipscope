// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched DeepSeek V4 mHC alpha application for official FP32 controls.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_apply_alpha_batched_f32weights(
    float*       __restrict__ c,
    const float* __restrict__ alpha,
    const float* __restrict__ base,
    int batch_size
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int i = threadIdx.x;
    if (i >= 24) return;
    const int seg = i < 4 ? 0 : (i < 8 ? 1 : 2);
    const float a = alpha[seg];
    const size_t idx = (size_t)b * 24 + i;
    c[idx] = a * c[idx] + (1.0f - a) * base[i];
}
