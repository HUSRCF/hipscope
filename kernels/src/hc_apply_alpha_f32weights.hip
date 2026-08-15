// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC alpha application for official FP32 controls.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_apply_alpha_f32weights(
    float*       __restrict__ c,
    const float* __restrict__ alpha,
    const float* __restrict__ base
) {
    if (blockIdx.x != 0) return;
    const int i = threadIdx.x;
    if (i >= 24) return;
    const int seg = i < 4 ? 0 : (i < 8 ? 1 : 2);
    const float a = alpha[seg];
    c[i] = a * c[i] + (1.0f - a) * base[i];
}
