// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void sigmoid_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// sigmoid_f32 of x round-tripped through BF16 first (bf16_round_trip_f32's
// RNE rounding; non-finite values unchanged): the round trip + sigmoid pair.
extern "C" __global__ void sigmoid_bf16_in_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        if (isfinite(v)) {
            const unsigned int u = __float_as_uint(v);
            v = __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
        }
        x[i] = 1.0f / (1.0f + expf(-v));
    }
}
