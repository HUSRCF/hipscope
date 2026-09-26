// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        out[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

// bf16_round_trip_f32's rounding (RNE; non-finite values unchanged).
static __device__ __forceinline__ float silu_mul_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// silu_mul_f32 with gate and up round-tripped through BF16 first and the
// product round-tripped after: the round trip / silu_mul / round trip
// sequence in one pass, the same value per element.
extern "C" __global__ void silu_mul_bf16_rt_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = silu_mul_bf16_rt(gate[i]);
        out[i] = silu_mul_bf16_rt((v / (1.0f + expf(-v))) * silu_mul_bf16_rt(up[i]));
    }
}

// The BF16 shared-expert activation of one decode token in one launch:
// `out` is silu_mul_bf16_rt_f32(gate, up) and thread 0 also replaces the
// selector scalar with sigmoid(its BF16 round trip) — bitwise the separate
// round-trip, sigmoid_f32 and silu_mul_bf16_rt_f32 launches.
extern "C" __global__ void shared_expert_activation_bf16_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    float* __restrict__ selector,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        selector[0] = 1.0f / (1.0f + expf(-silu_mul_bf16_rt(selector[0])));
    }
    if (i < n) {
        float v = silu_mul_bf16_rt(gate[i]);
        out[i] = silu_mul_bf16_rt((v / (1.0f + expf(-v))) * silu_mul_bf16_rt(up[i]));
    }
}
