// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Canonical EP4 finalization.  Keep the routed partial additions in the
// same order as the former sequence of three add_inplace_f32 launches, then
// add the completed routed value to the root's shared-expert output.
extern "C" __global__ void reduce4_add_shared_f32(
    float* __restrict__ shared,
    const float* __restrict__ p0,
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    const float* __restrict__ p3,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float routed = p0[i] + p1[i];
    routed = routed + p2[i];
    routed = routed + p3[i];
    shared[i] = shared[i] + routed;
}
