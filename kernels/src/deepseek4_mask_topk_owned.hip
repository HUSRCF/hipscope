// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Keep globally-computed routing weights unchanged, but replace remote routes
// with a sentinel before the indexed expert kernels dereference their table.
extern "C" __global__ void deepseek4_mask_topk_owned_i32(
    int* __restrict__ topk_indices,
    const unsigned char* __restrict__ expert_owned,
    int k_top,
    int n_exp
) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= k_top) return;
    const int eid = topk_indices[slot];
    if (eid < 0 || eid >= n_exp || expert_owned[eid] == 0) {
        topk_indices[slot] = -1;
    }
}
