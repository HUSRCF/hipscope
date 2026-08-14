// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

// Pair form of state_overlap_shift_f32_buf. grid.y selects KV or score.
extern "C" __global__ void state_overlap_shift_pair_f32_buf(
    float* __restrict__ state0,
    float* __restrict__ state1,
    const int* __restrict__ commit_slot_buf,
    int ratio,
    int proj_dim
) {
    if (commit_slot_buf[0] < 0) return;
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = ratio * proj_dim;
    if (idx >= total) return;

    float* state = blockIdx.y ? state1 : state0;
    state[idx] = state[total + idx];
}
