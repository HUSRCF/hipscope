// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

// Paired HIP-graph-safe compressor ring write. The first half copies KV and
// the second half copies score into the same ring slot.
extern "C" __global__ void state_ring_write_pair_f32_buf(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ state0,
    float* __restrict__ state1,
    const int* __restrict__ ring_slot_buf,
    int proj_dim
) {
    const int slot = ring_slot_buf[0];
    if (slot < 0) return;

    const int linear = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (linear >= 2 * proj_dim) return;

    const bool second = linear >= proj_dim;
    const int d = second ? linear - proj_dim : linear;
    float* state = second ? state1 : state0;
    const float* src = second ? src1 : src0;
    state[(size_t)slot * proj_dim + d] = src[d];
}
