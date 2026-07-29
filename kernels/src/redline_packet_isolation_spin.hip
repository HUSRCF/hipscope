// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Diagnostic-only independent work. Each retained dispatch receives a
// different output address, so removing CS_PARTIAL_FLUSH does not introduce
// cross-dispatch data dependencies.
extern "C" __global__ void redline_packet_isolation_spin(
    uint64_t* output,
    uint32_t iterations
) {
    uint64_t value = (uint64_t)threadIdx.x + 0x9e3779b97f4a7c15ULL;
#pragma unroll 1
    for (uint32_t i = 0; i < iterations; ++i) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        value *= 0x2545f4914f6cdd1dULL;
    }
    if (threadIdx.x == 0) {
        output[0] = value;
    }
}
