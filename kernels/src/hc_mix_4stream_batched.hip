// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HC 4-stream residual mix — BATCHED variant (Phase A5, 2026-05-18).
//
// Batched twin of `hc_mix_4stream`. Per batch position b:
//   x_out[b, dst, d] = sum_src(A[b, src, dst] * x_in[b, src, d])
//                      + scale[b, dst] * transform_out[b, d]
//
// Grid:  [hidden_blocks, HC_MULT, batch_size]
// Block: [BLOCK_HIDDEN, 1, 1]
//
// At batch_size == 1, byte-identical to hc_mix_4stream.

#include <hip/hip_runtime.h>

#define HC_MULT 4
#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream_batched(
    const float* __restrict__ x_in,           // [batch, HC_MULT, hidden]
    const float* __restrict__ A,              // [batch, HC_MULT, HC_MULT]
    const float* __restrict__ scale,          // [batch, HC_MULT]
    const float* __restrict__ transform_out,  // [batch, hidden]
    float*       __restrict__ x_out,          // [batch, HC_MULT, hidden]
    int hidden,
    int batch_size
) {
    const int stream_out = blockIdx.y;
    const int b          = blockIdx.z;
    const int tid        = threadIdx.x;
    const int d          = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden || stream_out >= HC_MULT || b >= batch_size) return;

    const size_t x_base     = (size_t)b * HC_MULT * hidden;
    const size_t a_base     = (size_t)b * HC_MULT * HC_MULT;
    const size_t scale_base = (size_t)b * HC_MULT;
    const size_t to_base    = (size_t)b * hidden;
    const size_t out_base   = (size_t)b * HC_MULT * hidden;

    float acc = 0.0f;
    #pragma unroll
    for (int s_in = 0; s_in < HC_MULT; ++s_in) {
        // A is contiguous [source_stream, destination_stream], matching the
        // official HC reference and the scalar kernel.
        float gate = A[a_base + s_in * HC_MULT + stream_out];
        float val  = x_in[x_base + s_in * hidden + d];
        acc += gate * val;
    }
    acc += scale[scale_base + stream_out] * transform_out[to_base + d];

    x_out[out_base + stream_out * hidden + d] = acc;
}
