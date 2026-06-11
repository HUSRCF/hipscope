// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Merged per-expert MIXED-PRECISION MoE down GEMV (expanded write).
//
// Each routed expert in a layer is EITHER MQ6 (hot) or MQ2-Lloyd (cold).
// The grid is block-per-(row, krank, token), so all 32 threads in a block
// share ONE expert_id → ONE dtype tag → the per-block branch is
// block-uniform → NO warp divergence.
//
//   dtype_tags[expert_id] == 0 → MQ6   (200 B/group: 4 B scale + 4 B zero
//                                        + 192 B of 6-bit packed quants)
//   dtype_tags[expert_id] == 1 → MQ2-Lloyd (72 B/group: 4×fp16 codebook
//                                        + 64 B of 2-bit indices)
//
// MQ6 dequant copied verbatim from
//   gemv_hfq6g256_moe_down_k8_indexed_batched_expanded.hip:62-88
// MQ2-Lloyd dequant copied verbatim (single-accumulator form, NOT the k4
// 4-acc form) from gemv_mq2g256_lloyd_moe_down_expanded_k4.hip:65-83.
//
// Both branches accumulate into a SINGLE `acc` and share one wave32 reduce
// + one expanded write — byte-identical to both reference expanded kernels.
// Feeds the dtype-agnostic `moe_down_combine_k8_batched`.
//
// Shapes:
//   expert_ptrs    : [n_exp]
//   dtype_tags     : [n_exp]                 u8, 0=MQ6 / 1=MQ2-Lloyd
//   topk_indices   : [N × K_TOP]
//   rot_batch      : [N × K_TOP × K]
//   expert_outputs : [N × K_TOP × M]         non-atomic, one writer per cell
//
// Grid: (M, K_TOP, N). Block = 32 threads.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mixed_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const unsigned char* __restrict__ dtype_tags,        // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const unsigned char tag = dtype_tags[expert_id]; // block-uniform
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;

    float acc = 0.0f;

    if (tag == 0) {
        // ---- MQ6 branch (200 B/group affine dequant) ----
        const int row_bytes = groups_per_row * 200;
        const char* row_ptr = A + (long long)row * row_bytes;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 200;
            float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
            float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const unsigned char* data = (const unsigned char*)(gptr + 8);

            int base_idx = g * 256 + tid * 8;
            int byte_off = tid * 6;

            unsigned char b0 = data[byte_off];
            unsigned char b1 = data[byte_off + 1];
            unsigned char b2 = data[byte_off + 2];
            unsigned char b3 = data[byte_off + 3];
            unsigned char b4 = data[byte_off + 4];
            unsigned char b5 = data[byte_off + 5];

            int q0 = b0 & 63;
            int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
            int q2 = (b1 >> 4) | ((b2 & 3) << 4);
            int q3 = b2 >> 2;
            int q4 = b3 & 63;
            int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
            int q6 = (b4 >> 4) | ((b5 & 3) << 4);
            int q7 = b5 >> 2;

            acc += (scale * (float)q0 + zero) * x[base_idx]
                 + (scale * (float)q1 + zero) * x[base_idx + 1]
                 + (scale * (float)q2 + zero) * x[base_idx + 2]
                 + (scale * (float)q3 + zero) * x[base_idx + 3]
                 + (scale * (float)q4 + zero) * x[base_idx + 4]
                 + (scale * (float)q5 + zero) * x[base_idx + 5]
                 + (scale * (float)q6 + zero) * x[base_idx + 6]
                 + (scale * (float)q7 + zero) * x[base_idx + 7];
        }
    } else {
        // ---- MQ2-Lloyd branch (72 B/group 2-bit codebook) ----
        const int row_bytes = groups_per_row * 72;
        const char* row_ptr = A + (long long)row * row_bytes;
        const int boff = tid * 2;
        const int xoff = tid * 8;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 72;
            const __half* cb_h = reinterpret_cast<const __half*>(gptr);
            float cb[4] = {
                __half2float(cb_h[0]),
                __half2float(cb_h[1]),
                __half2float(cb_h[2]),
                __half2float(cb_h[3]),
            };
            const unsigned char* data = (const unsigned char*)(gptr + 8);
            unsigned char b0 = data[boff];
            unsigned char b1 = data[boff + 1];
            const int base = g * 256 + xoff;
            acc += cb[b0 & 3]        * x[base]
                +  cb[(b0 >> 2) & 3] * x[base + 1]
                +  cb[(b0 >> 4) & 3] * x[base + 2]
                +  cb[b0 >> 6]       * x[base + 3]
                +  cb[b1 & 3]        * x[base + 4]
                +  cb[(b1 >> 2) & 3] * x[base + 5]
                +  cb[(b1 >> 4) & 3] * x[base + 6]
                +  cb[b1 >> 6]       * x[base + 7];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_offset =
            ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
