// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Merged per-expert MIXED-PRECISION MoE gate_up GEMV.
//
// Each routed expert in a layer has one of four precision tiers.
// The grid is block-per-(row, krank, token), so all 32 threads in a block
// share ONE expert_id → ONE dtype tag → the per-block branch is
// block-uniform → NO warp divergence.
//
//   dtype_tags[expert_id] == 0 → MQ6        (200 B/group: 4 B scale + 4 B zero
//                                             + 192 B of 6-bit packed quants)
//   dtype_tags[expert_id] == 1 → MQ2-Lloyd  (72 B/group: 4×fp16 codebook
//                                             + 64 B of 2-bit indices)
//   dtype_tags[expert_id] == 2 → MQ4        (136 B/group: 4 B scale + 4 B zero
//                                             + 128 B of 4-bit packed quants)
//   dtype_tags[expert_id] == 3 → MQ3-Lloyd  (112 B/group: 8×fp16 codebook
//                                             + 96 B of 3-bit packed quants)
//
// All branches accumulate into a SINGLE `acc` and share one wave32 reduce
// + one gate/up split write (M = 2*MI: first half → y_gate, second half →
// y_up) — byte-identical output layout to both reference kernels.
//
// x is indexed [bid * K] (per-token, NOT per-krank — gate_up reads the
// same input for every expert in a token's top-K).
//
// Shapes:
//   expert_ptrs : [n_exp]
//   dtype_tags  : [n_exp]                     u8, 0=MQ6 / 1=MQ2-Lloyd / 2=MQ4 / 3=MQ3-Lloyd
//   topk_indices: [N × K_TOP]
//   x           : [N × K]
//   y_gate      : [N × K_TOP × MI]            (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Block = 32 threads.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mixed_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const unsigned char* __restrict__ dtype_tags,        // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id]; // block-uniform
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;

    float acc = 0.0f;

    // cb_lds: 32 B shared, used only by tag==3 (MQ3-Lloyd); harmless to other branches.
    __shared__ float cb_lds[8];

    // Per-group dot (8 MACs) as a value, so the hot MQ6/MQ4 branches can carry
    // 4 INDEPENDENT accumulators across groups — restoring the quad-unroll ILP
    // the uniform gemv has (the single-`acc` chain serialized the FMA latency).
    // Self-contained: derives row_ptr from A/row/groups_per_row, offsets from tid.
#define _MQ6G(gg) ({ \
        const char* _gp = (const char*)A + ((long long)row * groups_per_row + (gg)) * 200; \
        float _s = __builtin_bit_cast(float, *(const unsigned int*)(_gp)); \
        float _z = __builtin_bit_cast(float, *(const unsigned int*)(_gp + 4)); \
        const unsigned char* _d = (const unsigned char*)(_gp + 8) + tid * 6; \
        int _bi = (gg) * 256 + tid * 8; \
        unsigned char _b0=_d[0],_b1=_d[1],_b2=_d[2],_b3=_d[3],_b4=_d[4],_b5=_d[5]; \
        int _q0=_b0&63,_q1=(_b0>>6)|((_b1&0xF)<<2),_q2=(_b1>>4)|((_b2&3)<<4),_q3=_b2>>2, \
            _q4=_b3&63,_q5=(_b3>>6)|((_b4&0xF)<<2),_q6=(_b4>>4)|((_b5&3)<<4),_q7=_b5>>2; \
        (_s*(float)_q0+_z)*x_row[_bi]   +(_s*(float)_q1+_z)*x_row[_bi+1] \
       +(_s*(float)_q2+_z)*x_row[_bi+2] +(_s*(float)_q3+_z)*x_row[_bi+3] \
       +(_s*(float)_q4+_z)*x_row[_bi+4] +(_s*(float)_q5+_z)*x_row[_bi+5] \
       +(_s*(float)_q6+_z)*x_row[_bi+6] +(_s*(float)_q7+_z)*x_row[_bi+7]; })
#define _MQ4G(gg) ({ \
        const char* _gp = (const char*)A + ((long long)row * groups_per_row + (gg)) * 136; \
        float _s = __builtin_bit_cast(float, *(const unsigned int*)(_gp)); \
        float _z = __builtin_bit_cast(float, *(const unsigned int*)(_gp + 4)); \
        unsigned int _pk = *(const unsigned int*)(_gp + 8 + tid * 4); \
        int _b = (gg) * 256 + tid * 8; \
        (_s*(float)(_pk&0xFu)+_z)*x_row[_b]        +(_s*(float)((_pk>>4)&0xFu)+_z)*x_row[_b+1] \
       +(_s*(float)((_pk>>8)&0xFu)+_z)*x_row[_b+2] +(_s*(float)((_pk>>12)&0xFu)+_z)*x_row[_b+3] \
       +(_s*(float)((_pk>>16)&0xFu)+_z)*x_row[_b+4]+(_s*(float)((_pk>>20)&0xFu)+_z)*x_row[_b+5] \
       +(_s*(float)((_pk>>24)&0xFu)+_z)*x_row[_b+6]+(_s*(float)((_pk>>28)&0xFu)+_z)*x_row[_b+7]; })

    if (tag == 0) {
        // ---- MQ6 branch (200 B/group affine dequant), 4-accumulator unroll ----
        float _a0=0.f,_a1=0.f,_a2=0.f,_a3=0.f;
        int g = 0;
        for (; g + 3 < groups_per_row; g += 4) {
            _a0 += _MQ6G(g);   _a1 += _MQ6G(g+1);
            _a2 += _MQ6G(g+2); _a3 += _MQ6G(g+3);
        }
        for (; g < groups_per_row; g++) _a0 += _MQ6G(g);
        acc = (_a0 + _a1) + (_a2 + _a3);
    } else if (tag == 1) {
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
            acc += cb[b0 & 3]        * x_row[base]
                +  cb[(b0 >> 2) & 3] * x_row[base + 1]
                +  cb[(b0 >> 4) & 3] * x_row[base + 2]
                +  cb[b0 >> 6]       * x_row[base + 3]
                +  cb[b1 & 3]        * x_row[base + 4]
                +  cb[(b1 >> 2) & 3] * x_row[base + 5]
                +  cb[(b1 >> 4) & 3] * x_row[base + 6]
                +  cb[b1 >> 6]       * x_row[base + 7];
        }
    } else if (tag == 2) {
        // ---- MQ4 branch (136 B/group), 4-accumulator unroll ----
        float _a0=0.f,_a1=0.f,_a2=0.f,_a3=0.f;
        int g = 0;
        for (; g + 3 < groups_per_row; g += 4) {
            _a0 += _MQ4G(g);   _a1 += _MQ4G(g+1);
            _a2 += _MQ4G(g+2); _a3 += _MQ4G(g+3);
        }
        for (; g < groups_per_row; g++) _a0 += _MQ4G(g);
        acc = (_a0 + _a1) + (_a2 + _a3);
    } else if (tag == 3) {
        // ---- MQ3-Lloyd branch (112 B/group: [0..16) 8xfp16 codebook, [16..112) 96 B 3-bit) ----
        const int row_bytes = groups_per_row * 112;
        const char* row_ptr = A + (long long)row * row_bytes;
        const int boff = tid * 3;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 112;
            if (tid < 8) cb_lds[tid] = __half2float(((const __half*)gptr)[tid]);
            __syncthreads();
            const unsigned char* d = (const unsigned char*)(gptr + 16) + boff;
            unsigned int pk = (unsigned int)d[0]
                            | ((unsigned int)d[1] << 8)
                            | ((unsigned int)d[2] << 16);
            const int base = g * 256 + tid * 8;
            acc += cb_lds[ pk        & 7u] * x_row[base]
                 + cb_lds[(pk >>  3) & 7u] * x_row[base + 1]
                 + cb_lds[(pk >>  6) & 7u] * x_row[base + 2]
                 + cb_lds[(pk >>  9) & 7u] * x_row[base + 3]
                 + cb_lds[(pk >> 12) & 7u] * x_row[base + 4]
                 + cb_lds[(pk >> 15) & 7u] * x_row[base + 5]
                 + cb_lds[(pk >> 18) & 7u] * x_row[base + 6]
                 + cb_lds[(pk >> 21) & 7u] * x_row[base + 7];
            __syncthreads();   // protect cb_lds before next group reload
        }
    }
#undef _MQ6G
#undef _MQ4G

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // M = 2 * MI, first half is gate, second half is up.
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
