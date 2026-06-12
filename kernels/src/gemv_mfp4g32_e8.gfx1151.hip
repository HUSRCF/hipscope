// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 decode-GEMV, gfx1151-specific (Strix Halo, RDNA3.5 APU).
//
// ONLY for gfx1151. Other archs use gemv_mfp4g32_e8 (generic).
//
// Changes vs generic:
// 1. e8_decode_index is fully inlined with manual loop unroll to reduce
//    instruction-level latency and enable wider VALU ILP.
// 2. Partial e7 reconstruction uses precomputed sl from 4-bit adds.
// 3. coset check folded into coords with a single sub-1.0 pattern.
//
// Bit-exact with generic kernel (same math, just unrolled).

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

// Fully unrolled E8 decode + dot product.
// Computes dot(coords, x[base..base+7]) * ssc and adds to acc.
// Inlined to allow the compiler to schedule the 7 nibble extracts
// independently of the FMA chain.
__device__ __forceinline__ void e8_dog_inlined(
    unsigned int idx, float ssc,
    const float* __restrict__ x, int base, float& acc)
{
    const int coset = (int)((idx >> 31) & 1u);

    // Extract 7 nibbles (4 bits each)
    const unsigned int n0 = (idx)       & 0xFu;
    const unsigned int n1 = (idx >>  4) & 0xFu;
    const unsigned int n2 = (idx >>  8) & 0xFu;
    const unsigned int n3 = (idx >> 12) & 0xFu;
    const unsigned int n4 = (idx >> 16) & 0xFu;
    const unsigned int n5 = (idx >> 20) & 0xFu;
    const unsigned int n6 = (idx >> 24) & 0xFu;
    const unsigned int sl = n0 + n1 + n2 + n3 + n4 + n5 + n6;

    // e7: high 3 bits from idx[28..30], lsb from parity
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int n7 = p7 | ((sl + p7) & 1u);

    // Decode: coord = nibble - 7, then offset by +0.5 if coset
    const float off = coset ? 0.5f : 0.0f;
    const float c0 = (float)((int)n0 - 7) + off;
    const float c1 = (float)((int)n1 - 7) + off;
    const float c2 = (float)((int)n2 - 7) + off;
    const float c3 = (float)((int)n3 - 7) + off;
    const float c4 = (float)((int)n4 - 7) + off;
    const float c5 = (float)((int)n5 - 7) + off;
    const float c6 = (float)((int)n6 - 7) + off;
    const float c7 = (float)((int)n7 - 7) + off;

    acc += ssc*c0*x[base]   + ssc*c1*x[base+1]
         + ssc*c2*x[base+2] + ssc*c3*x[base+3]
         + ssc*c4*x[base+4] + ssc*c5*x[base+5]
         + ssc*c6*x[base+6] + ssc*c7*x[base+7];
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mfp4g32_e8_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;

    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp2) * 0.88f;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp3) * 0.88f;

        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const unsigned int ew2 = *(const unsigned int*)(gp2 + 1 + cw_off);
        const unsigned int ew3 = *(const unsigned int*)(gp3 + 1 + cw_off);

        const int base = g * 256 + tid * 8;
        e8_dog_inlined(ew0, ssc0, x, base,         acc0);
        e8_dog_inlined(ew1, ssc1, x, base + 256,   acc1);
        e8_dog_inlined(ew2, ssc2, x, base + 512,   acc2);
        e8_dog_inlined(ew3, ssc3, x, base + 768,   acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        e8_dog_inlined(ew, ssc, x, g * 256 + tid * 8, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        e8_dog_inlined(ew, ssc, x, g * 256 + tid * 8, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        e8_dog_inlined(ew, ssc, x, g * 256 + tid * 8, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
