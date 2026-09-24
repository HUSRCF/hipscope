// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// gfx1201 symmetric qt49 A8: [d0,z0,d1,z1] f16 + 32 LE 24-bit octets
// per K256. u=0..7 widens to signed byte u-4 BEFORE LDS publication.
// Xq is block_i8_128 from the concatenated block_i8_128_quant.hip;
// activations AND weights have undergone the same upstream FWHT256 rotation.
// Tile 64 rows x 64 tokens, four wave32s, 128 threads. Per K128, 8
// fragment-order K16 slabs of 64*16 bytes per operand, 8192 B each.
// LDS: A 0..8192, W 8192..16384, row scales 16384..16640,
// token scales 16640..16896. Reuse 0..16384 for the 64x64 f32 epilogue.
// Grid [ceil(M/64),ceil(N/64)], block [128,1,1], dynamic LDS 16896 B.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#if defined(__gfx1201__)
typedef int32_t __attribute__((ext_vector_type(2))) i32x2;
typedef int32_t __attribute__((ext_vector_type(8))) i32x8;
typedef float __attribute__((ext_vector_type(8))) f32x8;

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_mq3v25_residual_mmq_iu8(
    const char* __restrict__ A, const block_i8_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add
) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int rs = blockIdx.x * 64, bs = blockIdx.y * 64;
    const int row_pair = wave >> 1, token_half = wave & 1;
    const int m_lane = lane & 15, k_half = lane >> 4;
    const int gpr = K / 256;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const a_lds = smem;
    unsigned char* const w_lds = smem + 8192;
    float* const scales_w = (float*)(smem + 16384);
    float* const scales_x = (float*)(smem + 16640);
    f32x8 acc[2][2];
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb)
        #pragma unroll
        for (int rg = 0; rg < 2; ++rg)
            acc[nb][rg] = (f32x8){0,0,0,0,0,0,0,0};

    // Each CTA stages every K128 activation/weight byte once. The old S0
    // probe's fragment-order map is retained; K16 doubles the bytes per
    // fragment (8 B/lane) and doubles the count of fragments per K128.
    for (int kb = 0; kb < K / 128; ++kb) {
        if (tid < 64) {
            const int row = rs + tid;
            const int safe = row < M ? row : M - 1;
            const unsigned short d = *(const unsigned short*)(A +
                (long long)safe * gpr * 104 + (kb / 2) * 104 + (kb & 1) * 4);
            scales_w[tid] = row < M ? __half2float(__ushort_as_half(d)) : 0.0f;
        } else {
            const int col = bs + tid - 64;
            scales_x[tid - 64] = col < N ? Xq[(long long)kb * N + col].d : 0.0f;
        }
        // 128 threads = 64 rows/tokens x two 8-byte K halves. The same
        // fragment-order address works for W and X; lane L = row-in-16 +
        // 16*(K-half), fragment group=(row/16)*8 + K16 step.
        const int r = tid >> 1, kh = tid & 1;
        const int wr = rs + r, xr = bs + r;
        const int safe_wr = wr < M ? wr : M - 1;
        const int safe_xr = xr < N ? xr : N - 1;
        const char* const wg = A + (long long)safe_wr * gpr * 104
            + (kb / 2) * 104 + 8 + (kb & 1) * 48;
        const block_i8_128* const xb = Xq + (long long)kb * N + safe_xr;
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int k8 = step * 16 + kh * 8;
            const unsigned char* p = (const unsigned char*)wg + (k8 / 8) * 3;
            const unsigned bits = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
            unsigned long long wide = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                wide |= (unsigned long long)(unsigned char)(int8_t)(((bits >> (3*j)) & 7u) - 4u) << (8*j);
            const int offset = ((r >> 4) * 8 + step) * 256 + ((r & 15) + 16 * kh) * 8;
            *(unsigned long long*)(w_lds + offset) = wide;
            const unsigned long long xv = *(const unsigned long long*)(xb->qs + k8);
            *(unsigned long long*)(a_lds + offset) = xr < N ? xv : 0ull;
        }
        __syncthreads();
        i32x8 c[2][2];
        const i32x8 zero = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            i32x2 wf[2], xf[2];
            #pragma unroll
            for (int rg = 0; rg < 2; ++rg) {
                const int off = ((row_pair * 2 + rg) * 8 + step) * 256 + lane * 8;
                wf[rg] = *(const i32x2*)(w_lds + off);
            }
            #pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                const int off = ((token_half * 2 + nb) * 8 + step) * 256 + lane * 8;
                xf[nb] = *(const i32x2*)(a_lds + off);
            }
            #pragma unroll
            for (int nb = 0; nb < 2; ++nb)
                #pragma unroll
                for (int rg = 0; rg < 2; ++rg)
                    c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, wf[rg], true, xf[nb], step == 0 ? zero : c[nb][rg], false);
        }
        // Exact signed i32 dot per K128; RN(d_w*d_x), RN(product), RN(sum).
        // No affine zero/sum correction and no cross-block i32 accumulation.
        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            const float dx = scales_x[token_half * 32 + nb * 16 + m_lane];
            #pragma unroll
            for (int rg = 0; rg < 2; ++rg) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ri = row_pair * 32 + rg * 16 + 8 * k_half + j;
                    const float ds = __fmul_rn(scales_w[ri], dx);
                    const float value = __fmul_rn(ds, (float)c[nb][rg][j]);
                    acc[nb][rg][j] = __fadd_rn(acc[nb][rg][j], value);
                }
            }
        }
        __syncthreads();
    }
    // CTA-wide transpose: one row-contiguous float stream per token.
    float* const slots = (float*)smem;
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb)
        #pragma unroll
        for (int rg = 0; rg < 2; ++rg)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = token_half * 32 + nb * 16 + m_lane;
                const int row = row_pair * 32 + rg * 16 + 8 * k_half + j;
                slots[col * 64 + row] = acc[nb][rg][j];
            }
    __syncthreads();
    for (int i = tid; i < 64 * 64; i += 128) {
        const int col = i >> 6, row = i & 63;
        if (bs + col < N && rs + row < M) {
            float* out = Y + (long long)(bs + col) * M + rs + row;
            const float v = slots[i];
            if (add) *out = __fadd_rn(*out, v);
            else *out = v;
        }
    }
}
#endif
