// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 grouped symmetric MQ4V2 experts x packed A4 activations, gfx1151
// (RDNA3.5) wmma_i32_16x16x16_iu4. Opt-in prefill route
// (`HIPFIRE_QWEN4_MOE_SYM_IU4=1`, fn-moe-sym) for artifacts whose every
// routed-expert header passed `qwen4_moe_sym_check_gfx1151`; the shipped F16
// WMMA route stays the default and the fallback. ABI: five pointers then five
// i32 (expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y_grouped,
// M, K, x_row_div, m_total, x_src_rows).
//
// Weights: QT44 (gate/up, 136 B per 256: [f16 sc0,zp0,sc1,zp1][128 B nibbles])
// and QT53 (down, 68 B per 128: [f16 sc,zp][64 B nibbles]); every header obeys
// zp == -8*sc, so w = sc*(u-8). A nibbles are rebiased with XOR 0x88888888
// (u ^ 8 is the two's-complement nibble u-8) and fed as SIGNED A; X is the
// signed block_i4_128 sidecar, epoch-major [K/128, x_src_rows], row
// slot / x_row_div. Per output and ascending K128 epoch h:
//   C_h = exact int32 sum_k128(A_signed * X_signed)
//   sum = fma(RN(float(sc_h) * d_h), float(C_h), sum), sum starts +0.
// Not bit-exact to the F16 WMMA route for general activations (A4 inputs).
//
// gfx11 WMMA contract (probe-verified in gemm_mq4g256v2_residual_mmq_iu4.gfx11):
// A/B int32x2 per lane = 16 nibbles (K16) of row/token lane&15, lanes r and
// r+16 identical; C acc[j] = C[row 2j+(lane>>4)][token lane&15].
//
// Operands are loaded straight into registers, the next epoch prefetched
// while the current one is consumed (measured by the fn-moe-sym kill run;
// the planned fragment-major LDS double buffer was not needed to clear it).
// LDS holds only the coalesced BF16 output transpose. Tiles: gate/up 32x16
// (block 64, two waves, each one 16-row gate/up pair: gate rows r and up rows
// r + M/2 of the unmodified weights), down 64x16 (block 128, four waves).
// Grid (M/2/32 resp. M/64, m_total/16). Sentinel tiles (expert id -1) return
// before any store; padding slots (sorted index -1) never load X row 0 and
// store +0.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1151__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_moe_iu4_sym.gfx1151.hip is gfx1151-only"
#endif

#ifndef HIPFIRE_BLOCK_I4_128_QUANT_HIP
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
#endif

typedef int sym_i32x2 __attribute__((ext_vector_type(2)));
typedef int sym_i32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short sym_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float sym_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}
__device__ __forceinline__ float sym_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}

// 64 nibble bytes (K128) as 16 dwords, 4-byte aligned source.
struct sym_k128 { unsigned int w[16]; };

__device__ __forceinline__ void sym_load_k128(sym_k128& o, const unsigned char* p) {
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 16; ++i) o.w[i] = q[i];
}

// Eight K16 steps of one K128 epoch into a zeroed accumulator.
__device__ __forceinline__ sym_i32x8 sym_epoch(const sym_k128& a, const sym_k128& x) {
    sym_i32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        sym_i32x2 av = {(int)(a.w[2 * t] ^ 0x88888888u), (int)(a.w[2 * t + 1] ^ 0x88888888u)};
        sym_i32x2 xv = {(int)x.w[2 * t], (int)x.w[2 * t + 1]};
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, av, true, xv, acc, false);
    }
    return acc;
}

// Fold one epoch: lane's rows are 2j+(lane>>4); sc_lane is this lane's own
// row (lane&15) scale, fetched per output row by shuffle.
__device__ __forceinline__ void sym_fold(float* sum, const sym_i32x8& acc, float sc_lane, float d, int lane) {
    const int hi = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float sc = __shfl(sc_lane, 2 * j + hi, 32);
        sum[j] = __fmaf_rn(__fmul_rn(sc, d), (float)acc[j], sum[j]);
    }
}

extern "C" __global__ __launch_bounds__(64) void qwen4_moe_gate_up_silu_iu4_sym_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const block_i4_128* __restrict__ Xq,
    unsigned short* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int x_src_rows) {
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0) return;
    __shared__ unsigned short sOut[16][32];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, lr = lane & 15;
    const int half = M >> 1;
    const int rbase = blockIdx.x * 32 + wave * 16;  // gate row of this wave's pair
    const size_t row_bytes = (size_t)(K >> 8) * 136;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    const unsigned char* wg = W + (size_t)(rbase + lr) * row_bytes;
    const unsigned char* wu = W + (size_t)(rbase + lr + half) * row_bytes;
    const int slot = sorted_slot_index[tile * 16 + lr];
    const bool live = slot >= 0;
    const block_i4_128* xrow = Xq + (live ? slot / x_row_div : 0);
    const int epochs = K >> 7;

    float sg[8], su[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { sg[j] = 0.0f; su[j] = 0.0f; }

    sym_k128 ag, au, xx, ag1, au1, xx1;
    unsigned int hg, hu, hg1, hu1;
    float dx, dx1;
    const sym_k128 zero = {};
    // epoch h: group h>>1, half h&1; header dword = sc_h | zp_h << 16.
#define SYM_LOAD(h, AG, AU, XX, HG, HU, DX) do { \
        const size_t go = (size_t)((h) >> 1) * 136; \
        HG = *(const unsigned int*)(wg + go + 4 * ((h) & 1)); \
        HU = *(const unsigned int*)(wu + go + 4 * ((h) & 1)); \
        sym_load_k128(AG, wg + go + 8 + 64 * ((h) & 1)); \
        sym_load_k128(AU, wu + go + 8 + 64 * ((h) & 1)); \
        if (live) { const block_i4_128* xb = xrow + (size_t)(h) * x_src_rows; \
            sym_load_k128(XX, xb->qs); DX = xb->d; } \
        else { XX = zero; DX = 0.0f; } } while (0)

    SYM_LOAD(0, ag, au, xx, hg, hu, dx);
    for (int h = 0; h < epochs; h += 2) {
        if (h + 1 < epochs) SYM_LOAD(h + 1, ag1, au1, xx1, hg1, hu1, dx1);
        {
            const sym_i32x8 cg = sym_epoch(ag, xx);
            const sym_i32x8 cu = sym_epoch(au, xx);
            sym_fold(sg, cg, sym_f16((unsigned short)(hg & 0xFFFFu)), dx, lane);
            sym_fold(su, cu, sym_f16((unsigned short)(hu & 0xFFFFu)), dx, lane);
        }
        if (h + 1 >= epochs) break;
        if (h + 2 < epochs) SYM_LOAD(h + 2, ag, au, xx, hg, hu, dx);
        {
            const sym_i32x8 cg = sym_epoch(ag1, xx1);
            const sym_i32x8 cu = sym_epoch(au1, xx1);
            sym_fold(sg, cg, sym_f16((unsigned short)(hg1 & 0xFFFFu)), dx1, lane);
            sym_fold(su, cu, sym_f16((unsigned short)(hu1 & 0xFFFFu)), dx1, lane);
        }
    }
#undef SYM_LOAD
    // SwiGLU with the shipped BF16 boundaries; padding slots store +0.
    const int hi = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = sym_bf16_rt(sg[j]);
        const float w = sym_bf16_rt(su[j]);
        const float r = live ? sym_bf16_rt((v / (1.0f + expf(-v))) * w) : 0.0f;
        sOut[lr][wave * 16 + 2 * j + hi] = sym_bf16_bits(r);
    }
    __syncthreads();
    // 16 rows x 64 B: thread t stores 16 B (row t>>2, quarter t&3).
    const int t = threadIdx.x, orow = t >> 2, q = t & 3;
    uint4 v = *(const uint4*)&sOut[orow][q * 8];
    *(uint4*)(Y_grouped + (size_t)(tile * 16 + orow) * half + blockIdx.x * 32 + q * 8) = v;
}

extern "C" __global__ __launch_bounds__(128) void qwen4_moe_down_iu4_sym_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const block_i4_128* __restrict__ Xq,
    unsigned short* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int x_src_rows) {
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0) return;
    __shared__ unsigned short sOut[16][64];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, lr = lane & 15;
    const int rbase = blockIdx.x * 64 + wave * 16;
    const size_t row_bytes = (size_t)(K >> 7) * 68;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    const unsigned char* wr = W + (size_t)(rbase + lr) * row_bytes;
    const int slot = sorted_slot_index[tile * 16 + lr];
    const bool live = slot >= 0;
    const block_i4_128* xrow = Xq + (live ? slot / x_row_div : 0);
    const int epochs = K >> 7;
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = 0.0f;
    sym_k128 a, x, a1, x1;
    unsigned int hw, hw1;
    float dx, dx1;
    const sym_k128 zero = {};
#define SYM_LOAD(h, A, XX, HW, DX) do { \
        HW = *(const unsigned int*)(wr + (size_t)(h) * 68); \
        sym_load_k128(A, wr + (size_t)(h) * 68 + 4); \
        if (live) { const block_i4_128* xb = xrow + (size_t)(h) * x_src_rows; \
            sym_load_k128(XX, xb->qs); DX = xb->d; } \
        else { XX = zero; DX = 0.0f; } } while (0)
    SYM_LOAD(0, a, x, hw, dx);
    for (int h = 0; h < epochs; h += 2) {
        if (h + 1 < epochs) SYM_LOAD(h + 1, a1, x1, hw1, dx1);
        sym_fold(s, sym_epoch(a, x), sym_f16((unsigned short)(hw & 0xFFFFu)), dx, lane);
        if (h + 1 >= epochs) break;
        if (h + 2 < epochs) SYM_LOAD(h + 2, a, x, hw, dx);
        sym_fold(s, sym_epoch(a1, x1), sym_f16((unsigned short)(hw1 & 0xFFFFu)), dx1, lane);
    }
#undef SYM_LOAD
    const int hi = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) sOut[lr][wave * 16 + 2 * j + hi] = live ? sym_bf16_bits(s[j]) : (unsigned short)0;
    __syncthreads();
    // 16 rows x 128 B: thread t stores 16 B (row t>>3, eighth t&7).
    const int t = threadIdx.x, orow = t >> 3, q = t & 7;
    uint4 v = *(const uint4*)&sOut[orow][q * 8];
    *(uint4*)(Y_grouped + (size_t)(tile * 16 + orow) * M + blockIdx.x * 64 + q * 8) = v;
}

// One status word per 256 checked K128 headers: nonzero = some header in that
// chunk violates zp == -8*sc or is non-finite. No atomics.
extern "C" __global__ void qwen4_moe_sym_check_gfx1151(
    const uint64_t* __restrict__ ptrs, uint32_t* __restrict__ bad,
    int M, int K, int E, int group_bytes) {
    const long long per_expert = (long long)M * (K >> 7);
    const long long total = per_expert * E;
    const long long base = (long long)blockIdx.x * 256;
    const long long i = base + threadIdx.x;
    bool viol = false;
    if (i < total) {
        const int e = (int)(i / per_expert);
        const long long r = i - (long long)e * per_expert;
        const int row = (int)(r / (K >> 7));
        const int h = (int)(r - (long long)row * (K >> 7));
        const unsigned char* W = (const unsigned char*)ptrs[e];
        size_t off;
        if (group_bytes == 136) off = (size_t)row * (K / 256) * 136 + (h / 2) * 136 + 4 * (h % 2);
        else off = (size_t)row * (K / 128) * 68 + (size_t)h * 68;
        const unsigned int hd = *(const unsigned int*)(W + off);
        const float sc = sym_f16((unsigned short)(hd & 0xFFFFu));
        const float zp = sym_f16((unsigned short)(hd >> 16));
        viol = !isfinite(sc) || !isfinite(zp) || zp != -8.0f * sc;
    }
    const int any = __syncthreads_or(viol ? 1 : 0);
    if (threadIdx.x == 0 && base < total) bad[blockIdx.x] = any ? 1u : 0u;
}
