#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Asymmetric attention: Q8 K cache + turbo4 V cache, head_dim=256.
// VGPR-optimized for RDNA wave32: targets ≤24 VGPRs for 10 waves/SIMD.
//
// Key optimizations vs previous version (40 VGPRs):
// - K dot: no inner loop, direct block access (8 dims always in 1 Q8 block)
// - V accumulate: vectorized u32 byte load instead of 4× individual
// - Q values loaded per-position (L1 cached), not hoisted to registers
// Grid: [n_heads]. Block: [32]. Shared: scores[seq_len].
extern "C" __launch_bounds__(32, 10)
__global__ void attention_q8k_turbo4v_256(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_q8,
    const unsigned char* __restrict__ v_t4,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int seq_len_hint,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn
) {
    const int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_h = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int d0 = tid * 8;

    // Q8 K layout constants
    const int q8_bph = (head_dim / 32) * 34;          // 272 bytes per head
    const int q8_bpp = n_kv_heads * q8_bph;            // bytes per position
    // Thread's Q8 block: d0/32 gives block index, d0%32 gives offset within block
    const int blk_idx = d0 / 32;
    const int blk_off = d0 % 32;
    const int blk_byte_off = kv_h * q8_bph + blk_idx * 34;

    // Turbo4 V layout constants
    const int t4_bph = 4 + head_dim / 2;               // 132 bytes per head
    const int t4_bpp = n_kv_heads * t4_bph;
    const int t4_head_off = kv_h * t4_bph;

    float* scores = sdata;
    const float* qh = q + h * head_dim + d0;

    // ── Phase 1: Q @ K^T — Q8 dequant, no inner loop ──
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* blk = k_q8 + (size_t)t * q8_bpp + blk_byte_off;
        float bs = (float)(*((const _Float16*)blk));
        const signed char* v = (const signed char*)(blk + 2) + blk_off;

        float p = qh[0] * (bs * (float)v[0])
                + qh[1] * (bs * (float)v[1])
                + qh[2] * (bs * (float)v[2])
                + qh[3] * (bs * (float)v[3])
                + qh[4] * (bs * (float)v[4])
                + qh[5] * (bs * (float)v[5])
                + qh[6] * (bs * (float)v[6])
                + qh[7] * (bs * (float)v[7]);

        for (int off = 16; off > 0; off >>= 1) p += __shfl_xor(p, off);
        if (tid == 0) scores[t] = p * scale_attn;
    }
    __syncthreads();

    // ── Softmax (warp shuffle) ──
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += 32) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += 32) {
        float e = expf(scores[t] - lmx); scores[t] = e; lsm += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float is = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += 32) scores[t] *= is;
    __syncthreads();

    // ── Phase 2: V weighted sum — turbo4 with sparse skip ──
    float m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0;
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_t4 + (size_t)t * t4_bpp + t4_head_off;
        float wn = w * *(const float*)vb;
        // Vectorized 4-byte load
        unsigned int pk = *(const unsigned int*)(vb + 4 + tid * 4);
        m0 += wn * TURBO_C4_256[(pk)       & 0xF];
        m1 += wn * TURBO_C4_256[(pk >> 4)  & 0xF];
        m2 += wn * TURBO_C4_256[(pk >> 8)  & 0xF];
        m3 += wn * TURBO_C4_256[(pk >> 12) & 0xF];
        m4 += wn * TURBO_C4_256[(pk >> 16) & 0xF];
        m5 += wn * TURBO_C4_256[(pk >> 20) & 0xF];
        m6 += wn * TURBO_C4_256[(pk >> 24) & 0xF];
        m7 += wn * TURBO_C4_256[(pk >> 28) & 0xF];
    }

    // ── Inverse FWHT-256 ──
    fwht_shfl_inverse_256(m0, m1, m2, m3, m4, m5, m6, m7, signs1, signs2, tid);

    float* oh = out + h * head_dim + d0;
    oh[0] = m0; oh[1] = m1; oh[2] = m2; oh[3] = m3;
    oh[4] = m4; oh[5] = m5; oh[6] = m6; oh[7] = m7;
}
