// Single-layer attention test: write known K/V, compute attention, compare output.
// Uses the SAME code as the production turbo4 kernels (copy-pasted, not linked).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Simplified single-head write: one head of K or V at one position
extern "C" __launch_bounds__(32, 1)
__global__ void test_turbo4_write_one(
    unsigned char* __restrict__ cache,    // [max_seq * bytes_per_head]
    const float* __restrict__ src,        // [head_dim]
    int pos,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int bytes_per_head = 4 + head_dim / 2;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    for (int d = tid; d < head_dim; d += 32) x[d] = src[d];
    __syncthreads();

    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid+s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    const int d0 = tid * 4;
    float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
    fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

    float local_rsq = 0.0f;
    int idx0 = turbo_quantize_4bit(fa); local_rsq += TURBO_C4[idx0] * TURBO_C4[idx0];
    int idx1 = turbo_quantize_4bit(fb); local_rsq += TURBO_C4[idx1] * TURBO_C4[idx1];
    int idx2 = turbo_quantize_4bit(fc); local_rsq += TURBO_C4[idx2] * TURBO_C4[idx2];
    int idx3 = turbo_quantize_4bit(fd); local_rsq += TURBO_C4[idx3] * TURBO_C4[idx3];
    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid+s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    unsigned char* out = cache + pos * bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;
    out[4 + tid * 2]     = (unsigned char)((idx1 << 4) | (idx0 & 0xF));
    out[4 + tid * 2 + 1] = (unsigned char)((idx3 << 4) | (idx2 & 0xF));
}

// Simplified single-head attention: Q dot K cache + V weighted sum + inverse FWHT
extern "C" __launch_bounds__(32, 1)
__global__ void test_turbo4_attend_one(
    const float* __restrict__ q,           // [head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,               // [head_dim]
    float* __restrict__ debug_scores,      // [seq_len] attention weights after softmax
    int seq_len,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int head_dim,
    float scale_attn
) {
    const int tid = threadIdx.x;
    const int bytes_per_head = 4 + head_dim / 2;
    const int d0 = tid * 4;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // FWHT on Q
    float mq0 = q[d0], mq1 = q[d0+1], mq2 = q[d0+2], mq3 = q[d0+3];
    fwht_shfl_forward(mq0, mq1, mq2, mq3, signs1, signs2, tid);
    float mq[4] = {mq0, mq1, mq2, mq3};

    // K dot products
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + t * bytes_per_head;
        float cnorm = *(const float*)kb;
        unsigned char b0 = kb[4 + tid * 2];
        unsigned char b1 = kb[4 + tid * 2 + 1];
        float partial = TURBO_C4[b0 & 0xF] * mq[0]
                       + TURBO_C4[b0 >> 4]  * mq[1]
                       + TURBO_C4[b1 & 0xF] * mq[2]
                       + TURBO_C4[b1 >> 4]  * mq[3];
        partial *= cnorm;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t] = partial * scale_attn;
    }
    __syncthreads();

    // Softmax
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += 32) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float mx = lmx;
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += 32) { float e = expf(scores[t]-mx); scores[t]=e; lsm+=e; }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float inv_sum = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += 32) scores[t] *= inv_sum;
    __syncthreads();

    // Save attention weights for debugging
    for (int t = tid; t < seq_len; t += 32) debug_scores[t] = scores[t];

    // V weighted sum
    float mv[4] = {0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_cache + t * bytes_per_head;
        float wn = w * *(const float*)vb;
        unsigned char b0 = vb[4 + tid * 2];
        unsigned char b1 = vb[4 + tid * 2 + 1];
        mv[0] += wn * TURBO_C4[b0 & 0xF];
        mv[1] += wn * TURBO_C4[b0 >> 4];
        mv[2] += wn * TURBO_C4[b1 & 0xF];
        mv[3] += wn * TURBO_C4[b1 >> 4];
    }

    // Inverse FWHT
    fwht_shfl_inverse(mv[0], mv[1], mv[2], mv[3], signs1, signs2, tid);

    out[d0] = mv[0]; out[d0+1] = mv[1]; out[d0+2] = mv[2]; out[d0+3] = mv[3];
}
