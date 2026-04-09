// Bandwidth microbenchmark: attention_turbo4_kv and attention_q8_0_kv
// Measures actual GB/s by timing the attention kernel at various context lengths.
// Uses dummy data — we only care about throughput, not correctness.

#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Minimal turbo4 attention — same memory access pattern as production kernel
extern "C" __launch_bounds__(32, 10)
__global__ void bw_attention_turbo4(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int seq_len, int n_kv_heads, int head_dim, float scale_attn
) {
    const int h = blockIdx.x;
    const int kv_h = h % n_kv_heads;
    const int tid = threadIdx.x;
    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;
    const int d0 = tid * 4;

    extern __shared__ float sdata[];
    float* scores = sdata;

    float mq0 = q[h * head_dim + d0];
    float mq1 = q[h * head_dim + d0 + 1];
    float mq2 = q[h * head_dim + d0 + 2];
    float mq3 = q[h * head_dim + d0 + 3];
    fwht_shfl_forward(mq0, mq1, mq2, mq3, signs1, signs2, tid);
    float mq[4] = {mq0, mq1, mq2, mq3};

    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + (size_t)t * bytes_per_pos + kv_head_off;
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

    float mv[4] = {0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float wn = w * *(const float*)vb;
        unsigned char b0 = vb[4 + tid * 2];
        unsigned char b1 = vb[4 + tid * 2 + 1];
        mv[0] += wn * TURBO_C4[b0 & 0xF];
        mv[1] += wn * TURBO_C4[b0 >> 4];
        mv[2] += wn * TURBO_C4[b1 & 0xF];
        mv[3] += wn * TURBO_C4[b1 >> 4];
    }
    fwht_shfl_inverse(mv[0], mv[1], mv[2], mv[3], signs1, signs2, tid);

    float* oh = out + h * head_dim;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
}

// Minimal Q8 attention for comparison
extern "C" __launch_bounds__(32, 10)
__global__ void bw_attention_q8(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    int seq_len, int n_kv_heads, int head_dim, float scale_attn
) {
    const int h = blockIdx.x;
    const int kv_h = h % n_kv_heads;
    const int tid = threadIdx.x;
    // Q8_0: 34 bytes per 32 elements = (4 byte scale + 32 int8) per block
    // For head_dim=128: 4 blocks = 136 bytes per head
    const int bytes_per_head = (head_dim / 32) * 34;  // 136
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;
    const int d0 = tid * 4;

    extern __shared__ float sdata[];
    float* scores = sdata;

    float mq[4] = {q[h*head_dim+d0], q[h*head_dim+d0+1], q[h*head_dim+d0+2], q[h*head_dim+d0+3]};

    // K dot products - read Q8 data
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + (size_t)t * bytes_per_pos + kv_head_off;
        // Decode 4 elements from Q8_0 format
        int block_idx = tid / 8;  // which 32-element block
        int in_block = tid % 8;   // position within block
        const unsigned char* bptr = kb + block_idx * 34;
        float block_scale = *(const float*)bptr;
        float partial = 0;
        for (int i = 0; i < 4; i++) {
            int elem = in_block * 4 + i;
            float val = block_scale * (float)(int8_t)bptr[4 + elem];
            partial += val * mq[i];
        }
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t] = partial * scale_attn;
    }
    __syncthreads();

    // softmax
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

    // V weighted sum
    float mv[4] = {0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_cache + (size_t)t * bytes_per_pos + kv_head_off;
        int block_idx = tid / 8;
        int in_block = tid % 8;
        const unsigned char* bptr = vb + block_idx * 34;
        float block_scale = *(const float*)bptr;
        for (int i = 0; i < 4; i++) {
            int elem = in_block * 4 + i;
            mv[i] += w * block_scale * (float)(int8_t)bptr[4 + elem];
        }
    }

    float* oh = out + h * head_dim;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
}
