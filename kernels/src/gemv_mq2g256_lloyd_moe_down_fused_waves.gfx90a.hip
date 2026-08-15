#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MQ2_FUSED_WAVES
#define HIPFIRE_MQ2_FUSED_WAVES 2
#endif
#ifndef HIPFIRE_MQ2_FUSED_WAVE_KERNEL
#define HIPFIRE_MQ2_FUSED_WAVE_KERNEL \
    gemv_mq2g256_lloyd_moe_down_fused_wave2_gfx90a
#endif

// One workgroup owns one output row. Native wave64s compute routed experts in
// parallel; LDS holds only their scalar dot products. Thread 0 combines those
// scalars in route order, preserving expanded+combine numerical ordering.
extern "C" __global__ __launch_bounds__(HIPFIRE_MQ2_FUSED_WAVES * 64)
void HIPFIRE_MQ2_FUSED_WAVE_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ residual,
    int M, int K, int K_TOP, int ROW_BASE, int ROW_COUNT
) {
    constexpr int WAVES = HIPFIRE_MQ2_FUSED_WAVES;
    const int local_row = blockIdx.x;
    if (local_row >= ROW_COUNT) return;
    const int row = ROW_BASE + local_row;
    if (row >= M) return;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int groups = K / 256;
    const int quads = groups >> 2;
    __shared__ float codebooks[WAVES][16];
    __shared__ float expert_acc[8];

    for (int krank = wave; krank < K_TOP; krank += WAVES) {
        const int expert_id = topk_indices[krank];
        if (expert_id < 0) {
            if (lane == 0) expert_acc[krank] = 0.0f;
            continue;
        }
        const char* A =
            reinterpret_cast<const char*>(expert_ptrs[expert_id]);
        const float* x = rot_batch + (long long)krank * K;
        const char* row_ptr = A + (long long)row * groups * 72;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        for (int q = 0; q < quads; ++q) {
            const int g = q << 2;
            const char* gp0 = row_ptr + g * 72;
            if (lane < 16) {
                const char* gp = gp0 + (lane >> 2) * 72;
                codebooks[wave][lane] =
                    __half2float(reinterpret_cast<const __half*>(gp)[lane & 3]);
            }
            __syncwarp();
            const int base = g * 256 + lane * 4;
#define DOT_GROUP(gp, cb, xb, acc) do {                                         \
            const unsigned char packed =                                        \
                *(const unsigned char*)((gp) + 8 + lane);                        \
            (acc) += codebooks[wave][(cb) + ( packed       & 3u)] * x[(xb)]     \
                   + codebooks[wave][(cb) + ((packed >> 2) & 3u)] * x[(xb) + 1]\
                   + codebooks[wave][(cb) + ((packed >> 4) & 3u)] * x[(xb) + 2]\
                   + codebooks[wave][(cb) + ((packed >> 6) & 3u)] * x[(xb) + 3];\
        } while (0)
            DOT_GROUP(gp0,       0, base,       acc0);
            DOT_GROUP(gp0 + 72,  4, base + 256, acc1);
            DOT_GROUP(gp0 + 144, 8, base + 512, acc2);
            DOT_GROUP(gp0 + 216, 12, base + 768, acc3);
#undef DOT_GROUP
            __syncwarp();
        }

        for (int g = quads << 2; g < groups; ++g) {
            const char* gp = row_ptr + g * 72;
            if (lane < 4)
                codebooks[wave][lane] =
                    __half2float(reinterpret_cast<const __half*>(gp)[lane]);
            __syncwarp();
            const unsigned char packed =
                *(const unsigned char*)(gp + 8 + lane);
            const int base = g * 256 + lane * 4;
            acc0 += codebooks[wave][ packed       & 3u] * x[base]
                  + codebooks[wave][(packed >> 2) & 3u] * x[base + 1]
                  + codebooks[wave][(packed >> 4) & 3u] * x[base + 2]
                  + codebooks[wave][(packed >> 6) & 3u] * x[base + 3];
            __syncwarp();
        }

        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (int offset = 32; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (lane == 0) expert_acc[krank] = acc;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        float weighted = 0.0f;
        for (int krank = 0; krank < K_TOP; ++krank)
            weighted += topk_weights[krank] * expert_acc[krank];
        residual[row] += weighted;
    }
}
