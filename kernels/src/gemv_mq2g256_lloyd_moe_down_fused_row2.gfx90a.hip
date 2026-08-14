#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MQ2_FUSED_ROW_TILE
#define HIPFIRE_MQ2_FUSED_ROW_TILE 2
#endif
#ifndef HIPFIRE_MQ2_FUSED_KERNEL
#define HIPFIRE_MQ2_FUSED_KERNEL gemv_mq2g256_lloyd_moe_down_fused_row2_gfx90a
#endif

// Deterministic fused MQ2-Lloyd MoE down for CDNA2. One wave owns ROW_TILE
// output rows, loops routed experts in fixed order, and commits each cell once.
extern "C" __global__ __launch_bounds__(64)
void HIPFIRE_MQ2_FUSED_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ residual,
    int M, int K, int K_TOP, int ROW_BASE, int ROW_COUNT
) {
    constexpr int ROW_TILE = HIPFIRE_MQ2_FUSED_ROW_TILE;
    const int local_row0 = blockIdx.x * ROW_TILE;
    if (local_row0 >= ROW_COUNT) return;
    const int row0 = ROW_BASE + local_row0;
    const int valid_rows = min(ROW_TILE, ROW_COUNT - local_row0);
    const int lane = threadIdx.x;
    const int groups = K / 256;
    const int quads = groups >> 2;
    __shared__ float codebooks[ROW_TILE][16];

    float weighted[ROW_TILE] = {};
    for (int krank = 0; krank < K_TOP; ++krank) {
        const char* A =
            reinterpret_cast<const char*>(expert_ptrs[topk_indices[krank]]);
        const float* x = rot_batch + (long long)krank * K;
        const float weight = topk_weights[krank];
        float acc0[ROW_TILE] = {};
        float acc1[ROW_TILE] = {};
        float acc2[ROW_TILE] = {};
        float acc3[ROW_TILE] = {};

        for (int q = 0; q < quads; ++q) {
            const int g = q << 2;
            if (lane < ROW_TILE * 16) {
                const int r = lane >> 4;
                const int cb = lane & 15;
                const int row = min(row0 + r, M - 1);
                const char* gp = A + (long long)row * groups * 72
                    + (g + (cb >> 2)) * 72;
                codebooks[r][cb] =
                    __half2float(reinterpret_cast<const __half*>(gp)[cb & 3]);
            }
            __syncwarp();
            const int base = g * 256 + lane * 4;

            #pragma unroll
            for (int r = 0; r < ROW_TILE; ++r) {
                const int row = min(row0 + r, M - 1);
                const char* gp0 = A + (long long)row * groups * 72 + g * 72;
#define DOT_GROUP(gp, cb, a, xb) do {                                           \
                const unsigned char packed =                                    \
                    *(const unsigned char*)((gp) + 8 + lane);                    \
                (a) += codebooks[r][(cb) + ( packed       & 3u)] * x[(xb)]      \
                     + codebooks[r][(cb) + ((packed >> 2) & 3u)] * x[(xb) + 1] \
                     + codebooks[r][(cb) + ((packed >> 4) & 3u)] * x[(xb) + 2] \
                     + codebooks[r][(cb) + ((packed >> 6) & 3u)] * x[(xb) + 3];\
            } while (0)
                DOT_GROUP(gp0,       0, acc0[r], base);
                DOT_GROUP(gp0 + 72,  4, acc1[r], base + 256);
                DOT_GROUP(gp0 + 144, 8, acc2[r], base + 512);
                DOT_GROUP(gp0 + 216, 12, acc3[r], base + 768);
#undef DOT_GROUP
            }
            __syncwarp();
        }

        for (int g = quads << 2; g < groups; ++g) {
            if (lane < ROW_TILE * 4) {
                const int r = lane >> 2;
                const int cb = lane & 3;
                const int row = min(row0 + r, M - 1);
                const char* gp = A + (long long)row * groups * 72 + g * 72;
                codebooks[r][cb] =
                    __half2float(reinterpret_cast<const __half*>(gp)[cb]);
            }
            __syncwarp();
            const int base = g * 256 + lane * 4;
            #pragma unroll
            for (int r = 0; r < ROW_TILE; ++r) {
                const int row = min(row0 + r, M - 1);
                const char* gp = A + (long long)row * groups * 72 + g * 72;
                const unsigned char packed =
                    *(const unsigned char*)(gp + 8 + lane);
                acc0[r] += codebooks[r][ packed       & 3u] * x[base]
                         + codebooks[r][(packed >> 2) & 3u] * x[base + 1]
                         + codebooks[r][(packed >> 4) & 3u] * x[base + 2]
                         + codebooks[r][(packed >> 6) & 3u] * x[base + 3];
            }
            __syncwarp();
        }

        #pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
            for (int offset = 32; offset > 0; offset >>= 1)
                acc += __shfl_down(acc, offset);
            if (lane == 0) weighted[r] += weight * acc;
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            if (r < valid_rows && row0 + r < M)
                residual[row0 + r] += weighted[r];
        }
    }
}
