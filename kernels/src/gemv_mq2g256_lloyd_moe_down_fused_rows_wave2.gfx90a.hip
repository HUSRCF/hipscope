#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Two independent native wave64s compute two output rows per workgroup. Each
// wave retains the row1 arithmetic order so the result matches the expanded
// down + ordered expert combine path bit for bit.
extern "C" __global__ __launch_bounds__(128)
void gemv_mq2g256_lloyd_moe_down_fused_rows_wave2_gfx90a(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float* __restrict__ residual,
    int M, int K, int K_TOP, int ROW_BASE, int ROW_COUNT
) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int local_row = blockIdx.x * 2 + wave;
    if (local_row >= ROW_COUNT) return;
    const int row = ROW_BASE + local_row;
    if (row >= M) return;

    const int groups = K / 256;
    const int quads = groups >> 2;
    __shared__ float codebooks[2][16];
    float weighted = 0.0f;

    for (int krank = 0; krank < K_TOP; ++krank) {
        const char* A =
            reinterpret_cast<const char*>(expert_ptrs[topk_indices[krank]]);
        const float* x = rot_batch + (long long)krank * K;
        const char* row_ptr = A + (long long)row * groups * 72;
        const float weight = topk_weights[krank];
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
#define DOT_GROUP(gp, cb, a, xb) do {                                          \
            const unsigned char packed =                                       \
                *(const unsigned char*)((gp) + 8 + lane);                       \
            (a) += codebooks[wave][(cb) + ( packed       & 3u)] * x[(xb)]      \
                 + codebooks[wave][(cb) + ((packed >> 2) & 3u)] * x[(xb) + 1] \
                 + codebooks[wave][(cb) + ((packed >> 4) & 3u)] * x[(xb) + 2] \
                 + codebooks[wave][(cb) + ((packed >> 6) & 3u)] * x[(xb) + 3];\
        } while (0)
            DOT_GROUP(gp0,       0, acc0, base);
            DOT_GROUP(gp0 + 72,  4, acc1, base + 256);
            DOT_GROUP(gp0 + 144, 8, acc2, base + 512);
            DOT_GROUP(gp0 + 216, 12, acc3, base + 768);
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
        if (lane == 0) weighted += weight * acc;
    }

    if (lane == 0) residual[row] += weighted;
}
