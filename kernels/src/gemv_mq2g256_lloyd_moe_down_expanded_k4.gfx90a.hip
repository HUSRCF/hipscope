#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx90a(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M, int K, int K_TOP, int ROW_BASE, int ROW_COUNT
) {
    const int local_row = blockIdx.x;
    if (local_row >= ROW_COUNT) return;
    const int row = ROW_BASE + local_row;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = threadIdx.x;
    const int route = bid * K_TOP + krank;
    const char* A = reinterpret_cast<const char*>(expert_ptrs[topk_indices[route]]);
    const float* x = rot_batch + (long long)route * K;
    const int groups = K / 256;
    const char* row_ptr = A + (long long)row * groups * 72;
    __shared__ float codebooks[16];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups >> 2;

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        if (lane < 16) {
            const __half* cb = reinterpret_cast<const __half*>(gp0 + (lane >> 2) * 72);
            codebooks[lane] = __half2float(cb[lane & 3]);
        }
        __syncwarp();
        const int base = g * 256 + lane * 4;
#define DOT_GROUP(gp, cb, xbase, acc) do {                                      \
        const unsigned char packed = *(const unsigned char*)((gp) + 8 + lane);  \
        (acc) += codebooks[(cb) + ( packed       & 3u)] * x[(xbase)]            \
               + codebooks[(cb) + ((packed >> 2) & 3u)] * x[(xbase) + 1]       \
               + codebooks[(cb) + ((packed >> 4) & 3u)] * x[(xbase) + 2]       \
               + codebooks[(cb) + ((packed >> 6) & 3u)] * x[(xbase) + 3];      \
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
            codebooks[lane] = __half2float(reinterpret_cast<const __half*>(gp)[lane]);
        __syncwarp();
        const unsigned char packed = *(const unsigned char*)(gp + 8 + lane);
        const int base = g * 256 + lane * 4;
        acc0 += codebooks[ packed       & 3u] * x[base]
              + codebooks[(packed >> 2) & 3u] * x[base + 1]
              + codebooks[(packed >> 4) & 3u] * x[base + 2]
              + codebooks[(packed >> 6) & 3u] * x[base + 3];
        __syncwarp();
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 32; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (lane == 0) expert_outputs[(long long)route * M + row] = acc;
}
