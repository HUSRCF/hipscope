#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Native wave64 MQ2-Lloyd GEMV: one complete wave computes one output row.
extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx90a(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int lane = threadIdx.x;
    const int expert_id = topk_indices[krank];
    if (expert_id < 0) {
        if (lane == 0) {
            const int mi = M >> 1;
            if (row < mi) y_gate[(long long)krank * mi + row] = 0.0f;
            else y_up[(long long)krank * mi + row - mi] = 0.0f;
        }
        return;
    }
    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
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
    if (lane == 0) {
        const int mi = M >> 1;
        if (row < mi) y_gate[krank * mi + row] = acc;
        else y_up[krank * mi + row - mi] = acc;
    }
}
