#include <hip/hip_runtime.h>

// Deterministic combine over a contiguous output-row range. M remains the
// full tensor stride; ROW_BASE/ROW_COUNT only contract the launched columns.
extern "C" __global__ void moe_down_combine_k8_batched_rows(
    const float* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    float* __restrict__ x_residual,
    int M, int K_TOP, int ROW_BASE, int ROW_COUNT
) {
    const int bid = blockIdx.y;
    const int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_col >= ROW_COUNT) return;
    const int col = ROW_BASE + local_col;
    if (col >= M) return;

    const long long token_base = (long long)bid * K_TOP * M;
    const int weights_base = bid * K_TOP;
    const long long residual_off = (long long)bid * M + col;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < K_TOP) {
            acc += topk_weights[weights_base + k]
                * expert_outputs[token_base + (long long)k * M + col];
        }
    }
    x_residual[residual_off] += acc;
}
