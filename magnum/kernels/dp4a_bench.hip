// dp4a microbenchmark: measure v_dot4_i32_iu8 throughput on gfx1100.
// Compares 4x INT8 dot product (dp4a) vs scalar FP32 multiply-accumulate.
//
// dp4a: v_dot4_i32_iu8 — 4 INT8 MACs per instruction per lane.
// scalar: v_fmac_f32 — 1 FP32 MAC per instruction per lane.
// Expected: dp4a achieves ~4x VALU throughput.

#include <hip/hip_runtime.h>

// dp4a kernel: each thread computes N dot products of 4-element INT8 vectors.
// Uses __builtin_amdgcn_sudot4(is_signed_a, a, is_signed_b, b, acc, clamp).
extern "C" __launch_bounds__(256, 8)
__global__ void bench_dp4a(
    const int* __restrict__ a,  // [N] packed INT8×4
    const int* __restrict__ b,  // [N] packed INT8×4
    int* __restrict__ out,      // [num_threads]
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // signed × signed dot product of 4 INT8 values packed in int32
        acc = __builtin_amdgcn_sudot4(true, a[i], true, b[i], acc, false);
    }
    out[tid] = acc;
}

// Scalar FP32 baseline: same number of MACs but using v_fmac_f32.
// Each iteration does 4 separate FP32 multiply-accumulates to match dp4a's 4 MACs.
extern "C" __launch_bounds__(256, 8)
__global__ void bench_fp32(
    const float* __restrict__ a,  // [N*4]
    const float* __restrict__ b,  // [N*4]
    float* __restrict__ out,      // [num_threads]
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int base = i * 4;
        acc += a[base]     * b[base];
        acc += a[base + 1] * b[base + 1];
        acc += a[base + 2] * b[base + 2];
        acc += a[base + 3] * b[base + 3];
    }
    out[tid] = acc;
}
