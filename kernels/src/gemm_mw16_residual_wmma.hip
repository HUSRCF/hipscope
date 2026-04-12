#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MagnumWave MW16 WMMA GEMM — pre-dequanted FP16 weights + K2 pipelining.
//
// W is FP16 [M, K] row-major (dequanted at model load time).
// X is FP16 [N, K] row-major (converted per-prefill).
// Y is FP32 [N, M] (residual add).
//
// No dequant → ~30 VGPRs (vs 47 with 4-bit dequant).
// K2 unroll: 2 WMMAs per iteration, vmcnt(2) pipelining.
// The A-load is now a simple FP16 vector load (no nibble extraction),
// so both A and B loads can pipeline with WMMA compute.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. K must be multiple of 32.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mw16_residual_wmma(
    const _Float16* __restrict__ W,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= N) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < N) ? (batch_start + (tid & 15)) : 0;

    const _Float16* wp = W + (long long)safe_row * K;
    const _Float16* xp = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    // K2: process 32 K-elements per iteration (2 WMMAs)
    for (int k = 0; k < K; k += 32) {
        // Load both A and B tiles for 2 consecutive K-tiles up front
        half16_t a0 = *(const half16_t*)(wp + k);
        half16_t b0 = *(const half16_t*)(xp + k);
        half16_t a1 = *(const half16_t*)(wp + k + 16);
        half16_t b1 = *(const half16_t*)(xp + k + 16);

        // First WMMA — a1/b1 loads may still be in flight
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
        // Second WMMA — fully pipelined
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }

    const int out_row = row_start + (tid & 15);
    if (out_row < M) {
        const int col_base = batch_start + (tid >> 4) * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (col_base + j < N)
                Y[(long long)(col_base + j) * M + out_row] += acc[j];
        }
    }
}
