#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// gfx1201 (RDNA4) sister of hyper_read_up_wmma.gfx1151.hip: the HC read tail,
// the BF16 [4 * hidden, K] up projection of `low` ([N, K] BF16 bits,
// hc_activation_fused_f32's packed output) with hyper_read_projected_f32's
// branch mix fused, reading the normalized rows from hyper_norm_f16's F16
// copy.  Same block decomposition, same LDS staging and the same epilogue
// expression; only the WMMA fragment mapping changes:
//   - half-width (8-element) operands and the `_w32_gfx12` builtin: lane l
//     holds row (l & 15), K elements [8 * (l >> 4), +8) of each 16-element
//     substep, so each lane loads one uint4 of X and one of W per substep;
//   - the accumulator is contiguous per lane-half:
//     acc[i] = C[8 * (l >> 4) + i][l & 15], i.e. token t0 + 8 * (l >> 4) + i.
// Both GEMM operands are exact in BF16, so the logits differ from
// hyper_read_up_fused_f32 only by F32 summation order.
//
// Grid [hidden / 16, ceil(N / 512)], block 256, dynamic LDS 64 * (K + 8) * 2
// bytes.  hidden % 16 == 0, K % 16 == 0, K <= 504.

typedef short __attribute__((ext_vector_type(8))) hru12_bf16x8_t;
typedef float __attribute__((ext_vector_type(8))) hru12_f8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)).
static __device__ __forceinline__ float hru12_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

extern "C" __global__ __launch_bounds__(256) void hyper_read_up_wmma_bf16_gfx1201(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int TOKENS = 512;
    extern __shared__ unsigned short hru12_w[];
    const int ldw = K + 8;
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int vpr = K / 8;
    for (int v = tid; v < 64 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        *(uint4*)(hru12_w + r * ldw + c * 8) = *(const uint4*)(W + (size_t)row * K + c * 8);
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int wide = 4 * hidden;
    const int j = j0 + l16;
    const int block_end = min(N, (int)blockIdx.y * TOKENS + TOKENS);
    for (int t0 = blockIdx.y * TOKENS + wave * 16; t0 < block_end; t0 += 128) {
        // The epilogue's normalized values, fetched before the K loop so the
        // loads overlap the WMMA work.
        _Float16 nv[4][8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = min(t0 + 8 * kh + i, N - 1);
#pragma unroll
            for (int br = 0; br < 4; br++)
                nv[br][i] = normalized[(size_t)token * wide + br * hidden + j];
        }
        const unsigned short* xr = X + (size_t)min(t0 + l16, N - 1) * K + 8 * kh;
        hru12_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = hru12_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
        for (int k = 0; k < K; k += 16) {
            const hru12_bf16x8_t a = __builtin_bit_cast(hru12_bf16x8_t, *(const uint4*)(xr + k));
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const hru12_bf16x8_t b =
                    *(const hru12_bf16x8_t*)(hru12_w + (br * 16 + l16) * ldw + k + 8 * kh);
                acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[br]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = t0 + 8 * kh + i;
            if (token >= N) continue;
            // hyper_read_projected_f32's mix, branch order from +0.0.
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float gate_input = hru12_bf16r(acc[br][i]);
                const float gate = hru12_bf16r(1.0f / (1.0f + expf(-gate_input)));
                const float normed = (float)nv[br][i];
                value += hru12_bf16r(gate * normed);
            }
            mixed[(size_t)token * hidden + j] = hru12_bf16r(value / 4.0f);
        }
    }
}
