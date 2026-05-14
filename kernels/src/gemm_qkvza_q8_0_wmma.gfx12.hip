#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused Q8_0 GEMM for the Qwen3.5
// DeltaNet LA preamble (wqkv + wz + w_beta + w_alpha projections).
//
// **gfx12 (RDNA4) variant** — sister kernel to gemm_qkvza_q8_0_wmma.hip.
// Compile with `hipcc --offload-arch=gfx1200` (or gfx1201, ...).
//
// SCAFFOLD STATUS (2026-05-14):
//   First-pass port from gemm_qkvza_q8_0_wmma.hip following the canonical
//   pattern in gemm_qkv_hfq4g256_wmma.gfx12.hip. C-output mapping is
//   silicon-validated for the Q8 QKV sibling (test_gemm_q8_qkv_wmma on
//   R9700 / gfx1201 — 22/22 PASS, 2026-05-14). Mapping reused here.
//
// Differences from the gfx11 Q8 kernel: see gemm_qkv_q8_0_wmma.gfx12.hip.
//   1. _w32_gfx12 builtin
//   2. half8_t A/B
//   3. K split across 2 lane-groups (k_grp = tid >> 4)
//   4. C mapping: acc[j] = C[8*(tid>>4) + j][tid & 15]

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_q8_0_wmma_gfx12(
    const char*     __restrict__ A_qkv,
    const char*     __restrict__ A_z,
    const char*     __restrict__ A_beta,
    const char*     __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y_qkv,
    float*          __restrict__ Y_z,
    float*          __restrict__ Y_beta,
    float*          __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m     = qkv_m + z_m + beta_m + alpha_m;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N)     return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row   = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < qkv_m) {
        A = A_qkv;   local_row = safe_row;
    } else if (safe_row < qkv_m + z_m) {
        A = A_z;     local_row = safe_row - qkv_m;
    } else if (safe_row < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = safe_row - (qkv_m + z_m);
    } else {
        A = A_alpha; local_row = safe_row - (qkv_m + z_m + beta_m);
    }

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    const int safe_batch = ((batch_start + m_lane) < N)
                           ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        // Tile 0: K[0..16], split between lane-groups.
        const signed char* w0 = (const signed char*)(bp + 2 + k_grp * 8);
        half8_t a0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }
        half8_t b0 = *(const half8_t*)(x_base + k_grp * 8);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, acc);

        // Tile 1: K[16..32], split between lane-groups.
        const signed char* w1 = (const signed char*)(bp + 2 + 16 + k_grp * 8);
        half8_t a1;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }
        half8_t b1 = *(const half8_t*)(x_base + 16 + k_grp * 8);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, acc);
    }

    // gfx12 C-output mapping (silicon-validated on R9700, 2026-05-14):
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    const int out_col = batch_start + m_lane;
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < total_m) {
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < qkv_m) {
                Y = Y_qkv;   proj_row = out_row;                         proj_m = qkv_m;
            } else if (out_row < qkv_m + z_m) {
                Y = Y_z;     proj_row = out_row - qkv_m;                  proj_m = z_m;
            } else if (out_row < qkv_m + z_m + beta_m) {
                Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);          proj_m = beta_m;
            } else {
                Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); proj_m = alpha_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
