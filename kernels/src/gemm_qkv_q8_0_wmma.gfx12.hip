#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused Q8_0 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections from one normed X).
//
// **gfx12 (RDNA4) variant** — sister kernel to gemm_qkv_q8_0_wmma.hip
// (gfx11 / RDNA3). Compile this file with
//   hipcc --offload-arch=gfx1200  (or gfx1201, ...).
//
// SCAFFOLD STATUS (2026-05-14):
//   First-pass port from gemm_qkv_q8_0_wmma.hip following the canonical
//   pattern in gemm_qkv_hfq4g256_wmma.gfx12.hip. C-output mapping is
//   the same HYPOTHESIS the HFQ4 sibling uses (derived from CK trait
//   kCM0/kCM1PerLane swap). MUST be runtime-validated on real RDNA4
//   silicon (R9700 / 9070 XT) via test_gemm_q8_qkv_wmma before this
//   file is allowed to win on any production dispatch site.
//
// Differences from the gfx11 Q8 kernel:
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (gfx11 uses _w32 without the _gfx12 suffix).
//   2. A and B operands: half8_t (8 fp16 per lane) instead of half16_t
//      (16 fp16 per lane). Accumulator C is unchanged at float8_t.
//   3. K dimension is split across 2 lane-groups (kABKLane=2) instead
//      of replicated across all 32 lanes (gfx11 kABKLane=1):
//        lane (tid >> 4) = 0  -> carries K = [0..7]  of the 16-K tile
//        lane (tid >> 4) = 1  -> carries K = [8..15] of the 16-K tile
//      Each lane therefore loads HALF as much per WMMA tile as gfx11.
//   4. C-output mapping (HYPOTHESIS — see below):
//        gfx11 (validated, commit b7ac66a):
//          acc[j] = C[2*j + (tid>>4)][tid & 15]
//        gfx12 (UNVERIFIED, derived from CK kCM0/kCM1PerLane swap):
//          acc[j] = C[8*(tid>>4) + j][tid & 15]
//
// Per Q8_0 block (K=32 elements, 34 bytes): 1 fp16 scale + 32 int8 weights.
// One Q8 block = 2 WMMA-K-tiles (each tile is 16 K-elements). Each gfx12
// lane carries 8 of those 16 K-elements per WMMA tile.
//
// Grid:  [ceil((q_m + k_m + v_m)/16), ceil(N/16)]
// Block: [32] (wave32).  LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_q8_0_wmma_gfx12(
    const char*     __restrict__ A_q,    // [q_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_k,    // [k_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_v,    // [v_m, K/32 * 34] Q8_0
    const _Float16* __restrict__ X,      // [N, K] FP16
    float*          __restrict__ Y_q,    // [N, q_m] FP32
    float*          __restrict__ Y_k,    // [N, k_m] FP32
    float*          __restrict__ Y_v,    // [N, v_m] FP32
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m     = q_m + k_m + v_m;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N)     return;

    // Lane decomposition (gfx12 wave32 WMMA):
    //   m_lane = tid & 15   (0..15)  -> row index within the 16-row tile
    //   k_grp  = tid >> 4   (0 or 1) -> which K half-tile this lane carries
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row   = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    // Route to the correct weight matrix for this thread's row.
    const char* A;
    int local_row;
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    const int safe_batch = ((batch_start + m_lane) < N)
                           ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        // fp16 scale at byte offset 0 (2 bytes). May be 2-byte aligned but
        // not 4-byte — load via memcpy to avoid unaligned-access UB.
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        // === WMMA tile 0 (K-elements 0..16, split between lane-groups) ===
        //   lane_grp 0 carries weights bp[2 + 0..7] and X[bi*32 + 0..7]
        //   lane_grp 1 carries weights bp[2 + 8..15] and X[bi*32 + 8..15]
        const signed char* w0 = (const signed char*)(bp + 2 + k_grp * 8);
        half8_t a0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }
        half8_t b0 = *(const half8_t*)(x_base + k_grp * 8);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, acc);

        // === WMMA tile 1 (K-elements 16..32, split between lane-groups) ===
        //   lane_grp 0 carries weights bp[2 + 16 + 0..7] and X[bi*32 + 16 + 0..7]
        //   lane_grp 1 carries weights bp[2 + 16 + 8..15] and X[bi*32 + 16 + 8..15]
        const signed char* w1 = (const signed char*)(bp + 2 + 16 + k_grp * 8);
        half8_t a1;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }
        half8_t b1 = *(const half8_t*)(x_base + 16 + k_grp * 8);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, acc);
    }

    // C-output mapping HYPOTHESIS — mirrors gemm_qkv_hfq4g256_wmma.gfx12:
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    // i.e. lane group 0 holds output rows 0..7, lane group 1 rows 8..15.
    // Validate against gfx11 reference for a small golden case before
    // trusting at scale.
    const int out_col = batch_start + m_lane;
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < total_m) {
            // Route this row to the correct output matrix.
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < q_m) {
                Y = Y_q;  proj_row = out_row;                 proj_m = q_m;
            } else if (out_row < q_m + k_m) {
                Y = Y_k;  proj_row = out_row - q_m;            proj_m = k_m;
            } else {
                Y = Y_v;  proj_row = out_row - q_m - k_m;      proj_m = v_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
