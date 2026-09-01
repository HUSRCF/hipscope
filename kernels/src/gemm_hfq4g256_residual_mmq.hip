// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Experimental HFQ4-G256 MMQ residual GEMM for RDNA3/RDNA3.5.
//
// This mirrors the important parts of llama.cpp's AMD MMQ path:
// - pre-quantize the activation matrix into block_q8_1_mmq DS4 layout
// - process 128 output rows x 128 batch columns per workgroup
// - stage both HFQ4 weights and Q8_1 activations in LDS
// - use i8 WMMA for the dot product and apply HFQ4 scale/zero via Q8_1
//   scale/sum correction.

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define HIPFIRE_RDNA3 1
#else
#define HIPFIRE_RDNA3 0
#endif

#ifndef MMQ_X
#define MMQ_X 128
#endif
#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#ifndef MMQ_NWARPS
#define MMQ_NWARPS 8
#endif
#ifndef MMQ_MIN_BLOCKS_PER_CU
#define MMQ_MIN_BLOCKS_PER_CU 2
#endif
#ifndef MMQ_ROW_FRAGS
#define MMQ_ROW_FRAGS 2
#endif
#ifndef MMQ_COL_GROUPS
#define MMQ_COL_GROUPS 2
#endif
#ifndef MMQ_PERM_NIBBLE
#define MMQ_PERM_NIBBLE 0
#endif
#ifndef MMQ_META_SINGLE_LOADER
#define MMQ_META_SINGLE_LOADER 0
#endif
#ifndef MMQ_WEIGHT_EXPANDED_I8
#define MMQ_WEIGHT_EXPANDED_I8 0
#endif
#ifndef MMQ_Q8_GROUP128
#define MMQ_Q8_GROUP128 0
#endif
#ifndef MMQ_DIRECT_PACKED_WEIGHT_N2
#define MMQ_DIRECT_PACKED_WEIGHT_N2 0
#endif
#ifndef MMQ_ACCUM_F16
#define MMQ_ACCUM_F16 0
#endif
#ifndef MMQ_DUAL_GATE_UP
#define MMQ_DUAL_GATE_UP 0
#endif
#ifndef MMQ_DUAL_PAYLOAD_ONLY
#define MMQ_DUAL_PAYLOAD_ONLY 0
#endif
#ifndef MMQ_DUAL_INTERLEAVED
#define MMQ_DUAL_INTERLEAVED 0
#endif
#ifndef MMQ_PACKED_DUAL_X256
#define MMQ_PACKED_DUAL_X256 0
#endif
#ifndef MMQ_PACKED_DUAL_ONE_PLANE
#define MMQ_PACKED_DUAL_ONE_PLANE 0
#endif
#ifndef MMQ_Q8_GROUP256_DIRECT
#define MMQ_Q8_GROUP256_DIRECT 0
#endif
#ifndef MMQ_Q8_GROUP256_SERIAL_ROW
#define MMQ_Q8_GROUP256_SERIAL_ROW 0
#endif
#ifndef MMQ_Q8_GROUP128_SERIAL_ROW
#define MMQ_Q8_GROUP128_SERIAL_ROW 0
#endif
#ifndef MMQ_Q8_GROUP128_DIRECT
#define MMQ_Q8_GROUP128_DIRECT 0
#endif
#ifndef MMQ_Q8_GROUP128_K32_STATIONARY
#define MMQ_Q8_GROUP128_K32_STATIONARY 0
#endif
#ifndef MMQ_SKIP_ZERO_CORRECTION
#define MMQ_SKIP_ZERO_CORRECTION 0
#endif
#ifndef MMQ_STREAM_K128
#define MMQ_STREAM_K128 0
#endif
#ifndef MMQ_STREAM_K128_PHASED_X256
#define MMQ_STREAM_K128_PHASED_X256 0
#endif
#ifndef MMQ_STREAM_K128_X256Y128
#define MMQ_STREAM_K128_X256Y128 0
#endif
#ifndef MMQ_Q8_GROUP256_STAGED
#define MMQ_Q8_GROUP256_STAGED 0
#endif
#ifndef MMQ_PACKED_WEIGHT_Y128
#define MMQ_PACKED_WEIGHT_Y128 0
#endif
#ifndef MMQ_WEIGHT_HALF_META
#define MMQ_WEIGHT_HALF_META 0
#endif
#ifndef MMQ_WEIGHT_DUAL_ROW_U32X2
#define MMQ_WEIGHT_DUAL_ROW_U32X2 0
#endif
#ifndef MMQ_WEIGHT_DUAL_ROW_SCALAR2
#define MMQ_WEIGHT_DUAL_ROW_SCALAR2 0
#endif
#ifndef MMQ_WEIGHT_QUAD_ROW_U32X2
#define MMQ_WEIGHT_QUAD_ROW_U32X2 0
#endif
#ifndef MMQ_WEIGHT_OCT_ROW_U32X2
#define MMQ_WEIGHT_OCT_ROW_U32X2 0
#endif
#ifndef MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4
#define MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4 0
#endif
#ifndef MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4
#define MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4 0
#endif
#ifndef MMQ_VECTOR_ACTIVATION_STAGING
#define MMQ_VECTOR_ACTIVATION_STAGING 0
#endif
#ifndef MMQ_VECTOR_ACTIVATION_STAGING_BATCH3
#define MMQ_VECTOR_ACTIVATION_STAGING_BATCH3 0
#endif
#ifndef MMQ_WARP_SPECIALIZED_STAGE
#define MMQ_WARP_SPECIALIZED_STAGE 0
#endif
#ifndef MMQ_WEIGHT_PREFETCH_NEXT
#define MMQ_WEIGHT_PREFETCH_NEXT 0
#endif
#define HFQ4_GROUP_BYTES (MMQ_WEIGHT_EXPANDED_I8 ? 264 : 136)
#define WARP_SIZE 32
#define QK8_1 32
#define QI8_1 8
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K 36
#define MMQ_STREAM_WEIGHT_K128 \
    (MMQ_STREAM_K128 || MMQ_STREAM_K128_PHASED_X256 || MMQ_STREAM_K128_X256Y128)
#define MMQ_TILE_X_K (MMQ_STREAM_WEIGHT_K128 ? 36 : 76)
#define MMQ_X_DM_OFFSET (MMQ_STREAM_WEIGHT_K128 ? MMQ_TILE_NE_K : 2 * MMQ_TILE_NE_K)

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Group128 activation format used by the packed dual-output probe. It keeps
// the shared scale and all four affine sums, but removes duplicated scales and
// padding from the 144-byte production wire format.
struct block_q8_1_group128_compact {
    half2 d_sum0;
    half2 sum1_sum2;
    half2 sum3_pad;
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_group128_compact) == 140,
              "bad compact group128 size");

// Q8_1 activation quantize: arch-portable, used by RDNA3 (this file's WMMA
// MMQ kernel) and gfx906 (the dp4a MMQ kernel in
// kernels/src/gemm_hfq4g256_residual_mmq_gfx906_*.hip). The body uses only
// width=32 shuffles, which are wave64-safe: each 32-lane half-wave reduces
// independently within the wave, and the kernel's amax+sum are per-32-lane
// (one Q8_1 sub-block per 32 lanes). Block dim is (256, 1, 1); each thread
// processes 4 float inputs. Per-32-lane group computes amax + sum over its
// 128-element Q8_1 sub-block.
extern "C" __global__ void quantize_q8_1_mmq_ds4(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}

// Diagnostic activation format retaining the block_q8_1_mmq wire layout but
// sharing one scale across each 128-value block. The four ds4 entries retain
// independent 32-value sums for the affine zero-point correction.
extern "C" __global__ void quantize_q8_1_mmq_ds4_group128(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;
    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}

// Standalone group256 probe. Adjacent 128-value Q8 blocks share one scale,
// while each 32-value subblock retains its own affine sum.
extern "C" __global__ void quantize_q8_1_mmq_ds4_group256(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    __shared__ float wave_amax[8];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    if (lane == 0) wave_amax[wave] = amax;
    __syncthreads();
    const float pair_amax = fmaxf(wave_amax[wave & ~1], wave_amax[wave | 1]);

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }
    const float d = pair_amax > 1.0e-20f ? pair_amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;
    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}

extern "C" __global__ void quantize_q8_1_group128_compact(
    const float* __restrict__ X,
    block_q8_1_group128_compact* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_group128_compact* out =
        Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;
    if (iqs == 0) {
        const float sum1 = __shfl(sum, 8, WARP_SIZE);
        const float sum2 = __shfl(sum, 16, WARP_SIZE);
        const float sum3 = __shfl(sum, 24, WARP_SIZE);
        out->d_sum0 = make_half2((_Float16)d, (_Float16)sum);
        out->sum1_sum2 = make_half2((_Float16)sum1, (_Float16)sum2);
        out->sum3_pad = make_half2((_Float16)sum3, (_Float16)0.0f);
    }
}

#if !HIPFIRE_RDNA3

// MMQ compute kernels — RDNA3 only (use WMMA i8 builtin). gfx906 dp4a
// implementation lives in gemm_hfq4g256_residual_mmq.gfx906.hip.
extern "C" __global__ void gemm_hfq4g256_residual_mmq(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_full_add(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_full_set(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_full_add_regzero(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_full_set_regzero(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_gate_up_full_set(
    const char*, const char*, const block_q8_1_mmq*, float*, float*, int, int, int) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_gate_up_dual_group128(
    const char*, const char*, const block_q8_1_mmq*, float*, float*, int, int, int) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_gate_up_dual_payload_only(
    const char*, const char*, const block_q8_1_mmq*, float*, float*, int, int, int) {}

extern "C" __global__ void gemm_hfq4g256_residual_mmq_gate_up_packed_dual_x256(
    const char*, const char*, const block_q8_1_group128_compact*,
    float*, float*, int, int, int) {}

#else

template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};

template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
#if HIPFIRE_RDNA3
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
#else
        return ne * (threadIdx.x / 16) + l;
#endif

    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return l;
    }
};

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride) {
#if HIPFIRE_RDNA3
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
#else
    static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad layout");
    static_assert(sizeof(t.x) == 16, "bad tile size");
    memcpy_aligned<16>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
#endif
}

template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}

// quantize_q8_1_mmq_ds4 hoisted above the HIPFIRE_RDNA3 split (line ~38)
// because the body is wave-portable. Phase 0 §3.5 verified.

template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    #if MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4
    static_assert((MMQ_Y % (4 * MMQ_NWARPS)) == 0,
                  "tile64 planar quad-row loader requires complete row groups");
    const int txi = threadIdx.x & 7;
    for (int i0 = 0; i0 < MMQ_Y; i0 += 4 * MMQ_NWARPS) {
        const int i = i0 + 4 * threadIdx.y + (threadIdx.x >> 3);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const long long group =
            (((long long)(row >> 6) * groups_per_row + kb) << 6) + (row & 63);
        const uint4 words = *(const uint4*)(A + group * 128 + txi * 16);
        int* dst = x_qs + i * MMQ_TILE_X_K + 8 * txi;
        unsigned int q = words.x;
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.z;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.w;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
    #elif MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4
    static_assert((MMQ_Y % (4 * MMQ_NWARPS)) == 0,
                  "planar quad-row loader requires complete row groups");
    const int txi = threadIdx.x & 7;
    for (int i0 = 0; i0 < MMQ_Y; i0 += 4 * MMQ_NWARPS) {
        const int i = i0 + 4 * threadIdx.y + (threadIdx.x >> 3);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const long long group = (long long)row * groups_per_row + kb;
        const uint4 words = *(const uint4*)(A + group * 128 + txi * 16);
        int* dst = x_qs + i * MMQ_TILE_X_K + 8 * txi;
        unsigned int q = words.x;
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.z;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.w;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
    #elif MMQ_WEIGHT_OCT_ROW_U32X2
    static_assert((MMQ_Y % (8 * MMQ_NWARPS)) == 0,
                  "oct-row loader requires complete row groups");
    const int txi = threadIdx.x & 3;
    for (int i0 = 0; i0 < MMQ_Y; i0 += 8 * MMQ_NWARPS) {
        const int i = i0 + 8 * threadIdx.y + (threadIdx.x >> 2);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const uint2* packed = (const uint2*)(gp + 8 + txi * 32);
        int* dst = x_qs + i * MMQ_TILE_X_K + 16 * txi;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const uint2 words = packed[pair];
            unsigned int q = words.x;
            unsigned int lo = q & 0x0F0F0F0Fu;
            unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[4 * pair + 0] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[4 * pair + 1] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            q = words.y;
            lo = q & 0x0F0F0F0Fu;
            hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[4 * pair + 2] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[4 * pair + 3] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    }
    #elif MMQ_WEIGHT_QUAD_ROW_U32X2
    #if MMQ_WARP_SPECIALIZED_STAGE
    static_assert(MMQ_NWARPS == 8 && MMQ_Y == 64,
                  "warp-specialized staging requires X256/Y64 with eight waves");
    if (threadIdx.y < 4) {
        const int txi = threadIdx.x & 7;
        for (int i0 = 0; i0 < MMQ_Y; i0 += 16) {
            const int i = i0 + 4 * threadIdx.y + (threadIdx.x >> 3);
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp =
                A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
            const uint2* packed = (const uint2*)(gp + 8 + txi * 16);
            int* dst = x_qs + i * MMQ_TILE_X_K + 8 * txi;
            uint2 words = packed[0];
            unsigned int q = words.x;
            unsigned int lo = q & 0x0F0F0F0Fu;
            unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            q = words.y;
            lo = q & 0x0F0F0F0Fu;
            hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            words = packed[1];
            q = words.x;
            lo = q & 0x0F0F0F0Fu;
            hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            q = words.y;
            lo = q & 0x0F0F0F0Fu;
            hi = (q >> 4) & 0x0F0F0F0Fu;
            dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    }
    #else
    static_assert((MMQ_Y % (4 * MMQ_NWARPS)) == 0,
                  "quad-row loader requires complete row groups");
    const int txi = threadIdx.x & 7;
    for (int i0 = 0; i0 < MMQ_Y; i0 += 4 * MMQ_NWARPS) {
        const int i = i0 + 4 * threadIdx.y + (threadIdx.x >> 3);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const uint2* packed = (const uint2*)(gp + 8 + txi * 16);
        int* dst = x_qs + i * MMQ_TILE_X_K + 8 * txi;
        uint2 words = packed[0];
        unsigned int q = words.x;
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        words = packed[1];
        q = words.x;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
    #endif
    #elif MMQ_WEIGHT_DUAL_ROW_U32X2 || MMQ_WEIGHT_DUAL_ROW_SCALAR2
    static_assert((MMQ_Y % (2 * MMQ_NWARPS)) == 0,
                  "dual-row loader requires complete row groups");
    const int txi = threadIdx.x & 15;
    for (int i0 = 0; i0 < MMQ_Y; i0 += 2 * MMQ_NWARPS) {
        const int i = i0 + 2 * threadIdx.y + (threadIdx.x >> 4);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        int* dst = x_qs + i * MMQ_TILE_X_K + 4 * txi;
        #if MMQ_WEIGHT_DUAL_ROW_U32X2
        const uint2 packed = *(const uint2*)(gp + 8 + txi * 8);
        unsigned int q = packed.x;
        #else
        const unsigned int* packed = (const unsigned int*)(gp + 8 + txi * 8);
        unsigned int q = packed[0];
        #endif
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        #if MMQ_WEIGHT_DUAL_ROW_U32X2
        q = packed.y;
        #else
        q = packed[1];
        #endif
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
    #else
    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;
    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        #if MMQ_WEIGHT_EXPANDED_I8
        const int2 expanded = *(const int2*)(gp + 8 + txi * 8);
        const int int_a = expanded.x;
        const int int_b = expanded.y;
        #else
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + txi * 4);
        #if MMQ_PERM_NIBBLE
        const unsigned int lo = qs0 & 0x0F0F0F0Fu;
        const unsigned int hi = (qs0 >> 4) & 0x0F0F0F0Fu;
        const int int_a = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        const int int_b = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        #else
        const unsigned int q0 = (qs0 >>  0) & 0xFu;
        const unsigned int q1 = (qs0 >>  4) & 0xFu;
        const unsigned int q2 = (qs0 >>  8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        const int int_a = (int)(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
        const int int_b = (int)(q4 | (q5 << 8) | (q6 << 16) | (q7 << 24));
        #endif
        #endif
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = int_a;
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = int_b;
    }
    #endif

    #if MMQ_WARP_SPECIALIZED_STAGE
    if (threadIdx.y < 4) {
        const int i = threadIdx.y * 16 + threadIdx.x / 2;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp =
            A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const int ksc = threadIdx.x & 1;
        const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
        }
    }
    #elif MMQ_WEIGHT_HALF_META
    constexpr int rows_per_wave_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_wave_meta) {
        const int i = i0 + threadIdx.y * rows_per_wave_meta + threadIdx.x / 2;
        if (i < MMQ_Y && (threadIdx.x & 1) == 0) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const long long group = (long long)row * groups_per_row + kb;
            #if MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4
            const long long planar_group =
                (((long long)(row >> 6) * groups_per_row + kb) << 6) + (row & 63);
            const char* gp =
                A + (long long)M * groups_per_row * 128 + planar_group * 8;
            #elif MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4
            const char* gp = A + (long long)M * groups_per_row * 128 + group * 8;
            #else
            const char* gp = A + group * HFQ4_GROUP_BYTES;
            #endif
            x_dm[i * MMQ_TILE_X_K] =
                __builtin_bit_cast(half2, *(const unsigned int*)gp);
        }
    }
    #else
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
#if MMQ_META_SINGLE_LOADER
            if ((threadIdx.x & 1) != 0) continue;
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const long long group = (long long)row * groups_per_row + kb;
            #if MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4
            const long long planar_group =
                (((long long)(row >> 6) * groups_per_row + kb) << 6) + (row & 63);
            const char* gp =
                A + (long long)M * groups_per_row * 128 + planar_group * 8;
            #elif MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4
            const char* gp = A + (long long)M * groups_per_row * 128 + group * 8;
            #else
            const char* gp = A + group * HFQ4_GROUP_BYTES;
            #endif
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
            const unsigned int dm_bits = __builtin_bit_cast(unsigned int, dm);
            const uint4 repeated = make_uint4(dm_bits, dm_bits, dm_bits, dm_bits);
            uint4* dst = reinterpret_cast<uint4*>(x_dm + i * MMQ_TILE_X_K);
            dst[0] = repeated;
            dst[1] = repeated;
#else
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const long long group = (long long)row * groups_per_row + kb;
            #if MMQ_WEIGHT_TILE64_PLANAR_QUAD_UINT4
            const long long planar_group =
                (((long long)(row >> 6) * groups_per_row + kb) << 6) + (row & 63);
            const char* gp =
                A + (long long)M * groups_per_row * 128 + planar_group * 8;
            #elif MMQ_WEIGHT_PLANAR_QUAD_ROW_UINT4
            const char* gp = A + (long long)M * groups_per_row * 128 + group * 8;
            #else
            const char* gp = A + group * HFQ4_GROUP_BYTES;
            #endif
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const int ksc = threadIdx.x & 1;
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
            }
#endif
        }
    }
    #endif
}

#if MMQ_WEIGHT_PREFETCH_NEXT
struct hfq4_quad_row_prefetch {
    uint2 payload0[2];
    uint2 payload1[2];
    uint2 metadata[2];
};

template <bool FULL>
static __device__ __forceinline__ void prefetch_hfq4_quad_row(
    const char* __restrict__ A,
    hfq4_quad_row_prefetch& prefetched,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    static_assert(MMQ_NWARPS == 8 && MMQ_Y == 64,
                  "next-group prefetch requires X256/Y64 with eight waves");
    const int txi = threadIdx.x & 7;
    #pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
        const int i = row_half * 32 + 4 * threadIdx.y + (threadIdx.x >> 3);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp =
            A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const uint2* packed = (const uint2*)(gp + 8 + txi * 16);
        prefetched.payload0[row_half] = packed[0];
        prefetched.payload1[row_half] = packed[1];
        if (txi < 2) {
            prefetched.metadata[row_half] = *(const uint2*)gp;
        }
    }
}

static __device__ __forceinline__ void commit_hfq4_quad_row(
    const hfq4_quad_row_prefetch& prefetched,
    int* __restrict__ tile_x
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int txi = threadIdx.x & 7;
    #pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
        const int i = row_half * 32 + 4 * threadIdx.y + (threadIdx.x >> 3);
        int* dst = x_qs + i * MMQ_TILE_X_K + 8 * txi;

        unsigned int q = prefetched.payload0[row_half].x;
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = prefetched.payload0[row_half].y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = prefetched.payload1[row_half].x;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = prefetched.payload1[row_half].y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);

        if (txi < 2) {
            const float sc =
                __builtin_bit_cast(float, prefetched.metadata[row_half].x);
            const float zp =
                __builtin_bit_cast(float, prefetched.metadata[row_half].y);
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                x_dm[i * MMQ_TILE_X_K + 4 * txi + l] = dm;
            }
        }
    }
}
#endif

#if MMQ_STREAM_WEIGHT_K128
// Load one 128-K half of an HFQ4 group. Each half-wave expands one row,
// allowing all lanes to participate while keeping only half a group in LDS.
template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile_window(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int window,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + MMQ_X_DM_OFFSET);
    const int lane = threadIdx.x;
    const int chunk = lane & 15;

    for (int i0 = 0; i0 < MMQ_Y; i0 += 2 * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y + (lane >> 4) * MMQ_NWARPS;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const unsigned int qs0 = *(const unsigned int*)(
            gp + 8 + window * 64 + chunk * 4);
#if MMQ_PERM_NIBBLE
        const unsigned int lo = qs0 & 0x0F0F0F0Fu;
        const unsigned int hi = (qs0 >> 4) & 0x0F0F0F0Fu;
        const int int_a = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        const int int_b = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
#else
        const unsigned int q0 = (qs0 >> 0) & 0xFu;
        const unsigned int q1 = (qs0 >> 4) & 0xFu;
        const unsigned int q2 = (qs0 >> 8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        const int int_a = (int)(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
        const int int_b = (int)(q4 | (q5 << 8) | (q6 << 16) | (q7 << 24));
#endif

        x_qs[i * MMQ_TILE_X_K + 2 * chunk] = int_a;
        x_qs[i * MMQ_TILE_X_K + 2 * chunk + 1] = int_b;
    }

    // The two windows share one affine header. Preserve it while swapping
    // only the packed data portion of the tile.
    if (window == 0) {
        const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
        if (tid < MMQ_Y) {
            const int row = FULL ? row0 + tid : min(row0 + tid, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            x_dm[tid * MMQ_TILE_X_K] = make_half2((_Float16)sc, (_Float16)zp);
        }
    }
}
#endif

#if MMQ_STREAM_K128_PHASED_X256
static constexpr int MMQ_PHASE_X = 128;

template <int PHASE>
static __device__ __forceinline__ void vec_dot_q8_1_group128_phased_x256(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    float* __restrict__ activation_sums
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    constexpr int j_step = col_groups * tile_C::J;
    constexpr int phase_accs = MMQ_PHASE_X / j_step;
    static_assert(MMQ_X == 256 && MMQ_Y == 64 && MMQ_NWARPS == 8,
                  "phased X256 requires X256/Y64 with eight waves");

    const int local_wave = threadIdx.y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    y += (local_wave % col_groups) * tile_C::J * MMQ_TILE_Y_K;
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + MMQ_X_DM_OFFSET);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

#pragma unroll
    for (int j0 = 0; j0 < MMQ_PHASE_X; j0 += j_step) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            activation_sums[PHASE * phase_accs + j0 / j_step] +=
                __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]).y;
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                load_ldmatrix_16x8(
                    a_frag,
                    x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k01,
                    MMQ_TILE_X_K);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + tile_C::get_j(0);
        const float d_x = __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
        const int global_acc = PHASE * phase_accs + j0 / j_step;
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float scale_w =
                    __half22float2(x_dm[i * MMQ_TILE_X_K]).x;
                const int acc = global_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    scale_w * d_x * (float)c_acc[n].x[l];
            }
        }
    }
}

static __device__ __forceinline__ void add_group_zero_correction_phased_x256(
    const int* __restrict__ tile_x,
    const float* __restrict__ activation_sums,
    float* __restrict__ sum
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    constexpr int zero_groups = MMQ_X / (col_groups * tile_C::J);
    const half2* x_dm = (const half2*)(tile_x + MMQ_X_DM_OFFSET);
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;

#pragma unroll
    for (int zero_acc = 0; zero_acc < zero_groups; ++zero_acc) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const float zero =
                    __half22float2(x_dm[i * MMQ_TILE_X_K]).y;
                const int acc = zero_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    zero * activation_sums[zero_acc];
            }
        }
    }
}
#endif

#if MMQ_DUAL_GATE_UP
static __device__ __forceinline__ void load_hfq4_gate_up_dual_tile(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    constexpr int plane_rows = MMQ_Y / 2;
    constexpr int plane_warps = MMQ_NWARPS / 2;
    const int plane = threadIdx.y / plane_warps;
    const int local_wave = threadIdx.y % plane_warps;
    const int plane_row = plane * plane_rows;
#if MMQ_DUAL_INTERLEAVED
    const char* A = A_gate;
#else
    const char* A = plane ? A_up : A_gate;
#endif
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    for (int local_i = local_wave; local_i < plane_rows; local_i += plane_warps) {
        const int i = plane_row + local_i;
        const int row = min(row0 + local_i, M - 1);
#if MMQ_DUAL_INTERLEAVED
        const int storage_row = (row0 / plane_rows) * MMQ_Y + plane_row + local_i;
#else
        const int storage_row = row;
#endif
        const char* gp = A + ((long long)storage_row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + threadIdx.x * 4);
        const unsigned int lo = qs0 & 0x0F0F0F0Fu;
        const unsigned int hi = (qs0 >> 4) & 0x0F0F0F0Fu;
        x_qs[i * MMQ_TILE_X_K + 2 * threadIdx.x] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        x_qs[i * MMQ_TILE_X_K + 2 * threadIdx.x + 1] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }

    const int local_i = local_wave * 16 + threadIdx.x / 2;
    if (local_i < plane_rows) {
        const int i = plane_row + local_i;
        const int row = min(row0 + local_i, M - 1);
#if MMQ_DUAL_INTERLEAVED
        const int storage_row = (row0 / plane_rows) * MMQ_Y + plane_row + local_i;
#else
        const int storage_row = row;
#endif
        const char* gp = A + ((long long)storage_row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const int ksc = threadIdx.x & 1;
        const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
        }
    }
}
#endif

#if MMQ_PACKED_DUAL_X256
static constexpr int PACKED_DUAL_ACT_STRIDE = 35;
static constexpr int PACKED_DUAL_EXP_STRIDE = 65;
static constexpr int PACKED_DUAL_PACK_STRIDE = 33;
static constexpr int PACKED_DUAL_PACK_FIRST = 16;
static constexpr int PACKED_DUAL_PACK_ROWS = 15;
static constexpr int PACKED_DUAL_EXP_ROWS = 64 - PACKED_DUAL_PACK_ROWS;
static constexpr int PACKED_DUAL_PLANE_INTS =
    PACKED_DUAL_EXP_ROWS * PACKED_DUAL_EXP_STRIDE +
    PACKED_DUAL_PACK_ROWS * PACKED_DUAL_PACK_STRIDE;

static __device__ __forceinline__ bool packed_dual_row_is_packed(int row) {
    return row >= PACKED_DUAL_PACK_FIRST &&
           row < PACKED_DUAL_PACK_FIRST + PACKED_DUAL_PACK_ROWS;
}

static __device__ __forceinline__ int packed_dual_exp_row(int row) {
    return row < PACKED_DUAL_PACK_FIRST ? row : row - PACKED_DUAL_PACK_ROWS;
}

static __device__ __forceinline__ int* packed_dual_pack_row_ptr(
    int* __restrict__ tile, int row
) {
    return tile + PACKED_DUAL_EXP_ROWS * PACKED_DUAL_EXP_STRIDE +
           (row - PACKED_DUAL_PACK_FIRST) * PACKED_DUAL_PACK_STRIDE;
}

static __device__ __forceinline__ const int* packed_dual_pack_row_ptr(
    const int* __restrict__ tile, int row
) {
    return tile + PACKED_DUAL_EXP_ROWS * PACKED_DUAL_EXP_STRIDE +
           (row - PACKED_DUAL_PACK_FIRST) * PACKED_DUAL_PACK_STRIDE;
}

static __device__ __forceinline__ int* packed_dual_exp_row_ptr(
    int* __restrict__ tile, int row
) {
    return tile + packed_dual_exp_row(row) * PACKED_DUAL_EXP_STRIDE;
}

static __device__ __forceinline__ const int* packed_dual_exp_row_ptr(
    const int* __restrict__ tile, int row
) {
    return tile + packed_dual_exp_row(row) * PACKED_DUAL_EXP_STRIDE;
}

static __device__ __forceinline__ void load_hfq4_packed_dual_weights(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    int* __restrict__ gate_tile,
    int* __restrict__ up_tile,
    int row0,
    int kb,
    int groups_per_row
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Only 15/64 rows per plane stay packed. This is the minimum compression
    // needed to fit X256 and both output planes below the gfx11 64 KiB limit.
    for (int l = tid; l < 2 * 64 * 32; l += MMQ_NWARPS * WARP_SIZE) {
        const int plane = l / (64 * 32);
        const int plane_l = l % (64 * 32);
        const int row = plane_l / 32;
        const int word = l % 32;
        const char* A = plane ? A_up : A_gate;
        int* tile = plane ? up_tile : gate_tile;
        const char* gp =
            A + ((long long)(row0 + row) * groups_per_row + kb) * 136;
        const unsigned int q = *(const unsigned int*)(gp + 8 + word * 4);
        if (packed_dual_row_is_packed(row)) {
            packed_dual_pack_row_ptr(tile, row)[word] = (int)q;
        } else {
            const unsigned int lo = q & 0x0F0F0F0Fu;
            const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
            int* dst = packed_dual_exp_row_ptr(tile, row) + 2 * word;
            dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    }

    if (tid < 128) {
        const bool is_up = tid >= 64;
        const int row = tid & 63;
        const char* A = is_up ? A_up : A_gate;
        const char* gp =
            A + ((long long)(row0 + row) * groups_per_row + kb) * 136;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zero =
            __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const int dm = __builtin_bit_cast(
            int, make_half2((_Float16)scale, (_Float16)zero));
        int* tile = is_up ? up_tile : gate_tile;
        if (packed_dual_row_is_packed(row)) {
            packed_dual_pack_row_ptr(tile, row)[32] = dm;
        } else {
            packed_dual_exp_row_ptr(tile, row)[64] = dm;
        }
    }
}

static __device__ __forceinline__ void load_packed_weight_fragment(
    tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>& frag,
    const int* __restrict__ packed_tile,
    int row,
    int k0
) {
    const int* src =
        packed_dual_pack_row_ptr(packed_tile, row) + k0 / 2;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const unsigned int q = (unsigned int)src[word];
        const unsigned int lo = q & 0x0F0F0F0Fu;
        const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        frag.x[2 * word] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        frag.x[2 * word + 1] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}

static __device__ __forceinline__ void load_expanded_weight_fragment(
    tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>& frag,
    const int* __restrict__ tile,
    int row,
    int k0
) {
    const int* src = packed_dual_exp_row_ptr(tile, row) + k0;
    memcpy_aligned<16>(frag.x, src);
    memcpy_aligned<16>(frag.x + 4, src + 4);
}

static __device__ __forceinline__ void vec_dot_packed_dual_group128(
    const int* __restrict__ weight_tile,
    const int* __restrict__ activation_tile,
    float* __restrict__ sum,
    int k00,
    float* __restrict__ activation_sums
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int local_wave = threadIdx.y & 7;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* activation_plane =
        activation_tile + (local_wave % col_groups) * tile_C::J *
                              PACKED_DUAL_ACT_STRIDE;
    const int* y_qs = activation_plane + 3;

#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < 32; k01 += QI8_1) {
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag,
                y_qs + j0 * PACKED_DUAL_ACT_STRIDE + k01,
                PACKED_DUAL_ACT_STRIDE);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                const int row = i0 + n * tile_A::I + tile_A::get_i(0);
                if (packed_dual_row_is_packed(row)) {
                    load_packed_weight_fragment(
                        a_frag, weight_tile, row, k00 + k01);
                } else {
                    load_expanded_weight_fragment(
                        a_frag, weight_tile, row, k00 + k01);
                }
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + tile_C::get_j(0);
        const half2 d_sum0 =
            __builtin_bit_cast(half2, activation_plane[j * PACKED_DUAL_ACT_STRIDE]);
        const half2 sum1_sum2 =
            __builtin_bit_cast(half2, activation_plane[j * PACKED_DUAL_ACT_STRIDE + 1]);
        const half2 sum3_pad =
            __builtin_bit_cast(half2, activation_plane[j * PACKED_DUAL_ACT_STRIDE + 2]);
        const float2 ds0 = __half22float2(d_sum0);
        const float2 ds12 = __half22float2(sum1_sum2);
        const float2 ds3 = __half22float2(sum3_pad);
        const int zero_acc = j0 / (col_groups * tile_C::J);
        activation_sums[zero_acc] += ds0.y + ds12.x + ds12.y + ds3.x;

#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                const int meta = packed_dual_row_is_packed(row)
                    ? packed_dual_pack_row_ptr(weight_tile, row)[32]
                    : packed_dual_exp_row_ptr(weight_tile, row)[64];
                const float scale =
                    __half22float2(__builtin_bit_cast(half2, meta)).x;
                const int acc = zero_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    scale * ds0.x * (float)c_acc[n].x[l];
            }
        }
    }
}
#endif

#if MMQ_PACKED_DUAL_ONE_PLANE
static constexpr int ONE_PLANE_ACT_STRIDE = 35;
static constexpr int ONE_PLANE_EXP_STRIDE = 65;
static constexpr int ONE_PLANE_PACK_STRIDE = 33;

static __device__ __forceinline__ void load_hfq4_one_packed_plane_weights(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    int* __restrict__ gate_tile,
    int* __restrict__ up_tile,
    int row0,
    int kb,
    int groups_per_row
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int l = tid; l < 2 * 64 * 32; l += MMQ_NWARPS * WARP_SIZE) {
        const int plane = l / (64 * 32);
        const int plane_l = l % (64 * 32);
        const int row = plane_l / 32;
        const int word = plane_l & 31;
        const char* A = plane ? A_up : A_gate;
        const char* gp =
            A + ((long long)(row0 + row) * groups_per_row + kb) * 136;
        const unsigned int q = *(const unsigned int*)(gp + 8 + word * 4);
        if (plane) {
            up_tile[row * ONE_PLANE_PACK_STRIDE + word] = (int)q;
        } else {
            const unsigned int lo = q & 0x0F0F0F0Fu;
            const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
            int* dst = gate_tile + row * ONE_PLANE_EXP_STRIDE + 2 * word;
            dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    }
    if (tid < 128) {
        const bool is_up = tid >= 64;
        const int row = tid & 63;
        const char* A = is_up ? A_up : A_gate;
        const char* gp =
            A + ((long long)(row0 + row) * groups_per_row + kb) * 136;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zero =
            __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const int dm = __builtin_bit_cast(
            int, make_half2((_Float16)scale, (_Float16)zero));
        if (is_up) {
            up_tile[row * ONE_PLANE_PACK_STRIDE + 32] = dm;
        } else {
            gate_tile[row * ONE_PLANE_EXP_STRIDE + 64] = dm;
        }
    }
}

template <bool PACKED>
static __device__ __forceinline__ void load_one_plane_weight_fragment(
    tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>& frag,
    const int* __restrict__ weight_tile,
    int row,
    int k0
) {
    if constexpr (PACKED) {
        const int* src = weight_tile + row * ONE_PLANE_PACK_STRIDE + k0 / 2;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const unsigned int q = (unsigned int)src[word];
            const unsigned int lo = q & 0x0F0F0F0Fu;
            const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
            frag.x[2 * word] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            frag.x[2 * word + 1] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    } else {
        const int* src = weight_tile + row * ONE_PLANE_EXP_STRIDE + k0;
        memcpy_aligned<16>(frag.x, src);
        memcpy_aligned<16>(frag.x + 4, src + 4);
    }
}

template <bool PACKED>
static __device__ __forceinline__ void vec_dot_one_packed_plane_group128(
    const int* __restrict__ weight_tile,
    const int* __restrict__ activation_tile,
    float* __restrict__ sum,
    int k00,
    float* __restrict__ activation_sums
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int local_wave = threadIdx.y & 7;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* activation_plane =
        activation_tile + (local_wave % col_groups) * tile_C::J *
                              ONE_PLANE_ACT_STRIDE;
    const int* y_qs = activation_plane + 3;

#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < 32; k01 += QI8_1) {
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag,
                y_qs + j0 * ONE_PLANE_ACT_STRIDE + k01,
                ONE_PLANE_ACT_STRIDE);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                const int row = i0 + n * tile_A::I + tile_A::get_i(0);
                load_one_plane_weight_fragment<PACKED>(
                    a_frag, weight_tile, row, k00 + k01);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + tile_C::get_j(0);
        const half2 d_sum0 = __builtin_bit_cast(
            half2, activation_plane[j * ONE_PLANE_ACT_STRIDE]);
        const half2 sum1_sum2 = __builtin_bit_cast(
            half2, activation_plane[j * ONE_PLANE_ACT_STRIDE + 1]);
        const half2 sum3_pad = __builtin_bit_cast(
            half2, activation_plane[j * ONE_PLANE_ACT_STRIDE + 2]);
        const float2 ds0 = __half22float2(d_sum0);
        const float2 ds12 = __half22float2(sum1_sum2);
        const float2 ds3 = __half22float2(sum3_pad);
        const int zero_acc = j0 / (col_groups * tile_C::J);
        activation_sums[zero_acc] += ds0.y + ds12.x + ds12.y + ds3.x;

#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                const int meta = PACKED
                    ? weight_tile[row * ONE_PLANE_PACK_STRIDE + 32]
                    : weight_tile[row * ONE_PLANE_EXP_STRIDE + 64];
                const float scale =
                    __half22float2(__builtin_bit_cast(half2, meta)).x;
                const int acc = zero_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    scale * ds0.x * (float)c_acc[n].x[l];
            }
        }
    }
}
#endif

template <bool APPLY_ZERO, bool COLLECT_SUMS = false>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00,
    float* __restrict__ activation_sums = nullptr
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    static_assert(MMQ_Y == (MMQ_NWARPS / col_groups) * row_frags * tile_C::I, "bad row topology");

    y += (threadIdx.y % col_groups) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        tile_A A_frag[row_frags];
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            load_ldmatrix_16x8(A_frag[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            if constexpr (COLLECT_SUMS) {
                const int zero_acc = j0 / (col_groups * tile_C::J);
                activation_sums[zero_acc] += dsB.y;
            }

            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);

                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
                    const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                    sum[acc * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    if constexpr (APPLY_ZERO) {
                        sum[acc * tile_C::ne + l] += dmA.y * dsB.y;
                    }
                }
            }
        }
    }
}

// Q8-group128 sibling: four 32-K i8 WMMA fragments share one activation
// scale, so accumulate them in int32 and apply the FP32 scale once.
static __device__ __forceinline__ half2 load_group128_weight_dm(
    const half2* __restrict__ x_dm,
    int row,
    int k_offset
) {
#if MMQ_WEIGHT_HALF_META
    (void)k_offset;
    return x_dm[row * MMQ_TILE_X_K];
#else
    return x_dm[row * MMQ_TILE_X_K + k_offset / QI8_1];
#endif
}

static __device__ __forceinline__ void mmq_accum_add(
    float* __restrict__ sum,
    int idx,
    float value
) {
    sum[idx] += value;
}

static __device__ __forceinline__ float mmq_accum_get(
    const float* __restrict__ sum,
    int idx
) {
    return sum[idx];
}

static __device__ __forceinline__ void mmq_accum_add(
    half2* __restrict__ sum,
    int idx,
    float value
) {
    float2 pair = __half22float2(sum[idx >> 1]);
    if (idx & 1) {
        pair.y += value;
    } else {
        pair.x += value;
    }
    sum[idx >> 1] = __floats2half2_rn(pair.x, pair.y);
}

static __device__ __forceinline__ float mmq_accum_get(
    const half2* __restrict__ sum,
    int idx
) {
    const float2 pair = __half22float2(sum[idx >> 1]);
    return (idx & 1) ? pair.y : pair.x;
}

template <typename SumT>
static __device__ __forceinline__ void vec_dot_q8_1_group128_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    SumT* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
#if MMQ_DUAL_GATE_UP
    constexpr int plane_warps = MMQ_NWARPS / 2;
    const int local_wave = threadIdx.y % plane_warps;
    const int plane_row = (threadIdx.y / plane_warps) * (MMQ_Y / 2);
#else
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
#endif
    y += (local_wave % col_groups) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + MMQ_X_DM_OFFSET);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;

#if MMQ_Q8_GROUP128_K32_STATIONARY
    // Keep one K32 weight fragment live while visiting every local N tile.
    // This reduces weight-side LDS loads at the cost of applying scale and
    // int32-to-FP32 conversion to each K32 partial independently.
    #pragma unroll 1
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        tile_A a_frag[row_frags];
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            load_ldmatrix_16x8(
                a_frag[n],
                x_qs + (plane_row + i0 + n * tile_A::I) * MMQ_TILE_X_K + k0,
                MMQ_TILE_X_K);
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float d_x = __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_C c_partial;
                mma_i8(c_partial, a_frag[n], b_frag);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float scale_w = __half22float2(load_group128_weight_dm(
                        x_dm, plane_row + i, k0)).x;
                    const int acc =
                        (j0 / (col_groups * tile_C::J)) * row_frags + n;
                    mmq_accum_add(
                        sum,
                        acc * tile_C::ne + l,
                        scale_w * d_x * (float)c_partial.x[l]);
                }
            }
        }
    }
#elif MMQ_Q8_GROUP128_N2_REUSE || MMQ_Q8_GROUP128_N4_REUSE
    // Reuse each K32 weight fragment across two neighboring activation
    // subtiles while preserving one int32 accumulator per full K128 group.
    // Unlike the K32-stationary probe, scaling and int32-to-FP32 conversion
    // still happen only after all four K32 fragments have accumulated.
    constexpr int j_step = col_groups * tile_C::J;
#if MMQ_Q8_GROUP128_N4_REUSE
    constexpr int reuse = 4;
#else
    constexpr int reuse = 2;
#endif
    static_assert(MMQ_X % (reuse * j_step) == 0, "N reuse requires complete subtile groups");
    #pragma unroll
    for (int j_base = 0; j_base < MMQ_X; j_base += reuse * j_step) {
        tile_C c_acc[reuse][row_frags];
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;
            tile_A a_frag[row_frags];
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                load_ldmatrix_16x8(
                    a_frag[n],
                    x_qs + (plane_row + i0 + n * tile_A::I) * MMQ_TILE_X_K + k0,
                    MMQ_TILE_X_K);
            }

            #pragma unroll
            for (int q = 0; q < reuse; ++q) {
                tile_B b_frag;
                const int j0 = j_base + q * j_step;
                load_ldmatrix_16x8(
                    b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
                #pragma unroll
                for (int n = 0; n < row_frags; ++n) {
                    mma_i8(c_acc[q][n], a_frag[n], b_frag);
                }
            }
        }

        #pragma unroll
        for (int q = 0; q < reuse; ++q) {
            const int j0 = j_base + q * j_step;
            const int j = j0 + tile_C::get_j(0);
            const float d_x = __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float scale_w = __half22float2(load_group128_weight_dm(
                        x_dm, plane_row + i, k00)).x;
                    const int acc = (j0 / j_step) * row_frags + n;
                    mmq_accum_add(
                        sum,
                        acc * tile_C::ne + l,
                        scale_w * d_x * (float)c_acc[q][n].x[l]);
                }
            }
        }
    }
#else
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#if MMQ_Q8_GROUP128_SERIAL_ROW
        const int j = j0 + tile_C::get_j(0);
        const float d_x = __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            tile_C c_acc;
            #pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;
                tile_B b_frag;
                load_ldmatrix_16x8(
                    b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
                tile_A a_frag;
                load_ldmatrix_16x8(
                    a_frag,
                    x_qs + (plane_row + i0 + n * tile_A::I) * MMQ_TILE_X_K + k0,
                    MMQ_TILE_X_K);
                mma_i8(c_acc, a_frag, b_frag);
            }

            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float scale_w = __half22float2(load_group128_weight_dm(
                    x_dm, plane_row + i, k00)).x;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                mmq_accum_add(
                    sum,
                    acc * tile_C::ne + l,
                    scale_w * d_x * (float)c_acc.x[l]);
            }
        }
#else
        tile_C c_acc[row_frags];
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                load_ldmatrix_16x8(
                    a_frag,
                    x_qs + (plane_row + i0 + n * tile_A::I) * MMQ_TILE_X_K + k0,
                    MMQ_TILE_X_K);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + tile_C::get_j(0);
        const float d_x = __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float scale_w = __half22float2(load_group128_weight_dm(
                    x_dm, plane_row + i, k00)).x;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                mmq_accum_add(
                    sum,
                    acc * tile_C::ne + l,
                    scale_w * d_x * (float)c_acc[n].x[l]);
            }
        }
#endif
    }
#endif
}

#if MMQ_Q8_GROUP256_DIRECT
static __device__ __forceinline__ void vec_dot_q8_1_group256_direct_mma(
    const int* __restrict__ x,
    const block_q8_1_mmq* __restrict__ by0,
    const block_q8_1_mmq* __restrict__ by1,
    float* __restrict__ sum
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int local_wave = threadIdx.y;
    const int col_lane_base = (local_wave % col_groups) * tile_C::J;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);

#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int half_block = 0; half_block < 2; ++half_block) {
            const int* y = (const int*)(half_block ? by1 : by0);
            y += col_lane_base * MMQ_TILE_Y_K;
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                tile_B b_frag;
                load_ldmatrix_16x8(
                    b_frag, y + j0 * MMQ_TILE_Y_K + 4 + k01,
                    MMQ_TILE_Y_K);
#pragma unroll
                for (int n = 0; n < row_frags; ++n) {
                    tile_A a_frag;
                    load_ldmatrix_16x8(
                        a_frag,
                        x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K
                            + half_block * MMQ_TILE_NE_K + k01,
                        MMQ_TILE_X_K);
                    mma_i8(c_acc[n], a_frag, b_frag);
                }
            }
        }

        const int j = j0 + col_lane_base + tile_C::get_j(0);
        const block_q8_1_mmq* token0 = by0 + j;
        const block_q8_1_mmq* token1 = by1 + j;
        const float d_x = __half22float2(token0->ds4[0]).x;
        float activation_sum = 0.0f;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            activation_sum += __half22float2(token0->ds4[sub]).y;
            activation_sum += __half22float2(token1->ds4[sub]).y;
        }
        const int zero_acc = j0 / (col_groups * tile_C::J);
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float2 dm_w =
                    __half22float2(x_dm[i * MMQ_TILE_X_K]);
                const int acc = zero_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    dm_w.x * d_x * (float)c_acc[n].x[l]
                    + dm_w.y * activation_sum;
            }
        }
    }
}

#if MMQ_Q8_GROUP256_SERIAL_ROW
static __device__ __forceinline__ void vec_dot_q8_1_group256_serial_row_mma(
    const int* __restrict__ x,
    const block_q8_1_mmq* __restrict__ by0,
    const block_q8_1_mmq* __restrict__ by1,
    float* __restrict__ sum
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int local_wave = threadIdx.y;
    const int col_lane_base = (local_wave % col_groups) * tile_C::J;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);

#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        const int j = j0 + col_lane_base + tile_C::get_j(0);
        const block_q8_1_mmq* token0 = by0 + j;
        const block_q8_1_mmq* token1 = by1 + j;
        const float d_x = __half22float2(token0->ds4[0]).x;
        float activation_sum = 0.0f;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            activation_sum += __half22float2(token0->ds4[sub]).y;
            activation_sum += __half22float2(token1->ds4[sub]).y;
        }
        const int zero_acc = j0 / (col_groups * tile_C::J);

#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            tile_C c_acc;
#pragma unroll
            for (int half_block = 0; half_block < 2; ++half_block) {
                const int* y = (const int*)(half_block ? by1 : by0);
                y += col_lane_base * MMQ_TILE_Y_K;
#pragma unroll
                for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                    tile_B b_frag;
                    load_ldmatrix_16x8(
                        b_frag,
                        y + j0 * MMQ_TILE_Y_K + 4 + k01,
                        MMQ_TILE_Y_K);
                    tile_A a_frag;
                    load_ldmatrix_16x8(
                        a_frag,
                        x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K
                            + half_block * MMQ_TILE_NE_K + k01,
                        MMQ_TILE_X_K);
                    mma_i8(c_acc, a_frag, b_frag);
                }
            }

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float2 dm_w =
                    __half22float2(x_dm[i * MMQ_TILE_X_K]);
                const int acc = zero_acc * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    dm_w.x * d_x * (float)c_acc.x[l]
                    + dm_w.y * activation_sum;
            }
        }
    }
}
#endif
#endif

#if MMQ_Q8_GROUP128_DIRECT
static __device__ __forceinline__ void vec_dot_q8_1_group128_direct_mma(
    const int* __restrict__ x,
    const block_q8_1_mmq* __restrict__ by0,
    const block_q8_1_mmq* __restrict__ by1,
    float* __restrict__ sum
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int local_wave = threadIdx.y;
    const int col_lane_base = (local_wave % col_groups) * tile_C::J;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);

#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        const int zero_acc = j0 / (col_groups * tile_C::J);
        // Keep the two group128 halves sequential so their metadata and WMMA
        // temporaries share one live range instead of being duplicated by
        // inlining two fully unrolled helper calls.
#pragma unroll 1
        for (int half_block = 0; half_block < 2; ++half_block) {
            const block_q8_1_mmq* by = half_block ? by1 : by0;
            const int k00 = half_block * MMQ_TILE_NE_K;
            const int j = j0 + col_lane_base + tile_C::get_j(0);
            const block_q8_1_mmq* token = by + j;
            const float d_x = __half22float2(token->ds4[0]).x;
            float activation_sum = 0.0f;
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                activation_sum += __half22float2(token->ds4[sub]).y;
            }
            const int* y = (const int*)by + col_lane_base * MMQ_TILE_Y_K;

#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_C c_acc;
#pragma unroll
                for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                    tile_B b_frag;
                    load_ldmatrix_16x8(
                        b_frag,
                        y + j0 * MMQ_TILE_Y_K + 4 + k01,
                        MMQ_TILE_Y_K);
                    tile_A a_frag;
                    load_ldmatrix_16x8(
                        a_frag,
                        x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K
                            + k00 + k01,
                        MMQ_TILE_X_K);
                    mma_i8(c_acc, a_frag, b_frag);
                }

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dm_w = __half22float2(
                        x_dm[i * MMQ_TILE_X_K + k00 / QI8_1]);
                    const int acc = zero_acc * row_frags + n;
                    sum[acc * tile_C::ne + l] +=
                        dm_w.x * d_x * (float)c_acc.x[l]
                        + dm_w.y * activation_sum;
                }
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void
gemm_hfq4g256_group128_direct_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    extern __shared__ int tile_x[];
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        __syncthreads();
        const block_q8_1_mmq* by0 = Xq + (long long)(2 * kb) * N + col0;
        const block_q8_1_mmq* by1 = Xq + (long long)(2 * kb + 1) * N + col0;
        vec_dot_q8_1_group128_direct_mma(tile_x, by0, by1, sum);
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

#if MMQ_Q8_GROUP256_DIRECT
template <bool ADD>
static __device__ __forceinline__ void
gemm_hfq4g256_group256_direct_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    extern __shared__ int tile_x[];
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        __syncthreads();
        const block_q8_1_mmq* by0 = Xq + (long long)(2 * kb) * N + col0;
        const block_q8_1_mmq* by1 = Xq + (long long)(2 * kb + 1) * N + col0;
#if MMQ_Q8_GROUP256_SERIAL_ROW
        vec_dot_q8_1_group256_serial_row_mma(tile_x, by0, by1, sum);
#else
        vec_dot_q8_1_group256_direct_mma(tile_x, by0, by1, sum);
#endif
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

#if MMQ_Q8_GROUP256_STAGED
static __device__ __forceinline__ void vec_dot_q8_1_group256_staged_slice(
    const int* __restrict__ x,
    const int* __restrict__ y0,
    const int* __restrict__ y1,
    float* __restrict__ sum,
    int slice
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int local_wave = threadIdx.y;
    const int col_lane_base = (local_wave % col_groups) * tile_C::J;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    tile_C c_acc[row_frags];

#pragma unroll
    for (int half_block = 0; half_block < 2; ++half_block) {
        const int* y = (half_block ? y1 : y0)
            + col_lane_base * MMQ_TILE_Y_K;
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y + 4 + k01, MMQ_TILE_Y_K);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                load_ldmatrix_16x8(
                    a_frag,
                    x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K
                        + half_block * MMQ_TILE_NE_K + k01,
                    MMQ_TILE_X_K);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }
    }

    const int j = col_lane_base + tile_C::get_j(0);
    const block_q8_1_mmq* token0 =
        (const block_q8_1_mmq*)y0 + j;
    const block_q8_1_mmq* token1 =
        (const block_q8_1_mmq*)y1 + j;
    const float d_x = __half22float2(token0->ds4[0]).x;
    float activation_sum = 0.0f;
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        activation_sum += __half22float2(token0->ds4[sub]).y;
        activation_sum += __half22float2(token1->ds4[sub]).y;
    }
#pragma unroll
    for (int n = 0; n < row_frags; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + n * tile_A::I + tile_C::get_i(l);
            const float2 dm_w = __half22float2(x_dm[i * MMQ_TILE_X_K]);
            const int acc = slice * row_frags + n;
            sum[acc * tile_C::ne + l] +=
                dm_w.x * d_x * (float)c_acc[n].x[l]
                + dm_w.y * activation_sum;
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void
gemm_hfq4g256_group256_staged_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    static_assert(MMQ_X == 256 && MMQ_Y == 64 && MMQ_NWARPS == 8,
                  "group256 staged requires X256/Y64/8 waves");
    static_assert(MMQ_ROW_FRAGS == 2 && MMQ_COL_GROUPS == 4,
                  "group256 staged requires row2/col4");
    constexpr int slice_cols = 64;
    constexpr int slice_ints = slice_cols * MMQ_TILE_Y_K;
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    extern __shared__ int smem[];
    int* tile_x = smem;
    int* tile_y0 = tile_x + MMQ_Y * MMQ_TILE_X_K;
    int* tile_y1 = tile_y0 + slice_ints;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
#pragma unroll
        for (int slice = 0; slice < MMQ_X / slice_cols; ++slice) {
            const int* by0 = (const int*)(
                Xq + (long long)(2 * kb) * N + col0 + slice * slice_cols);
            const int* by1 = (const int*)(
                Xq + (long long)(2 * kb + 1) * N + col0 + slice * slice_cols);
            for (int l = tid; l < slice_ints; l += MMQ_NWARPS * WARP_SIZE) {
                tile_y0[l] = by0[l];
                tile_y1[l] = by1[l];
            }
            __syncthreads();
            vec_dot_q8_1_group256_staged_slice(
                tile_x, tile_y0, tile_y1, sum, slice);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

static __device__ __forceinline__ void load_group_row_zeros(
    const int* __restrict__ tile_x,
    float* __restrict__ row_zeros
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const half2* x_dm = (const half2*)(tile_x + MMQ_X_DM_OFFSET);
#if MMQ_DUAL_GATE_UP
    constexpr int plane_warps = MMQ_NWARPS / 2;
    const int local_wave = threadIdx.y % plane_warps;
    const int plane_row = (threadIdx.y / plane_warps) * (MMQ_Y / 2);
#else
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
#endif
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;

    #pragma unroll
    for (int n = 0; n < row_frags; ++n) {
        #pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + n * tile_C::I + tile_C::get_i(l);
            row_zeros[n * tile_C::ne + l] =
                __half22float2(x_dm[(plane_row + i) * MMQ_TILE_X_K]).y;
        }
    }
}

template <typename SumT>
static __device__ __forceinline__ void add_group_zero_correction_regs(
    const float* __restrict__ row_zeros,
    const float* __restrict__ activation_sums,
    SumT* __restrict__ sum
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        const int zero_acc = j0 / (col_groups * tile_C::J);
        const float activation_sum = activation_sums[zero_acc];
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int acc = zero_acc * row_frags + n;
                mmq_accum_add(
                    sum,
                    acc * tile_C::ne + l,
                    row_zeros[n * tile_C::ne + l] * activation_sum);
            }
        }
    }
}

static __device__ __forceinline__ void collect_q8_1_sums(
    const int* __restrict__ tile_y,
    float* __restrict__ zero_sums,
    bool reset
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (tid < MMQ_X) {
        const half2* y_dm = (const half2*)tile_y;
        float value = 0.0f;
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            value += __half22float2(y_dm[tid * MMQ_TILE_Y_K + k01 / QI8_1]).y;
        }
        zero_sums[tid] = reset ? value : zero_sums[tid] + value;
    }
}

template <typename SumT>
static __device__ __forceinline__ void add_group_zero_correction(
    const int* __restrict__ tile_x,
    const float* __restrict__ zero_sums,
    SumT* __restrict__ sum
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const half2* x_dm = (const half2*)(tile_x + MMQ_X_DM_OFFSET);
#if MMQ_DUAL_GATE_UP
    constexpr int plane_warps = MMQ_NWARPS / 2;
    const int local_wave = threadIdx.y % plane_warps;
    const int plane_row = (threadIdx.y / plane_warps) * (MMQ_Y / 2);
#else
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
#endif
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        const int j = j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(0);
        const float activation_sum = zero_sums[j];
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const float zero = __half22float2(x_dm[(plane_row + i) * MMQ_TILE_X_K]).y;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                mmq_accum_add(sum, acc * tile_C::ne + l, zero * activation_sum);
            }
        }
    }
}

#if MMQ_DIRECT_PACKED_WEIGHT_N2
static __device__ __forceinline__ void load_hfq4_direct_packed_fragment(
    tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>& frag,
    const char* __restrict__ A,
    int row,
    int kb,
    int k0,
    int groups_per_row
) {
    const char* gp = A + ((long long)row * groups_per_row + kb) * HFQ4_GROUP_BYTES;
    const unsigned int* src =
        (const unsigned int*)(gp + 8) + k0 / 2;
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        const unsigned int q = src[word];
        const unsigned int lo = q & 0x0F0F0F0Fu;
        const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        frag.x[2 * word] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        frag.x[2 * word + 1] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}

static __device__ __forceinline__ void vec_dot_direct_packed_weight_n2(
    const char* __restrict__ A,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int row0,
    int kb,
    int k00,
    int groups_per_row
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    constexpr int j_step = col_groups * tile_C::J;
    static_assert(MMQ_X == 256 && MMQ_Y == 64 && row_frags == 2 && col_groups == 4,
                  "direct packed N2 requires X256/Y64 row2/col4");

    const int local_wave = threadIdx.y;
    y += (local_wave % col_groups) * tile_C::J * MMQ_TILE_Y_K;
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    #pragma unroll
    for (int j_base = 0; j_base < MMQ_X; j_base += 2 * j_step) {
        tile_C c_acc[2][row_frags];
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_A a_frag[row_frags];
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                const int row = row0 + i0 + n * tile_C::I + (threadIdx.x & 15);
                load_hfq4_direct_packed_fragment(
                    a_frag[n], A, row, kb, k00 + k01, groups_per_row);
            }
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int j0 = j_base + q * j_step;
                tile_B b_frag;
                load_ldmatrix_16x8(
                    b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
                #pragma unroll
                for (int n = 0; n < row_frags; ++n) {
                    mma_i8(c_acc[q][n], a_frag[n], b_frag);
                }
            }
        }

        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int j0 = j_base + q * j_step;
            const int j = j0 + tile_C::get_j(0);
            const float activation_scale =
                __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
            #pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = row0 + i0 + n * tile_C::I
                        + tile_C::get_i(l);
                    const char* gp = A
                        + ((long long)row * groups_per_row + kb)
                            * HFQ4_GROUP_BYTES;
                    const float weight_scale = __builtin_bit_cast(
                        float, *(const unsigned int*)gp);
                    const int acc = (j0 / j_step) * row_frags + n;
                    sum[acc * tile_C::ne + l] += weight_scale
                        * activation_scale * (float)c_acc[q][n].x[l];
                }
            }
        }
    }
}

static __device__ __forceinline__ void add_direct_weight_zero_correction(
    const char* __restrict__ A,
    const float* __restrict__ zero_sums,
    float* __restrict__ sum,
    int row0,
    int kb,
    int groups_per_row
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int local_wave = threadIdx.y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        const int j = j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(0);
        const float activation_sum = zero_sums[j];
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = row0 + i0 + n * tile_C::I
                    + tile_C::get_i(l);
                const char* gp = A
                    + ((long long)row * groups_per_row + kb)
                        * HFQ4_GROUP_BYTES;
                const float weight_zero = __builtin_bit_cast(
                    float, *(const unsigned int*)(gp + 4));
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                sum[acc * tile_C::ne + l] += weight_zero * activation_sum;
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_hfq4g256_direct_packed_weight_n2_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    extern __shared__ int smem[];
    int* tile_y = smem;
    float* zero_sums = (float*)(tile_y + MMQ_X * MMQ_TILE_Y_K);
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K;
             l += MMQ_NWARPS * WARP_SIZE) {
            tile_y[l] = by0[l];
        }
        __syncthreads();
        collect_q8_1_sums(tile_y, zero_sums, true);
        vec_dot_direct_packed_weight_n2(
            A, tile_y, sum, row0, kb, 0, groups_per_row);
        __syncthreads();

        const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K;
             l += MMQ_NWARPS * WARP_SIZE) {
            tile_y[l] = by1[l];
        }
        __syncthreads();
        collect_q8_1_sums(tile_y, zero_sums, false);
        vec_dot_direct_packed_weight_n2(
            A, tile_y, sum, row0, kb, MMQ_TILE_NE_K, groups_per_row);
        __syncthreads();
        add_direct_weight_zero_correction(
            A, zero_sums, sum, row0, kb, groups_per_row);
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, MMQ_MIN_BLOCKS_PER_CU)
void gemm_hfq4g256_residual_mmq(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_tile<false>(A, tile_x, row0, kb, M, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<true>(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            vec_dot_q8_1_q8_1_mma<true>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                    const float v = sum[acc * tile_C::ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

#if MMQ_PACKED_WEIGHT_Y128
static constexpr int PACKED_WEIGHT_STRIDE = 33;

static __device__ __forceinline__ void load_hfq4_packed_weight_tile_y128(
    const char* __restrict__ A,
    int* __restrict__ tile,
    int row0,
    int kb,
    int groups_per_row
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int l = tid; l < MMQ_Y * 32; l += MMQ_NWARPS * WARP_SIZE) {
        const int row = l / 32;
        const int word = l & 31;
        const char* gp =
            A + ((long long)(row0 + row) * groups_per_row + kb) * 136;
        tile[row * PACKED_WEIGHT_STRIDE + word] =
            *(const int*)(gp + 8 + 4 * word);
    }
    if (tid < MMQ_Y) {
        const char* gp =
            A + ((long long)(row0 + tid) * groups_per_row + kb) * 136;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zero =
            __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        tile[tid * PACKED_WEIGHT_STRIDE + 32] = __builtin_bit_cast(
            int, make_half2((_Float16)scale, (_Float16)zero));
    }
}

static __device__ __forceinline__ void load_hfq4_packed_weight_fragment_y128(
    tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>& frag,
    const int* __restrict__ weight_tile,
    int row,
    int k0
) {
    const int* src = weight_tile + row * PACKED_WEIGHT_STRIDE + k0 / 2;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const unsigned int q = (unsigned int)src[word];
        const unsigned int lo = q & 0x0F0F0F0Fu;
        const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        frag.x[2 * word] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        frag.x[2 * word + 1] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}

static __device__ __forceinline__ void vec_dot_packed_weight_group128_y128(
    const int* __restrict__ weight_tile,
    const int* __restrict__ activation_tile,
    float* __restrict__ sum,
    int k00,
    float* __restrict__ activation_sums
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int local_wave = threadIdx.y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    const int* y = activation_tile +
        (local_wave % col_groups) * tile_C::J * MMQ_TILE_Y_K;
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 ds_b =
                __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            const int zero_acc = j0 / (col_groups * tile_C::J);
            activation_sums[zero_acc] += ds_b.y;
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                const int row = i0 + n * tile_A::I + (threadIdx.x & 15);
                tile_A a_frag;
                load_hfq4_packed_weight_fragment_y128(
                    a_frag, weight_tile, row, k0);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + tile_C::get_j(0);
        const float activation_scale =
            __half22float2(y_dm[j * MMQ_TILE_Y_K]).x;
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = i0 + n * tile_A::I + tile_C::get_i(l);
                const half2 dm = __builtin_bit_cast(
                    half2, weight_tile[row * PACKED_WEIGHT_STRIDE + 32]);
                const int acc =
                    (j0 / (col_groups * tile_C::J)) * row_frags + n;
                sum[acc * tile_C::ne + l] +=
                    __half22float2(dm).x * activation_scale *
                    (float)c_acc[n].x[l];
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_hfq4g256_packed_weight_y128_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int* activation_tile = smem;
    int* weight_tile = activation_tile + MMQ_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};
    float activation_sums[MMQ_X / (MMQ_COL_GROUPS * 16)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#pragma unroll
        for (int z = 0; z < MMQ_X / (MMQ_COL_GROUPS * 16); ++z) {
            activation_sums[z] = 0.0f;
        }
        load_hfq4_packed_weight_tile_y128(
            A, weight_tile, row0, kb, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by0[l];
        }
        __syncthreads();
        vec_dot_packed_weight_group128_y128(
            weight_tile, activation_tile, sum, 0, activation_sums);
        __syncthreads();

        const int* by1 =
            (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by1[l];
        }
        __syncthreads();
        vec_dot_packed_weight_group128_y128(
            weight_tile, activation_tile, sum, MMQ_TILE_NE_K, activation_sums);
        __syncthreads();

        using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
        constexpr int row_frags = 2;
        constexpr int col_groups = 4;
        const int i0 =
            (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
            const int zero_acc = j0 / (col_groups * tile_C::J);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row =
                        i0 + n * tile_C::I + tile_C::get_i(l);
                    const half2 dm = __builtin_bit_cast(
                        half2,
                        weight_tile[row * PACKED_WEIGHT_STRIDE + 32]);
                    const int acc = zero_acc * row_frags + n;
                    sum[acc * tile_C::ne + l] +=
                        __half22float2(dm).y * activation_sums[zero_acc];
                }
            }
        }
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) *
                    tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const int acc =
                    (j0 / (col_groups * tile_C::J)) * row_frags + n;
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 1)
void gemm_hfq4g256_packed_weight_y128_full_set(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 1)
void gemm_hfq4g256_packed_weight_y128_full_add(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 1)
void gemm_hfq4g256_packed_weight_y64_full_set(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 1)
void gemm_hfq4g256_packed_weight_y64_full_add(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<true>(A, Xq, Y, M, K, N);
}

// X128/Y64 residency probe. The packed-weight body is unchanged; only the
// activation tile width and launch bound differ so the 26.25 KiB LDS footprint
// can admit two workgroups per CU on gfx11.
extern "C" __global__ __launch_bounds__(
    MMQ_NWARPS * WARP_SIZE, MMQ_MIN_BLOCKS_PER_CU)
void gemm_hfq4g256_packed_weight_x128y64_full_set(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(
    MMQ_NWARPS * WARP_SIZE, MMQ_MIN_BLOCKS_PER_CU)
void gemm_hfq4g256_packed_weight_x128y64_full_add(
    const char* A,
    const block_q8_1_mmq* Xq,
    float* Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_packed_weight_y128_body<true>(A, Xq, Y, M, K, N);
}
#endif

// COMBINE_ZERO folds the eight Q8_1 activation sums covered by one HFQ4-G256
// group into one zero-point correction. The scale-dependent WMMA terms remain
// unchanged; only repeated affine zero-point work is removed.
template <bool ADD, bool COMBINE_ZERO = false, bool REG_ZERO = false,
          typename OutputT = float>
static __device__ __forceinline__ void gemm_hfq4g256_residual_mmq_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    OutputT* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float* zero_sums = (float*)(tile_x + MMQ_Y * MMQ_TILE_X_K);

#if MMQ_ACCUM_F16
    half2 sum[MMQ_X * MMQ_Y / (2 * MMQ_NWARPS * WARP_SIZE)] = {};
#else
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {};
#endif
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    constexpr int zero_groups = MMQ_X / (col_groups * tile_C::J);
    float activation_sums[zero_groups] = {};
    float row_zeros[row_frags * tile_C::ne] = {};

#if MMQ_WEIGHT_PREFETCH_NEXT
    static_assert(MMQ_WEIGHT_QUAD_ROW_U32X2,
                  "next-group prefetch requires the quad-row payload mapping");
    hfq4_quad_row_prefetch current_weight;
    prefetch_hfq4_quad_row<true>(
        A, current_weight, row0, 0, M, groups_per_row);
#endif

    for (int kb = 0; kb < groups_per_row; ++kb) {
#if MMQ_WEIGHT_PREFETCH_NEXT
        hfq4_quad_row_prefetch next_weight;
#endif
        if constexpr (REG_ZERO) {
            #pragma unroll
            for (int z = 0; z < zero_groups; ++z) activation_sums[z] = 0.0f;
        }
#if MMQ_WEIGHT_PREFETCH_NEXT
        commit_hfq4_quad_row(current_weight, tile_x);
#else
        load_hfq4_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
#endif

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        #if MMQ_WARP_SPECIALIZED_STAGE
        if (threadIdx.y >= 4) {
            const uint4* by0_vec = (const uint4*)by0;
            uint4* tile_y_vec = (uint4*)tile_y;
            constexpr int activation_threads = 4 * WARP_SIZE;
            static_assert((MMQ_X * MMQ_TILE_Y_K) % (4 * 3 * activation_threads) == 0,
                          "warp-specialized activation staging requires complete triplets");
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
                 l0 += 3 * activation_threads) {
                const int l = l0 + (threadIdx.y - 4) * WARP_SIZE + threadIdx.x;
                const uint4 v0 = by0_vec[l];
                const uint4 v1 = by0_vec[l + activation_threads];
                const uint4 v2 = by0_vec[l + 2 * activation_threads];
                tile_y_vec[l] = v0;
                tile_y_vec[l + activation_threads] = v1;
                tile_y_vec[l + 2 * activation_threads] = v2;
            }
        }
        #elif MMQ_VECTOR_ACTIVATION_STAGING_BATCH3
        static_assert((MMQ_X * MMQ_TILE_Y_K) %
                          (4 * 3 * MMQ_NWARPS * WARP_SIZE) == 0,
                      "batch3 activation staging requires complete triplets");
        const uint4* by0_vec = (const uint4*)by0;
        uint4* tile_y_vec = (uint4*)tile_y;
        constexpr int activation_threads = MMQ_NWARPS * WARP_SIZE;
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
             l0 += 3 * activation_threads) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const uint4 v0 = by0_vec[l];
            const uint4 v1 = by0_vec[l + activation_threads];
            const uint4 v2 = by0_vec[l + 2 * activation_threads];
            tile_y_vec[l] = v0;
            tile_y_vec[l + activation_threads] = v1;
            tile_y_vec[l + 2 * activation_threads] = v2;
        }
        #elif MMQ_VECTOR_ACTIVATION_STAGING
        static_assert((MMQ_X * MMQ_TILE_Y_K) % 4 == 0,
                      "vector activation staging requires uint4-aligned tile size");
        const uint4* by0_vec = (const uint4*)by0;
        uint4* tile_y_vec = (uint4*)tile_y;
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
             l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y_vec[l] = by0_vec[l];
        }
        #else
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        #endif
        __syncthreads();
#if MMQ_WEIGHT_PREFETCH_NEXT
        if (kb + 1 < groups_per_row) {
            prefetch_hfq4_quad_row<true>(
                A, next_weight, row0, kb + 1, M, groups_per_row);
        }
#endif
        if constexpr (REG_ZERO) {
            load_group_row_zeros(tile_x, row_zeros);
        } else if constexpr (COMBINE_ZERO && !MMQ_SKIP_ZERO_CORRECTION) {
            collect_q8_1_sums(tile_y, zero_sums, true);
        }
        if constexpr (MMQ_Q8_GROUP128) {
            vec_dot_q8_1_group128_mma(tile_x, tile_y, sum, 0);
        } else {
            vec_dot_q8_1_q8_1_mma<!COMBINE_ZERO, REG_ZERO>(
                tile_x, tile_y, sum, 0, activation_sums);
        }
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            #if MMQ_WARP_SPECIALIZED_STAGE
            if (threadIdx.y >= 4) {
                const uint4* by1_vec = (const uint4*)by1;
                uint4* tile_y_vec = (uint4*)tile_y;
                constexpr int activation_threads = 4 * WARP_SIZE;
                for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
                     l0 += 3 * activation_threads) {
                    const int l = l0 + (threadIdx.y - 4) * WARP_SIZE + threadIdx.x;
                    const uint4 v0 = by1_vec[l];
                    const uint4 v1 = by1_vec[l + activation_threads];
                    const uint4 v2 = by1_vec[l + 2 * activation_threads];
                    tile_y_vec[l] = v0;
                    tile_y_vec[l + activation_threads] = v1;
                    tile_y_vec[l + 2 * activation_threads] = v2;
                }
            }
            #elif MMQ_VECTOR_ACTIVATION_STAGING_BATCH3
            const uint4* by1_vec = (const uint4*)by1;
            uint4* tile_y_vec = (uint4*)tile_y;
            constexpr int activation_threads = MMQ_NWARPS * WARP_SIZE;
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
                 l0 += 3 * activation_threads) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const uint4 v0 = by1_vec[l];
                const uint4 v1 = by1_vec[l + activation_threads];
                const uint4 v2 = by1_vec[l + 2 * activation_threads];
                tile_y_vec[l] = v0;
                tile_y_vec[l + activation_threads] = v1;
                tile_y_vec[l + 2 * activation_threads] = v2;
            }
            #elif MMQ_VECTOR_ACTIVATION_STAGING
            const uint4* by1_vec = (const uint4*)by1;
            uint4* tile_y_vec = (uint4*)tile_y;
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K / 4;
                 l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y_vec[l] = by1_vec[l];
            }
            #else
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            #endif
            __syncthreads();
            if constexpr (COMBINE_ZERO && !REG_ZERO && !MMQ_SKIP_ZERO_CORRECTION) {
                collect_q8_1_sums(tile_y, zero_sums, false);
            }
            if constexpr (MMQ_Q8_GROUP128) {
                vec_dot_q8_1_group128_mma(
                    tile_x, tile_y, sum, MMQ_TILE_NE_K);
            } else {
                vec_dot_q8_1_q8_1_mma<!COMBINE_ZERO, REG_ZERO>(
                    tile_x, tile_y, sum, MMQ_TILE_NE_K, activation_sums);
            }
            __syncthreads();
        }
        if constexpr (REG_ZERO) {
            add_group_zero_correction_regs(row_zeros, activation_sums, sum);
        } else if constexpr (COMBINE_ZERO && !MMQ_SKIP_ZERO_CORRECTION) {
            add_group_zero_correction(tile_x, zero_sums, sum);
            // All waves must finish consuming this group's LDS before the
            // next iteration overwrites tile_x and zero_sums.
            __syncthreads();
        }
#if MMQ_WEIGHT_PREFETCH_NEXT
        if (kb + 1 < groups_per_row) {
            #pragma unroll
            for (int row_half = 0; row_half < 2; ++row_half) {
                current_weight.payload0[row_half] = next_weight.payload0[row_half];
                current_weight.payload1[row_half] = next_weight.payload1[row_half];
                if ((threadIdx.x & 7) < 2) {
                    current_weight.metadata[row_half] = next_weight.metadata[row_half];
                }
            }
        }
#endif
    }

    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                OutputT* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                const float v = mmq_accum_get(sum, acc * tile_C::ne + l);
                if constexpr (ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

#if MMQ_STREAM_K128 || MMQ_STREAM_K128_X256Y128
template <bool ADD>
static __device__ __forceinline__ void gemm_hfq4g256_stream_k128_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    static_assert(
        (MMQ_X == 128 && MMQ_Y == 64 && MMQ_NWARPS == 8)
            || (MMQ_X == 256 && MMQ_Y == 128 && MMQ_NWARPS == 16),
        "stream K128 requires X128/Y64/8 waves or X256/Y128/16 waves");
    static_assert(MMQ_ROW_FRAGS == 2 && MMQ_COL_GROUPS == 4,
                  "stream K128 requires row2/col4");
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float* zero_sums = (float*)(tile_x + MMQ_Y * MMQ_TILE_X_K);
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#pragma unroll
        for (int window = 0; window < 2; ++window) {
            load_hfq4_tile_window<true>(
                A, tile_x, row0, kb, window, M, groups_per_row);
            const int* src = (const int*)(
                Xq + (long long)(2 * kb + window) * N + col0);
            for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K;
                 l += MMQ_NWARPS * WARP_SIZE) {
                tile_y[l] = src[l];
            }
            __syncthreads();
            collect_q8_1_sums(tile_y, zero_sums, window == 0);
            vec_dot_q8_1_group128_mma(tile_x, tile_y, sum, 0);
            __syncthreads();
        }
        add_group_zero_correction(tile_x, zero_sums, sum);
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

#if MMQ_STREAM_K128_PHASED_X256
template <bool ADD>
static __device__ __forceinline__ void gemm_hfq4g256_stream_k128_phased_x256_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    const int groups_per_row = K / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_PHASE_X * MMQ_TILE_Y_K;
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        float activation_sums[MMQ_X / (MMQ_COL_GROUPS * 16)] = {0.0f};
#pragma unroll
        for (int window = 0; window < 2; ++window) {
            load_hfq4_tile_window<true>(
                A, tile_x, row0, kb, window, M, groups_per_row);
#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
                const int* src = (const int*)(
                    Xq + (long long)(2 * kb + window) * N + col0
                    + phase * MMQ_PHASE_X);
                for (int l = tid; l < MMQ_PHASE_X * MMQ_TILE_Y_K;
                     l += MMQ_NWARPS * WARP_SIZE) {
                    tile_y[l] = src[l];
                }
                __syncthreads();
                if (phase == 0) {
                    vec_dot_q8_1_group128_phased_x256<0>(
                        tile_x, tile_y, sum, activation_sums);
                } else {
                    vec_dot_q8_1_group128_phased_x256<1>(
                        tile_x, tile_y, sum, activation_sums);
                }
                __syncthreads();
            }
        }
        add_group_zero_correction_phased_x256(
            tile_x, activation_sums, sum);
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* dst = Y + (long long)(col0 + j) * M + row0 + i;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                if constexpr (ADD) {
                    *dst += sum[acc * tile_C::ne + l];
                } else {
                    *dst = sum[acc * tile_C::ne + l];
                }
            }
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
#if MMQ_STREAM_K128_PHASED_X256
    gemm_hfq4g256_stream_k128_phased_x256_body<true>(A, Xq, Y, M, K, N);
#elif MMQ_STREAM_K128 || MMQ_STREAM_K128_X256Y128
    gemm_hfq4g256_stream_k128_full_body<true>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP256_STAGED
    gemm_hfq4g256_group256_staged_full_body<true>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP128_DIRECT
    gemm_hfq4g256_group128_direct_full_body<true>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP256_DIRECT
    gemm_hfq4g256_group256_direct_full_body<true>(A, Xq, Y, M, K, N);
#elif MMQ_DIRECT_PACKED_WEIGHT_N2
    gemm_hfq4g256_direct_packed_weight_n2_body<true>(A, Xq, Y, M, K, N);
#else
    gemm_hfq4g256_residual_mmq_full_body<true, true>(A, Xq, Y, M, K, N);
#endif
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
#if MMQ_STREAM_K128_PHASED_X256
    gemm_hfq4g256_stream_k128_phased_x256_body<false>(A, Xq, Y, M, K, N);
#elif MMQ_STREAM_K128 || MMQ_STREAM_K128_X256Y128
    gemm_hfq4g256_stream_k128_full_body<false>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP256_STAGED
    gemm_hfq4g256_group256_staged_full_body<false>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP128_DIRECT
    gemm_hfq4g256_group128_direct_full_body<false>(A, Xq, Y, M, K, N);
#elif MMQ_Q8_GROUP256_DIRECT
    gemm_hfq4g256_group256_direct_full_body<false>(A, Xq, Y, M, K, N);
#elif MMQ_DIRECT_PACKED_WEIGHT_N2
    gemm_hfq4g256_direct_packed_weight_n2_body<false>(A, Xq, Y, M, K, N);
#else
    gemm_hfq4g256_residual_mmq_full_body<false, true>(A, Xq, Y, M, K, N);
#endif
}

// Standalone FFN dataflow probe: retain FP32 accumulation but halve the
// gate/up intermediate footprint at the final global-store boundary.
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_full_set_f16(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    half* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_residual_mmq_full_body<false, true, false, half>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_full_add_regzero(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_residual_mmq_full_body<true, true, true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_full_set_regzero(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_hfq4g256_residual_mmq_full_body<false, true, true>(A, Xq, Y, M, K, N);
}

// Dispatch gate and up as two planes of one grid while retaining the exact
// single-output kernel body. This removes one host/kernel boundary without
// extending accumulator live ranges or changing arithmetic order.
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gate_up_full_set(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int M,
    int K,
    int N
) {
    const bool is_up = blockIdx.z != 0;
    const char* A = is_up ? A_up : A_gate;
    float* Y = is_up ? Y_up : Y_gate;
    gemm_hfq4g256_residual_mmq_full_body<false, true>(A, Xq, Y, M, K, N);
}

#if MMQ_DUAL_GATE_UP
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_mmq_gate_up_dual_group128(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int M,
    int K,
    int N
) {
    constexpr int plane_rows = MMQ_Y / 2;
    constexpr int plane_warps = MMQ_NWARPS / 2;
    const int row0 = blockIdx.x * plane_rows;
    const int col0 = blockIdx.y * MMQ_X;
    const int plane = threadIdx.y / plane_warps;
    const int local_wave = threadIdx.y % plane_warps;
    const int groups_per_row = K / 256;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;
    float* zero_sums = (float*)(tile_x + MMQ_Y * MMQ_TILE_X_K);
    float sum[MMQ_X * plane_rows / (plane_warps * WARP_SIZE)] = {0.0f};

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_hfq4_gate_up_dual_tile(A_gate, A_up, tile_x, row0, kb, M, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K; l += MMQ_NWARPS * WARP_SIZE) {
            tile_y[l] = by0[l];
        }
        __syncthreads();
        collect_q8_1_sums(tile_y, zero_sums, true);
        vec_dot_q8_1_group128_mma(tile_x, tile_y, sum, 0);
        __syncthreads();

        const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        for (int l = tid; l < MMQ_X * MMQ_TILE_Y_K; l += MMQ_NWARPS * WARP_SIZE) {
            tile_y[l] = by1[l];
        }
        __syncthreads();
        collect_q8_1_sums(tile_y, zero_sums, false);
        vec_dot_q8_1_group128_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
        add_group_zero_correction(tile_x, zero_sums, sum);
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = MMQ_ROW_FRAGS;
    constexpr int col_groups = MMQ_COL_GROUPS;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    float* Y = plane ? Y_up : Y_gate;

#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                Y[(long long)(col0 + j) * M + row0 + i] = sum[acc * tile_C::ne + l];
            }
        }
    }
}
#endif

#if MMQ_DUAL_PAYLOAD_ONLY
static constexpr int DUAL_PAYLOAD_X = 256;
static constexpr int DUAL_PAYLOAD_ROWS = 64;
static constexpr int DUAL_PAYLOAD_WAVES_PER_PLANE = 8;
static constexpr int DUAL_PAYLOAD_ACT_STRIDE = 32;
static constexpr int DUAL_PAYLOAD_WEIGHT_STRIDE = 64;

static __device__ __forceinline__ int dual_payload_swizzle_k(
    int logical_k, int row, int fragment_mask
) {
    const int fragment = logical_k >> 3;
    return ((fragment ^ (row & fragment_mask)) << 3) + (logical_k & 7);
}

static __device__ __forceinline__ void load_dual_payload_weights(
    const char* __restrict__ gate,
    const char* __restrict__ up,
    int* __restrict__ payload,
    int row0,
    int kb,
    int groups_per_row
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int l = tid; l < 2 * DUAL_PAYLOAD_ROWS * 32;
         l += 2 * DUAL_PAYLOAD_WAVES_PER_PLANE * WARP_SIZE) {
        const int plane = l / (DUAL_PAYLOAD_ROWS * 32);
        const int plane_l = l % (DUAL_PAYLOAD_ROWS * 32);
        const int row = plane_l / 32;
        const int word = plane_l & 31;
        const char* weights = plane ? up : gate;
        const char* group = weights
            + ((long long)(row0 + row) * groups_per_row + kb)
                * HFQ4_GROUP_BYTES;
        const unsigned int packed =
            *(const unsigned int*)(group + 8 + word * 4);
        const unsigned int lo = packed & 0x0F0F0F0Fu;
        const unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
        const int logical_k = 2 * word;
        const int physical_k = dual_payload_swizzle_k(logical_k, row, 7);
        int* dst = payload
            + (plane * DUAL_PAYLOAD_ROWS + row)
                * DUAL_PAYLOAD_WEIGHT_STRIDE
            + physical_k;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}

static __device__ __forceinline__ void load_dual_payload_activations(
    const block_q8_1_mmq* __restrict__ source,
    int* __restrict__ payload
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int l = tid; l < DUAL_PAYLOAD_X * DUAL_PAYLOAD_ACT_STRIDE;
         l += 2 * DUAL_PAYLOAD_WAVES_PER_PLANE * WARP_SIZE) {
        const int token = l / DUAL_PAYLOAD_ACT_STRIDE;
        const int word = l & (DUAL_PAYLOAD_ACT_STRIDE - 1);
        const int physical_k = dual_payload_swizzle_k(word, token, 3);
        payload[token * DUAL_PAYLOAD_ACT_STRIDE + physical_k] =
            ((const int*)source[token].qs)[word];
    }
}

template <typename Tile>
static __device__ __forceinline__ void load_dual_payload_fragment(
    Tile& fragment,
    const int* __restrict__ payload,
    int row_base,
    int logical_k,
    int stride,
    int fragment_mask
) {
    const int row = row_base + (threadIdx.x & 15);
    const int physical_k =
        dual_payload_swizzle_k(logical_k, row, fragment_mask);
    const int* src = payload + row * stride + physical_k;
    memcpy_aligned<16>(fragment.x, src);
    memcpy_aligned<16>(fragment.x + 4, src + 4);
}

static __device__ __forceinline__ float dual_payload_activation_sum(
    const block_q8_1_mmq& block
) {
    const float2 ds0 = __half22float2(block.ds4[0]);
    const float2 ds1 = __half22float2(block.ds4[1]);
    const float2 ds2 = __half22float2(block.ds4[2]);
    const float2 ds3 = __half22float2(block.ds4[3]);
    return ds0.y + ds1.y + ds2.y + ds3.y;
}

static __device__ __forceinline__ void vec_dot_dual_payload_group128(
    const int* __restrict__ weight_payload,
    const int* __restrict__ activation_payload,
    const block_q8_1_mmq* __restrict__ activation_meta,
    const char* __restrict__ weights,
    float* __restrict__ sum,
    float* __restrict__ activation_sums,
    int weight_k_base,
    int row0,
    int kb,
    int groups_per_row
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int local_wave = threadIdx.y & (DUAL_PAYLOAD_WAVES_PER_PLANE - 1);
    const int col_lane_base = (local_wave % col_groups) * tile_C::J;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;

#pragma unroll
    for (int j0 = 0; j0 < DUAL_PAYLOAD_X;
         j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < DUAL_PAYLOAD_ACT_STRIDE; k01 += QI8_1) {
            tile_B b_frag;
            load_dual_payload_fragment(
                b_frag, activation_payload, col_lane_base + j0, k01,
                DUAL_PAYLOAD_ACT_STRIDE, 3);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                load_dual_payload_fragment(
                    a_frag, weight_payload, i0 + n * tile_A::I,
                    weight_k_base + k01, DUAL_PAYLOAD_WEIGHT_STRIDE, 7);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }

        const int j = j0 + col_lane_base + tile_C::get_j(0);
        const float activation_scale =
            __half22float2(activation_meta[j].ds4[0]).x;
        const int zero_acc = j0 / (col_groups * tile_C::J);
        activation_sums[zero_acc] +=
            dual_payload_activation_sum(activation_meta[j]);
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const int acc = zero_acc * row_frags + n;
                const char* group = weights
                    + ((long long)(row0 + i) * groups_per_row + kb)
                        * HFQ4_GROUP_BYTES;
                const float weight_scale = (float)(_Float16)__builtin_bit_cast(
                    float, *(const unsigned int*)group);
                sum[acc * tile_C::ne + l] +=
                    weight_scale * activation_scale
                        * (float)c_acc[n].x[l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(
    2 * DUAL_PAYLOAD_WAVES_PER_PLANE * WARP_SIZE, 1)
void gemm_hfq4g256_residual_mmq_gate_up_dual_payload_only(
    const char* __restrict__ gate,
    const char* __restrict__ up,
    const block_q8_1_mmq* __restrict__ activations,
    float* __restrict__ gate_output,
    float* __restrict__ up_output,
    int M,
    int K,
    int N
) {
    static_assert(MMQ_Q8_GROUP128, "payload-only dual requires group128 Q8");
    const int row0 = blockIdx.x * DUAL_PAYLOAD_ROWS;
    const int col0 = blockIdx.y * DUAL_PAYLOAD_X;
    const int plane = threadIdx.y / DUAL_PAYLOAD_WAVES_PER_PLANE;
    const int local_wave = threadIdx.y & (DUAL_PAYLOAD_WAVES_PER_PLANE - 1);
    const int groups_per_row = K / 256;

    extern __shared__ int smem[];
    int* activation_payload = smem;
    int* weight_payload =
        activation_payload + DUAL_PAYLOAD_X * DUAL_PAYLOAD_ACT_STRIDE;
    float sum[64] = {};
    float activation_sums[4] = {};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#pragma unroll
        for (int z = 0; z < 4; ++z) activation_sums[z] = 0.0f;
        load_dual_payload_weights(
            gate, up, weight_payload, row0, kb, groups_per_row);

        const block_q8_1_mmq* by0 =
            activations + (long long)(2 * kb) * N + col0;
        load_dual_payload_activations(by0, activation_payload);
        __syncthreads();
        const int* plane_weights = weight_payload
            + plane * DUAL_PAYLOAD_ROWS * DUAL_PAYLOAD_WEIGHT_STRIDE;
        const char* weights = plane ? up : gate;
        vec_dot_dual_payload_group128(
            plane_weights, activation_payload, by0, weights, sum,
            activation_sums, 0, row0, kb, groups_per_row);
        __syncthreads();

        const block_q8_1_mmq* by1 =
            activations + (long long)(2 * kb + 1) * N + col0;
        load_dual_payload_activations(by1, activation_payload);
        __syncthreads();
        vec_dot_dual_payload_group128(
            plane_weights, activation_payload, by1, weights, sum,
            activation_sums, 32, row0, kb, groups_per_row);
        __syncthreads();

        using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
        constexpr int row_frags = 2;
        constexpr int col_groups = 4;
        const int i0 =
            (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
        for (int j0 = 0; j0 < DUAL_PAYLOAD_X;
             j0 += col_groups * tile_C::J) {
            const int zero_acc = j0 / (col_groups * tile_C::J);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                    const char* group = weights
                        + ((long long)(row0 + i) * groups_per_row + kb)
                            * HFQ4_GROUP_BYTES;
                    const float zero = (float)(_Float16)__builtin_bit_cast(
                        float, *(const unsigned int*)(group + 4));
                    const int acc = zero_acc * row_frags + n;
                    const int idx = acc * tile_C::ne + l;
                    sum[idx] += zero * activation_sums[zero_acc];
                }
            }
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    float* output = plane ? up_output : gate_output;
#pragma unroll
    for (int j0 = 0; j0 < DUAL_PAYLOAD_X;
         j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j =
                    j0 + (local_wave % col_groups) * tile_C::J
                    + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const int acc = (j0 / (col_groups * tile_C::J))
                    * row_frags + n;
                output[(long long)(col0 + j) * M + row0 + i] =
                    sum[acc * tile_C::ne + l];
            }
        }
    }
}
#endif

#if MMQ_PACKED_DUAL_X256
extern "C" __global__ __launch_bounds__(16 * WARP_SIZE, 1)
void gemm_hfq4g256_residual_mmq_gate_up_packed_dual_x256(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_group128_compact* __restrict__ Xq,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int M,
    int K,
    int N
) {
    static_assert(MMQ_X == 256 && MMQ_Y == 128 && MMQ_NWARPS == 16,
                  "packed dual requires X256/Y128/16 waves");
    static_assert(MMQ_ROW_FRAGS == 2 && MMQ_COL_GROUPS == 4,
                  "packed dual requires row2/col4 topology");
    constexpr int plane_rows = 64;
    constexpr int plane_warps = 8;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    const int row0 = blockIdx.x * plane_rows;
    const int col0 = blockIdx.y * MMQ_X;
    const int plane = threadIdx.y / plane_warps;
    const int local_wave = threadIdx.y % plane_warps;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int groups_per_row = K / 256;

    extern __shared__ int smem[];
    int* activation_tile = smem;
    int* gate_tile = activation_tile + MMQ_X * PACKED_DUAL_ACT_STRIDE;
    int* up_tile = gate_tile + PACKED_DUAL_PLANE_INTS;
    float sum[MMQ_X * plane_rows / (plane_warps * WARP_SIZE)] = {0.0f};
    float activation_sums[MMQ_X / (col_groups * tile_C::J)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#pragma unroll
        for (int z = 0; z < 4; ++z) activation_sums[z] = 0.0f;
        load_hfq4_packed_dual_weights(
            A_gate, A_up, gate_tile, up_tile, row0, kb, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l = tid; l < MMQ_X * PACKED_DUAL_ACT_STRIDE;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by0[l];
        }
        __syncthreads();
        if (plane == 0) {
            vec_dot_packed_dual_group128(
                gate_tile, activation_tile, sum, 0, activation_sums);
        } else {
            vec_dot_packed_dual_group128(
                up_tile, activation_tile, sum, 0, activation_sums);
        }
        __syncthreads();

        const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        for (int l = tid; l < MMQ_X * PACKED_DUAL_ACT_STRIDE;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by1[l];
        }
        __syncthreads();
        if (plane == 0) {
            vec_dot_packed_dual_group128(
                gate_tile, activation_tile, sum, 32, activation_sums);
        } else {
            vec_dot_packed_dual_group128(
                up_tile, activation_tile, sum, 32, activation_sums);
        }

        const int* weight_tile = plane == 0 ? gate_tile : up_tile;
        const int i0 =
            (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
            const int zero_acc = j0 / (col_groups * tile_C::J);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                    const int meta = packed_dual_row_is_packed(row)
                        ? packed_dual_pack_row_ptr(weight_tile, row)[32]
                        : packed_dual_exp_row_ptr(weight_tile, row)[64];
                    const half2 dm = __builtin_bit_cast(half2, meta);
                    const int acc = zero_acc * row_frags + n;
                    sum[acc * tile_C::ne + l] +=
                        __half22float2(dm).y * activation_sums[zero_acc];
                }
            }
        }
        __syncthreads();
    }

    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    float* Y = plane == 0 ? Y_gate : Y_up;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j =
                    j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(l);
                const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                const int acc =
                    (j0 / (col_groups * tile_C::J)) * row_frags + n;
                Y[(long long)(col0 + j) * M + row0 + row] =
                    sum[acc * tile_C::ne + l];
            }
        }
    }
}
#endif

#if MMQ_PACKED_DUAL_ONE_PLANE
extern "C" __global__ __launch_bounds__(16 * WARP_SIZE, 1)
void gemm_hfq4g256_residual_mmq_gate_up_one_packed_plane(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_group128_compact* __restrict__ Xq,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int M,
    int K,
    int N
) {
    static_assert(MMQ_X == 256 && MMQ_Y == 128 && MMQ_NWARPS == 16,
                  "one-packed-plane requires X256/Y128/16 waves");
    constexpr int plane_rows = 64;
    constexpr int plane_warps = 8;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    const int row0 = blockIdx.x * plane_rows;
    const int col0 = blockIdx.y * MMQ_X;
    const int plane = threadIdx.y / plane_warps;
    const int local_wave = threadIdx.y % plane_warps;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int groups_per_row = K / 256;

    extern __shared__ int smem[];
    int* activation_tile = smem;
    int* gate_tile = activation_tile + MMQ_X * ONE_PLANE_ACT_STRIDE;
    int* up_tile = gate_tile + plane_rows * ONE_PLANE_EXP_STRIDE;
    float sum[MMQ_X * plane_rows / (plane_warps * WARP_SIZE)] = {0.0f};
    float activation_sums[MMQ_X / (col_groups * tile_C::J)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#pragma unroll
        for (int z = 0; z < 4; ++z) activation_sums[z] = 0.0f;
        load_hfq4_one_packed_plane_weights(
            A_gate, A_up, gate_tile, up_tile, row0, kb, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l = tid; l < MMQ_X * ONE_PLANE_ACT_STRIDE;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by0[l];
        }
        __syncthreads();
        if (plane == 0) {
            vec_dot_one_packed_plane_group128<false>(
                gate_tile, activation_tile, sum, 0, activation_sums);
        } else {
            vec_dot_one_packed_plane_group128<true>(
                up_tile, activation_tile, sum, 0, activation_sums);
        }
        __syncthreads();

        const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        for (int l = tid; l < MMQ_X * ONE_PLANE_ACT_STRIDE;
             l += MMQ_NWARPS * WARP_SIZE) {
            activation_tile[l] = by1[l];
        }
        __syncthreads();
        if (plane == 0) {
            vec_dot_one_packed_plane_group128<false>(
                gate_tile, activation_tile, sum, 32, activation_sums);
        } else {
            vec_dot_one_packed_plane_group128<true>(
                up_tile, activation_tile, sum, 32, activation_sums);
        }

        const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
            const int zero_acc = j0 / (col_groups * tile_C::J);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                    const int meta = plane
                        ? up_tile[row * ONE_PLANE_PACK_STRIDE + 32]
                        : gate_tile[row * ONE_PLANE_EXP_STRIDE + 64];
                    const half2 dm = __builtin_bit_cast(half2, meta);
                    const int acc = zero_acc * row_frags + n;
                    sum[acc * tile_C::ne + l] +=
                        __half22float2(dm).y * activation_sums[zero_acc];
                }
            }
        }
        __syncthreads();
    }

    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
    float* Y = plane == 0 ? Y_gate : Y_up;
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j =
                    j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(l);
                const int row = i0 + n * tile_C::I + tile_C::get_i(l);
                const int acc =
                    (j0 / (col_groups * tile_C::J)) * row_frags + n;
                Y[(long long)(col0 + j) * M + row0 + row] =
                    sum[acc * tile_C::ne + l];
            }
        }
    }
}
#endif

#endif
