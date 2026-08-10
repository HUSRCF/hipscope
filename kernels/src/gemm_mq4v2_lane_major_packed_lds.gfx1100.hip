// SPDX-License-Identifier: Apache-2.0
// Standalone exact-affine MQ4-v2 execution-format probe for gfx1100.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if !defined(__gfx1100__)
#error "gemm_mq4v2_lane_major_packed_lds requires gfx1100"
#endif

#define WARP_SIZE 32
#define TILE_N 256
#define TILE_M 64
#define WAVES 8
#define Q8_STRIDE_I32 36
#define PACKED_WORDS_PER_ROW 32
#define PACKED_WORDS_PER_TILE16 (16 * PACKED_WORDS_PER_ROW)
#define EXEC_GROUP_BYTES (16 * 136)

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad Q8 MMQ block size");

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

struct frag_i8_16x8 {
    int x[8] = {0};
};

struct frag_i32_16x16 {
    int x[8] = {0};
};

static __device__ __forceinline__ void load_i8_fragment(
    frag_i8_16x8& frag,
    const int* __restrict__ src,
    int stride
) {
    const int row = threadIdx.x & 15;
    *(int4*)&frag.x[0] = *(const int4*)(src + row * stride);
    *(int4*)&frag.x[4] = *(const int4*)(src + row * stride + 4);
}

static __device__ __forceinline__ void mma_i8(
    frag_i32_16x16& dst,
    const frag_i8_16x8& a,
    const frag_i8_16x8& b
) {
    int32x8_t* acc = (int32x8_t*)dst.x;
    const int32x4_t* av = (const int32x4_t*)a.x;
    const int32x4_t* bv = (const int32x4_t*)b.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, av[0], true, bv[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, av[1], true, bv[1], acc[0], true);
}

static __device__ __forceinline__ int output_i(int l) {
    return 2 * l + (threadIdx.x >> 4);
}

static __device__ __forceinline__ int output_j(int) {
    return threadIdx.x & 15;
}

// Execution layout per (tile16, group256):
//   scale[16] f32, zero[16] f32,
//   payload[subK32=8][packed_word=4][row=16] u32.
// The LDS payload drops the two FP32 metadata planes and stores one half2 per
// row after conversion, preserving the retained kernel's arithmetic contract.
static __device__ __forceinline__ void stage_weight_group(
    const char* __restrict__ execution,
    int* __restrict__ payload,
    int* __restrict__ metadata,
    int row0,
    int kb,
    int groups_per_row
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int local_tile = 0; local_tile < 4; ++local_tile) {
        const long long tile = (row0 >> 4) + local_tile;
        const char* group = execution
            + (tile * groups_per_row + kb) * (long long)EXEC_GROUP_BYTES;
        const int* src = (const int*)(group + 128);
        int* dst = payload + local_tile * PACKED_WORDS_PER_TILE16;
        for (int word = tid; word < PACKED_WORDS_PER_TILE16;
             word += WAVES * WARP_SIZE) {
            dst[word] = src[word];
        }
    }
    if (tid < TILE_M) {
        const int local_tile = tid >> 4;
        const int row = tid & 15;
        const long long tile = (row0 >> 4) + local_tile;
        const char* group = execution
            + (tile * groups_per_row + kb) * (long long)EXEC_GROUP_BYTES;
        const float scale = *(const float*)(group + 4 * row);
        const float zero = *(const float*)(group + 64 + 4 * row);
        metadata[tid] = __builtin_bit_cast(
            int, make_half2((_Float16)scale, (_Float16)zero));
    }
}

static __device__ __forceinline__ void load_weight_fragment(
    frag_i8_16x8& frag,
    const int* __restrict__ payload,
    int local_row_base,
    int k0
) {
    const int local_row = local_row_base + (threadIdx.x & 15);
    const int local_tile = local_row >> 4;
    const int row = local_row & 15;
    const int subtile = k0 >> 5;
    const int* src = payload
        + local_tile * PACKED_WORDS_PER_TILE16
        + subtile * 64;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const unsigned int q = (unsigned int)src[word * 16 + row];
        const unsigned int lo = q & 0x0F0F0F0Fu;
        const unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        frag.x[2 * word] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        frag.x[2 * word + 1] =
            (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}

static __device__ __forceinline__ void collect_sums(
    const int* __restrict__ activation,
    float* __restrict__ sums,
    bool reset
) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (tid < TILE_N) {
        const half2* ds = (const half2*)(activation + tid * Q8_STRIDE_I32);
        float value = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) value += __half22float2(ds[i]).y;
        sums[tid] = reset ? value : sums[tid] + value;
    }
}

static __device__ __forceinline__ void dot_half(
    const int* __restrict__ payload,
    const int* __restrict__ metadata,
    const int* __restrict__ activation,
    float* __restrict__ accum,
    int k_base
) {
    const int wave = threadIdx.y;
    const int row_base = (wave >> 2) * 32;
    const int col_group = wave & 3;
    const int* activation_group = activation + col_group * 16 * Q8_STRIDE_I32;
    const int* activation_qs = activation_group + 4;
    const half2* activation_ds = (const half2*)activation_group;

#pragma unroll
    for (int col_base = 0; col_base < TILE_N; col_base += 64) {
        frag_i32_16x16 c[2];
#pragma unroll
        for (int k_local = 0; k_local < 128; k_local += 32) {
            frag_i8_16x8 b;
            load_i8_fragment(
                b,
                activation_qs + col_base * Q8_STRIDE_I32 + k_local / 4,
                Q8_STRIDE_I32);
#pragma unroll
            for (int row_frag = 0; row_frag < 2; ++row_frag) {
                frag_i8_16x8 a;
                load_weight_fragment(
                    a, payload, row_base + row_frag * 16, k_base + k_local);
                mma_i8(c[row_frag], a, b);
            }
        }

        const int local_col = col_base + output_j(0);
        const float activation_scale =
            __half22float2(activation_ds[local_col * Q8_STRIDE_I32]).x;
#pragma unroll
        for (int row_frag = 0; row_frag < 2; ++row_frag) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int local_row = row_base + row_frag * 16 + output_i(l);
                const half2 dm = __builtin_bit_cast(half2, metadata[local_row]);
                const int acc = (col_base >> 6) * 16 + row_frag * 8 + l;
                accum[acc] += __half22float2(dm).x * activation_scale
                    * (float)c[row_frag].x[l];
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_body(
    const char* __restrict__ execution,
    const block_q8_1_mmq* __restrict__ xq,
    float* __restrict__ y,
    int m,
    int k,
    int n
) {
    const int row0 = blockIdx.x * TILE_M;
    const int col0 = blockIdx.y * TILE_N;
    const int groups_per_row = k / 256;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    extern __shared__ int smem[];
    int* activation = smem;
    int* weight_payload = activation + TILE_N * Q8_STRIDE_I32;
    int* weight_metadata = weight_payload + TILE_M * PACKED_WORDS_PER_ROW;
    float* activation_sums = (float*)(weight_metadata + TILE_M);
    float accum[64] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        stage_weight_group(
            execution, weight_payload, weight_metadata,
            row0, kb, groups_per_row);
        const int* x0 = (const int*)(xq + (long long)(2 * kb) * n + col0);
        for (int i = tid; i < TILE_N * Q8_STRIDE_I32;
             i += WAVES * WARP_SIZE) {
            activation[i] = x0[i];
        }
        __syncthreads();
        collect_sums(activation, activation_sums, true);
        dot_half(weight_payload, weight_metadata, activation, accum, 0);
        __syncthreads();

        const int* x1 = (const int*)(xq + (long long)(2 * kb + 1) * n + col0);
        for (int i = tid; i < TILE_N * Q8_STRIDE_I32;
             i += WAVES * WARP_SIZE) {
            activation[i] = x1[i];
        }
        __syncthreads();
        collect_sums(activation, activation_sums, false);
        dot_half(weight_payload, weight_metadata, activation, accum, 128);
        __syncthreads();

        const int wave = threadIdx.y;
        const int row_base = (wave >> 2) * 32;
        const int col_group = wave & 3;
#pragma unroll
        for (int col_base = 0; col_base < TILE_N; col_base += 64) {
            const int local_col = col_base + output_j(0);
            const float sum_x = activation_sums[col_group * 16 + local_col];
#pragma unroll
            for (int row_frag = 0; row_frag < 2; ++row_frag) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int local_row = row_base + row_frag * 16 + output_i(l);
                    const half2 dm =
                        __builtin_bit_cast(half2, weight_metadata[local_row]);
                    const int acc = (col_base >> 6) * 16 + row_frag * 8 + l;
                    accum[acc] += __half22float2(dm).y * sum_x;
                }
            }
        }
        __syncthreads();
    }

    const int wave = threadIdx.y;
    const int row_base = (wave >> 2) * 32;
    const int col_group = wave & 3;
#pragma unroll
    for (int col_base = 0; col_base < TILE_N; col_base += 64) {
#pragma unroll
        for (int row_frag = 0; row_frag < 2; ++row_frag) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int local_col = col_base + col_group * 16 + output_j(l);
                const int local_row = row_base + row_frag * 16 + output_i(l);
                const int acc = (col_base >> 6) * 16 + row_frag * 8 + l;
                float* dst = y + (long long)(col0 + local_col) * m + row0 + local_row;
                if constexpr (ADD) {
                    *dst += accum[acc];
                } else {
                    *dst = accum[acc];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(WAVES * WARP_SIZE, 1)
void gemm_mq4v2_lane_major_packed_lds_set(
    const char* execution,
    const block_q8_1_mmq* xq,
    float* y,
    int m,
    int k,
    int n
) {
    gemm_body<false>(execution, xq, y, m, k, n);
}

extern "C" __global__ __launch_bounds__(WAVES * WARP_SIZE, 1)
void gemm_mq4v2_lane_major_packed_lds_add(
    const char* execution,
    const block_q8_1_mmq* xq,
    float* y,
    int m,
    int k,
    int n
) {
    gemm_body<true>(execution, xq, y, m, k, n);
}
