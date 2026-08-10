// SPDX-License-Identifier: MIT OR Apache-2.0
// Standalone gfx11 execution-format probe. This is not a serving kernel.

#include <hip/hip_runtime.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__)

using int32x4_t = int __attribute__((ext_vector_type(4)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using int8x4_t = signed char __attribute__((ext_vector_type(4)));

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 128;
static constexpr int WAVES = 8;

struct wmma_accum {
    int32x8_t value = {0, 0, 0, 0, 0, 0, 0, 0};
};

static __device__ __forceinline__ int unpack_centered_i4(unsigned int packed,
                                                         unsigned int selector) {
    const unsigned int lo = packed & 0x0F0F0F0Fu;
    const unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
    const int unpacked = (int)__builtin_amdgcn_perm(hi, lo, selector);
    const int8x4_t values = __builtin_bit_cast(int8x4_t, unpacked);
    const int8x4_t offset = {8, 8, 8, 8};
    return __builtin_bit_cast(int, values - offset);
}

static __device__ __forceinline__ int32x4_t load_i8x16(
    const signed char* base, int row, int stride, int k) {
    return *reinterpret_cast<const int32x4_t*>(base + row * stride + k);
}

static __device__ __forceinline__ void wmma_i8_k32(
    wmma_accum& acc,
    const signed char* weights,
    const signed char* activations,
    int weight_row,
    int activation_row,
    int k) {
    const int32x4_t w0 = load_i8x16(weights, weight_row, TILE_K, k);
    const int32x4_t x0 = load_i8x16(activations, activation_row, TILE_K, k);
    acc.value = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, w0, true, x0, acc.value, true);

    const int32x4_t w1 = load_i8x16(weights, weight_row, TILE_K, k + 16);
    const int32x4_t x1 = load_i8x16(activations, activation_row, TILE_K, k + 16);
    acc.value = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, w1, true, x1, acc.value, true);
}

template <bool ADD, bool WEIGHT_Q4 = false>
static __device__ __forceinline__ void row_i8_q8_body(
    const signed char* __restrict__ weights,
    const float* __restrict__ weight_scales,
    const signed char* __restrict__ activations,
    const float* __restrict__ activation_scales,
    float* __restrict__ output,
    int output_dim,
    int k_dim,
    int tokens) {
    const int output_tile = blockIdx.x * TILE_M;
    const int token_tile = blockIdx.y * TILE_N;
    if (output_tile >= output_dim || token_tile >= tokens) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int wave = threadIdx.y;
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int token_group = wave & 3;
    const int output_group = (wave >> 2) * 32;

    extern __shared__ unsigned char shared_bytes[];
    signed char* weight_tile = reinterpret_cast<signed char*>(shared_bytes);
    signed char* activation_tile = weight_tile + TILE_M * TILE_K;

    // Four token subtiles x two output fragments. Unlike the retained MQ4
    // path, these int32 accumulators remain live across the complete K range.
    wmma_accum accum[4][2];

    constexpr int vec_bytes = 16;
    constexpr int activation_vecs = TILE_N * TILE_K / vec_bytes;

    for (int k_tile = 0; k_tile < k_dim; k_tile += TILE_K) {
        if constexpr (WEIGHT_Q4) {
            // Exactly one 16-byte transaction per work-item expands 32 Q4
            // weights into one aligned 32-byte LDS segment.
            const int row = tid >> 2;
            const int k_vec = tid & 3;
            const int32x4_t packed = *reinterpret_cast<const int32x4_t*>(
                weights + (long long)(output_tile + row) * (k_dim / 2)
                + k_tile / 2 + k_vec * vec_bytes);
            int* destination = reinterpret_cast<int*>(
                weight_tile + row * TILE_K + k_vec * 32);
            const unsigned int words[4] = {
                (unsigned int)packed[0], (unsigned int)packed[1],
                (unsigned int)packed[2], (unsigned int)packed[3]};
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                destination[2 * word] =
                    unpack_centered_i4(words[word], 0x05010400u);
                destination[2 * word + 1] =
                    unpack_centered_i4(words[word], 0x07030602u);
            }
        } else {
            constexpr int weight_vecs = TILE_M * TILE_K / vec_bytes;
            for (int index = tid; index < weight_vecs; index += WAVES * 32) {
                const int row = index / (TILE_K / vec_bytes);
                const int k_vec = index % (TILE_K / vec_bytes);
                const int32x4_t value = *reinterpret_cast<const int32x4_t*>(
                    weights + (long long)(output_tile + row) * k_dim + k_tile
                    + k_vec * vec_bytes);
                *reinterpret_cast<int32x4_t*>(
                    weight_tile + row * TILE_K + k_vec * vec_bytes) = value;
            }
        }
        for (int index = tid; index < activation_vecs; index += WAVES * 32) {
            const int row = index / (TILE_K / vec_bytes);
            const int k_vec = index % (TILE_K / vec_bytes);
            const int32x4_t value = *reinterpret_cast<const int32x4_t*>(
                activations + (long long)(token_tile + row) * k_dim + k_tile
                + k_vec * vec_bytes);
            *reinterpret_cast<int32x4_t*>(
                activation_tile + row * TILE_K + k_vec * vec_bytes) = value;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_K; k += 32) {
#pragma unroll
            for (int token_subtile = 0; token_subtile < 4; ++token_subtile) {
                const int activation_row =
                    token_subtile * 64 + token_group * 16 + lane_row;
#pragma unroll
                for (int output_fragment = 0; output_fragment < 2;
                     ++output_fragment) {
                    const int weight_row =
                        output_group + output_fragment * 16 + lane_row;
                    wmma_i8_k32(
                        accum[token_subtile][output_fragment], weight_tile,
                        activation_tile, weight_row, activation_row, k);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int token_subtile = 0; token_subtile < 4; ++token_subtile) {
        const int token = token_tile + token_subtile * 64
            + token_group * 16 + lane_row;
        const float activation_scale = activation_scales[token];
#pragma unroll
        for (int output_fragment = 0; output_fragment < 2;
             ++output_fragment) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int output_row = output_tile + output_group
                    + output_fragment * 16 + 2 * element + lane_half;
                const float value =
                    (float)accum[token_subtile][output_fragment].value[element]
                    * weight_scales[output_row] * activation_scale;
                float* destination =
                    output + (long long)token * output_dim + output_row;
                if constexpr (ADD) {
                    *destination += value;
                } else {
                    *destination = value;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_mq4v2_row_i8_q8_wmma_256x64_set(
    const signed char* weights,
    const float* weight_scales,
    const signed char* activations,
    const float* activation_scales,
    float* output,
    int output_dim,
    int k_dim,
    int tokens) {
    row_i8_q8_body<false>(weights, weight_scales, activations,
                          activation_scales, output, output_dim, k_dim, tokens);
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_mq4v2_row_i8_q8_wmma_256x64_add(
    const signed char* weights,
    const float* weight_scales,
    const signed char* activations,
    const float* activation_scales,
    float* output,
    int output_dim,
    int k_dim,
    int tokens) {
    row_i8_q8_body<true>(weights, weight_scales, activations,
                         activation_scales, output, output_dim, k_dim, tokens);
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_mq4v2_row_q4_q8_wmma_256x64_set(
    const signed char* weights,
    const float* weight_scales,
    const signed char* activations,
    const float* activation_scales,
    float* output,
    int output_dim,
    int k_dim,
    int tokens) {
    row_i8_q8_body<false, true>(weights, weight_scales, activations,
                                activation_scales, output, output_dim, k_dim,
                                tokens);
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_mq4v2_row_q4_q8_wmma_256x64_add(
    const signed char* weights,
    const float* weight_scales,
    const signed char* activations,
    const float* activation_scales,
    float* output,
    int output_dim,
    int k_dim,
    int tokens) {
    row_i8_q8_body<true, true>(weights, weight_scales, activations,
                               activation_scales, output, output_dim, k_dim,
                               tokens);
}

#else
extern "C" __global__ void gemm_mq4v2_row_i8_q8_wmma_256x64_set(
    const signed char*, const float*, const signed char*, const float*, float*,
    int, int, int) {}
extern "C" __global__ void gemm_mq4v2_row_i8_q8_wmma_256x64_add(
    const signed char*, const float*, const signed char*, const float*, float*,
    int, int, int) {}
extern "C" __global__ void gemm_mq4v2_row_q4_q8_wmma_256x64_set(
    const signed char*, const float*, const signed char*, const float*, float*,
    int, int, int) {}
extern "C" __global__ void gemm_mq4v2_row_q4_q8_wmma_256x64_add(
    const signed char*, const float*, const signed char*, const float*, float*,
    int, int, int) {}
#endif
