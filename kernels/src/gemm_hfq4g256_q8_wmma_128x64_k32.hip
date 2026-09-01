// SPDX-License-Identifier: MIT OR Apache-2.0
// gfx11 structural probe: Q8_1 activations, HFQ4 weights, 128M x 64N, K32.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

using int4x_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int8x_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

template <bool ADD_RESIDUAL, int K_TILE, bool GROUP128 = false,
          int TOKEN_TILE = 128>
static __device__ __forceinline__ void gemm_hfq4g256_q8_wmma_128x64_k32_impl(
    const char* __restrict__ weights,
    const block_q8_1_mmq* __restrict__ activations,
    float* __restrict__ output, int output_dim, int k_dim, int tokens) {
    static_assert(K_TILE == 32 || K_TILE == 64 || K_TILE == 128,
                  "unsupported K tile");
    static_assert(!GROUP128 || K_TILE == 128,
                  "group128 path requires K128 staging");
    static_assert(TOKEN_TILE == 128 || TOKEN_TILE == 256,
                  "unsupported token tile");
    constexpr int k_ints = K_TILE / 4;
    constexpr int k_stride = k_ints + 2;
    constexpr int q8_subblocks = K_TILE / 32;
    constexpr int output_tile_size = 64;
    constexpr int token_tile_size = TOKEN_TILE;

    const int output_tile = blockIdx.x * output_tile_size;
    const int token_tile = blockIdx.y * token_tile_size;
    if (output_tile >= output_dim || token_tile >= tokens) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int groups_per_row = k_dim / 256;
    const int actual_token = token_tile + wave * 16 + lane_lo;

    float sum[4][8] = {};
    int8x_t group_dot[4] = {};
    float group_activation_sum = 0.0f;
    float group_activation_scale = 0.0f;
    __shared__ int weight_lds[2][output_tile_size][k_stride];
    __shared__ half2 weight_dm[2][output_tile_size];

    auto stage_k32 = [&](int buffer, int k_tile) __attribute__((always_inline)) {
        // A 64xK_TILE weight tile needs only 256 staging lanes.  The X256
        // probe launches 16 waves so the upper half skips staging and joins
        // the compute after the workgroup barrier.
        const int weight_row = tid >> 2;
        const int weight_segment = tid & 3;
        const int actual_output = output_tile + weight_row;
        if (tid < 256 && actual_output < output_dim) {
            const int group_index = k_tile >> 8;
            const char* group = weights
                + ((long long)actual_output * groups_per_row + group_index) * 136;
            const int packed_offset =
                ((k_tile & 255) >> 1) + weight_segment * (K_TILE / 8);
#pragma unroll
            for (int word = 0; word < K_TILE / 32; ++word) {
                const unsigned int packed = *reinterpret_cast<const unsigned int*>(
                    group + 8 + packed_offset + word * 4);
                const unsigned int lo = packed & 0x0F0F0F0Fu;
                const unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
                const int out = weight_segment * (K_TILE / 16) + word * 2;
                weight_lds[buffer][weight_row][out] =
                    (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
                weight_lds[buffer][weight_row][out + 1] =
                    (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            }
            if (weight_segment == 0) {
                const float scale = __builtin_bit_cast(
                    float, *reinterpret_cast<const unsigned int*>(group));
                const float zero = __builtin_bit_cast(
                    float, *reinterpret_cast<const unsigned int*>(group + 4));
                weight_dm[buffer][weight_row] =
                    make_half2((_Float16)scale, (_Float16)zero);
            }
        }

    };

    stage_k32(0, 0);
    __syncthreads();

    int current = 0;
    for (int k_tile = 0; k_tile < k_dim; k_tile += K_TILE) {
        const int next = current ^ 1;
        if (k_tile + K_TILE < k_dim) stage_k32(next, k_tile + K_TILE);

#pragma unroll
        for (int subblock = 0; subblock < q8_subblocks; ++subblock) {
            int8x_t activation_fragment = {0, 0, 0, 0, 0, 0, 0, 0};
            float2 ds = make_float2(0, 0);
            const int subblock_k = k_tile + subblock * 32;
            if (actual_token < tokens) {
                const int q8_block = subblock_k >> 7;
                const int q8_subblock = (subblock_k & 127) >> 5;
                const block_q8_1_mmq* block =
                    activations + (long long)q8_block * tokens + actual_token;
                activation_fragment = *reinterpret_cast<const int8x_t*>(
                    block->qs + q8_subblock * 32);
                ds = __half22float2(block->ds4[q8_subblock]);
            }
            const int4x_t* activation_halves =
                reinterpret_cast<const int4x_t*>(&activation_fragment);
            if constexpr (GROUP128) {
                if ((subblock_k & 127) == 0) {
                    group_activation_sum = 0.0f;
                    group_activation_scale = ds.x;
                }
                group_activation_sum += ds.y;
            }

#pragma unroll
            for (int output_group = 0; output_group < 4; ++output_group) {
                const int weight_row = output_group * 16 + lane_lo;
                const int8x_t weight_fragment = *reinterpret_cast<const int8x_t*>(
                    &weight_lds[current][weight_row][subblock * 8]);
                const int4x_t* weight_halves =
                    reinterpret_cast<const int4x_t*>(&weight_fragment);
                int8x_t dot = GROUP128
                    ? group_dot[output_group]
                    : int8x_t{0, 0, 0, 0, 0, 0, 0, 0};
                dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, weight_halves[0], true, activation_halves[0], dot, true);
                dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, weight_halves[1], true, activation_halves[1], dot, true);
                if constexpr (GROUP128) group_dot[output_group] = dot;

#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    const int output_row = output_group * 16 + 2 * element + lane_hi;
                    const float2 dm = __half22float2(weight_dm[current][output_row]);
                    if constexpr (GROUP128) {
                        if ((subblock_k & 127) == 96) {
                            sum[output_group][element] +=
                                dm.x * group_activation_scale * (float)dot[element]
                                + dm.y * group_activation_sum;
                        }
                    } else {
                        sum[output_group][element] +=
                            dm.x * ds.x * (float)dot[element] + dm.y * ds.y;
                    }
                }
            }
            if constexpr (GROUP128) {
                if ((subblock_k & 127) == 96) {
#pragma unroll
                    for (int output_group = 0; output_group < 4; ++output_group) {
                        group_dot[output_group] = int8x_t{0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }
        }

        __syncthreads();
        current = next;
    }

    if (actual_token >= tokens) return;
#pragma unroll
    for (int output_group = 0; output_group < 4; ++output_group) {
#pragma unroll
        for (int element = 0; element < 8; ++element) {
            const int local_output = output_group * 16 + 2 * element + lane_hi;
            const int actual_output = output_tile + local_output;
            if (actual_output < output_dim) {
                float* dst = output + (long long)actual_token * output_dim + actual_output;
                if constexpr (ADD_RESIDUAL) {
                    *dst += sum[output_group][element];
                } else {
                    *dst = sum[output_group][element];
                }
            }
        }
    }
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k32_set(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<false, 32>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k32_add(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<true, 32>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(512, 1) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_256x64_k32_set(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<false, 32, false, 256>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(512, 1) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_256x64_k32_add(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<true, 32, false, 256>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k128_group128_set(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<false, 128, true>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k128_group128_add(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<true, 128, true>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k64_set(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<false, 64>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 2) extern "C" __global__ void
gemm_hfq4g256_q8_wmma_128x64_k64_add(
    const char* weights, const block_q8_1_mmq* activations, float* output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_q8_wmma_128x64_k32_impl<true, 64>(
        weights, activations, output, output_dim, k_dim, tokens);
}

static __device__ __forceinline__ void
gemm_gate_up_hfq4g256_q8_wmma_128x64_k32_impl(
    const char* __restrict__ gate_weights,
    const char* __restrict__ up_weights,
    const block_q8_1_mmq* __restrict__ activations,
    float* __restrict__ gate_output, float* __restrict__ up_output,
    int output_dim, int k_dim, int tokens) {
    constexpr int k_tile_size = 32;
    constexpr int k_ints = k_tile_size / 4;
    constexpr int k_stride = k_ints + 2;
    constexpr int output_tile_size = 64;
    constexpr int token_tile_size = 128;

    const int output_tile = blockIdx.x * output_tile_size;
    const int token_tile = blockIdx.y * token_tile_size;
    if (output_tile >= output_dim || token_tile >= tokens) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int groups_per_row = k_dim / 256;
    const int actual_token = token_tile + wave * 16 + lane_lo;

    float sum[2][4][8] = {};
    __shared__ int weight_lds[2][2][output_tile_size][k_stride];
    __shared__ half2 weight_dm[2][2][output_tile_size];

    auto stage_pair = [&](int buffer, int k_tile) __attribute__((always_inline)) {
        const int weight_row = tid >> 2;
        const int weight_segment = tid & 3;
        const int actual_output = output_tile + weight_row;
        if (actual_output >= output_dim) return;
        const int group_index = k_tile >> 8;
#pragma unroll
        for (int projection = 0; projection < 2; ++projection) {
            const char* weights = projection == 0 ? gate_weights : up_weights;
            const char* group = weights
                + ((long long)actual_output * groups_per_row + group_index) * 136;
            const int packed_offset = ((k_tile & 255) >> 1) + weight_segment * 4;
            const unsigned int packed = *reinterpret_cast<const unsigned int*>(
                group + 8 + packed_offset);
            const unsigned int lo = packed & 0x0F0F0F0Fu;
            const unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
            const int out = weight_segment * 2;
            weight_lds[projection][buffer][weight_row][out] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
            weight_lds[projection][buffer][weight_row][out + 1] =
                (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
            if (weight_segment == 0) {
                const float scale = __builtin_bit_cast(
                    float, *reinterpret_cast<const unsigned int*>(group));
                const float zero = __builtin_bit_cast(
                    float, *reinterpret_cast<const unsigned int*>(group + 4));
                weight_dm[projection][buffer][weight_row] =
                    make_half2((_Float16)scale, (_Float16)zero);
            }
        }
    };

    stage_pair(0, 0);
    __syncthreads();

    int current = 0;
    for (int k_tile = 0; k_tile < k_dim; k_tile += k_tile_size) {
        const int next = current ^ 1;
        if (k_tile + k_tile_size < k_dim) stage_pair(next, k_tile + k_tile_size);

        int8x_t activation_fragment = {0, 0, 0, 0, 0, 0, 0, 0};
        float2 ds = make_float2(0, 0);
        if (actual_token < tokens) {
            const int q8_block = k_tile >> 7;
            const int q8_subblock = (k_tile & 127) >> 5;
            const block_q8_1_mmq* block =
                activations + (long long)q8_block * tokens + actual_token;
            activation_fragment = *reinterpret_cast<const int8x_t*>(
                block->qs + q8_subblock * 32);
            ds = __half22float2(block->ds4[q8_subblock]);
        }
        const int4x_t* activation_halves =
            reinterpret_cast<const int4x_t*>(&activation_fragment);

#pragma unroll
        for (int output_group = 0; output_group < 4; ++output_group) {
            const int weight_row = output_group * 16 + lane_lo;
#pragma unroll
            for (int projection = 0; projection < 2; ++projection) {
                const int8x_t weight_fragment = *reinterpret_cast<const int8x_t*>(
                    &weight_lds[projection][current][weight_row][0]);
                const int4x_t* weight_halves =
                    reinterpret_cast<const int4x_t*>(&weight_fragment);
                int8x_t dot = {0, 0, 0, 0, 0, 0, 0, 0};
                dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, weight_halves[0], true, activation_halves[0], dot, true);
                dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, weight_halves[1], true, activation_halves[1], dot, true);

#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    const int output_row = output_group * 16 + 2 * element + lane_hi;
                    const float2 dm = __half22float2(
                        weight_dm[projection][current][output_row]);
                    sum[projection][output_group][element] +=
                        dm.x * ds.x * (float)dot[element] + dm.y * ds.y;
                }
            }
        }

        __syncthreads();
        current = next;
    }

    if (actual_token >= tokens) return;
#pragma unroll
    for (int projection = 0; projection < 2; ++projection) {
        float* output = projection == 0 ? gate_output : up_output;
#pragma unroll
        for (int output_group = 0; output_group < 4; ++output_group) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int local_output = output_group * 16 + 2 * element + lane_hi;
                const int actual_output = output_tile + local_output;
                if (actual_output < output_dim) {
                    output[(long long)actual_token * output_dim + actual_output] =
                        sum[projection][output_group][element];
                }
            }
        }
    }
}

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_gate_up_hfq4g256_q8_wmma_128x64_k32_set(
    const char* gate_weights, const char* up_weights,
    const block_q8_1_mmq* activations, float* gate_output, float* up_output,
    int output_dim, int k_dim, int tokens) {
    gemm_gate_up_hfq4g256_q8_wmma_128x64_k32_impl(
        gate_weights, up_weights, activations, gate_output, up_output,
        output_dim, k_dim, tokens);
}

#else

struct block_q8_1_mmq;
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k32_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k32_add(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_256x64_k32_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_256x64_k32_add(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k128_group128_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k128_group128_add(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k64_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_q8_wmma_128x64_k64_add(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void gemm_gate_up_hfq4g256_q8_wmma_128x64_k32_set(
    const char*, const char*, const block_q8_1_mmq*, float*, float*,
    int, int, int) {}

#endif
