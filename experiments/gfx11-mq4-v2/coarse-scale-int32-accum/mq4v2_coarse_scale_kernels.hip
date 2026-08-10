// SPDX-License-Identifier: MIT
// Standalone-only gfx1100 MQ4-v2 coarse-scale / int32-accumulation probe.

#include <hip/hip_runtime.h>

#include <stdint.h>

namespace mq4v2_probe {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kGroupK = 128;
constexpr int kCoarseGroups = 4;
constexpr int kCoarseK = kGroupK * kCoarseGroups;
constexpr int kWaveSize = 32;

struct ExactWeightGroup128 {
    float scale;
    float zero;
    uint32_t payload_words[16];
};

struct CoarseWeightGroup512 {
    float scale;
    float zero;
    uint32_t payload_words[64];
};

struct ExactActGroup128 {
    float scale;
    float sum;
    uint32_t q_words[32];
};

struct CoarseActGroup512 {
    float scale;
    float sum;
    uint32_t q_words[128];
};

static_assert(sizeof(ExactWeightGroup128) == 72, "bad ExactWeightGroup128 size");
static_assert(sizeof(CoarseWeightGroup512) == 264, "bad CoarseWeightGroup512 size");
static_assert(sizeof(ExactActGroup128) == 136, "bad ExactActGroup128 size");
static_assert(sizeof(CoarseActGroup512) == 520, "bad CoarseActGroup512 size");

using int8x16_t = int8_t __attribute__((ext_vector_type(16)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ int32x4_t as_i32x4(const int8x16_t& value) {
    return *reinterpret_cast<const int32x4_t*>(&value);
}

static __device__ __forceinline__ int32x4_t load_q_words4(
    const uint32_t* __restrict__ words,
    int word_index
) {
    int32x4_t out;
    out[0] = static_cast<int32_t>(words[word_index + 0]);
    out[1] = static_cast<int32_t>(words[word_index + 1]);
    out[2] = static_cast<int32_t>(words[word_index + 2]);
    out[3] = static_cast<int32_t>(words[word_index + 3]);
    return out;
}

static __device__ __forceinline__ int32x8_t wmma_i8(
    const int32x4_t& a_words,
    const int32x4_t& b_words,
    const int32x8_t& acc
) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, a_words, true, b_words, acc, true);
}

static __device__ __forceinline__ int32x4_t unpack_u4x16_to_i32x4(
    uint32_t packed0,
    uint32_t packed1
) {
    int8x16_t values;
    values[0] = static_cast<int8_t>((packed0 >> 0) & 0xFu);
    values[1] = static_cast<int8_t>((packed0 >> 4) & 0xFu);
    values[2] = static_cast<int8_t>((packed0 >> 8) & 0xFu);
    values[3] = static_cast<int8_t>((packed0 >> 12) & 0xFu);
    values[4] = static_cast<int8_t>((packed0 >> 16) & 0xFu);
    values[5] = static_cast<int8_t>((packed0 >> 20) & 0xFu);
    values[6] = static_cast<int8_t>((packed0 >> 24) & 0xFu);
    values[7] = static_cast<int8_t>((packed0 >> 28) & 0xFu);
    values[8] = static_cast<int8_t>((packed1 >> 0) & 0xFu);
    values[9] = static_cast<int8_t>((packed1 >> 4) & 0xFu);
    values[10] = static_cast<int8_t>((packed1 >> 8) & 0xFu);
    values[11] = static_cast<int8_t>((packed1 >> 12) & 0xFu);
    values[12] = static_cast<int8_t>((packed1 >> 16) & 0xFu);
    values[13] = static_cast<int8_t>((packed1 >> 20) & 0xFu);
    values[14] = static_cast<int8_t>((packed1 >> 24) & 0xFu);
    values[15] = static_cast<int8_t>((packed1 >> 28) & 0xFu);
    return as_i32x4(values);
}

template <bool AddOutput>
static __device__ __forceinline__ void exact_group128_body(
    const ExactWeightGroup128* __restrict__ weights,
    const ExactActGroup128* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int lane16 = tid & 15;
    const int parity = tid >> 4;
    const int row_tile = blockIdx.x * kTileM;
    const int col_tile = blockIdx.y * kTileN;
    const int row_lane = row_tile + lane16;
    const int col_lane = col_tile + lane16;
    const bool row_valid = row_lane < M;
    const bool col_valid = col_lane < N;
    const int groups128 = K / kGroupK;

    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups128; ++g) {
        const ExactWeightGroup128* row_group =
            row_valid ? (weights + static_cast<long long>(row_lane) * groups128 + g)
                      : nullptr;
        const ExactActGroup128* col_group =
            col_valid ? (acts + static_cast<long long>(g) * N + col_lane)
                      : nullptr;

        const float scale_self = row_group != nullptr ? row_group->scale : 0.0f;
        const float zero_self = row_group != nullptr ? row_group->zero : 0.0f;
        float row_scale[8];
        float row_zero[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src_lane = 2 * j + parity;
            row_scale[j] = __shfl(scale_self, src_lane, kWaveSize);
            row_zero[j] = __shfl(zero_self, src_lane, kWaveSize);
        }

        const float act_scale = col_group != nullptr ? col_group->scale : 0.0f;
        const float act_sum = col_group != nullptr ? col_group->sum : 0.0f;

        int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int k16 = 0; k16 < 8; ++k16) {
            const uint32_t packed0 =
                row_group != nullptr ? row_group->payload_words[2 * k16 + 0] : 0u;
            const uint32_t packed1 =
                row_group != nullptr ? row_group->payload_words[2 * k16 + 1] : 0u;
            const int32x4_t a_words = unpack_u4x16_to_i32x4(packed0, packed1);
            const int32x4_t b_words =
                col_group != nullptr ? load_q_words4(col_group->q_words, 4 * k16)
                                    : int32x4_t{0, 0, 0, 0};
            cacc = wmma_i8(a_words, b_words, cacc);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            accum[j] += row_scale[j] * act_scale * static_cast<float>(cacc[j]) +
                        row_zero[j] * act_sum;
        }
    }

    if (!col_valid) {
        return;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int out_row = row_tile + 2 * j + parity;
        if (out_row < M) {
            const long long out_index = static_cast<long long>(col_lane) * M + out_row;
            float value = accum[j];
            if constexpr (AddOutput) {
                value += output[out_index];
            }
            output[out_index] = value;
        }
    }
}

template <bool AddOutput>
static __device__ __forceinline__ void coarse_group512_body(
    const CoarseWeightGroup512* __restrict__ weights,
    const CoarseActGroup512* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int lane16 = tid & 15;
    const int parity = tid >> 4;
    const int row_tile = blockIdx.x * kTileM;
    const int col_tile = blockIdx.y * kTileN;
    const int row_lane = row_tile + lane16;
    const int col_lane = col_tile + lane16;
    const bool row_valid = row_lane < M;
    const bool col_valid = col_lane < N;
    const int coarse_groups = K / kCoarseK;

    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int cg = 0; cg < coarse_groups; ++cg) {
        const CoarseWeightGroup512* row_group =
            row_valid ? (weights + static_cast<long long>(row_lane) * coarse_groups + cg)
                      : nullptr;
        const CoarseActGroup512* col_group =
            col_valid ? (acts + static_cast<long long>(cg) * N + col_lane)
                      : nullptr;

        const float scale_self = row_group != nullptr ? row_group->scale : 0.0f;
        const float zero_self = row_group != nullptr ? row_group->zero : 0.0f;
        float row_scale[8];
        float row_zero[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src_lane = 2 * j + parity;
            row_scale[j] = __shfl(scale_self, src_lane, kWaveSize);
            row_zero[j] = __shfl(zero_self, src_lane, kWaveSize);
        }

        const float act_scale = col_group != nullptr ? col_group->scale : 0.0f;
        const float act_sum = col_group != nullptr ? col_group->sum : 0.0f;

        int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int local_group = 0; local_group < kCoarseGroups; ++local_group) {
            const int weight_word_base = local_group * 16;
            const int act_word_base = local_group * 32;
            #pragma unroll
            for (int k16 = 0; k16 < 8; ++k16) {
                const uint32_t packed0 =
                    row_group != nullptr
                        ? row_group->payload_words[weight_word_base + 2 * k16 + 0]
                        : 0u;
                const uint32_t packed1 =
                    row_group != nullptr
                        ? row_group->payload_words[weight_word_base + 2 * k16 + 1]
                        : 0u;
                const int32x4_t a_words = unpack_u4x16_to_i32x4(packed0, packed1);
                const int32x4_t b_words =
                    col_group != nullptr
                        ? load_q_words4(col_group->q_words, act_word_base + 4 * k16)
                        : int32x4_t{0, 0, 0, 0};
                cacc = wmma_i8(a_words, b_words, cacc);
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            accum[j] += row_scale[j] * act_scale * static_cast<float>(cacc[j]) +
                        row_zero[j] * act_sum;
        }
    }

    if (!col_valid) {
        return;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int out_row = row_tile + 2 * j + parity;
        if (out_row < M) {
            const long long out_index = static_cast<long long>(col_lane) * M + out_row;
            float value = accum[j];
            if constexpr (AddOutput) {
                value += output[out_index];
            }
            output[out_index] = value;
        }
    }
}

extern "C" __global__ __launch_bounds__(32) void mq4v2_exact_group128_full_set(
    const ExactWeightGroup128* __restrict__ weights,
    const ExactActGroup128* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    exact_group128_body<false>(weights, acts, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(32) void mq4v2_exact_group128_full_add(
    const ExactWeightGroup128* __restrict__ weights,
    const ExactActGroup128* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    exact_group128_body<true>(weights, acts, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(32) void mq4v2_coarse_group512_full_set(
    const CoarseWeightGroup512* __restrict__ weights,
    const CoarseActGroup512* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    coarse_group512_body<false>(weights, acts, output, M, K, N);
}

extern "C" __global__ __launch_bounds__(32) void mq4v2_coarse_group512_full_add(
    const CoarseWeightGroup512* __restrict__ weights,
    const CoarseActGroup512* __restrict__ acts,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    coarse_group512_body<true>(weights, acts, output, M, K, N);
}

}  // namespace mq4v2_probe
