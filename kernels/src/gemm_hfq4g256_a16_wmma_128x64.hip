// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Standalone gfx11 large-M probe. Each block computes 128 activation rows by
// 64 output columns. Four waves decode one 16K x 64N HFQ4 tile into LDS while
// all eight waves reuse it for independent 16-row activation slices.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(16))) float16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

template <bool ACTIVATION_F32>
static __device__ __forceinline__ half16_t load_activation_16(
    const void* __restrict__ activations, long long offset) {
    if constexpr (ACTIVATION_F32) {
        const float16_t values =
            *reinterpret_cast<const float16_t*>(
                static_cast<const float*>(activations) + offset);
        return __builtin_convertvector(values, half16_t);
    } else {
        return *reinterpret_cast<const half16_t*>(
            static_cast<const _Float16*>(activations) + offset);
    }
}

static __device__ __forceinline__ half16_t load_hfq4_16(
    const char* __restrict__ group, int packed_offset, _Float16 scale,
    _Float16 zero) {
    const unsigned int p0 =
        *reinterpret_cast<const unsigned int*>(group + 8 + packed_offset);
    const unsigned int p1 =
        *reinterpret_cast<const unsigned int*>(group + 12 + packed_offset);
    half16_t out;
#define DQ(I, P, S) \
    out[I] = scale * (_Float16)(float)(((P) >> (S)) & 0xFu) + zero
    DQ(0, p0, 0);
    DQ(1, p0, 4);
    DQ(2, p0, 8);
    DQ(3, p0, 12);
    DQ(4, p0, 16);
    DQ(5, p0, 20);
    DQ(6, p0, 24);
    DQ(7, p0, 28);
    DQ(8, p1, 0);
    DQ(9, p1, 4);
    DQ(10, p1, 8);
    DQ(11, p1, 12);
    DQ(12, p1, 16);
    DQ(13, p1, 20);
    DQ(14, p1, 24);
    DQ(15, p1, 28);
#undef DQ
    return out;
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__)
__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_a16_wmma_128x64_set(const char* __restrict__ weights,
                                  const _Float16* __restrict__ activations,
                                  float* __restrict__ output, int output_dim,
                                  int k_dim, int tokens) {
    const int output_tile = blockIdx.x * 64;
    const int token_tile = blockIdx.y * 128;
    if (output_tile >= output_dim || token_tile >= tokens) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int groups_per_row = k_dim / 256;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // N-major/K-contiguous layout enables vector loads of a full WMMA B
    // fragment after cooperative HFQ4 decode.
    __shared__ _Float16 weight_lds[2][64][16];

    const int decoded_output = wave * 16 + lane_lo;
    const int actual_output = output_tile + decoded_output;
    const bool decode_lane = wave < 4 && actual_output < output_dim;
    int cached_group = -1;
    _Float16 cached_scale = 0;
    _Float16 cached_zero = 0;

    auto decode_into = [&](int buffer, int k_tile) __attribute__((always_inline)) {
        if (!decode_lane) return;
        const int group_index = k_tile >> 8;
        const char* group = weights +
            ((long long)actual_output * groups_per_row + group_index) * 136;
        if (group_index != cached_group) {
            cached_group = group_index;
            cached_scale = (_Float16)__builtin_bit_cast(
                float, *reinterpret_cast<const unsigned int*>(group));
            cached_zero = (_Float16)__builtin_bit_cast(
                float, *reinterpret_cast<const unsigned int*>(group + 4));
        }
        const int packed_offset = (k_tile & 255) >> 1;
        const half16_t decoded =
            load_hfq4_16(group, packed_offset, cached_scale, cached_zero);
        *reinterpret_cast<half16_t*>(&weight_lds[buffer][decoded_output][0]) =
            decoded;
    };

    decode_into(0, 0);
    __syncthreads();

    const int token_row = token_tile + wave * 16 + lane_lo;
    const _Float16* activation_row = token_row < tokens
        ? activations + (long long)token_row * k_dim
        : nullptr;
    int current = 0;
    for (int k_tile = 0; k_tile < k_dim; k_tile += 16) {
        const int next = current ^ 1;
        if (k_tile + 16 < k_dim) decode_into(next, k_tile + 16);

        half16_t a_frag;
        if (activation_row) {
            a_frag = *reinterpret_cast<const half16_t*>(activation_row + k_tile);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }
        const half16_t b0 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 0][0]);
        const half16_t b1 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 16][0]);
        const half16_t b2 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 32][0]);
        const half16_t b3 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 48][0]);

        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b1, acc1);
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b3, acc3);

        __syncthreads();
        current = next;
    }

    const int token_base = token_tile + wave * 16;
    auto store = [&](const float8_t& acc, int output_base) {
        const int out_col = output_tile + output_base + lane_lo;
        if (out_col >= output_dim) return;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_token = token_base + 2 * i + lane_hi;
            if (out_token < tokens) {
                output[(long long)out_token * output_dim + out_col] = acc[i];
            }
        }
    };
    store(acc0, 0);
    store(acc1, 16);
    store(acc2, 32);
    store(acc3, 48);
}

// K32 sibling: all eight waves decode one half of a 32K x 64N tile. The
// padded K stride avoids the power-of-two LDS row stride used by the K16
// probe and halves the number of block barriers.
template <bool ADD_RESIDUAL, bool ACTIVATION_F32 = false>
static __device__ __forceinline__ void
gemm_hfq4g256_a16_wmma_128x64_k32_impl(
    const char* __restrict__ weights,
    const void* __restrict__ activations, float* __restrict__ output,
    int output_dim, int k_dim, int tokens) {
    const int output_tile = blockIdx.x * 64;
    const int token_tile = blockIdx.y * 128;
    if (output_tile >= output_dim || token_tile >= tokens) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int groups_per_row = k_dim / 256;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    __shared__ _Float16 weight_lds[2][64][34];

    const int decoded_output = (wave & 3) * 16 + lane_lo;
    const int decode_half = wave >> 2;
    const int actual_output = output_tile + decoded_output;
    const bool decode_lane = actual_output < output_dim;
    int cached_group = -1;
    _Float16 cached_scale = 0;
    _Float16 cached_zero = 0;

    auto decode_into = [&](int buffer, int k_tile) __attribute__((always_inline)) {
        if (!decode_lane) return;
        const int group_index = k_tile >> 8;
        const char* group = weights +
            ((long long)actual_output * groups_per_row + group_index) * 136;
        if (group_index != cached_group) {
            cached_group = group_index;
            cached_scale = (_Float16)__builtin_bit_cast(
                float, *reinterpret_cast<const unsigned int*>(group));
            cached_zero = (_Float16)__builtin_bit_cast(
                float, *reinterpret_cast<const unsigned int*>(group + 4));
        }
        const int k_half = decode_half * 16;
        const int packed_offset = ((k_tile + k_half) & 255) >> 1;
        const half16_t decoded =
            load_hfq4_16(group, packed_offset, cached_scale, cached_zero);
        *reinterpret_cast<half16_t*>(
            &weight_lds[buffer][decoded_output][k_half]) = decoded;
    };

    decode_into(0, 0);
    __syncthreads();

    const int token_row = token_tile + wave * 16 + lane_lo;
    const long long activation_offset = (long long)token_row * k_dim;
    const bool valid_token = token_row < tokens;
    int current = 0;
    for (int k_tile = 0; k_tile < k_dim; k_tile += 32) {
        const int next = current ^ 1;
        if (k_tile + 32 < k_dim) decode_into(next, k_tile + 32);

        half16_t a_lo;
        half16_t a_hi;
        if (valid_token) {
            a_lo = load_activation_16<ACTIVATION_F32>(
                activations, activation_offset + k_tile);
            a_hi = load_activation_16<ACTIVATION_F32>(
                activations, activation_offset + k_tile + 16);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_lo[i] = 0;
                a_hi[i] = 0;
            }
        }

        half16_t b0 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 0][0]);
        half16_t b1 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 16][0]);
        half16_t b2 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 32][0]);
        half16_t b3 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 48][0]);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b1, acc1);
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b3, acc3);

        b0 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 0][16]);
        b1 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 16][16]);
        b2 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 32][16]);
        b3 = *reinterpret_cast<const half16_t*>(
            &weight_lds[current][lane_lo + 48][16]);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b0, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b1, acc1);
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b2, acc2);
        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b3, acc3);

        __syncthreads();
        current = next;
    }

    const int token_base = token_tile + wave * 16;
    auto store = [&](const float8_t& acc, int output_base) {
        const int out_col = output_tile + output_base + lane_lo;
        if (out_col >= output_dim) return;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_token = token_base + 2 * i + lane_hi;
            if (out_token < tokens) {
                float* dst = output + (long long)out_token * output_dim + out_col;
                if constexpr (ADD_RESIDUAL) {
                    *dst += acc[i];
                } else {
                    *dst = acc[i];
                }
            }
        }
    };
    store(acc0, 0);
    store(acc1, 16);
    store(acc2, 32);
    store(acc3, 48);
}

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_a16_wmma_128x64_k32_set(
    const char* __restrict__ weights,
    const _Float16* __restrict__ activations, float* __restrict__ output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_a16_wmma_128x64_k32_impl<false>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_a16_wmma_128x64_k32_add(
    const char* __restrict__ weights,
    const _Float16* __restrict__ activations, float* __restrict__ output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_a16_wmma_128x64_k32_impl<true>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_f32a_wmma_128x64_k32_set(
    const char* __restrict__ weights,
    const float* __restrict__ activations, float* __restrict__ output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_a16_wmma_128x64_k32_impl<false, true>(
        weights, activations, output, output_dim, k_dim, tokens);
}

__launch_bounds__(256, 1) extern "C" __global__ void
gemm_hfq4g256_f32a_wmma_128x64_k32_add(
    const char* __restrict__ weights,
    const float* __restrict__ activations, float* __restrict__ output,
    int output_dim, int k_dim, int tokens) {
    gemm_hfq4g256_a16_wmma_128x64_k32_impl<true, true>(
        weights, activations, output, output_dim, k_dim, tokens);
}
#else
extern "C" __global__ void gemm_hfq4g256_a16_wmma_128x64_set(
    const char*, const _Float16*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_a16_wmma_128x64_k32_set(
    const char*, const _Float16*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_a16_wmma_128x64_k32_add(
    const char*, const _Float16*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_f32a_wmma_128x64_k32_set(
    const char*, const float*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_f32a_wmma_128x64_k32_add(
    const char*, const float*, float*, int, int, int) {}
#endif
