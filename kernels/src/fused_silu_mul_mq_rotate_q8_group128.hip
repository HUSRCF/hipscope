// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Fused SwiGLU + FWHT + group128 Q8 packing for the gfx11 HFQ4 MMQ path.
// Each wave owns one 256-value FWHT group and emits two k-major
// block_q8_1_mmq records. This avoids materializing the rotated F32 hidden
// activation before the down projection.

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

template <typename InputT>
__device__ __forceinline__ void fused_silu_mul_mq_rotate_q8_group128_body(
    const InputT* __restrict__ gate,
    const InputT* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_q8_1_mmq* __restrict__ out_q8,
    int K,
    int N
) {
    const int group = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    if (token >= N || group >= K / 256) return;

    const long long batch_off = (long long)token * K;
    const int d0 = tid * 8;
    const int base = group * 256 + d0;

    #define SILU_MUL_AT(i) ({ \
        const float _g = (float)gate[batch_off + base + (i)]; \
        const float _u = (float)up[batch_off + base + (i)]; \
        (_g / (1.0f + expf(-_g)) * _u) * signs1[d0 + (i)]; \
    })
    float v0 = SILU_MUL_AT(0);
    float v1 = SILU_MUL_AT(1);
    float v2 = SILU_MUL_AT(2);
    float v3 = SILU_MUL_AT(3);
    float v4 = SILU_MUL_AT(4);
    float v5 = SILU_MUL_AT(5);
    float v6 = SILU_MUL_AT(6);
    float v7 = SILU_MUL_AT(7);
    #undef SILU_MUL_AT

    float t;
    t = v0; v0 += v1; v1 = t - v1;
    t = v2; v2 += v3; v3 = t - v3;
    t = v4; v4 += v5; v5 = t - v5;
    t = v6; v6 += v7; v7 = t - v7;
    t = v0; v0 += v2; v2 = t - v2;
    t = v1; v1 += v3; v3 = t - v3;
    t = v4; v4 += v6; v6 = t - v6;
    t = v5; v5 += v7; v7 = t - v7;
    t = v0; v0 += v4; v4 = t - v4;
    t = v1; v1 += v5; v5 = t - v5;
    t = v2; v2 += v6; v6 = t - v6;
    t = v3; v3 += v7; v7 = t - v7;

    #define HBFLY(v, pat, bit) do { \
        const float _p = __int_as_float( \
            __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        (v) = (tid & (bit)) ? _p - (v) : (v) + _p; \
    } while (0)
    #define HBFLY8(pat, bit) \
        HBFLY(v0, pat, bit); HBFLY(v1, pat, bit); \
        HBFLY(v2, pat, bit); HBFLY(v3, pat, bit); \
        HBFLY(v4, pat, bit); HBFLY(v5, pat, bit); \
        HBFLY(v6, pat, bit); HBFLY(v7, pat, bit)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float fwht_scale = 0.0625f;
    v0 *= fwht_scale * signs2[d0];
    v1 *= fwht_scale * signs2[d0 + 1];
    v2 *= fwht_scale * signs2[d0 + 2];
    v3 *= fwht_scale * signs2[d0 + 3];
    v4 *= fwht_scale * signs2[d0 + 4];
    v5 *= fwht_scale * signs2[d0 + 5];
    v6 *= fwht_scale * signs2[d0 + 6];
    v7 *= fwht_scale * signs2[d0 + 7];

    float amax = fmaxf(
        fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
        fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    for (int offset = 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 16));
    }

    float sum = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    for (int offset = 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 4);
    }

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    #define Q8(v) ((int8_t)lrintf(fmaxf(-127.0f, fminf(127.0f, (v) * id))))
    const char4 q0 = make_char4(Q8(v0), Q8(v1), Q8(v2), Q8(v3));
    const char4 q1 = make_char4(Q8(v4), Q8(v5), Q8(v6), Q8(v7));
    #undef Q8

    const int half = tid >> 4;
    const int lane16 = tid & 15;
    block_q8_1_mmq* out = out_q8 + (long long)(2 * group + half) * N + token;
    ((char4*)out->qs)[2 * lane16] = q0;
    ((char4*)out->qs)[2 * lane16 + 1] = q1;
    if ((lane16 & 3) == 0) {
        out->ds4[lane16 >> 2] = make_half2((_Float16)d, (_Float16)sum);
    }
}

extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_q8_group128(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_q8_1_mmq* __restrict__ out_q8,
    int K,
    int N
) {
    fused_silu_mul_mq_rotate_q8_group128_body(
        gate, up, signs1, signs2, out_q8, K, N);
}

extern "C" __launch_bounds__(32, 16)
__global__ void fused_silu_mul_mq_rotate_q8_group128_f16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_q8_1_mmq* __restrict__ out_q8,
    int K,
    int N
) {
    fused_silu_mul_mq_rotate_q8_group128_body(
        gate, up, signs1, signs2, out_q8, K, N);
}
