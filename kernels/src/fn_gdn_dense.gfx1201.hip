// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FN dense GDN scan adapter kernels (gfx1201, Qwen3.5 dense geometry:
// 16 q/k heads + 48 v heads, 128 dims, conv channels 10240 = q 2048 | k 2048 |
// v 6144, K=4 convolution with a 3-row history ring, 48 gate heads).  Compiled
// with default flags (no -ffp-contract=off), exactly like tensor_ops, whose
// expressions are copied here; kernels.rs prepends mq_fwht256.h.
//
// fn_gdn_dense_producer = gated_delta_conv_bf16_f32_batched_k4 (+ its params
//   blocks) + the dense adapter's BF16 Q/K l2 norm (serial 128-channel sums,
//   pipe128 row-parameter expression sequence) + F16 packing + chunk-local
//   inclusive cumsum of the log decay, in one launch. The packed F16 q/k/v
//   rows [rows, n64) are written zero, and the scan's EF plane is zeroed. No
//   DPP/swizzle is used by the norm.
// fn_gdn_dense_gate_bf16in / fn_gdn_dense_gate_rotate_bf16in = gated_delta_gate_bf16_f32_batched[/_rotate]
//   reading the chunk scan's packed BF16 output plane directly (the widening
//   hc_state_bf16_to_f32 and the in-place bf16 rounding of the recurrent
//   plane are exact no-ops on those bits).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float bf(float x) {
    return static_cast<float>(static_cast<hip_bfloat16>(x));
}

static __device__ __forceinline__ unsigned short f16bits(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

#define FNGD_CHANNELS 10240
#define FNGD_HEADS 48

// gated_delta_conv_bf16_f32_batched_k4's body for one channel and the tile
// rows [r0, min(r0 + 16, rows)) (r0 < rows); `emit(r, bf16_out)` receives
// each output row.  Only the r0 == 0 tile reads the history and then stores
// next_history (it may alias history), as in the reference kernel.
template <typename Emit>
static __device__ __forceinline__ void fn_gdn_dense_conv_rows(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    int rows,
    int start_cursor,
    int input_bf16,
    int channel,
    int r0,
    Emit emit) {
    const int channels = FNGD_CHANNELS;
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
            b.data = ((const unsigned short*)input)[j * channels + channel];
            return b;
        }
        return bf16_round(input[j * channels + channel]);
    };
    auto x = [&](int j) {
        return j >= 0 ? in(j)
                      : bf16_round(history[((start_cursor + j + 3) % 3) * channels + channel]);
    };
    const float k0 = bf16_to_f32(kernel[channel * 4 + 0]);
    const float k1 = bf16_to_f32(kernel[channel * 4 + 1]);
    const float k2 = bf16_to_f32(kernel[channel * 4 + 2]);
    const float k3 = bf16_to_f32(kernel[channel * 4 + 3]);
    __hip_bfloat16 h0 = x(r0 - 3), h1 = x(r0 - 2), h2 = x(r0 - 1);
    __hip_bfloat16 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    const int r1 = min(r0 + 16, rows);
    for (int r = r0; r < r1; ++r) {
        const __hip_bfloat16 current_bf16 = in(r);
        float value = k3 * bf16_to_f32(current_bf16);
        value += k0 * bf16_to_f32(h0);
        value += k1 * bf16_to_f32(h1);
        value += k2 * bf16_to_f32(h2);
        const __hip_bfloat16 out = bf16_round(value * (1.0f / (1.0f + expf(-value))));
        emit(r, out);
        h0 = h1;
        h1 = h2;
        h2 = current_bf16;
    }
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s)
            next_history[s * channels + channel] = bf16_to_f32(tail[s]);
    }
}

// gated_delta_params_flat's body; the log decay is returned (the cumsum
// consumes it from LDS, i.e. after the same float store the reference makes).
static __device__ __forceinline__ float fn_gdn_dense_params_flat(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* beta,
    int flat) {
    const int head = flat % FNGD_HEADS;
    const float al = bf16_to_f32(a_log[head]);
    const float dt = bf16_to_f32(dt_bias[head]);
    const float value = bf16_to_f32(bf16_round(a[flat])) + dt;
    const float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    const float gate = -expf(al) * softplus;
    beta[flat] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[flat]))));
    return gate;
}

// Grid [82, n64 / 32, 1], block 256, no dynamic LDS (32-row tiles).
//   x < 32  : q|k head x (q heads 0-15, k heads 16-31), 128 channels; thread =
//             (row half, channel), 16 rows each; serial norm on 32 threads.
//   x < 80  : v head x - 32, 128 channels, same thread mapping.
//   x == 80 : params + beta + chunk64 cumsum into g for chunk y / 2 (even y).
//   x == 81 : zero `ef` (ef_words u32 words, a multiple of 4), grid-strided
//             over y.
// input [rows][10240] F32 (input_bf16 == 0) or BF16 bits; q16/k16 [n64][2048]
// and v16 [n64][6144] F16 bits (q scaled by query_scale; pad rows [rows, n64)
// zeroed); g and beta [rows][48] (pad rows are not written).
extern "C" __global__ __launch_bounds__(256) void fn_gdn_dense_producer(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* next_history,
    unsigned short* q16,
    unsigned short* k16,
    unsigned short* v16,
    float* g,
    float* beta,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    unsigned int* ef,
    int rows,
    int n64,
    int start_cursor,
    int input_bf16,
    float query_scale,
    int ef_words) {
    // conv staging [32 rows][129] (q|k) or the chunk's log decay [64][48].
    __shared__ float smem[32 * 129];
    __shared__ float inv[32];
    const int t = (int)threadIdx.x;
    const int x = (int)blockIdx.x;
    const int row0 = (int)blockIdx.y * 32;
    if (x >= 80) {
        if (x == 81) {
            const int stride = (int)gridDim.y * 256;
            uint4* p = reinterpret_cast<uint4*>(ef);
            for (int i = (int)blockIdx.y * 256 + t; i < ef_words / 4; i += stride)
                p[i] = make_uint4(0u, 0u, 0u, 0u);
            return;
        }
        if (blockIdx.y & 1) return;
        const int chunk = (int)blockIdx.y >> 1;
        const int total = rows * FNGD_HEADS;
        for (int i = t; i < 64 * FNGD_HEADS; i += 256) {
            const int flat = chunk * 64 * FNGD_HEADS + i;
            smem[i] = flat < total
                ? fn_gdn_dense_params_flat(param_a, param_b, a_log, dt_bias, beta, flat)
                : 0.0f;
        }
        __syncthreads();
        if (t < FNGD_HEADS) {
            // The dense adapter's chunk-local inclusive cumsum, serial.
            float s = 0.0f;
            for (int r = 0; r < 64; ++r) {
                const int flat = chunk * 64 * FNGD_HEADS + r * FNGD_HEADS + t;
                if (flat < total) {
                    s += smem[r * FNGD_HEADS + t];
                    g[flat] = s;
                }
            }
        }
        return;
    }
    const int c = t & 127;
    const int half = t >> 7;
    const int r0 = row0 + half * 16;
    if (x >= 32) {
        // v: BF16 conv value -> F16.
        const int channel = 4096 + (x - 32) * 128 + c;
        unsigned short* dst = v16 + (x - 32) * 128 + c;
        if (r0 < rows) {
            fn_gdn_dense_conv_rows(input, kernel, history, next_history, rows, start_cursor,
                                   input_bf16, channel, r0,
                                   [&](int r, __hip_bfloat16 out) {
                                       dst[(size_t)r * 6144] = f16bits(bf(bf16_to_f32(out)));
                                   });
        }
        for (int r = r0 > rows ? r0 : rows; r < r0 + 16; ++r) dst[(size_t)r * 6144] = 0;
        return;
    }
    // q|k head x.
    const int channel = x * 128 + c;
    if (r0 < rows) {
        fn_gdn_dense_conv_rows(input, kernel, history, next_history, rows, start_cursor,
                               input_bf16, channel, r0,
                               [&](int r, __hip_bfloat16 out) {
                                   smem[(r - row0) * 129 + c] = bf16_to_f32(out);
                               });
    }
    __syncthreads();
    if (t < 32) {
        // pipe128 row params: serial sum of BF16-rounded squares, BF16 rsqrt.
        const float* h = smem + t * 129;
        float s = 0.0f;
        if (row0 + t < rows) {
            for (int cc = 0; cc < 128; ++cc) {
                const float xv = bf(h[cc]);
                s += bf(xv * xv);
            }
        }
        inv[t] = bf(rsqrtf(bf(s) + bf(1.0e-6f)));
    }
    __syncthreads();
    unsigned short* dst = x < 16 ? q16 + (size_t)x * 128 + c : k16 + (size_t)(x - 16) * 128 + c;
    for (int r = r0; r < r0 + 16; ++r) {
        unsigned short bits = 0;
        if (r < rows) {
            const float n = bf(bf(smem[(r - row0) * 129 + c]) * inv[r - row0]);
            bits = f16bits(x < 16 ? n * query_scale : n);
        }
        dst[(size_t)r * 2048] = bits;
    }
}

// gated_delta_gate_head128 with `recurrent` the packed BF16 bits plane.
static __device__ __forceinline__ float4 fn_gdn_dense_gate_head128(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int start,
    int lane,
    float* rounded) {
    const uint2 raw = *reinterpret_cast<const uint2*>(recurrent + start + lane * 4);
    const float4 value = make_float4(
        __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
        __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
    *reinterpret_cast<float4*>(rounded + lane * 4) = value;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 128; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(rounded + i);
        sum += v.x * v.x;
        sum += v.y * v.y;
        sum += v.z * v.z;
        sum += v.w * v.w;
    }
    const float inv = rsqrtf(sum / 128.0f + 1.0e-6f);
    const float4 zv = *reinterpret_cast<const float4*>(z + start + lane * 4);
    const float in[4] = {value.x, value.y, value.z, value.w};
    const float zs[4] = {zv.x, zv.y, zv.z, zv.w};
    float out[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int channel = lane * 4 + e;
        const float normalized = bf16_to_f32(bf16_round(
            in[e] * inv * bf16_to_f32(norm[channel])));
        const float gate = bf16_to_f32(bf16_round(zs[e]));
        out[e] = bf16_to_f32(bf16_round(normalized / (1.0f + expf(-gate))));
    }
    const float4 result = make_float4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<float4*>(output + start + lane * 4) = result;
    return result;
}

// Grid [rows * 48], block 32.
extern "C" __global__ __launch_bounds__(32) void fn_gdn_dense_gate_bf16in(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int rows,
    int value_heads,
    int value_dim) {
    if ((int)blockIdx.x >= rows * value_heads || value_dim != 128) return;
    __shared__ float rounded[128];
    fn_gdn_dense_gate_head128(recurrent, z, norm, output, (int)blockIdx.x * 128,
                              (int)threadIdx.x, rounded);
}

// Grid [rows * 48 / 2], block 64: the head pair's 256 outputs form one
// rotation group, which wave 0 rotates from LDS.
extern "C" __global__ __launch_bounds__(64) void fn_gdn_dense_gate_rotate_bf16in(
    const unsigned short* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    float* rotated,
    const float* signs1,
    const float* signs2) {
    const int wave = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    __shared__ float rounded[2][128];
    __shared__ float pair[256];
    const float4 result = fn_gdn_dense_gate_head128(
        recurrent, z, norm, output, ((int)blockIdx.x * 2 + wave) * 128, lane, rounded[wave]);
    *reinterpret_cast<float4*>(pair + wave * 128 + lane * 4) = result;
    __syncthreads();
    if (wave != 0) return;
    const int d0 = lane * 8;
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) values[e] = pair[d0 + e] * signs1[d0 + e];
    mq_fwht256_lane(values, lane);
#pragma unroll
    for (int e = 0; e < 8; ++e)
        rotated[(int)blockIdx.x * 256 + d0 + e] = values[e] * 0.0625f * signs2[d0 + e];
}
