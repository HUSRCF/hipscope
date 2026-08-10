// SPDX-License-Identifier: MIT
// Standalone gfx11 packed-MQ4 x signed-A4 WMMA experiment.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x8_t = int __attribute__((ext_vector_type(8)));

static constexpr int TILE_X = 256;
static constexpr int TILE_Y = 64;
static constexpr int NWARPS = 8;
static constexpr int COL_GROUPS = 4;
static constexpr int ROW_FRAGS = 2;
static constexpr int HFQ4_GROUP_BYTES = 136;

struct block_q4_1_group128 {
    half2 d_sum;
    uint32_t pad;
    uint8_t qs[64];
};

static_assert(sizeof(block_q4_1_group128) == 72, "bad q4 activation block");

struct block_q4_1_group32 {
    half d[4];
    half sum;
    uint16_t pad;
    uint8_t qs[64];
};

static_assert(sizeof(block_q4_1_group32) == 76, "bad q4 group32 activation block");

// Exact Q8 activation represented as two packed 4-bit planes:
// q8 = low_u4 + 16 * high_i4. The metadata matches block_q8_1_mmq so the
// quantization scale and four 32-value affine sums remain unchanged.
struct block_q8_1_iu4_planes {
    half2 ds4[4];
    uint8_t low[64];
    uint8_t high[64];
};

static_assert(sizeof(block_q8_1_iu4_planes) == 144,
              "bad exact Q8 IU4-plane activation block");

extern "C" __global__ void quantize_q8_1_group128_iu4_planes(
    const float* __restrict__ x,
    block_q8_1_iu4_planes* __restrict__ y,
    int k,
    int n
) {
    const int token = blockIdx.y;
    const int kblock = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= n || lane >= 32 || 128 * kblock >= k) return;

    const int k0 = 128 * kblock + 4 * lane;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k0 + 3 < k) v = *(const float4*)(x + (long long)token * k + k0);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                       fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float sum = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
    const int q0 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v.x * inv_d)));
    const int q1 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v.y * inv_d)));
    const int q2 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v.z * inv_d)));
    const int q3 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v.w * inv_d)));

    block_q8_1_iu4_planes* out = y + (long long)kblock * n + token;
    out->low[2 * lane] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
    out->low[2 * lane + 1] = (uint8_t)((q2 & 15) | ((q3 & 15) << 4));
    out->high[2 * lane] =
        (uint8_t)(((q0 >> 4) & 15) | (((q1 >> 4) & 15) << 4));
    out->high[2 * lane + 1] =
        (uint8_t)(((q2 >> 4) & 15) | (((q3 >> 4) & 15) << 4));
    if ((lane & 7) == 0) {
        out->ds4[lane >> 3] = make_half2((_Float16)d, (_Float16)sum);
    }
}

extern "C" __global__ void quantize_q4_1_group128(
    const float* __restrict__ x,
    block_q4_1_group128* __restrict__ y,
    int k,
    int n
) {
    const int token = blockIdx.y;
    const int kblock = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= n || lane >= 32 || 128 * kblock >= k) return;

    const int k0 = 128 * kblock + 4 * lane;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k0 + 3 < k) v = *(const float4*)(x + (long long)token * k + k0);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                       fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 7.0f : 0.0f;
    const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
    const int q0 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.x * inv_d)));
    const int q1 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.y * inv_d)));
    const int q2 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.z * inv_d)));
    const int q3 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.w * inv_d)));
    block_q4_1_group128* out = y + (long long)kblock * n + token;
    out->qs[2 * lane] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
    out->qs[2 * lane + 1] = (uint8_t)((q2 & 15) | ((q3 & 15) << 4));
    if (lane == 0) {
        out->d_sum = make_half2((_Float16)d, (_Float16)sum);
        out->pad = 0;
    }
}

extern "C" __global__ void quantize_q4_1_group32(
    const float* __restrict__ x,
    block_q4_1_group32* __restrict__ y,
    int k,
    int n
) {
    const int token = blockIdx.y;
    const int kblock = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= n || lane >= 32 || 128 * kblock >= k) return;

    const int k0 = 128 * kblock + 4 * lane;
    const float4 v = *(const float4*)(x + (long long)token * k + k0);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                       fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 7.0f : 0.0f;
    const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
    const int q0 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.x * inv_d)));
    const int q1 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.y * inv_d)));
    const int q2 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.z * inv_d)));
    const int q3 = (int)lrintf(fmaxf(-7.0f, fminf(7.0f, v.w * inv_d)));
    block_q4_1_group32* out = y + (long long)kblock * n + token;
    out->qs[2 * lane] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
    out->qs[2 * lane + 1] = (uint8_t)((q2 & 15) | ((q3 & 15) << 4));
    if ((lane & 7) == 0) out->d[lane >> 3] = (_Float16)d;
    if (lane == 0) {
        out->sum = (_Float16)sum;
        out->pad = 0;
    }
}

static __device__ __forceinline__ int32x2_t load_q4_fragment(
    const uint8_t* __restrict__ ptr
) {
    return *(const int32x2_t*)ptr;
}

static __device__ __forceinline__ int c_row(int lane, int l) {
    return 2 * l + (lane >> 4);
}

template <bool SIGNED_ACTIVATION>
static __device__ __forceinline__ int32x8_t mma_iu4_plane(
    int32x2_t a,
    int32x2_t b,
    int32x8_t c
) {
    if constexpr (SIGNED_ACTIVATION) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
            false, a, true, b, c, false);
    } else {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
            false, a, false, b, c, false);
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_body_q8_iu4_exact(
    const char* __restrict__ weights,
    const block_q8_1_iu4_planes* __restrict__ activations,
    float* __restrict__ output,
    int m,
    int k,
    int n
) {
    const int row0 = blockIdx.x * TILE_Y;
    const int col0 = blockIdx.y * TILE_X;
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int lane16 = lane & 15;
    const int row_base = (wave / COL_GROUPS) * ROW_FRAGS * 16;
    const int col_lane_base = (wave % COL_GROUPS) * 16;
    const int groups = k / 256;
    const int tid = wave * 32 + lane;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* weight_tile = smem;
    block_q8_1_iu4_planes* activation_tile =
        (block_q8_1_iu4_planes*)(weight_tile + TILE_Y * HFQ4_GROUP_BYTES);
    float accum[TILE_X * TILE_Y / (NWARPS * 32)] = {0.0f};

    for (int group = 0; group < groups; ++group) {
        for (int row = wave; row < TILE_Y; row += NWARPS) {
            const char* src = weights +
                ((long long)(row0 + row) * groups + group) * HFQ4_GROUP_BYTES;
            int* dst_q = (int*)(weight_tile + row * HFQ4_GROUP_BYTES + 8);
            dst_q[lane] = *(const int*)(src + 8 + 4 * lane);
            if (lane == 0) {
                const float2 meta = *(const float2*)src;
                *(half2*)(weight_tile + row * HFQ4_GROUP_BYTES) =
                    make_half2((_Float16)meta.x, (_Float16)meta.y);
                *(uint32_t*)(weight_tile + row * HFQ4_GROUP_BYTES + 4) = 0;
            }
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const block_q8_1_iu4_planes* src =
                activations + (long long)(2 * group + half) * n + col0;
            const int activation_ints =
                TILE_X * (int)sizeof(block_q8_1_iu4_planes) / 4;
            for (int idx = tid; idx < activation_ints; idx += NWARPS * 32) {
                ((int*)activation_tile)[idx] = ((const int*)src)[idx];
            }
            __syncthreads();

#pragma unroll
            for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
                const int token = j0 + col_lane_base + lane16;
                const float2 activation_meta0 =
                    __half22float2(activation_tile[token].ds4[0]);

                {
#pragma unroll
                    for (int rf = 0; rf < ROW_FRAGS; ++rf) {
                        int32x8_t c = {};
#pragma unroll
                        for (int kk = 0; kk < 128; kk += 16) {
                            const int32x2_t b = load_q4_fragment(
                                activation_tile[token].low + kk / 2);
                            const int input_row = row_base + rf * 16 + lane16;
                            const uint8_t* wq = weight_tile +
                                input_row * HFQ4_GROUP_BYTES + 8 + half * 64 + kk / 2;
                            c = mma_iu4_plane<false>(load_q4_fragment(wq), b, c);
                        }
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            const int local_row = row_base + rf * 16 + c_row(lane, l);
                            const float weight_scale = __half2float(*(const __half*)(
                                weight_tile + local_row * HFQ4_GROUP_BYTES));
                            const int acc_group = j0 / (COL_GROUPS * 16);
                            const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                            accum[acc_index] += weight_scale * activation_meta0.x *
                                (float)c[l];
                        }
                    }
                }

                {
#pragma unroll
                    for (int rf = 0; rf < ROW_FRAGS; ++rf) {
                        int32x8_t c = {};
#pragma unroll
                        for (int kk = 0; kk < 128; kk += 16) {
                            const int32x2_t b = load_q4_fragment(
                                activation_tile[token].high + kk / 2);
                            const int input_row = row_base + rf * 16 + lane16;
                            const uint8_t* wq = weight_tile +
                                input_row * HFQ4_GROUP_BYTES + 8 + half * 64 + kk / 2;
                            c = mma_iu4_plane<true>(load_q4_fragment(wq), b, c);
                        }
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            const int local_row = row_base + rf * 16 + c_row(lane, l);
                            const float weight_scale = __half2float(*(const __half*)(
                                weight_tile + local_row * HFQ4_GROUP_BYTES));
                            const int acc_group = j0 / (COL_GROUPS * 16);
                            const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                            accum[acc_index] += 16.0f * weight_scale * activation_meta0.x *
                                (float)c[l];
                        }
                    }
                }

                float activation_sum = 0.0f;
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    activation_sum += __half22float2(activation_tile[token].ds4[sub]).y;
                }
#pragma unroll
                for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int local_row = row_base + rf * 16 + c_row(lane, l);
                        const float weight_zero = __half2float(*(const __half*)(
                            weight_tile + local_row * HFQ4_GROUP_BYTES + 2));
                        const int acc_group = j0 / (COL_GROUPS * 16);
                        const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                        accum[acc_index] += weight_zero * activation_sum;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
        const int token = j0 + col_lane_base + lane16;
#pragma unroll
        for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int local_row = row_base + rf * 16 + c_row(lane, l);
                const int acc_group = j0 / (COL_GROUPS * 16);
                const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                float* dst = output +
                    (long long)(col0 + token) * m + row0 + local_row;
                if constexpr (ADD) {
                    *dst += accum[acc_index];
                } else {
                    *dst = accum[acc_index];
                }
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_body(
    const char* __restrict__ weights,
    const block_q4_1_group128* __restrict__ activations,
    float* __restrict__ output,
    int m,
    int k,
    int n
) {
    const int row0 = blockIdx.x * TILE_Y;
    const int col0 = blockIdx.y * TILE_X;
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int lane16 = lane & 15;
    const int row_base = (wave / COL_GROUPS) * ROW_FRAGS * 16;
    const int col_lane_base = (wave % COL_GROUPS) * 16;
    const int groups = k / 256;
    const int tid = wave * 32 + lane;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* weight_tile = smem;
    block_q4_1_group128* activation_tile =
        (block_q4_1_group128*)(weight_tile + TILE_Y * HFQ4_GROUP_BYTES);
    float accum[TILE_X * TILE_Y / (NWARPS * 32)] = {0.0f};

    for (int group = 0; group < groups; ++group) {
        for (int row = wave; row < TILE_Y; row += NWARPS) {
            const char* src = weights +
                ((long long)(row0 + row) * groups + group) * HFQ4_GROUP_BYTES;
            int* dst_q = (int*)(weight_tile + row * HFQ4_GROUP_BYTES + 8);
            dst_q[lane] = *(const int*)(src + 8 + 4 * lane);
            if (lane == 0) {
                const float2 meta = *(const float2*)src;
                *(half2*)(weight_tile + row * HFQ4_GROUP_BYTES) =
                    make_half2((_Float16)meta.x, (_Float16)meta.y);
                *(uint32_t*)(weight_tile + row * HFQ4_GROUP_BYTES + 4) = 0;
            }
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const block_q4_1_group128* src =
                activations + (long long)(2 * group + half) * n + col0;
            const int activation_ints = TILE_X * (int)sizeof(block_q4_1_group128) / 4;
            for (int idx = tid; idx < activation_ints; idx += NWARPS * 32) {
                ((int*)activation_tile)[idx] = ((const int*)src)[idx];
            }
            __syncthreads();

#pragma unroll
            for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
                int32x8_t c[ROW_FRAGS] = {};
#pragma unroll
                for (int kk = 0; kk < 128; kk += 16) {
                    const int token = j0 + col_lane_base + lane16;
                    const int32x2_t b = load_q4_fragment(
                        activation_tile[token].qs + kk / 2);
#pragma unroll
                    for (int rf = 0; rf < ROW_FRAGS; ++rf) {
                        const int input_row = row_base + rf * 16 + lane16;
                        const uint8_t* wq = weight_tile +
                            input_row * HFQ4_GROUP_BYTES + 8 + half * 64 + kk / 2;
                        const int32x2_t a = load_q4_fragment(wq);
                        c[rf] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                            false, a, true, b, c[rf], false);
                    }
                }

                const int token = j0 + col_lane_base + lane16;
                const float2 activation_meta =
                    __half22float2(activation_tile[token].d_sum);
#pragma unroll
                for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int local_row = row_base + rf * 16 + c_row(lane, l);
                        const float2 weight_meta = __half22float2(*(const half2*)(
                            weight_tile + local_row * HFQ4_GROUP_BYTES));
                        const int acc_group = j0 / (COL_GROUPS * 16);
                        const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                        accum[acc_index] += weight_meta.x * activation_meta.x *
                            (float)c[rf][l] + weight_meta.y * activation_meta.y;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
        const int token = j0 + col_lane_base + lane16;
#pragma unroll
        for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int local_row = row_base + rf * 16 + c_row(lane, l);
                const int acc_group = j0 / (COL_GROUPS * 16);
                const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                float* dst = output +
                    (long long)(col0 + token) * m + row0 + local_row;
                if constexpr (ADD) {
                    *dst += accum[acc_index];
                } else {
                    *dst = accum[acc_index];
                }
            }
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_body_group32(
    const char* __restrict__ weights,
    const block_q4_1_group32* __restrict__ activations,
    float* __restrict__ output,
    int m,
    int k,
    int n
) {
    const int row0 = blockIdx.x * TILE_Y;
    const int col0 = blockIdx.y * TILE_X;
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int lane16 = lane & 15;
    const int row_base = (wave / COL_GROUPS) * ROW_FRAGS * 16;
    const int col_lane_base = (wave % COL_GROUPS) * 16;
    const int groups = k / 256;
    const int tid = wave * 32 + lane;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* weight_tile = smem;
    block_q4_1_group32* activation_tile =
        (block_q4_1_group32*)(weight_tile + TILE_Y * HFQ4_GROUP_BYTES);
    float accum[TILE_X * TILE_Y / (NWARPS * 32)] = {0.0f};

    for (int group = 0; group < groups; ++group) {
        for (int row = wave; row < TILE_Y; row += NWARPS) {
            const char* src = weights +
                ((long long)(row0 + row) * groups + group) * HFQ4_GROUP_BYTES;
            int* dst_q = (int*)(weight_tile + row * HFQ4_GROUP_BYTES + 8);
            dst_q[lane] = *(const int*)(src + 8 + 4 * lane);
            if (lane == 0) {
                const float2 meta = *(const float2*)src;
                *(half2*)(weight_tile + row * HFQ4_GROUP_BYTES) =
                    make_half2((_Float16)meta.x, (_Float16)meta.y);
                *(uint32_t*)(weight_tile + row * HFQ4_GROUP_BYTES + 4) = 0;
            }
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const block_q4_1_group32* src =
                activations + (long long)(2 * group + half) * n + col0;
            const int activation_ints = TILE_X * (int)sizeof(block_q4_1_group32) / 4;
            for (int idx = tid; idx < activation_ints; idx += NWARPS * 32) {
                ((int*)activation_tile)[idx] = ((const int*)src)[idx];
            }
            __syncthreads();

#pragma unroll
            for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
                const int token = j0 + col_lane_base + lane16;
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    int32x8_t c[ROW_FRAGS] = {};
#pragma unroll
                    for (int kk = 0; kk < 32; kk += 16) {
                        const int ksub = sub * 32 + kk;
                        const int32x2_t b = load_q4_fragment(
                            activation_tile[token].qs + ksub / 2);
#pragma unroll
                        for (int rf = 0; rf < ROW_FRAGS; ++rf) {
                            const int input_row = row_base + rf * 16 + lane16;
                            const uint8_t* wq = weight_tile +
                                input_row * HFQ4_GROUP_BYTES + 8 + half * 64 + ksub / 2;
                            const int32x2_t a = load_q4_fragment(wq);
                            c[rf] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                                false, a, true, b, c[rf], false);
                        }
                    }

                    const float activation_scale =
                        __half2float(activation_tile[token].d[sub]);
#pragma unroll
                    for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            const int local_row = row_base + rf * 16 + c_row(lane, l);
                            const float weight_scale = __half2float(*(const __half*)(
                                weight_tile + local_row * HFQ4_GROUP_BYTES));
                            const int acc_group = j0 / (COL_GROUPS * 16);
                            const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                            accum[acc_index] +=
                                weight_scale * activation_scale * (float)c[rf][l];
                        }
                    }
                }

                const float activation_sum = __half2float(activation_tile[token].sum);
#pragma unroll
                for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int local_row = row_base + rf * 16 + c_row(lane, l);
                        const float weight_zero = __half2float(*(const __half*)(
                            weight_tile + local_row * HFQ4_GROUP_BYTES + 2));
                        const int acc_group = j0 / (COL_GROUPS * 16);
                        const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                        accum[acc_index] += weight_zero * activation_sum;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < TILE_X; j0 += COL_GROUPS * 16) {
        const int token = j0 + col_lane_base + lane16;
#pragma unroll
        for (int rf = 0; rf < ROW_FRAGS; ++rf) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int local_row = row_base + rf * 16 + c_row(lane, l);
                const int acc_group = j0 / (COL_GROUPS * 16);
                const int acc_index = (acc_group * ROW_FRAGS + rf) * 8 + l;
                float* dst = output +
                    (long long)(col0 + token) * m + row0 + local_row;
                if constexpr (ADD) {
                    *dst += accum[acc_index];
                } else {
                    *dst = accum[acc_index];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_a4_set(
    const char* weights,
    const block_q4_1_group128* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body<false>(weights, activations, output, m, k, n);
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_a4_add(
    const char* weights,
    const block_q4_1_group128* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body<true>(weights, activations, output, m, k, n);
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_a4_group32_set(
    const char* weights,
    const block_q4_1_group32* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body_group32<false>(weights, activations, output, m, k, n);
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_a4_group32_add(
    const char* weights,
    const block_q4_1_group32* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body_group32<true>(weights, activations, output, m, k, n);
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_q8_exact_set(
    const char* weights,
    const block_q8_1_iu4_planes* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body_q8_iu4_exact<false>(weights, activations, output, m, k, n);
}

extern "C" __global__ __launch_bounds__(NWARPS * 32, 2)
void gemm_hfq4g256_mmq_iu4_q8_exact_add(
    const char* weights,
    const block_q8_1_iu4_planes* activations,
    float* output,
    int m,
    int k,
    int n
) {
    gemm_body_q8_iu4_exact<true>(weights, activations, output, m, k, n);
}

#else

struct block_q4_1_group128;
struct block_q4_1_group32;
struct block_q8_1_iu4_planes;
extern "C" __global__ void quantize_q4_1_group128(
    const float*, block_q4_1_group128*, int, int) {}
extern "C" __global__ void quantize_q4_1_group32(
    const float*, block_q4_1_group32*, int, int) {}
extern "C" __global__ void quantize_q8_1_group128_iu4_planes(
    const float*, block_q8_1_iu4_planes*, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_a4_set(
    const char*, const block_q4_1_group128*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_a4_add(
    const char*, const block_q4_1_group128*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_a4_group32_set(
    const char*, const block_q4_1_group32*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_a4_group32_add(
    const char*, const block_q4_1_group32*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_q8_exact_set(
    const char*, const block_q8_1_iu4_planes*, float*, int, int, int) {}
extern "C" __global__ void gemm_hfq4g256_mmq_iu4_q8_exact_add(
    const char*, const block_q8_1_iu4_planes*, float*, int, int, int) {}

#endif
