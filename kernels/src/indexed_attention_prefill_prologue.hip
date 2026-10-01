// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multirow QSA norm/RoPE and source-cache append. The arithmetic helpers below
// retain tensor_ops.hip's serial RMS sum, RoPE association, BF16 index-key
// boundary and FP8 encoding. Pooling is a separate incumbent launch.
// Launch [index_heads + heads + kv_heads + 1, rows], block 256; the host owns
// geometry checks and excludes recording/capture.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

// QSA state formats (`QsaKvFormat` in tensor_ops.rs). K and V are separate
// token-major caches, one row per position:
//   F32  kv_heads * head_dim f32.
//   FP8  kv_heads * head_dim E4M3 codes, then kv_heads f16 scales: the qwen35
//        native fp8 row, quantized as kv_cache_write_fp8_e4m3 does (gfx12).
// Index keys retain the same BF16 rounding boundary in both cache formats:
// F32 arenas store the rounded F32 value; FP8 arenas store BF16 bits.
#define QSA_KV_F32 0
#define QSA_KV_FP8 1
#if defined(__GFX12__)
#define QSA_HAS_FP8 1
#endif

template <int F>
static __device__ __forceinline__ int qsa_kv_row_bytes(int kv_heads, int head_dim) {
    if constexpr (F == QSA_KV_F32) {
        return kv_heads * head_dim * 4;
    } else {
        return kv_heads * (head_dim + 2);
    }
}

// Store one channel of a K or V row in the fp8 format. Every thread of the
// block calls it (block 256 over one KV head, `channel` = threadIdx.x; lanes
// at or past head_dim pass any value and store nothing). The head's absmax is
// reduced through wave shuffles and `amax_lds` (one float per wave), which
// synchronizes.
template <int F>
static __device__ __forceinline__ void qsa_kv_store(
    unsigned char* row,
    float value,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel,
    float* amax_lds) {
    static_assert(F == QSA_KV_FP8, "F32 rows are stored directly");
#if defined(QSA_HAS_FP8)
    const bool active = channel < head_dim;
    float amax = active ? fabsf(value) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if ((threadIdx.x & 31) == 0) amax_lds[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = amax_lds[0];
    for (int wave = 1; wave < ((int)blockDim.x + 31) / 32; ++wave) {
        amax = fmaxf(amax, amax_lds[wave]);
    }
    __syncthreads();
    _Float16 s16 = amax == 0.0f ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f) {
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    }
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = bits >= 0x7bff ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }
    if (!active) return;
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value / sf, 0.0f, 0u, false);
    if (channel == 0) {
        *reinterpret_cast<_Float16*>(row + kv_heads * head_dim + 2 * kv_head) = s16;
    }
#endif
}

// One head's RMSNorm + HalfSplit RoPE at `position` (block 256, one channel
// per thread; returns the channel's value, `head_dim` <= 256). Every block
// thread must call it (it synchronizes).
static __device__ __forceinline__ float qsa_norm_rope_head(
    const float* head_values,
    const __hip_bfloat16* norm,
    int head_dim,
    int position,
    int rotary_dim,
    float* input_row,
    float* row_sum) {
    const int channel = (int)threadIdx.x;
    if (channel < head_dim) {
        input_row[channel] = head_values[channel];
    }
    __syncthreads();
    // One wave runs the serial i-order sum (identical in every lane) instead
    // of every wave repeating it.
    if (threadIdx.x < 32) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float row_value = input_row[i];
            sum += row_value * row_value;
        }
        if (threadIdx.x == 0) *row_sum = sum;
    }
    __syncthreads();
    if (channel >= head_dim) return 0.0f;
    float inv = rsqrtf(*row_sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)position
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    return value;
}

// QSA prologue of `rows` consecutive rows in one launch (block 256, one block
// per head and row): the index-query, query and key RMSNorm + RoPE (the three
// indexed_attention_norm_rope_f32_batched launches), the key/value cache
// append at `position + row`, and the index key's BF16 round trip and raw-key
// copy (bf16_round_trip_f32_strided + copy_rows_strided_f32). Same
// per-element operations. Grid [index_heads + heads + kv_heads + 1, rows];
// row r's inputs sit at r times each buffer's row width. `F` is the K/V cache
// format; the raw index keys are F32 for F32 caches and BF16 otherwise, and
// the fp8 rows are written exactly as the prefill append writes them.
template <int F, typename RawKey>
static __device__ __forceinline__ void qsa_prefill_prologue(
    float* index_row,
    float* qgate,
    float* keys,
    const float* values,
    unsigned char* full_keys,
    unsigned char* full_values,
    RawKey* raw_index_keys,
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    int position_start) {
    __shared__ float input_row[256];
    __shared__ float row_sum;
    __shared__ float kv_amax[8];
    const int c = (int)threadIdx.x;
    const int r = (int)blockIdx.y;
    const int position = position_start + r;
    const int index_q_width = index_heads * index_dim;
    const int kv_width = kv_heads * head_dim;
    index_row += r * (index_q_width + index_kv_width);
    qgate += r * 2 * heads * head_dim;
    keys += r * kv_width;
    values += r * kv_width;
    int block = (int)blockIdx.x;
    if (block < index_heads) {
        float* row = index_row + block * index_dim;
        const float value = qsa_norm_rope_head(row, index_q_norm, index_dim, position,
                                               min(index_dim, 64), input_row, &row_sum);
        if (c < index_dim) row[c] = value;
        return;
    }
    block -= index_heads;
    if (block < heads) {
        float* row = qgate + block * 2 * head_dim;
        const float value = qsa_norm_rope_head(row, q_norm, head_dim, position,
                                               min(head_dim, 64), input_row, &row_sum);
        if (c < head_dim) row[c] = value;
        return;
    }
    block -= heads;
    if (block < kv_heads) {
        float* row = keys + block * head_dim;
        const float value = qsa_norm_rope_head(row, k_norm, head_dim, position,
                                               min(head_dim, 64), input_row, &row_sum);
        if constexpr (F == QSA_KV_F32) {
            if (c < head_dim) {
                const int channel = block * head_dim + c;
                row[c] = value;
                reinterpret_cast<float*>(full_keys)[position * kv_width + channel] = value;
                reinterpret_cast<float*>(full_values)[position * kv_width + channel] =
                    values[channel];
            }
        } else {
            const float v = c < head_dim ? values[block * head_dim + c] : 0.0f;
            if (c < head_dim) row[c] = value;
            const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(kv_heads, head_dim);
            qsa_kv_store<F>(full_keys + (size_t)position * row_bytes, value, kv_heads, head_dim,
                            block, c, kv_amax);
            qsa_kv_store<F>(full_values + (size_t)position * row_bytes, v, kv_heads, head_dim,
                            block, c, kv_amax);
        }
        return;
    }
    for (int i = c; i < index_kv_width; i += 256) {
        float v = index_row[index_q_width + i];
        if (isfinite(v)) {
            v = __uint_as_float((__float_as_uint(v) + 0x7FFFu + ((__float_as_uint(v) >> 16) & 1u))
                                & 0xFFFF0000u);
            index_row[index_q_width + i] = v;
        }
        if constexpr (F == QSA_KV_F32) {
            raw_index_keys[position * index_kv_width + i] = v;
        } else {
            raw_index_keys[(size_t)position * index_kv_width + i] = bf16_round(v);
        }
    }
}

#define QSA_PREFILL_PROLOGUE(NAME, FORMAT, CACHE, RAW)                                      \
    extern "C" __global__ __launch_bounds__(256) void NAME(                               \
        float* index_row, float* qgate, float* keys, const float* values, CACHE* full_keys, \
        CACHE* full_values, RAW* raw_index_keys, const __hip_bfloat16* index_q_norm,       \
        const __hip_bfloat16* q_norm, const __hip_bfloat16* k_norm, int index_heads,       \
        int index_dim, int index_kv_width, int heads, int kv_heads, int head_dim,          \
        int position_start) {                                                              \
        qsa_prefill_prologue<FORMAT, RAW>(                                                  \
            index_row, qgate, keys, values, reinterpret_cast<unsigned char*>(full_keys),   \
            reinterpret_cast<unsigned char*>(full_values), raw_index_keys, index_q_norm,   \
            q_norm, k_norm, index_heads, index_dim, index_kv_width, heads, kv_heads,       \
            head_dim, position_start);                                                     \
    }
QSA_PREFILL_PROLOGUE(indexed_attention_prefill_prologue_f32, QSA_KV_F32, float, float)
#if defined(QSA_HAS_FP8)
QSA_PREFILL_PROLOGUE(indexed_attention_prefill_prologue_fp8, QSA_KV_FP8, unsigned char,
                    __hip_bfloat16)
#endif
#undef QSA_PREFILL_PROLOGUE
