// SPDX-License-Identifier: Apache-2.0

#if !defined(__HIP_DEVICE_COMPILE__) && defined(HIPFIRE_CK_TARGET_GFX11) && \
    !defined(__gfx11__)
#define __gfx11__
#endif

#include "quantized_ck_buffer_view.hpp"
#include "hipfire_flash_attn_ck_quantized.h"
#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
#include "quantized_kv_predecode.hpp"
#include "../hipfire_flash_attn_ck.h"
#endif

#include "fmha_fwd.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string_view>
#include <vector>

extern "C" __global__ void attention_flash_asym3_tile_batched(
    const float*, const unsigned char*, const unsigned char*, float*, const int*, const float*,
    const float*, const float*, int, int, int, int, float, int, int, int, int, int, int);
extern "C" __global__ void attention_flash_asym_reduce_batched(
    const float*, float*, const int*, int, int, int, int, int, int, int);

namespace {

constexpr int kHeadDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr int kAsym3HeadBytes = 100;
constexpr int kQ8HeadBytes = 272;

static_assert(sizeof(hipfire_flash_attn_ck_quantized_prefill_params) == 120,
              "quantized CK ABI parameter layout changed");
static_assert(offsetof(hipfire_flash_attn_ck_quantized_prefill_params, q) == 8);
static_assert(offsetof(hipfire_flash_attn_ck_quantized_prefill_params, workspace_bytes) == 48);
static_assert(offsetof(hipfire_flash_attn_ck_quantized_prefill_params, softmax_scale) == 80);
static_assert(offsetof(hipfire_flash_attn_ck_quantized_prefill_params,
                       v_row_stride_bytes) == 112);
static_assert(sizeof(hipfire_flash_attn_ck_asym4_loader_params) == 72,
              "Asym4 loader ABI parameter layout changed");
static_assert(offsetof(hipfire_flash_attn_ck_asym4_loader_params, packed_k) == 8);
static_assert(offsetof(hipfire_flash_attn_ck_asym4_loader_params, seqlen_k) == 48);
static_assert(offsetof(hipfire_flash_attn_ck_asym4_loader_params,
                       v_row_stride_bytes) == 64);

#ifndef HIPFIRE_CK_FMHA_BM
#define HIPFIRE_CK_FMHA_BM 64
#endif

#ifndef HIPFIRE_CK_FMHA_BN
#define HIPFIRE_CK_FMHA_BN 64
#endif

#ifndef HIPFIRE_CK_FMHA_OUTPUT_F32
#define HIPFIRE_CK_FMHA_OUTPUT_F32 0
#endif

constexpr int kFmhaBlockM = HIPFIRE_CK_FMHA_BM;
constexpr int kFmhaBlockN = HIPFIRE_CK_FMHA_BN;
static_assert(kFmhaBlockM == 16 || kFmhaBlockM == 32 || kFmhaBlockM == 64);
static_assert(kFmhaBlockN == 32 || kFmhaBlockN == 64);
static_assert(kFmhaBlockM % 16 == 0);
using FmhaOutput = std::conditional_t<HIPFIRE_CK_FMHA_OUTPUT_F32 != 0,
                                      float,
                                      ck_tile::half_t>;

float output_to_float(FmhaOutput value)
{
    if constexpr(std::is_same_v<FmhaOutput, float>)
    {
        return value;
    }
    else
    {
        return __half2float(value);
    }
}

__global__ void rotate_q_givens_f32_to_f16(const float* __restrict__ input,
                                            __half* __restrict__ output,
                                            const float* __restrict__ cos_theta,
                                            const float* __restrict__ sin_theta,
                                            int query_rows,
                                            int query_heads)
{
    const int row_head = blockIdx.x;
    const int head = row_head % query_heads;
    const int row = row_head / query_heads;
    const int lane = threadIdx.x;
    if(row >= query_rows || lane >= 32)
    {
        return;
    }

    const int base = (row * query_heads + head) * kHeadDim + lane * 8;
#pragma unroll
    for(int pair = 0; pair < 4; ++pair)
    {
        const int index = base + pair * 2;
        const float a = input[index];
        const float b = input[index + 1];
        const float c = cos_theta[lane * 4 + pair];
        const float s = sin_theta[lane * 4 + pair];
        output[index] = __float2half(a * c - b * s);
        output[index + 1] = __float2half(a * s + b * c);
    }
}

__global__ void convert_f16_to_f32(const __half* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t pair_count)
{
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(pair < pair_count)
    {
        reinterpret_cast<float2*>(output)[pair] =
            __half22float2(reinterpret_cast<const __half2*>(input)[pair]);
    }
}

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};

static_assert(sizeof(block_q8_1_mmq) == 144);

__device__ __forceinline__ char quantize_q8_value(float value, float inverse_scale)
{
    return static_cast<char>(lrintf(fmaxf(-127.0f, fminf(127.0f, value * inverse_scale))));
}

__global__ __launch_bounds__(32, 16)
void gated_mq_rotate_q8_1(const __half* __restrict__ attention,
                         const float* __restrict__ gate,
                         const float* __restrict__ signs1,
                         const float* __restrict__ signs2,
                         block_q8_1_mmq* __restrict__ output,
                         int rows,
                         int heads)
{
    const int row_head = blockIdx.x;
    const int lane = threadIdx.x;
    if(row_head >= rows * heads || lane >= 32) return;
    const int row = row_head / heads;
    const int head = row_head - row * heads;
    const int d0 = lane * 8;
    const int base = row_head * kHeadDim + d0;
    const __half2* half_input = reinterpret_cast<const __half2*>(attention + base);
    const float2 h0 = __half22float2(half_input[0]);
    const float2 h1 = __half22float2(half_input[1]);
    const float2 h2 = __half22float2(half_input[2]);
    const float2 h3 = __half22float2(half_input[3]);
    float v0 = h0.x * (1.0f / (1.0f + expf(-gate[base])));
    float v1 = h0.y * (1.0f / (1.0f + expf(-gate[base + 1])));
    float v2 = h1.x * (1.0f / (1.0f + expf(-gate[base + 2])));
    float v3 = h1.y * (1.0f / (1.0f + expf(-gate[base + 3])));
    float v4 = h2.x * (1.0f / (1.0f + expf(-gate[base + 4])));
    float v5 = h2.y * (1.0f / (1.0f + expf(-gate[base + 5])));
    float v6 = h3.x * (1.0f / (1.0f + expf(-gate[base + 6])));
    float v7 = h3.y * (1.0f / (1.0f + expf(-gate[base + 7])));
    v0 *= signs1[d0]; v1 *= signs1[d0 + 1];
    v2 *= signs1[d0 + 2]; v3 *= signs1[d0 + 3];
    v4 *= signs1[d0 + 4]; v5 *= signs1[d0 + 5];
    v6 *= signs1[d0 + 6]; v7 *= signs1[d0 + 7];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

#define HIPFIRE_HBFLY(v, pat, stride) do {                                      \
    const float peer = __int_as_float(                                           \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat)));                  \
    (v) = (lane & (stride)) ? peer - (v) : (v) + peer;                           \
} while(0)
#define HIPFIRE_HBFLY8(pat, stride)                                              \
    HIPFIRE_HBFLY(v0, pat, stride); HIPFIRE_HBFLY(v1, pat, stride);              \
    HIPFIRE_HBFLY(v2, pat, stride); HIPFIRE_HBFLY(v3, pat, stride);              \
    HIPFIRE_HBFLY(v4, pat, stride); HIPFIRE_HBFLY(v5, pat, stride);              \
    HIPFIRE_HBFLY(v6, pat, stride); HIPFIRE_HBFLY(v7, pat, stride)
    HIPFIRE_HBFLY8(0x041F, 1);
    HIPFIRE_HBFLY8(0x081F, 2);
    HIPFIRE_HBFLY8(0x101F, 4);
    HIPFIRE_HBFLY8(0x201F, 8);
    HIPFIRE_HBFLY8(0x401F, 16);
#undef HIPFIRE_HBFLY8
#undef HIPFIRE_HBFLY

    constexpr float rotate_scale = 0.0625f;
    v0 *= rotate_scale * signs2[d0]; v1 *= rotate_scale * signs2[d0 + 1];
    v2 *= rotate_scale * signs2[d0 + 2]; v3 *= rotate_scale * signs2[d0 + 3];
    v4 *= rotate_scale * signs2[d0 + 4]; v5 *= rotate_scale * signs2[d0 + 5];
    v6 *= rotate_scale * signs2[d0 + 6]; v7 *= rotate_scale * signs2[d0 + 7];

    float amax0 = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax1 = fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7)));
    float sum0 = v0 + v1 + v2 + v3;
    float sum1 = v4 + v5 + v6 + v7;
    amax0 = fmaxf(amax0, __shfl_xor(amax0, 2, 4));
    amax1 = fmaxf(amax1, __shfl_xor(amax1, 2, 4));
    sum0 += __shfl_xor(sum0, 2, 4);
    sum1 += __shfl_xor(sum1, 2, 4);
    amax0 = fmaxf(amax0, __shfl_xor(amax0, 1, 4));
    amax1 = fmaxf(amax1, __shfl_xor(amax1, 1, 4));
    sum0 += __shfl_xor(sum0, 1, 4);
    sum1 += __shfl_xor(sum1, 1, 4);
    const float amax = fmaxf(amax0, amax1);
    const float sum = sum0 + sum1;
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    const char4 q0 = make_char4(quantize_q8_value(v0, id), quantize_q8_value(v1, id),
                                quantize_q8_value(v2, id), quantize_q8_value(v3, id));
    const char4 q1 = make_char4(quantize_q8_value(v4, id), quantize_q8_value(v5, id),
                                quantize_q8_value(v6, id), quantize_q8_value(v7, id));
    const int half = lane >> 4;
    const int lane_in_half = lane & 15;
    block_q8_1_mmq* out = output + static_cast<long long>(head * 2 + half) * rows + row;
    reinterpret_cast<char4*>(out->qs)[lane_in_half * 2] = q0;
    reinterpret_cast<char4*>(out->qs)[lane_in_half * 2 + 1] = q1;
    if((lane & 3) == 0)
        out->ds4[lane_in_half >> 2] = make_half2(static_cast<__half>(d), static_cast<__half>(sum));
}

using FmhaBlockTile =
    ck_tile::sequence<kFmhaBlockM, kFmhaBlockN, 32, 256, 32, 256>;
using FmhaShape = ck_tile::TileFmhaShape<FmhaBlockTile,
                                         ck_tile::sequence<kFmhaBlockM / 16, 1, 1>,
                                         ck_tile::sequence<16, 16, 16>,
                                         ck_tile::sequence<kFmhaBlockM / 16, 1, 1>,
                                         ck_tile::sequence<16, 16, 16>,
                                         true>;
using FmhaTraits = ck_tile::TileFmhaTraits<true,
                                           true,
                                           true,
                                           true,
                                           false,
                                           ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                           false,
                                           false,
                                           false,
                                           ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                           -1,
                                           false>;
using FmhaVariant =
    ck_tile::ComposedAttention<false * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;
using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<true>;
using FmhaProblem = ck_tile::BlockFmhaPipelineProblem<ck_tile::half_t,
                                                      ck_tile::half_t,
                                                      ck_tile::half_t,
                                                      float,
                                                      float,
                                                      ck_tile::half_t,
                                                      uint8_t,
                                                      float,
                                                      ck_tile::half_t,
                                                      float,
                                                      FmhaOutput,
                                                      FmhaShape,
                                                      false,
                                                      FmhaVariant,
                                                      FmhaMask,
                                                      false,
                                                      FmhaTraits>;
using FmhaPipeline = ck_tile::BlockFmhaPipelineQRKSVS<FmhaProblem>;
using FmhaEpilogue = ck_tile::DefaultGemm2DEpilogue<ck_tile::DefaultGemm2DEpilogueProblem<
    ck_tile::half_t,
    ck_tile::half_t,
    ck_tile::tuple<>,
    float,
    FmhaOutput,
    ck_tile::tuple<>,
    ck_tile::tensor_layout::gemm::RowMajor,
    ck_tile::element_wise::PassThrough,
    FmhaShape::kM0,
    FmhaShape::kN1,
    true,
    true,
    FmhaShape::Gemm1WarpTile::at(ck_tile::number<0>{}),
    FmhaShape::Gemm1WarpTile::at(ck_tile::number<1>{}),
    FmhaShape::Gemm1WarpTile::at(ck_tile::number<2>{}),
    true>>;

struct QuantizedFmhaKernel
{
    struct Kargs
    {
        const ck_tile::half_t* q;
        const uint8_t* packed_k;
        const uint8_t* packed_v;
        FmhaOutput* out;
        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t batch;
        ck_tile::index_t num_query_heads;
        ck_tile::index_t num_kv_heads;
        ck_tile::index_t stride_q;
        ck_tile::index_t stride_out;
        ck_tile::long_index_t batch_stride_q;
        ck_tile::long_index_t batch_stride_k_bytes;
        ck_tile::long_index_t batch_stride_v_bytes;
        ck_tile::long_index_t batch_stride_out;
        ck_tile::index_t k_row_stride_bytes;
        ck_tile::index_t v_row_stride_bytes;
        float scale_s;
    };

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    static constexpr ck_tile::index_t kBlockSize = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    CK_TILE_HOST static dim3 GridSize(const Kargs& args)
    {
        return dim3(args.num_query_heads,
                    ck_tile::integer_divide_ceil(args.seqlen_q, FmhaPipeline::kM0),
                    args.batch);
    }

    CK_TILE_HOST static dim3 BlockSize() { return dim3(FmhaPipeline::kBlockSize / 2); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
#if defined(HIPFIRE_CK_ASYM3_LDS_CODEBOOK)
        return ck_tile::max(FmhaPipeline::GetSmemSize(), FmhaEpilogue::GetSmemSize()) + 32;
#else
        return ck_tile::max(FmhaPipeline::GetSmemSize(), FmhaEpilogue::GetSmemSize());
#endif
    }

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
#if defined(__HIP_DEVICE_COMPILE__)
        alignas(16) __shared__ char smem[GetSmemSize()];

#if defined(HIPFIRE_CK_ASYM3_LDS_CODEBOOK)
        float* asym3_codebook = reinterpret_cast<float*>(smem);
        if(threadIdx.x < 8)
        {
            constexpr float centroids[8] = {
                -0.134860f,
                -0.083320f,
                -0.046469f,
                -0.015176f,
                0.015176f,
                0.046469f,
                0.083320f,
                0.134860f,
            };
            asym3_codebook[threadIdx.x] = centroids[threadIdx.x];
        }
        __syncthreads();
        char* pipeline_smem = smem + 32;
#else
        const float* asym3_codebook = nullptr;
        char* pipeline_smem = smem;
#endif

        const ck_tile::index_t query_head = blockIdx.x;
        const ck_tile::index_t batch = blockIdx.z;
        const ck_tile::index_t query_tile = gridDim.y - 1 - blockIdx.y;
        const ck_tile::index_t query_start = query_tile * FmhaPipeline::kM0;
        const ck_tile::index_t kv_head = query_head / (args.num_query_heads / args.num_kv_heads);

        const ck_tile::half_t* q_head =
            args.q + batch * args.batch_stride_q + query_head * kHeadDim;
        FmhaOutput* out_head =
            args.out + batch * args.batch_stride_out + query_head * kHeadDim;
        const uint8_t* k_head = args.packed_k + batch * args.batch_stride_k_bytes +
                                kv_head * kAsym3HeadBytes;
        const uint8_t* v_head = args.packed_v + batch * args.batch_stride_v_bytes +
                                kv_head * kQ8HeadBytes;

        const auto q_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            q_head,
            ck_tile::make_tuple(args.seqlen_q, ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(args.stride_q, ck_tile::number<1>{}),
            ck_tile::number<FmhaPipeline::kAlignmentQ>{},
            ck_tile::number<1>{});
        const auto q_view = ck_tile::pad_tensor_view(
            q_naive,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kSubQKHeaddim>{}),
            ck_tile::sequence<true, true>{});
        const auto k_naive =
            hipfire::ck_attention::make_packed_kv_tensor_view<
                hipfire::ck_attention::PackedKvKind::Asym3K>(
                k_head, args.seqlen_k, args.k_row_stride_bytes, asym3_codebook);
        const auto k_view = ck_tile::pad_tensor_view(
            k_naive,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kN0>{},
                                ck_tile::number<FmhaPipeline::kK0>{}),
            ck_tile::sequence<true, true>{});
        const auto v_naive =
            hipfire::ck_attention::make_packed_kv_tensor_view<
                hipfire::ck_attention::PackedKvKind::Q8V>(
                v_head, args.seqlen_k, args.v_row_stride_bytes);
        const auto v_transposed = ck_tile::transform_tensor_view(
            v_naive,
            ck_tile::make_tuple(ck_tile::make_pass_through_transform(kHeadDim),
                                ck_tile::make_pass_through_transform(args.seqlen_k)),
            ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        const auto v_view = ck_tile::pad_tensor_view(
            v_transposed,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kN1>{},
                                ck_tile::number<FmhaPipeline::kK1>{}),
            ck_tile::sequence<true, true>{});

        auto q_window = ck_tile::make_tile_window(
            q_view,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kSubQKHeaddim>{}),
            {query_start, 0});
        auto k_window = ck_tile::make_tile_window(
            k_view,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kN0>{},
                                ck_tile::number<FmhaPipeline::kK0>{}),
            {0, 0});
        auto v_window = ck_tile::make_tile_window(
            v_view,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kN1>{},
                                ck_tile::number<FmhaPipeline::kK1>{}),
            {0, 0});
        auto bias_window = ck_tile::make_null_tile_window(
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kN0>{}));
        auto rand_window = ck_tile::make_null_tile_window(
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kN0>{}));
        auto lse_window = ck_tile::make_null_tile_window(
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{}));

        auto mask = ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
            -1, 0, args.seqlen_q, args.seqlen_k, false);
        ck_tile::EmptyPositionEncoding<float> position_encoding{};
        FmhaVariant variant{};
        ck_tile::StandardAttentionParams<FmhaMask> variant_params{mask, args.scale_s};
        BlockIndices block_indices{batch, query_head, kv_head};
        ck_tile::NullBlockDropout dropout{};

        auto output_tile = FmhaPipeline{}(q_window,
                                          k_window,
                                          v_window,
                                          bias_window,
                                          rand_window,
                                          lse_window,
                                          mask,
                                          position_encoding,
                                          args.scale_s,
                                          variant,
                                          variant_params,
                                          block_indices,
                                          pipeline_smem,
                                          dropout);

        const auto out_naive =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                out_head,
                ck_tile::make_tuple(args.seqlen_q, ck_tile::number<kHeadDim>{}),
                ck_tile::make_tuple(args.stride_out, ck_tile::number<1>{}),
                ck_tile::number<FmhaPipeline::kAlignmentO>{},
                ck_tile::number<1>{});
        const auto out_view = ck_tile::pad_tensor_view(
            out_naive,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kN1>{}),
            ck_tile::sequence<true, true>{});
        auto out_window = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<FmhaPipeline::kM0>{},
                                ck_tile::number<FmhaPipeline::kN1>{}),
            {query_start, 0});
        FmhaEpilogue{}(out_window, output_tile, nullptr);
#else
        (void)args;
#endif
    }
};

size_t quantized_workspace_bytes(int32_t seqlen_q, int32_t nhead_q, int32_t head_dim)
{
    if(seqlen_q <= 0 || nhead_q <= 0 || head_dim <= 0)
    {
        return 0;
    }
    const size_t values = static_cast<size_t>(seqlen_q) * nhead_q * head_dim;
    if(values > SIZE_MAX / (2 * sizeof(__half)))
    {
        return 0;
    }
    return 2 * values * sizeof(__half);
}

#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
size_t staged_workspace_bytes(int32_t seqlen_q,
                              int32_t seqlen_k,
                              int32_t nhead_q,
                              int32_t nhead_k,
                              int32_t head_dim)
{
    if(seqlen_q <= 0 || seqlen_k <= 0 || nhead_q <= 0 || nhead_k <= 0 ||
       head_dim <= 0)
    {
        return 0;
    }
    const size_t q_values = static_cast<size_t>(seqlen_q) * nhead_q * head_dim;
    const size_t kv_values = static_cast<size_t>(seqlen_k) * nhead_k * head_dim;
    if(q_values > SIZE_MAX / (2 * sizeof(__half)) ||
       kv_values > SIZE_MAX / (2 * sizeof(__half)) ||
       2 * q_values > SIZE_MAX - 2 * kv_values)
    {
        return 0;
    }
    return (2 * q_values + 2 * kv_values) * sizeof(__half);
}
#endif

void set_quantized_error(char* error, size_t capacity, const char* message)
{
    if(error == nullptr || capacity == 0)
    {
        return;
    }
    const size_t count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(error, message, count);
    error[count] = '\0';
}

int validate_quantized_prefill(const hipfire_flash_attn_ck_quantized_prefill_params* p,
                               char* error,
                               size_t error_capacity)
{
    if(p == nullptr)
    {
        set_quantized_error(error, error_capacity, "params is null");
        return 1;
    }
    if(p->abi_version != HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION ||
       p->struct_size < sizeof(*p))
    {
        set_quantized_error(error, error_capacity, "unsupported quantized ABI layout");
        return 1;
    }
    if(p->q == nullptr || p->packed_k == nullptr || p->packed_v == nullptr ||
       p->out == nullptr || p->workspace == nullptr || p->cos_theta == nullptr ||
       p->sin_theta == nullptr)
    {
        set_quantized_error(error, error_capacity, "quantized prefill pointers must be non-null");
        return 1;
    }
    if(p->head_dim != kHeadDim || p->nhead_q != kQueryHeads || p->nhead_k != kKvHeads)
    {
        set_quantized_error(error, error_capacity, "only D256 24Q/4KV GQA is validated");
        return 1;
    }
    if(p->causal != 1 || p->seqlen_q < 128 || p->seqlen_k < p->seqlen_q)
    {
        set_quantized_error(
            error, error_capacity, "requires bottom-right causal prefill with Q>=128 and K>=Q");
        return 1;
    }
    if(p->k_row_stride_bytes != kKvHeads * kAsym3HeadBytes ||
       p->v_row_stride_bytes != kKvHeads * kQ8HeadBytes)
    {
        set_quantized_error(error, error_capacity, "unsupported packed KV row stride");
        return 1;
    }
    if(!(p->softmax_scale > 0.0f))
    {
        set_quantized_error(error, error_capacity, "softmax_scale must be positive");
        return 1;
    }
    const size_t required = quantized_workspace_bytes(p->seqlen_q, p->nhead_q, p->head_dim);
    if(required == 0 || p->workspace_bytes < required)
    {
        set_quantized_error(error, error_capacity, "quantized prefill workspace is too small");
        return 1;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}

int launch_quantized_prefill_impl(
    const hipfire_flash_attn_ck_quantized_prefill_params* p,
    const hipfire_flash_attn_ck_quantized_mq_q8_params* mq_q8,
    char* error,
    size_t error_capacity)
{
#if defined(HIPFIRE_QUANTIZED_CK_SIDECAR)
    static_assert(std::is_same_v<FmhaOutput, ck_tile::half_t>,
                  "the quantized ABI requires the FP16 CK epilogue");
#endif
    const size_t values =
        static_cast<size_t>(p->seqlen_q) * p->nhead_q * p->head_dim;
    auto* q_rotated = reinterpret_cast<__half*>(p->workspace);
    auto* out_half = q_rotated + values;
    auto stream = reinterpret_cast<hipStream_t>(p->stream);

    hipLaunchKernelGGL(rotate_q_givens_f32_to_f16,
                       dim3(p->nhead_q * p->seqlen_q),
                       dim3(32),
                       0,
                       stream,
                       p->q,
                       q_rotated,
                       p->cos_theta,
                       p->sin_theta,
                       p->seqlen_q,
                       p->nhead_q);

    QuantizedFmhaKernel::Kargs args{
        reinterpret_cast<const ck_tile::half_t*>(q_rotated),
        p->packed_k,
        p->packed_v,
        reinterpret_cast<FmhaOutput*>(out_half),
        p->seqlen_q,
        p->seqlen_k,
        1,
        p->nhead_q,
        p->nhead_k,
        p->nhead_q * p->head_dim,
        p->nhead_q * p->head_dim,
        static_cast<ck_tile::long_index_t>(p->seqlen_q) * p->nhead_q * p->head_dim,
        static_cast<ck_tile::long_index_t>(p->seqlen_k) * p->k_row_stride_bytes,
        static_cast<ck_tile::long_index_t>(p->seqlen_k) * p->v_row_stride_bytes,
        static_cast<ck_tile::long_index_t>(p->seqlen_q) * p->nhead_q * p->head_dim,
        p->k_row_stride_bytes,
        p->v_row_stride_bytes,
#if CK_TILE_FMHA_FWD_FAST_EXP2
        static_cast<float>(p->softmax_scale * ck_tile::log2e_v<>),
#else
        p->softmax_scale,
#endif
    };
    ck_tile::stream_config stream_config{stream};
    const float result = ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<QuantizedFmhaKernel::kBlockPerCu, ck_tile::gfx11_t>(
            QuantizedFmhaKernel{},
            QuantizedFmhaKernel::GridSize(args),
            QuantizedFmhaKernel::BlockSize(),
            0,
            args));
    if(result < 0.0f)
    {
        set_quantized_error(error, error_capacity, "CK quantized prefill launch failed");
        return 2;
    }

    if(mq_q8 != nullptr)
    {
        hipLaunchKernelGGL(gated_mq_rotate_q8_1,
                           dim3(p->seqlen_q * p->nhead_q),
                           dim3(32),
                           0,
                           stream,
                           out_half,
                           mq_q8->gate,
                           mq_q8->signs1,
                           mq_q8->signs2,
                           reinterpret_cast<block_q8_1_mmq*>(mq_q8->q8_1_out),
                           p->seqlen_q,
                           p->nhead_q);
    }
    else
    {
        const size_t pair_count = values / 2;
        hipLaunchKernelGGL(convert_f16_to_f32,
                           dim3((pair_count + 255) / 256),
                           dim3(256),
                           0,
                           stream,
                           out_half,
                           p->out,
                           pair_count);
    }
    const hipError_t status = hipPeekAtLastError();
    if(status != hipSuccess)
    {
        set_quantized_error(error, error_capacity, hipGetErrorString(status));
        return 2;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}

#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
int validate_staged_prefill(const hipfire_flash_attn_ck_quantized_prefill_params* p,
                            char* error,
                            size_t error_capacity)
{
    if(const int status = validate_quantized_prefill(p, error, error_capacity); status != 0)
    {
        return status;
    }
    const size_t required = staged_workspace_bytes(
        p->seqlen_q, p->seqlen_k, p->nhead_q, p->nhead_k, p->head_dim);
    if(required == 0 || p->workspace_bytes < required)
    {
        set_quantized_error(error, error_capacity, "staged prefill workspace is too small");
        return 1;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}

int launch_staged_prefill_impl(
    const hipfire_flash_attn_ck_quantized_prefill_params* p,
    char* error,
    size_t error_capacity)
{
    const size_t q_values =
        static_cast<size_t>(p->seqlen_q) * p->nhead_q * p->head_dim;
    const size_t kv_values =
        static_cast<size_t>(p->seqlen_k) * p->nhead_k * p->head_dim;
    auto* q_rotated = reinterpret_cast<__half*>(p->workspace);
    auto* out_half = q_rotated + q_values;
    auto* dense_k = out_half + q_values;
    auto* dense_v = dense_k + kv_values;
    auto stream = reinterpret_cast<hipStream_t>(p->stream);

    hipLaunchKernelGGL(rotate_q_givens_f32_to_f16,
                       dim3(p->nhead_q * p->seqlen_q),
                       dim3(32),
                       0,
                       stream,
                       p->q,
                       q_rotated,
                       p->cos_theta,
                       p->sin_theta,
                       p->seqlen_q,
                       p->nhead_q);
    hipLaunchKernelGGL(hipfire::ck_attention::predecode_asym3_k_q8_v_f16,
                       dim3(p->seqlen_k, p->nhead_k),
                       dim3(32),
                       0,
                       stream,
                       p->packed_k,
                       p->packed_v,
                       dense_k,
                       dense_v,
                       p->seqlen_k,
                       p->k_row_stride_bytes,
                       p->v_row_stride_bytes);

    hipfire_flash_attn_ck_fwd_params dense{};
    dense.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    dense.struct_size = sizeof(dense);
    dense.q = q_rotated;
    dense.k = dense_k;
    dense.v = dense_v;
    dense.out = out_half;
    dense.stream = p->stream;
    dense.dtype = HIPFIRE_FLASH_ATTN_CK_F16;
    dense.batch = 1;
    dense.seqlen_q = p->seqlen_q;
    dense.seqlen_k = p->seqlen_k;
    dense.nhead_q = p->nhead_q;
    dense.nhead_k = p->nhead_k;
    dense.head_dim = p->head_dim;
    dense.causal = p->causal;
    dense.softmax_scale = p->softmax_scale;
    dense.stride_q = static_cast<int64_t>(p->nhead_q) * p->head_dim;
    dense.stride_k = static_cast<int64_t>(p->nhead_k) * p->head_dim;
    dense.stride_v = dense.stride_k;
    dense.stride_out = dense.stride_q;
    dense.nhead_stride_q = p->head_dim;
    dense.nhead_stride_k = p->head_dim;
    dense.nhead_stride_v = p->head_dim;
    dense.nhead_stride_out = p->head_dim;
    dense.batch_stride_q = static_cast<int64_t>(q_values);
    dense.batch_stride_k = static_cast<int64_t>(kv_values);
    dense.batch_stride_v = static_cast<int64_t>(kv_values);
    dense.batch_stride_out = static_cast<int64_t>(q_values);
    const int dense_status = hipfire_flash_attn_ck_fwd(&dense, error, error_capacity);
    if(dense_status != 0)
    {
        return dense_status;
    }

    const size_t pair_count = q_values / 2;
    hipLaunchKernelGGL(convert_f16_to_f32,
                       dim3((pair_count + 255) / 256),
                       dim3(256),
                       0,
                       stream,
                       out_half,
                       p->out,
                       pair_count);
    const hipError_t status = hipPeekAtLastError();
    if(status != hipSuccess)
    {
        set_quantized_error(error, error_capacity, hipGetErrorString(status));
        return 2;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}
#endif

#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
int validate_asym4_loader(const hipfire_flash_attn_ck_asym4_loader_params* p,
                          char* error,
                          size_t error_capacity)
{
    if(p == nullptr || p->abi_version != HIPFIRE_FLASH_ATTN_CK_ASYM4_LOADER_ABI_VERSION ||
       p->struct_size < sizeof(*p))
    {
        set_quantized_error(error, error_capacity, "unsupported Asym4 loader ABI layout");
        return 1;
    }
    if(p->packed_k == nullptr || p->packed_v == nullptr || p->dense_k_f16 == nullptr ||
       p->dense_v_f16 == nullptr)
    {
        set_quantized_error(error, error_capacity, "Asym4 loader pointers must be non-null");
        return 1;
    }
    if(p->seqlen_k <= 0 || p->nhead_k != kKvHeads || p->head_dim != kHeadDim ||
       p->k_row_stride_bytes != kKvHeads * hipfire::ck_attention::kPredecodeAsym4HeadBytes ||
       p->v_row_stride_bytes != kKvHeads * kQ8HeadBytes)
    {
        set_quantized_error(error, error_capacity, "unsupported Asym4/Q8 KV layout");
        return 1;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}
#endif

} // namespace

extern "C" uint32_t hipfire_flash_attn_ck_quantized_abi_version(void)
{
    return HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION;
}

extern "C" size_t hipfire_flash_attn_ck_quantized_prefill_workspace_bytes(
    int32_t seqlen_q,
    int32_t nhead_q,
    int32_t head_dim)
{
    return quantized_workspace_bytes(seqlen_q, nhead_q, head_dim);
}

#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
extern "C" size_t hipfire_flash_attn_ck_quantized_staged_workspace_bytes(
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t head_dim)
{
    return staged_workspace_bytes(seqlen_q, seqlen_k, nhead_q, nhead_k, head_dim);
}

extern "C" int hipfire_flash_attn_ck_quantized_staged_supported(
    const hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity)
{
    return validate_staged_prefill(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_quantized_staged_prefill(
    const hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate_staged_prefill(params, error, error_capacity); status != 0)
    {
        return status;
    }
    return launch_staged_prefill_impl(params, error, error_capacity);
}
#endif

extern "C" int hipfire_flash_attn_ck_quantized_prefill_supported(
    const hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity)
{
    return validate_quantized_prefill(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_quantized_prefill(
    const hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate_quantized_prefill(params, error, error_capacity); status != 0)
    {
        return status;
    }
    try
    {
        return launch_quantized_prefill_impl(params, nullptr, error, error_capacity);
    }
    catch(const std::exception& exception)
    {
        set_quantized_error(error, error_capacity, exception.what());
        return 2;
    }
    catch(...)
    {
        set_quantized_error(error, error_capacity, "unknown CK quantized prefill exception");
        return 2;
    }
}

extern "C" int hipfire_flash_attn_ck_quantized_mq_q8_supported(
    const hipfire_flash_attn_ck_quantized_mq_q8_params* params,
    char* error,
    size_t error_capacity)
{
    if(params == nullptr || params->abi_version != HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION ||
       params->struct_size < sizeof(*params) || params->gate == nullptr ||
       params->signs1 == nullptr || params->signs2 == nullptr || params->q8_1_out == nullptr)
    {
        set_quantized_error(error, error_capacity, "invalid MQ-Q8 projection bridge parameters");
        return 1;
    }
    return validate_quantized_prefill(&params->prefill, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_quantized_prefill_mq_q8(
    const hipfire_flash_attn_ck_quantized_mq_q8_params* params,
    char* error,
    size_t error_capacity)
{
    const int status = hipfire_flash_attn_ck_quantized_mq_q8_supported(
        params, error, error_capacity);
    if(status != 0) return status;
    try
    {
        return launch_quantized_prefill_impl(&params->prefill, params, error, error_capacity);
    }
    catch(const std::exception& exception)
    {
        set_quantized_error(error, error_capacity, exception.what());
        return 2;
    }
    catch(...)
    {
        set_quantized_error(error, error_capacity, "unknown CK MQ-Q8 prefill exception");
        return 2;
    }
}

#if defined(HIPFIRE_ENABLE_STAGED_QUANTIZED_CK)
extern "C" int hipfire_flash_attn_ck_asym4_loader_supported(
    const hipfire_flash_attn_ck_asym4_loader_params* params,
    char* error,
    size_t error_capacity)
{
    return validate_asym4_loader(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_asym4_predecode(
    const hipfire_flash_attn_ck_asym4_loader_params* params,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate_asym4_loader(params, error, error_capacity); status != 0)
    {
        return status;
    }
    auto stream = reinterpret_cast<hipStream_t>(params->stream);
    hipLaunchKernelGGL(hipfire::ck_attention::predecode_asym4_k_q8_v_f16,
                       dim3(params->seqlen_k, params->nhead_k),
                       dim3(32),
                       0,
                       stream,
                       params->packed_k,
                       params->packed_v,
                       reinterpret_cast<__half*>(params->dense_k_f16),
                       reinterpret_cast<__half*>(params->dense_v_f16),
                       params->seqlen_k,
                       params->k_row_stride_bytes,
                       params->v_row_stride_bytes);
    const hipError_t status = hipPeekAtLastError();
    if(status != hipSuccess)
    {
        set_quantized_error(error, error_capacity, hipGetErrorString(status));
        return 2;
    }
    set_quantized_error(error, error_capacity, "");
    return 0;
}
#endif

namespace {

#if !defined(HIPFIRE_QUANTIZED_CK_SIDECAR)

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
T load_unaligned(const uint8_t* source)
{
    T value{};
    std::memcpy(&value, source, sizeof(value));
    return value;
}

template <typename T>
void store_unaligned(uint8_t* destination, T value)
{
    std::memcpy(destination, &value, sizeof(value));
}

constexpr float kCodebook[8] = {-0.134860f,
                                -0.083320f,
                                -0.046469f,
                                -0.015176f,
                                0.015176f,
                                0.046469f,
                                0.083320f,
                                0.134860f};

float decode_k(const std::vector<uint8_t>& packed,
               int seqlen_k,
               int position,
               int kv_head,
               int dim)
{
    const int row_bytes = kKvHeads * kAsym3HeadBytes;
    const uint8_t* head = packed.data() + static_cast<size_t>(position) * row_bytes +
                          kv_head * kAsym3HeadBytes;
    const float cnorm = load_unaligned<float>(head);
    const int pack = dim / 8;
    const int lane = dim % 8;
    const uint8_t* bytes = head + 4 + pack * 3;
    const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16);
    (void)seqlen_k;
    return cnorm * kCodebook[(word >> (lane * 3)) & 7u];
}

float decode_v(const std::vector<uint8_t>& packed,
               int seqlen_k,
               int position,
               int kv_head,
               int dim)
{
    const int row_bytes = kKvHeads * kQ8HeadBytes;
    const uint8_t* head = packed.data() + static_cast<size_t>(position) * row_bytes +
                          kv_head * kQ8HeadBytes;
    const int block = dim / 32;
    const int lane = dim % 32;
    const uint8_t* q8 = head + block * 34;
    const float scale = __half2float(load_unaligned<__half>(q8));
    const int value = static_cast<int>(static_cast<int8_t>(q8[2 + lane]));
    (void)seqlen_k;
    return scale * value;
}

size_t dense_offset(int position, int head, int dim, int heads)
{
    return (static_cast<size_t>(position) * heads + head) * kHeadDim + dim;
}

void run_case(int seqlen_q, int seqlen_k)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const size_t q_values = static_cast<size_t>(seqlen_q) * kQueryHeads * kHeadDim;
    const size_t k_bytes = static_cast<size_t>(seqlen_k) * kKvHeads * kAsym3HeadBytes;
    const size_t v_bytes = static_cast<size_t>(seqlen_k) * kKvHeads * kQ8HeadBytes;
    std::vector<__half> q(q_values);
    std::vector<uint8_t> packed_k(k_bytes);
    std::vector<uint8_t> packed_v(v_bytes);
    std::vector<FmhaOutput> output(q_values);
    std::vector<float> expected(q_values);

    std::mt19937 rng(20260810 + seqlen_q + seqlen_k);
    std::uniform_real_distribution<float> q_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> cnorm_dist(0.25f, 3.0f);
    std::uniform_real_distribution<float> v_scale_dist(0.0005f, 0.02f);
    std::uniform_int_distribution<int> code_dist(0, 7);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    for(__half& value : q)
    {
        value = __float2half(q_dist(rng));
    }
    for(int position = 0; position < seqlen_k; ++position)
    {
        for(int head = 0; head < kKvHeads; ++head)
        {
            uint8_t* k_head = packed_k.data() +
                              (static_cast<size_t>(position) * kKvHeads + head) *
                                  kAsym3HeadBytes;
            store_unaligned(k_head, cnorm_dist(rng));
            for(int pack = 0; pack < kHeadDim / 8; ++pack)
            {
                uint32_t word = 0;
                for(int lane = 0; lane < 8; ++lane)
                {
                    word |= static_cast<uint32_t>(code_dist(rng)) << (lane * 3);
                }
                k_head[4 + pack * 3 + 0] = static_cast<uint8_t>(word);
                k_head[4 + pack * 3 + 1] = static_cast<uint8_t>(word >> 8);
                k_head[4 + pack * 3 + 2] = static_cast<uint8_t>(word >> 16);
            }
            uint8_t* v_head = packed_v.data() +
                              (static_cast<size_t>(position) * kKvHeads + head) * kQ8HeadBytes;
            for(int block = 0; block < kHeadDim / 32; ++block)
            {
                uint8_t* q8 = v_head + block * 34;
                store_unaligned(q8, __float2half(v_scale_dist(rng)));
                for(int lane = 0; lane < 32; ++lane)
                {
                    q8[2 + lane] =
                        static_cast<uint8_t>(static_cast<int8_t>(q8_dist(rng)));
                }
            }
        }
    }

    const int causal_shift = seqlen_k - seqlen_q;
    for(int query_position = 0; query_position < seqlen_q; ++query_position)
    {
        for(int query_head = 0; query_head < kQueryHeads; ++query_head)
        {
            const int kv_head = query_head / kQueriesPerKv;
            std::vector<float> scores(seqlen_k, -INFINITY);
            float maximum = -INFINITY;
            const int key_end = std::min(seqlen_k, query_position + causal_shift + 1);
            for(int key_position = 0; key_position < key_end; ++key_position)
            {
                float score = 0.0f;
                for(int dim = 0; dim < kHeadDim; ++dim)
                {
                    score += __half2float(q[dense_offset(
                                 query_position, query_head, dim, kQueryHeads)]) *
                             decode_k(packed_k, seqlen_k, key_position, kv_head, dim);
                }
                scores[key_position] = score * scale;
                maximum = std::max(maximum, scores[key_position]);
            }
            float denominator = 0.0f;
            for(int key_position = 0; key_position < key_end; ++key_position)
            {
                scores[key_position] = std::exp(scores[key_position] - maximum);
                denominator += scores[key_position];
            }
            for(int dim = 0; dim < kHeadDim; ++dim)
            {
                float value = 0.0f;
                for(int key_position = 0; key_position < key_end; ++key_position)
                {
                    value += scores[key_position] / denominator *
                             decode_v(packed_v, seqlen_k, key_position, kv_head, dim);
                }
                expected[dense_offset(query_position, query_head, dim, kQueryHeads)] = value;
            }
        }
    }

    __half* q_device = nullptr;
    uint8_t* k_device = nullptr;
    uint8_t* v_device = nullptr;
    FmhaOutput* out_device = nullptr;
    check_hip(hipMalloc(&q_device, q_values * sizeof(__half)), "hipMalloc(q)");
    check_hip(hipMalloc(&k_device, k_bytes), "hipMalloc(k)");
    check_hip(hipMalloc(&v_device, v_bytes), "hipMalloc(v)");
    check_hip(hipMalloc(&out_device, q_values * sizeof(FmhaOutput)), "hipMalloc(out)");
    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
    check_hip(hipMemcpyAsync(q_device,
                            q.data(),
                            q_values * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(q)");
    check_hip(hipMemcpyAsync(
                  k_device, packed_k.data(), k_bytes, hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(k)");
    check_hip(hipMemcpyAsync(
                  v_device, packed_v.data(), v_bytes, hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(v)");

    QuantizedFmhaKernel::Kargs args{
        reinterpret_cast<const ck_tile::half_t*>(q_device),
        k_device,
        v_device,
        out_device,
        seqlen_q,
        seqlen_k,
        1,
        kQueryHeads,
        kKvHeads,
        kQueryHeads * kHeadDim,
        kQueryHeads * kHeadDim,
        static_cast<ck_tile::long_index_t>(seqlen_q) * kQueryHeads * kHeadDim,
        static_cast<ck_tile::long_index_t>(seqlen_k) * kKvHeads * kAsym3HeadBytes,
        static_cast<ck_tile::long_index_t>(seqlen_k) * kKvHeads * kQ8HeadBytes,
        static_cast<ck_tile::long_index_t>(seqlen_q) * kQueryHeads * kHeadDim,
        kKvHeads * kAsym3HeadBytes,
        kKvHeads * kQ8HeadBytes,
#if CK_TILE_FMHA_FWD_FAST_EXP2
        static_cast<float>(scale * ck_tile::log2e_v<>),
#else
        scale,
#endif
    };
    ck_tile::stream_config stream_config{stream, true, 0, 5, 20, true};
    const float elapsed = ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<QuantizedFmhaKernel::kBlockPerCu, ck_tile::gfx11_t>(
            QuantizedFmhaKernel{},
            QuantizedFmhaKernel::GridSize(args),
            QuantizedFmhaKernel::BlockSize(),
            0,
            args));
    if(elapsed < 0.0f)
    {
        std::fprintf(stderr, "quantized CK launch failed\n");
        std::exit(3);
    }
    check_hip(hipMemcpyAsync(output.data(),
                            out_device,
                            q_values * sizeof(FmhaOutput),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(out)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t worst = 0;
    for(size_t index = 0; index < q_values; ++index)
    {
        const float error = std::abs(output_to_float(output[index]) - expected[index]);
        if(error > max_abs)
        {
            max_abs = error;
            worst = index;
        }
        mean_abs += error;
    }
    mean_abs /= q_values;
    std::printf("case=ck-d256-asym3k-q8v-causal-gqa seqlen_q=%d seqlen_k=%d "
                "q_heads=%d kv_heads=%d gpu_ms=%.9g max_abs=%.9g mean_abs=%.9g worst=%zu "
                "actual=%.9g expected=%.9g\n",
                seqlen_q,
                seqlen_k,
                kQueryHeads,
                kKvHeads,
                elapsed,
                max_abs,
                mean_abs,
                worst,
                output_to_float(output[worst]),
                expected[worst]);
    if(max_abs > 0.025f)
    {
        std::exit(4);
    }

    check_hip(hipFree(q_device), "hipFree(q)");
    check_hip(hipFree(k_device), "hipFree(k)");
    check_hip(hipFree(v_device), "hipFree(v)");
    check_hip(hipFree(out_device), "hipFree(out)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
}

float time_q_rotation(hipStream_t stream,
                      const float* input,
                      __half* output,
                      const float* cos_theta,
                      const float* sin_theta,
                      int query_rows)
{
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(rotate start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(rotate stop)");
    for(int iteration = 0; iteration < 5; ++iteration)
    {
        hipLaunchKernelGGL(rotate_q_givens_f32_to_f16,
                           dim3(kQueryHeads * query_rows),
                           dim3(32),
                           0,
                           stream,
                           input,
                           output,
                           cos_theta,
                           sin_theta,
                           query_rows,
                           kQueryHeads);
    }
    check_hip(hipEventRecord(start, stream), "hipEventRecord(rotate start)");
    for(int iteration = 0; iteration < 100; ++iteration)
    {
        hipLaunchKernelGGL(rotate_q_givens_f32_to_f16,
                           dim3(kQueryHeads * query_rows),
                           dim3(32),
                           0,
                           stream,
                           input,
                           output,
                           cos_theta,
                           sin_theta,
                           query_rows,
                           kQueryHeads);
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(rotate stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(rotate)");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, start, stop), "hipEventElapsedTime(rotate)");
    check_hip(hipEventDestroy(start), "hipEventDestroy(rotate start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(rotate stop)");
    return elapsed / 100.0f;
}

float time_ck_production_path(
    hipStream_t stream,
    const hipfire_flash_attn_ck_quantized_prefill_params& params)
{
    auto launch = [&]() {
        char error[256]{};
        const int status =
            hipfire_flash_attn_ck_quantized_prefill(&params, error, sizeof(error));
        if(status != 0)
        {
            std::fprintf(stderr, "quantized CK ABI launch failed: %s\n", error);
            std::exit(3);
        }
    };

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(CK production start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(CK production stop)");
    for(int iteration = 0; iteration < 5; ++iteration)
    {
        launch();
    }
    check_hip(hipEventRecord(start, stream), "hipEventRecord(CK production start)");
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        launch();
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(CK production stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(CK production)");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, start, stop),
              "hipEventElapsedTime(CK production)");
    check_hip(hipEventDestroy(start), "hipEventDestroy(CK production start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(CK production stop)");
    return elapsed / 20.0f;
}

float time_native_asym3(hipStream_t stream,
                        const float* q,
                        const uint8_t* packed_k,
                        const uint8_t* packed_v,
                        float* partials,
                        float* output,
                        const int* positions,
                        const float* cos_theta,
                        const float* sin_theta,
                        int seqlen_k,
                        int query_rows,
                        int tile_size,
                        int max_tiles)
{
    auto launch = [&]() {
        hipLaunchKernelGGL(attention_flash_asym3_tile_batched,
                           dim3(kQueryHeads, max_tiles, query_rows),
                           dim3(32),
                           static_cast<size_t>(tile_size) * sizeof(float),
                           stream,
                           q,
                           packed_k,
                           packed_v,
                           partials,
                           positions,
                           cos_theta,
                           sin_theta,
                           nullptr,
                           kQueryHeads,
                           kKvHeads,
                           kHeadDim,
                           seqlen_k,
                           1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                           tile_size,
                           max_tiles,
                           0,
                           0,
                           0,
                           8);
        hipLaunchKernelGGL(attention_flash_asym_reduce_batched,
                           dim3(kQueryHeads, query_rows),
                           dim3(32),
                           0,
                           stream,
                           partials,
                           output,
                           positions,
                           kQueryHeads,
                           kHeadDim,
                           tile_size,
                           max_tiles,
                           0,
                           0,
                           0);
    };

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(native start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(native stop)");
    for(int iteration = 0; iteration < 5; ++iteration)
    {
        launch();
    }
    check_hip(hipEventRecord(start, stream), "hipEventRecord(native start)");
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        launch();
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(native stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(native)");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, start, stop), "hipEventElapsedTime(native)");
    check_hip(hipEventDestroy(start), "hipEventDestroy(native start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(native stop)");
    return elapsed / 20.0f;
}

void run_native_ab(int seqlen_q, int seqlen_k)
{
    constexpr int tile_size = 128;
    const int max_tiles = (seqlen_k + tile_size - 1) / tile_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const size_t q_values = static_cast<size_t>(seqlen_q) * kQueryHeads * kHeadDim;
    const size_t k_bytes = static_cast<size_t>(seqlen_k) * kKvHeads * kAsym3HeadBytes;
    const size_t v_bytes = static_cast<size_t>(seqlen_k) * kKvHeads * kQ8HeadBytes;
    const size_t partial_values =
        static_cast<size_t>(seqlen_q) * kQueryHeads * max_tiles * (2 + kHeadDim);

    std::vector<float> q(q_values);
    std::vector<float> cos_theta(kHeadDim / 2);
    std::vector<float> sin_theta(kHeadDim / 2);
    std::vector<uint8_t> packed_k(k_bytes);
    std::vector<uint8_t> packed_v(v_bytes);
    std::vector<float> ck_output(q_values);
    std::vector<float> native_output(q_values);

    std::mt19937 rng(20260811 + seqlen_k);
    std::uniform_real_distribution<float> q_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> angle_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> cnorm_dist(0.25f, 3.0f);
    std::uniform_real_distribution<float> v_scale_dist(0.0005f, 0.02f);
    std::uniform_int_distribution<int> code_dist(0, 7);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    for(float& value : q)
    {
        value = q_dist(rng);
    }
    for(int pair = 0; pair < kHeadDim / 2; ++pair)
    {
        const float angle = angle_dist(rng);
        cos_theta[pair] = std::cos(angle);
        sin_theta[pair] = std::sin(angle);
    }
    for(int position = 0; position < seqlen_k; ++position)
    {
        for(int head = 0; head < kKvHeads; ++head)
        {
            uint8_t* k_head = packed_k.data() +
                              (static_cast<size_t>(position) * kKvHeads + head) *
                                  kAsym3HeadBytes;
            store_unaligned(k_head, cnorm_dist(rng));
            for(int pack = 0; pack < kHeadDim / 8; ++pack)
            {
                uint32_t word = 0;
                for(int lane = 0; lane < 8; ++lane)
                {
                    word |= static_cast<uint32_t>(code_dist(rng)) << (lane * 3);
                }
                k_head[4 + pack * 3 + 0] = static_cast<uint8_t>(word);
                k_head[4 + pack * 3 + 1] = static_cast<uint8_t>(word >> 8);
                k_head[4 + pack * 3 + 2] = static_cast<uint8_t>(word >> 16);
            }
            uint8_t* v_head = packed_v.data() +
                              (static_cast<size_t>(position) * kKvHeads + head) * kQ8HeadBytes;
            for(int block = 0; block < kHeadDim / 32; ++block)
            {
                uint8_t* q8 = v_head + block * 34;
                store_unaligned(q8, __float2half(v_scale_dist(rng)));
                for(int lane = 0; lane < 32; ++lane)
                {
                    q8[2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q8_dist(rng)));
                }
            }
        }
    }

    float* q_device = nullptr;
    void* ck_workspace_device = nullptr;
    __half* q_rotated_device = nullptr;
    float* cos_device = nullptr;
    float* sin_device = nullptr;
    uint8_t* k_device = nullptr;
    uint8_t* v_device = nullptr;
    FmhaOutput* ck_out_device = nullptr;
    float* ck_float_out_device = nullptr;
    float* native_out_device = nullptr;
    float* partials_device = nullptr;
    int* positions_device = nullptr;
    check_hip(hipMalloc(&q_device, q_values * sizeof(float)), "hipMalloc(ab q)");
    const size_t ck_workspace_bytes =
        quantized_workspace_bytes(seqlen_q, kQueryHeads, kHeadDim);
    check_hip(hipMalloc(&ck_workspace_device, ck_workspace_bytes), "hipMalloc(ab CK workspace)");
    q_rotated_device = reinterpret_cast<__half*>(ck_workspace_device);
    check_hip(hipMalloc(&cos_device, cos_theta.size() * sizeof(float)), "hipMalloc(ab cos)");
    check_hip(hipMalloc(&sin_device, sin_theta.size() * sizeof(float)), "hipMalloc(ab sin)");
    check_hip(hipMalloc(&k_device, k_bytes), "hipMalloc(ab k)");
    check_hip(hipMalloc(&v_device, v_bytes), "hipMalloc(ab v)");
    ck_out_device = reinterpret_cast<FmhaOutput*>(q_rotated_device + q_values);
    if constexpr(std::is_same_v<FmhaOutput, float>)
    {
        ck_float_out_device = reinterpret_cast<float*>(ck_out_device);
    }
    else
    {
        check_hip(hipMalloc(&ck_float_out_device, q_values * sizeof(float)),
                  "hipMalloc(ab ck float out)");
    }
    check_hip(hipMalloc(&native_out_device, q_values * sizeof(float)), "hipMalloc(ab native out)");
    check_hip(hipMalloc(&partials_device, partial_values * sizeof(float)), "hipMalloc(ab partials)");
    check_hip(hipMalloc(&positions_device, seqlen_q * sizeof(int)), "hipMalloc(ab positions)");
    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate(ab)");
    std::vector<int> positions(seqlen_q);
    for(int row = 0; row < seqlen_q; ++row)
    {
        positions[row] = seqlen_k - seqlen_q + row;
    }
    check_hip(hipMemcpyAsync(q_device, q.data(), q_values * sizeof(float), hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab q)");
    check_hip(hipMemcpyAsync(cos_device, cos_theta.data(), cos_theta.size() * sizeof(float), hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab cos)");
    check_hip(hipMemcpyAsync(sin_device, sin_theta.data(), sin_theta.size() * sizeof(float), hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab sin)");
    check_hip(hipMemcpyAsync(k_device, packed_k.data(), k_bytes, hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab k)");
    check_hip(hipMemcpyAsync(v_device, packed_v.data(), v_bytes, hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab v)");
    check_hip(hipMemcpyAsync(positions_device, positions.data(), seqlen_q * sizeof(int), hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(ab position)");

    const float rotate_ms =
        time_q_rotation(stream, q_device, q_rotated_device, cos_device, sin_device, seqlen_q);
    QuantizedFmhaKernel::Kargs args{
        reinterpret_cast<const ck_tile::half_t*>(q_rotated_device),
        k_device,
        v_device,
        ck_out_device,
        seqlen_q,
        seqlen_k,
        1,
        kQueryHeads,
        kKvHeads,
        kQueryHeads * kHeadDim,
        kQueryHeads * kHeadDim,
        kQueryHeads * kHeadDim,
        static_cast<ck_tile::long_index_t>(seqlen_k) * kKvHeads * kAsym3HeadBytes,
        static_cast<ck_tile::long_index_t>(seqlen_k) * kKvHeads * kQ8HeadBytes,
        kQueryHeads * kHeadDim,
        kKvHeads * kAsym3HeadBytes,
        kKvHeads * kQ8HeadBytes,
#if CK_TILE_FMHA_FWD_FAST_EXP2
        static_cast<float>(scale * ck_tile::log2e_v<>),
#else
        scale,
#endif
    };
    ck_tile::stream_config stream_config{stream, true, 0, 5, 20, true};
    const float ck_ms = ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<QuantizedFmhaKernel::kBlockPerCu, ck_tile::gfx11_t>(
            QuantizedFmhaKernel{},
            QuantizedFmhaKernel::GridSize(args),
            QuantizedFmhaKernel::BlockSize(),
            0,
            args));
    hipfire_flash_attn_ck_quantized_prefill_params quantized_params{};
    quantized_params.abi_version = HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION;
    quantized_params.struct_size = sizeof(quantized_params);
    quantized_params.q = q_device;
    quantized_params.packed_k = k_device;
    quantized_params.packed_v = v_device;
    quantized_params.out = ck_float_out_device;
    quantized_params.workspace = ck_workspace_device;
    quantized_params.workspace_bytes = ck_workspace_bytes;
    quantized_params.cos_theta = cos_device;
    quantized_params.sin_theta = sin_device;
    quantized_params.stream = stream;
    quantized_params.softmax_scale = scale;
    quantized_params.seqlen_q = seqlen_q;
    quantized_params.seqlen_k = seqlen_k;
    quantized_params.nhead_q = kQueryHeads;
    quantized_params.nhead_k = kKvHeads;
    quantized_params.head_dim = kHeadDim;
    quantized_params.causal = 1;
    quantized_params.k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    quantized_params.v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    const float ck_total_ms = time_ck_production_path(stream, quantized_params);
    const float native_ms = time_native_asym3(stream,
                                               q_device,
                                               k_device,
                                               v_device,
                                               partials_device,
                                               native_out_device,
                                               positions_device,
                                               cos_device,
                                               sin_device,
                                               seqlen_k,
                                               seqlen_q,
                                               tile_size,
                                               max_tiles);
    check_hip(hipMemcpyAsync(ck_output.data(),
                            ck_float_out_device,
                            q_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(ab ck out)");
    check_hip(hipMemcpyAsync(native_output.data(), native_out_device, q_values * sizeof(float), hipMemcpyDeviceToHost, stream),
              "hipMemcpyAsync(ab native out)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(ab)");

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t index = 0; index < q_values; ++index)
    {
        const float error = std::abs(ck_output[index] - native_output[index]);
        max_abs = std::max(max_abs, error);
        mean_abs += error;
    }
    mean_abs /= q_values;
    std::printf("case=native-ab-d256-asym3k-q8v seqlen_q=%d seqlen_k=%d q_heads=%d kv_heads=%d "
                "rotate_ms=%.9g ck_ms=%.9g ck_total_ms=%.9g native_ms=%.9g speedup=%.9g "
                "max_abs=%.9g mean_abs=%.9g\n",
                seqlen_q,
                seqlen_k,
                kQueryHeads,
                kKvHeads,
                rotate_ms,
                ck_ms,
                ck_total_ms,
                native_ms,
                native_ms / ck_total_ms,
                max_abs,
                mean_abs);
    if(max_abs > 0.03f)
    {
        std::exit(5);
    }

    check_hip(hipFree(q_device), "hipFree(ab q)");
    check_hip(hipFree(ck_workspace_device), "hipFree(ab CK workspace)");
    check_hip(hipFree(cos_device), "hipFree(ab cos)");
    check_hip(hipFree(sin_device), "hipFree(ab sin)");
    check_hip(hipFree(k_device), "hipFree(ab k)");
    check_hip(hipFree(v_device), "hipFree(ab v)");
    if constexpr(!std::is_same_v<FmhaOutput, float>)
    {
        check_hip(hipFree(ck_float_out_device), "hipFree(ab ck float out)");
    }
    check_hip(hipFree(native_out_device), "hipFree(ab native out)");
    check_hip(hipFree(partials_device), "hipFree(ab partials)");
    check_hip(hipFree(positions_device), "hipFree(ab positions)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy(ab)");
}

} // namespace

int main(int argc, char** argv)
{
    if(argc >= 4 && std::string_view(argv[1]) == "--native-ab-qrows")
    {
        const int query_rows = std::atoi(argv[2]);
        if(query_rows <= 0)
        {
            std::fprintf(stderr, "query rows must be positive\n");
            return 2;
        }
        for(int index = 3; index < argc; ++index)
        {
            run_native_ab(query_rows, std::atoi(argv[index]));
        }
        return 0;
    }
    if(argc >= 3 && std::string_view(argv[1]) == "--native-ab")
    {
        for(int index = 2; index < argc; ++index)
        {
            run_native_ab(1, std::atoi(argv[index]));
        }
        return 0;
    }
    if(argc >= 3 && std::string_view(argv[1]) == "--native-ab-q16")
    {
        for(int index = 2; index < argc; ++index)
        {
            run_native_ab(16, std::atoi(argv[index]));
        }
        return 0;
    }
    run_case(64, 96);
    run_case(73, 137);
    return 0;
}

#else

} // namespace

#endif
