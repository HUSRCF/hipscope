// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hipfire::ck_attention {

inline constexpr int kPredecodeHeadDim = 256;
inline constexpr int kPredecodeKvHeads = 4;
inline constexpr int kPredecodeAsym3HeadBytes = 100;
inline constexpr int kPredecodeQ8HeadBytes = 272;

static __device__ __constant__ float kPredecodeAsym3Codebook[8] = {
    -0.134860f,
    -0.083320f,
    -0.046469f,
    -0.015176f,
    0.015176f,
    0.046469f,
    0.083320f,
    0.134860f,
};

// One Wave32 owns one physical KV row/head. Each lane decodes eight dimensions,
// so K emits one 3-byte pack per lane while V shares one Q8 scale per four lanes.
// The output is dense [position, kv_head, 256] FP16 for the mature CK pipeline.
static __global__ void predecode_asym3_k_q8_v_f16(
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ packed_v,
    __half* __restrict__ dense_k,
    __half* __restrict__ dense_v,
    int seqlen,
    int k_row_stride_bytes,
    int v_row_stride_bytes)
{
    const int row = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= seqlen || lane >= 32)
    {
        return;
    }

    const uint8_t* k_head = packed_k + static_cast<size_t>(row) * k_row_stride_bytes +
                            kv_head * kPredecodeAsym3HeadBytes;
    const uint8_t* v_head = packed_v + static_cast<size_t>(row) * v_row_stride_bytes +
                            kv_head * kPredecodeQ8HeadBytes;
    float cnorm = lane == 0 ? *reinterpret_cast<const float*>(k_head) : 0.0f;
    cnorm = __shfl(cnorm, 0, 32);

    const uint8_t* k_pack = k_head + 4 + lane * 3;
    const uint32_t k_word = static_cast<uint32_t>(k_pack[0]) |
                            (static_cast<uint32_t>(k_pack[1]) << 8) |
                            (static_cast<uint32_t>(k_pack[2]) << 16);

    const int v_block = lane / 4;
    const int lane_in_v_block = lane % 4;
    const uint8_t* q8 = v_head + v_block * 34;
    float v_scale = lane_in_v_block == 0
                        ? __half2float(*reinterpret_cast<const __half*>(q8))
                        : 0.0f;
    v_scale = __shfl(v_scale, lane - lane_in_v_block, 32);
    const int8_t* v_values = reinterpret_cast<const int8_t*>(q8 + 2) +
                             lane_in_v_block * 8;

    const size_t output =
        (static_cast<size_t>(row) * kPredecodeKvHeads + kv_head) *
            kPredecodeHeadDim +
        lane * 8;
#pragma unroll
    for(int index = 0; index < 8; ++index)
    {
        const uint32_t code = (k_word >> (index * 3)) & 7u;
        dense_k[output + index] =
            __float2half_rn(cnorm * kPredecodeAsym3Codebook[code]);
        dense_v[output + index] =
            __float2half_rn(v_scale * static_cast<float>(v_values[index]));
    }
}

static __device__ __forceinline__ uint32_t pack_half2_bits(__half low, __half high)
{
    return __builtin_bit_cast(uint32_t, __halves2half2(low, high));
}

// Exact sidecar probe: retain the scalar decode arithmetic but combine each
// lane's eight contiguous FP16 results into one 16-byte store per output plane.
static __global__ void predecode_asym3_k_q8_v_f16_packet_store(
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ packed_v,
    __half* __restrict__ dense_k,
    __half* __restrict__ dense_v,
    int seqlen,
    int k_row_stride_bytes,
    int v_row_stride_bytes)
{
    const int row = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= seqlen || lane >= 32)
    {
        return;
    }

    const uint8_t* k_head = packed_k + static_cast<size_t>(row) * k_row_stride_bytes +
                            kv_head * kPredecodeAsym3HeadBytes;
    const uint8_t* v_head = packed_v + static_cast<size_t>(row) * v_row_stride_bytes +
                            kv_head * kPredecodeQ8HeadBytes;
    float cnorm = lane == 0 ? *reinterpret_cast<const float*>(k_head) : 0.0f;
    cnorm = __shfl(cnorm, 0, 32);

    const uint8_t* k_pack = k_head + 4 + lane * 3;
    const uint32_t k_word = static_cast<uint32_t>(k_pack[0]) |
                            (static_cast<uint32_t>(k_pack[1]) << 8) |
                            (static_cast<uint32_t>(k_pack[2]) << 16);

    const int v_block = lane / 4;
    const int lane_in_v_block = lane % 4;
    const uint8_t* q8 = v_head + v_block * 34;
    float v_scale = lane_in_v_block == 0
                        ? __half2float(*reinterpret_cast<const __half*>(q8))
                        : 0.0f;
    v_scale = __shfl(v_scale, lane - lane_in_v_block, 32);
    const int8_t* v_values = reinterpret_cast<const int8_t*>(q8 + 2) +
                             lane_in_v_block * 8;

    __half k_values[8];
    __half v_results[8];
#pragma unroll
    for(int index = 0; index < 8; ++index)
    {
        const uint32_t code = (k_word >> (index * 3)) & 7u;
        k_values[index] = __float2half_rn(cnorm * kPredecodeAsym3Codebook[code]);
        v_results[index] =
            __float2half_rn(v_scale * static_cast<float>(v_values[index]));
    }

    const uint4 k_packet = make_uint4(pack_half2_bits(k_values[0], k_values[1]),
                                      pack_half2_bits(k_values[2], k_values[3]),
                                      pack_half2_bits(k_values[4], k_values[5]),
                                      pack_half2_bits(k_values[6], k_values[7]));
    const uint4 v_packet = make_uint4(pack_half2_bits(v_results[0], v_results[1]),
                                      pack_half2_bits(v_results[2], v_results[3]),
                                      pack_half2_bits(v_results[4], v_results[5]),
                                      pack_half2_bits(v_results[6], v_results[7]));
    const size_t output =
        (static_cast<size_t>(row) * kPredecodeKvHeads + kv_head) *
            kPredecodeHeadDim +
        lane * 8;
    *reinterpret_cast<uint4*>(dense_k + output) = k_packet;
    *reinterpret_cast<uint4*>(dense_v + output) = v_packet;
}

} // namespace hipfire::ck_attention
