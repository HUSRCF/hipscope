// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" __global__ void kv_cache_write_asym_k_givens3_batched(
    unsigned char* k_dst,
    const float* k_src,
    const int* positions,
    const float* cos_theta,
    const float* sin_theta,
    int n_kv_heads,
    int head_dim,
    int batch_size);

extern "C" __global__ void kv_cache_write_q8_0_batched(unsigned char* dst,
                                                        const float* src,
                                                        const int* positions,
                                                        int n_kv_heads,
                                                        int head_dim,
                                                        int batch_size);

namespace {

constexpr int kHeadDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr int kAsym3ValuesPerPack = 8;
constexpr int kAsym3BytesPerPack = 3;
constexpr int kAsym3HeadBytes = 4 + (kHeadDim * 3) / 8;
constexpr int kQ8BlockValues = 32;
constexpr int kQ8BlockBytes = 2 + kQ8BlockValues;
constexpr int kQ8BlocksPerHead = kHeadDim / kQ8BlockValues;
constexpr int kQ8HeadBytes = kQ8BlocksPerHead * kQ8BlockBytes;
constexpr int kTileRows = 64;
constexpr int kTileDims = 32;

static_assert(kAsym3HeadBytes == 100);
static_assert(kQ8HeadBytes == 272);
static_assert(kQueriesPerKv == 6);

constexpr float kAsym3Codebook[8] = {
    -0.134860f,
    -0.083320f,
    -0.046469f,
    -0.015176f,
    0.015176f,
    0.046469f,
    0.083320f,
    0.134860f,
};

__device__ __constant__ float kAsym3CodebookDevice[8] = {
    -0.134860f,
    -0.083320f,
    -0.046469f,
    -0.015176f,
    0.015176f,
    0.046469f,
    0.083320f,
    0.134860f,
};

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

__device__ float load_asym3_k(const uint8_t* packed,
                              int position,
                              int kv_head,
                              int dim,
                              int k_row_stride_bytes)
{
    const uint8_t* head = packed + static_cast<size_t>(position) * k_row_stride_bytes +
                          kv_head * kAsym3HeadBytes;
    const float cnorm = *reinterpret_cast<const float*>(head);
    const int pack = dim / kAsym3ValuesPerPack;
    const int lane = dim % kAsym3ValuesPerPack;
    const uint8_t* bytes = head + 4 + pack * kAsym3BytesPerPack;
    const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16);
    return cnorm * kAsym3CodebookDevice[(word >> (lane * 3)) & 7u];
}

__device__ float load_q8_v(const uint8_t* packed,
                           int position,
                           int kv_head,
                           int dim,
                           int v_row_stride_bytes)
{
    const uint8_t* head = packed + static_cast<size_t>(position) * v_row_stride_bytes +
                          kv_head * kQ8HeadBytes;
    const int block = dim / kQ8BlockValues;
    const int lane = dim % kQ8BlockValues;
    const uint8_t* q8 = head + block * kQ8BlockBytes;
    const float scale = __half2float(*reinterpret_cast<const __half*>(q8));
    const int value = static_cast<int>(*reinterpret_cast<const int8_t*>(q8 + 2 + lane));
    return scale * value;
}

// The launch shape mirrors the selected CK D256 instance: 64 rows per N tile
// and 32 dimensions per K tile. Query heads intentionally map onto four KV
// heads here so loader ABI errors cannot hide behind a one-to-one head layout.
__global__ void decode_quantized_kv_tiles(const uint8_t* __restrict__ packed_k,
                                          const uint8_t* __restrict__ packed_v,
                                          float* __restrict__ decoded_k,
                                          float* __restrict__ decoded_v,
                                          int seqlen,
                                          int k_row_stride_bytes,
                                          int v_row_stride_bytes)
{
    const int row_tile = blockIdx.x;
    const int query_head = blockIdx.y;
    const int dim_tile = blockIdx.z;
    const int kv_head = query_head / kQueriesPerKv;

    for(int linear = threadIdx.x; linear < kTileRows * kTileDims; linear += blockDim.x)
    {
        const int row = row_tile * kTileRows + linear / kTileDims;
        const int dim = dim_tile * kTileDims + linear % kTileDims;
        if(row >= seqlen)
        {
            continue;
        }
        const size_t output =
            (static_cast<size_t>(row) * kQueryHeads + query_head) * kHeadDim + dim;
        decoded_k[output] =
            load_asym3_k(packed_k, row, kv_head, dim, k_row_stride_bytes);
        decoded_v[output] = load_q8_v(packed_v, row, kv_head, dim, v_row_stride_bytes);
    }
}

// One work item owns one aligned 32-dimension slice. This matches one Q8 block
// and four Asym3 packs, so scale/cnorm and packed words are loaded once per
// batch instead of once per decoded scalar.
__global__ void decode_quantized_kv_tiles_vectorized(const uint8_t* __restrict__ packed_k,
                                                     const uint8_t* __restrict__ packed_v,
                                                     float* __restrict__ decoded_k,
                                                     float* __restrict__ decoded_v,
                                                     int seqlen,
                                                     int k_row_stride_bytes,
                                                     int v_row_stride_bytes)
{
    const int row_tile = blockIdx.x;
    const int query_head = blockIdx.y;
    const int block = blockIdx.z;
    const int kv_head = query_head / kQueriesPerKv;

    for(int row_in_tile = threadIdx.x; row_in_tile < kTileRows; row_in_tile += blockDim.x)
    {
        const int row = row_tile * kTileRows + row_in_tile;
        if(row >= seqlen)
        {
            continue;
        }

        const uint8_t* k_head = packed_k + static_cast<size_t>(row) * k_row_stride_bytes +
                                kv_head * kAsym3HeadBytes;
        const uint8_t* v_head = packed_v + static_cast<size_t>(row) * v_row_stride_bytes +
                                kv_head * kQ8HeadBytes;
        const float cnorm = *reinterpret_cast<const float*>(k_head);
        const uint8_t* q8 = v_head + block * kQ8BlockBytes;
        const float v_scale = __half2float(*reinterpret_cast<const __half*>(q8));
        const int dim_base = block * kQ8BlockValues;
        const size_t output_base =
            (static_cast<size_t>(row) * kQueryHeads + query_head) * kHeadDim + dim_base;

#pragma unroll
        for(int pack_in_block = 0; pack_in_block < 4; ++pack_in_block)
        {
            const int pack = block * 4 + pack_in_block;
            const uint8_t* bytes = k_head + 4 + pack * kAsym3BytesPerPack;
            const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                                  (static_cast<uint32_t>(bytes[1]) << 8) |
                                  (static_cast<uint32_t>(bytes[2]) << 16);
#pragma unroll
            for(int lane = 0; lane < kAsym3ValuesPerPack; ++lane)
            {
                const int local_dim = pack_in_block * kAsym3ValuesPerPack + lane;
                decoded_k[output_base + local_dim] =
                    cnorm * kAsym3CodebookDevice[(word >> (lane * 3)) & 7u];
                const int value = static_cast<int>(
                    *reinterpret_cast<const int8_t*>(q8 + 2 + local_dim));
                decoded_v[output_base + local_dim] = v_scale * value;
            }
        }
    }
}

size_t dense_offset(int position, int query_head, int dim)
{
    return (static_cast<size_t>(position) * kQueryHeads + query_head) * kHeadDim + dim;
}

void pack_test_data(std::vector<uint8_t>& packed_k,
                    std::vector<uint8_t>& packed_v,
                    std::vector<float>& expected_k,
                    std::vector<float>& expected_v,
                    int seqlen,
                    int k_row_stride_bytes,
                    int v_row_stride_bytes)
{
    std::mt19937 rng(20260808);
    std::uniform_real_distribution<float> cnorm_dist(0.25f, 9.0f);
    std::uniform_real_distribution<float> scale_dist(0.0005f, 0.08f);
    std::uniform_int_distribution<int> code_dist(0, 7);
    std::uniform_int_distribution<int> q8_dist(-127, 127);

    for(int position = 0; position < seqlen; ++position)
    {
        for(int kv_head = 0; kv_head < kKvHeads; ++kv_head)
        {
            uint8_t* k_head = packed_k.data() +
                              static_cast<size_t>(position) * k_row_stride_bytes +
                              kv_head * kAsym3HeadBytes;
            const float cnorm = cnorm_dist(rng);
            store_unaligned(k_head, cnorm);
            for(int pack = 0; pack < kHeadDim / kAsym3ValuesPerPack; ++pack)
            {
                uint32_t word = 0;
                for(int lane = 0; lane < kAsym3ValuesPerPack; ++lane)
                {
                    word |= static_cast<uint32_t>(code_dist(rng)) << (lane * 3);
                }
                k_head[4 + pack * 3 + 0] = static_cast<uint8_t>(word);
                k_head[4 + pack * 3 + 1] = static_cast<uint8_t>(word >> 8);
                k_head[4 + pack * 3 + 2] = static_cast<uint8_t>(word >> 16);
            }

            uint8_t* v_head = packed_v.data() +
                              static_cast<size_t>(position) * v_row_stride_bytes +
                              kv_head * kQ8HeadBytes;
            for(int block = 0; block < kQ8BlocksPerHead; ++block)
            {
                uint8_t* q8 = v_head + block * kQ8BlockBytes;
                const __half scale = __float2half(scale_dist(rng));
                store_unaligned(q8, scale);
                for(int lane = 0; lane < kQ8BlockValues; ++lane)
                {
                    q8[2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q8_dist(rng)));
                }
            }
        }
    }

    for(int position = 0; position < seqlen; ++position)
    {
        for(int query_head = 0; query_head < kQueryHeads; ++query_head)
        {
            const int kv_head = query_head / kQueriesPerKv;
            const uint8_t* k_head = packed_k.data() +
                                    static_cast<size_t>(position) * k_row_stride_bytes +
                                    kv_head * kAsym3HeadBytes;
            const float cnorm = load_unaligned<float>(k_head);
            const uint8_t* v_head = packed_v.data() +
                                    static_cast<size_t>(position) * v_row_stride_bytes +
                                    kv_head * kQ8HeadBytes;
            for(int dim = 0; dim < kHeadDim; ++dim)
            {
                const int pack = dim / kAsym3ValuesPerPack;
                const int pack_lane = dim % kAsym3ValuesPerPack;
                const uint8_t* bytes = k_head + 4 + pack * kAsym3BytesPerPack;
                const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                                      (static_cast<uint32_t>(bytes[1]) << 8) |
                                      (static_cast<uint32_t>(bytes[2]) << 16);
                expected_k[dense_offset(position, query_head, dim)] =
                    cnorm * kAsym3Codebook[(word >> (pack_lane * 3)) & 7u];

                const int block = dim / kQ8BlockValues;
                const int block_lane = dim % kQ8BlockValues;
                const uint8_t* q8 = v_head + block * kQ8BlockBytes;
                const float scale = __half2float(load_unaligned<__half>(q8));
                const int value =
                    static_cast<int>(static_cast<int8_t>(q8[2 + block_lane]));
                expected_v[dense_offset(position, query_head, dim)] = scale * value;
            }
        }
    }
}

void compare(const char* name,
             const std::vector<float>& actual,
             const std::vector<float>& expected)
{
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t worst = 0;
    for(size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - expected[index]);
        if(error > max_abs)
        {
            max_abs = error;
            worst = index;
        }
        mean_abs += error;
    }
    mean_abs /= actual.size();
    std::printf("tensor=%s max_abs=%.9g mean_abs=%.9g worst_index=%zu actual=%.9g expected=%.9g\n",
                name,
                max_abs,
                mean_abs,
                worst,
                actual[worst],
                expected[worst]);
    if(max_abs > 1.0e-6f)
    {
        std::exit(4);
    }
}

void decode_packed_reference(const std::vector<uint8_t>& packed_k,
                             const std::vector<uint8_t>& packed_v,
                             std::vector<float>& expected_k,
                             std::vector<float>& expected_v,
                             int seqlen,
                             int k_row_stride_bytes,
                             int v_row_stride_bytes)
{
    for(int position = 0; position < seqlen; ++position)
    {
        for(int query_head = 0; query_head < kQueryHeads; ++query_head)
        {
            const int kv_head = query_head / kQueriesPerKv;
            const uint8_t* k_head = packed_k.data() +
                                    static_cast<size_t>(position) * k_row_stride_bytes +
                                    kv_head * kAsym3HeadBytes;
            const float cnorm = load_unaligned<float>(k_head);
            const uint8_t* v_head = packed_v.data() +
                                    static_cast<size_t>(position) * v_row_stride_bytes +
                                    kv_head * kQ8HeadBytes;
            for(int dim = 0; dim < kHeadDim; ++dim)
            {
                const int pack = dim / kAsym3ValuesPerPack;
                const int pack_lane = dim % kAsym3ValuesPerPack;
                const uint8_t* bytes = k_head + 4 + pack * kAsym3BytesPerPack;
                const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                                      (static_cast<uint32_t>(bytes[1]) << 8) |
                                      (static_cast<uint32_t>(bytes[2]) << 16);
                expected_k[dense_offset(position, query_head, dim)] =
                    cnorm * kAsym3Codebook[(word >> (pack_lane * 3)) & 7u];

                const int block = dim / kQ8BlockValues;
                const int block_lane = dim % kQ8BlockValues;
                const uint8_t* q8 = v_head + block * kQ8BlockBytes;
                const float scale = __half2float(load_unaligned<__half>(q8));
                const int value =
                    static_cast<int>(static_cast<int8_t>(q8[2 + block_lane]));
                expected_v[dense_offset(position, query_head, dim)] = scale * value;
            }
        }
    }
}

float time_decode_loader(bool vectorized,
                         dim3 grid,
                         hipStream_t stream,
                         const uint8_t* packed_k,
                         const uint8_t* packed_v,
                         float* decoded_k,
                         float* decoded_v,
                         int seqlen,
                         int k_row_stride_bytes,
                         int v_row_stride_bytes)
{
    auto launch = [&]() {
        if(vectorized)
        {
            hipLaunchKernelGGL(decode_quantized_kv_tiles_vectorized,
                               dim3(grid.x, grid.y, kHeadDim / kQ8BlockValues),
                               dim3(64),
                               0,
                               stream,
                               packed_k,
                               packed_v,
                               decoded_k,
                               decoded_v,
                               seqlen,
                               k_row_stride_bytes,
                               v_row_stride_bytes);
        }
        else
        {
            hipLaunchKernelGGL(decode_quantized_kv_tiles,
                               dim3(grid.x, grid.y, kHeadDim / kTileDims),
                               dim3(256),
                               0,
                               stream,
                               packed_k,
                               packed_v,
                               decoded_k,
                               decoded_v,
                               seqlen,
                               k_row_stride_bytes,
                               v_row_stride_bytes);
        }
    };
    for(int iteration = 0; iteration < 5; ++iteration)
    {
        launch();
    }
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(loader start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(loader stop)");
    check_hip(hipEventRecord(start, stream), "hipEventRecord(loader start)");
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        launch();
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(loader stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(loader)");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, start, stop), "hipEventElapsedTime(loader)");
    check_hip(hipEventDestroy(start), "hipEventDestroy(loader start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(loader stop)");
    return elapsed / 20.0f;
}

void run_case(int seqlen)
{
    const int k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    const int v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    const size_t k_bytes = static_cast<size_t>(seqlen) * k_row_stride_bytes;
    const size_t v_bytes = static_cast<size_t>(seqlen) * v_row_stride_bytes;
    const size_t dense_values =
        static_cast<size_t>(seqlen) * kQueryHeads * kHeadDim;
    std::vector<uint8_t> packed_k(k_bytes);
    std::vector<uint8_t> packed_v(v_bytes);
    std::vector<float> expected_k(dense_values);
    std::vector<float> expected_v(dense_values);
    std::vector<float> decoded_k(dense_values);
    std::vector<float> decoded_v(dense_values);
    std::vector<float> vector_k(dense_values);
    std::vector<float> vector_v(dense_values);
    pack_test_data(packed_k,
                   packed_v,
                   expected_k,
                   expected_v,
                   seqlen,
                   k_row_stride_bytes,
                   v_row_stride_bytes);

    uint8_t* k_device = nullptr;
    uint8_t* v_device = nullptr;
    float* decoded_k_device = nullptr;
    float* decoded_v_device = nullptr;
    float* vector_k_device = nullptr;
    float* vector_v_device = nullptr;
    check_hip(hipMalloc(&k_device, k_bytes), "hipMalloc(k)");
    check_hip(hipMalloc(&v_device, v_bytes), "hipMalloc(v)");
    check_hip(hipMalloc(&decoded_k_device, dense_values * sizeof(float)),
              "hipMalloc(decoded_k)");
    check_hip(hipMalloc(&decoded_v_device, dense_values * sizeof(float)),
              "hipMalloc(decoded_v)");
    check_hip(hipMalloc(&vector_k_device, dense_values * sizeof(float)),
              "hipMalloc(vector_k)");
    check_hip(hipMalloc(&vector_v_device, dense_values * sizeof(float)),
              "hipMalloc(vector_v)");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
    check_hip(hipMemcpyAsync(k_device,
                            packed_k.data(),
                            k_bytes,
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(k)");
    check_hip(hipMemcpyAsync(v_device,
                            packed_v.data(),
                            v_bytes,
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(v)");

    const dim3 grid((seqlen + kTileRows - 1) / kTileRows, kQueryHeads, 1);
    const float scalar_ms = time_decode_loader(false,
                                               grid,
                                               stream,
                                               k_device,
                                               v_device,
                                               decoded_k_device,
                                               decoded_v_device,
                                               seqlen,
                                               k_row_stride_bytes,
                                               v_row_stride_bytes);
    const float vector_ms = time_decode_loader(true,
                                               grid,
                                               stream,
                                               k_device,
                                               v_device,
                                               vector_k_device,
                                               vector_v_device,
                                               seqlen,
                                               k_row_stride_bytes,
                                               v_row_stride_bytes);
    check_hip(hipGetLastError(), "decode loader launch");
    check_hip(hipMemcpyAsync(decoded_k.data(),
                            decoded_k_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(decoded_k)");
    check_hip(hipMemcpyAsync(decoded_v.data(),
                            decoded_v_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(decoded_v)");
    check_hip(hipMemcpyAsync(vector_k.data(),
                            vector_k_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(vector_k)");
    check_hip(hipMemcpyAsync(vector_v.data(),
                            vector_v_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(vector_v)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");

    std::printf("case=asym3-k-q8-v seqlen=%d head_dim=%d q_heads=%d kv_heads=%d "
                "k_head_bytes=%d v_head_bytes=%d tile=%dx%d stream=non-default "
                "scalar_ms=%.9g vector_ms=%.9g speedup=%.9g\n",
                seqlen,
                kHeadDim,
                kQueryHeads,
                kKvHeads,
                kAsym3HeadBytes,
                kQ8HeadBytes,
                kTileRows,
                kTileDims,
                scalar_ms,
                vector_ms,
                scalar_ms / vector_ms);
    compare("asym3_k", decoded_k, expected_k);
    compare("q8_v", decoded_v, expected_v);
    compare("asym3_k_vectorized", vector_k, expected_k);
    compare("q8_v_vectorized", vector_v, expected_v);

    check_hip(hipFree(k_device), "hipFree(k)");
    check_hip(hipFree(v_device), "hipFree(v)");
    check_hip(hipFree(decoded_k_device), "hipFree(decoded_k)");
    check_hip(hipFree(decoded_v_device), "hipFree(decoded_v)");
    check_hip(hipFree(vector_k_device), "hipFree(vector_k)");
    check_hip(hipFree(vector_v_device), "hipFree(vector_v)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
}

void run_writer_contract_case()
{
    constexpr int seqlen = 73;
    constexpr int batch_size = seqlen;
    constexpr int kv_dim = kKvHeads * kHeadDim;
    constexpr int angle_count = kHeadDim / 2;
    const int k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    const int v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    const size_t k_bytes = static_cast<size_t>(seqlen) * k_row_stride_bytes;
    const size_t v_bytes = static_cast<size_t>(seqlen) * v_row_stride_bytes;
    const size_t source_values = static_cast<size_t>(batch_size) * kv_dim;
    const size_t dense_values =
        static_cast<size_t>(seqlen) * kQueryHeads * kHeadDim;

    std::mt19937 rng(20260809);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);
    std::vector<float> source_k(source_values);
    std::vector<float> source_v(source_values);
    for(float& value : source_k)
    {
        value = value_dist(rng);
    }
    for(float& value : source_v)
    {
        value = value_dist(rng);
    }
    std::vector<int> positions(batch_size);
    for(int index = 0; index < batch_size; ++index)
    {
        positions[index] = (index * 17) % seqlen;
    }
    std::vector<float> cos_theta(angle_count);
    std::vector<float> sin_theta(angle_count);
    for(int index = 0; index < angle_count; ++index)
    {
        const float angle = 0.001f * static_cast<float>(index + 1);
        cos_theta[index] = std::cos(angle);
        sin_theta[index] = std::sin(angle);
    }

    std::vector<uint8_t> packed_k(k_bytes, 0);
    std::vector<uint8_t> packed_v(v_bytes, 0);
    std::vector<float> expected_k(dense_values);
    std::vector<float> expected_v(dense_values);
    std::vector<float> decoded_k(dense_values);
    std::vector<float> decoded_v(dense_values);

    float* source_k_device = nullptr;
    float* source_v_device = nullptr;
    int* positions_device = nullptr;
    float* cos_device = nullptr;
    float* sin_device = nullptr;
    uint8_t* packed_k_device = nullptr;
    uint8_t* packed_v_device = nullptr;
    float* decoded_k_device = nullptr;
    float* decoded_v_device = nullptr;
    check_hip(hipMalloc(&source_k_device, source_values * sizeof(float)),
              "hipMalloc(source_k)");
    check_hip(hipMalloc(&source_v_device, source_values * sizeof(float)),
              "hipMalloc(source_v)");
    check_hip(hipMalloc(&positions_device, positions.size() * sizeof(int)),
              "hipMalloc(positions)");
    check_hip(hipMalloc(&cos_device, cos_theta.size() * sizeof(float)),
              "hipMalloc(cos_theta)");
    check_hip(hipMalloc(&sin_device, sin_theta.size() * sizeof(float)),
              "hipMalloc(sin_theta)");
    check_hip(hipMalloc(&packed_k_device, k_bytes), "hipMalloc(packed_k)");
    check_hip(hipMalloc(&packed_v_device, v_bytes), "hipMalloc(packed_v)");
    check_hip(hipMalloc(&decoded_k_device, dense_values * sizeof(float)),
              "hipMalloc(decoded_k)");
    check_hip(hipMalloc(&decoded_v_device, dense_values * sizeof(float)),
              "hipMalloc(decoded_v)");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
    check_hip(hipMemcpyAsync(source_k_device,
                            source_k.data(),
                            source_values * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(source_k)");
    check_hip(hipMemcpyAsync(source_v_device,
                            source_v.data(),
                            source_values * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(source_v)");
    check_hip(hipMemcpyAsync(positions_device,
                            positions.data(),
                            positions.size() * sizeof(int),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(positions)");
    check_hip(hipMemcpyAsync(cos_device,
                            cos_theta.data(),
                            cos_theta.size() * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(cos_theta)");
    check_hip(hipMemcpyAsync(sin_device,
                            sin_theta.data(),
                            sin_theta.size() * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(sin_theta)");
    check_hip(hipMemsetAsync(packed_k_device, 0, k_bytes, stream), "hipMemsetAsync(k)");
    check_hip(hipMemsetAsync(packed_v_device, 0, v_bytes, stream), "hipMemsetAsync(v)");

    const dim3 asym3_grid(kKvHeads, batch_size);
    constexpr size_t asym3_shared_bytes = (kHeadDim + 32) * sizeof(float);
    hipLaunchKernelGGL(kv_cache_write_asym_k_givens3_batched,
                       asym3_grid,
                       dim3(32),
                       asym3_shared_bytes,
                       stream,
                       packed_k_device,
                       source_k_device,
                       positions_device,
                       cos_device,
                       sin_device,
                       kKvHeads,
                       kHeadDim,
                       batch_size);
    check_hip(hipGetLastError(), "kv_cache_write_asym_k_givens3_batched launch");

    const dim3 q8_grid(kKvHeads * kQ8BlocksPerHead, batch_size);
    hipLaunchKernelGGL(kv_cache_write_q8_0_batched,
                       q8_grid,
                       dim3(32),
                       0,
                       stream,
                       packed_v_device,
                       source_v_device,
                       positions_device,
                       kKvHeads,
                       kHeadDim,
                       batch_size);
    check_hip(hipGetLastError(), "kv_cache_write_q8_0_batched launch");

    const dim3 decode_grid((seqlen + kTileRows - 1) / kTileRows,
                           kQueryHeads,
                           kHeadDim / kTileDims);
    hipLaunchKernelGGL(decode_quantized_kv_tiles,
                       decode_grid,
                       dim3(256),
                       0,
                       stream,
                       packed_k_device,
                       packed_v_device,
                       decoded_k_device,
                       decoded_v_device,
                       seqlen,
                       k_row_stride_bytes,
                       v_row_stride_bytes);
    check_hip(hipGetLastError(), "decode writer output launch");

    check_hip(hipMemcpyAsync(packed_k.data(),
                            packed_k_device,
                            k_bytes,
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(packed_k)");
    check_hip(hipMemcpyAsync(packed_v.data(),
                            packed_v_device,
                            v_bytes,
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(packed_v)");
    check_hip(hipMemcpyAsync(decoded_k.data(),
                            decoded_k_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(decoded_k)");
    check_hip(hipMemcpyAsync(decoded_v.data(),
                            decoded_v_device,
                            dense_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(decoded_v)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");

    decode_packed_reference(packed_k,
                            packed_v,
                            expected_k,
                            expected_v,
                            seqlen,
                            k_row_stride_bytes,
                            v_row_stride_bytes);
    std::printf("case=real-hipfire-writers seqlen=%d batch=%d position_order=permute17 "
                "head_dim=%d q_heads=%d kv_heads=%d stream=non-default\n",
                seqlen,
                batch_size,
                kHeadDim,
                kQueryHeads,
                kKvHeads);
    compare("writer_asym3_k", decoded_k, expected_k);
    compare("writer_q8_v", decoded_v, expected_v);

    check_hip(hipFree(source_k_device), "hipFree(source_k)");
    check_hip(hipFree(source_v_device), "hipFree(source_v)");
    check_hip(hipFree(positions_device), "hipFree(positions)");
    check_hip(hipFree(cos_device), "hipFree(cos_theta)");
    check_hip(hipFree(sin_device), "hipFree(sin_theta)");
    check_hip(hipFree(packed_k_device), "hipFree(packed_k)");
    check_hip(hipFree(packed_v_device), "hipFree(packed_v)");
    check_hip(hipFree(decoded_k_device), "hipFree(decoded_k)");
    check_hip(hipFree(decoded_v_device), "hipFree(decoded_v)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
}

} // namespace

int main()
{
    run_case(64);
    run_case(73);
    run_case(128);
    run_writer_contract_case();
    return 0;
}
