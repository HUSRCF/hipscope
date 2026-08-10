// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "quantized_kv_predecode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kKvHeads = 4;
constexpr int kAsym3ValuesPerPack = 8;
constexpr int kAsym3BytesPerPack = 3;
constexpr int kAsym3HeadBytes = 4 + (kHeadDim * 3) / 8;
constexpr int kQ8BlockValues = 32;
constexpr int kQ8BlockBytes = 2 + kQ8BlockValues;
constexpr int kQ8BlocksPerHead = kHeadDim / kQ8BlockValues;
constexpr int kQ8HeadBytes = kQ8BlocksPerHead * kQ8BlockBytes;

static_assert(kAsym3HeadBytes == 100);
static_assert(kQ8HeadBytes == 272);

constexpr float kAsym3CodebookHost[8] = {
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
        std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
void store_unaligned(uint8_t* destination, T value)
{
    std::memcpy(destination, &value, sizeof(value));
}

template <typename T>
T load_unaligned(const uint8_t* source)
{
    T value{};
    std::memcpy(&value, source, sizeof(value));
    return value;
}

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void make_packed(std::vector<uint8_t>& packed_k,
                 std::vector<uint8_t>& packed_v,
                 std::vector<__half>& expected_k,
                 std::vector<__half>& expected_v,
                 int seqlen)
{
    const int k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    const int v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    for(int row = 0; row < seqlen; ++row)
    {
        for(int head = 0; head < kKvHeads; ++head)
        {
            uint8_t* k_head = packed_k.data() +
                              static_cast<size_t>(row) * k_row_stride_bytes +
                              head * kAsym3HeadBytes;
            uint8_t* v_head = packed_v.data() +
                              static_cast<size_t>(row) * v_row_stride_bytes +
                              head * kQ8HeadBytes;
            const float cnorm = 0.25f + static_cast<float>((row + head * 7) % 19) * 0.01f;
            store_unaligned(k_head, cnorm);
            for(int pack = 0; pack < kHeadDim / kAsym3ValuesPerPack; ++pack)
            {
                uint32_t word = 0;
                for(int index = 0; index < kAsym3ValuesPerPack; ++index)
                {
                    const int dim = pack * kAsym3ValuesPerPack + index;
                    const uint32_t code = static_cast<uint32_t>(
                        (row * 13 + head * 17 + dim * 5) & 7);
                    word |= code << (index * 3);
                    const size_t output =
                        (static_cast<size_t>(row) * kKvHeads + head) * kHeadDim + dim;
                    expected_k[output] =
                        __float2half(cnorm * kAsym3CodebookHost[code]);
                }
                uint8_t* bytes = k_head + 4 + pack * kAsym3BytesPerPack;
                bytes[0] = static_cast<uint8_t>(word);
                bytes[1] = static_cast<uint8_t>(word >> 8);
                bytes[2] = static_cast<uint8_t>(word >> 16);
            }
            for(int block = 0; block < kQ8BlocksPerHead; ++block)
            {
                uint8_t* q8 = v_head + block * kQ8BlockBytes;
                const float scale = 0.01f + static_cast<float>((row + head + block) % 7) * 0.002f;
                store_unaligned(q8, __float2half(scale));
                for(int index = 0; index < kQ8BlockValues; ++index)
                {
                    const int dim = block * kQ8BlockValues + index;
                    const int8_t value = static_cast<int8_t>(
                        ((row * 11 + head * 23 + dim * 3) % 127) - 63);
                    q8[2 + index] = static_cast<uint8_t>(value);
                    const size_t output =
                        (static_cast<size_t>(row) * kKvHeads + head) * kHeadDim + dim;
                    expected_v[output] = __float2half(scale * static_cast<float>(value));
                }
            }
        }
    }
}

void compare(const char* label,
             const std::vector<__half>& actual,
             const std::vector<__half>& expected)
{
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    for(size_t index = 0; index < actual.size(); ++index)
    {
        const float error =
            std::abs(__half2float(actual[index]) - __half2float(expected[index]));
        max_abs = std::max(max_abs, error);
        sum_abs += error;
    }
    std::printf("%s_max_abs=%.9g %s_mean_abs=%.9g\n",
                label,
                max_abs,
                label,
                sum_abs / static_cast<double>(actual.size()));
    if(max_abs > 1.0e-3f)
    {
        std::fprintf(stderr, "%s correctness failed\n", label);
        std::exit(3);
    }
}

#ifndef HIPFIRE_PREDECODE_LIBRARY_ONLY

void run_case(int seqlen)
{
    const int k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    const int v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    const size_t k_bytes = static_cast<size_t>(seqlen) * k_row_stride_bytes;
    const size_t v_bytes = static_cast<size_t>(seqlen) * v_row_stride_bytes;
    const size_t dense_values = static_cast<size_t>(seqlen) * kKvHeads * kHeadDim;

    std::vector<uint8_t> packed_k(k_bytes);
    std::vector<uint8_t> packed_v(v_bytes);
    std::vector<__half> expected_k(dense_values);
    std::vector<__half> expected_v(dense_values);
    std::vector<__half> actual_k(dense_values);
    std::vector<__half> actual_v(dense_values);
    make_packed(packed_k, packed_v, expected_k, expected_v, seqlen);

    uint8_t* packed_k_device = nullptr;
    uint8_t* packed_v_device = nullptr;
    __half* dense_k_device = nullptr;
    __half* dense_v_device = nullptr;
    check_hip(hipMalloc(&packed_k_device, k_bytes), "hipMalloc(packed K)");
    check_hip(hipMalloc(&packed_v_device, v_bytes), "hipMalloc(packed V)");
    check_hip(hipMalloc(&dense_k_device, dense_values * sizeof(__half)),
              "hipMalloc(dense K)");
    check_hip(hipMalloc(&dense_v_device, dense_values * sizeof(__half)),
              "hipMalloc(dense V)");
    check_hip(hipMemcpy(packed_k_device,
                        packed_k.data(),
                        k_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy(packed K)");
    check_hip(hipMemcpy(packed_v_device,
                        packed_v.data(),
                        v_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy(packed V)");

    const auto launch = [&]() {
        hipLaunchKernelGGL(hipfire::ck_attention::predecode_asym3_k_q8_v_f16,
                           dim3(seqlen, kKvHeads),
                           dim3(32),
                           0,
                           nullptr,
                           packed_k_device,
                           packed_v_device,
                           dense_k_device,
                           dense_v_device,
                           seqlen,
                           k_row_stride_bytes,
                           v_row_stride_bytes);
    };
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        launch();
    }
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(warmup)");

    constexpr int trials = 9;
    constexpr int iterations = 100;
    std::vector<float> samples;
    samples.reserve(trials);
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    for(int trial = 0; trial < trials; ++trial)
    {
        check_hip(hipEventRecord(start), "hipEventRecord(start)");
        for(int iteration = 0; iteration < iterations; ++iteration)
        {
            launch();
        }
        check_hip(hipEventRecord(stop), "hipEventRecord(stop)");
        check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        check_hip(hipEventElapsedTime(&elapsed_ms, start, stop),
                  "hipEventElapsedTime");
        samples.push_back(elapsed_ms / static_cast<float>(iterations));
    }
    check_hip(hipGetLastError(), "predecode launch");
    check_hip(hipMemcpy(actual_k.data(),
                        dense_k_device,
                        dense_values * sizeof(__half),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(dense K)");
    check_hip(hipMemcpy(actual_v.data(),
                        dense_v_device,
                        dense_values * sizeof(__half),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(dense V)");

    std::printf("seqlen=%d,kv_heads=%d,head_dim=%d,packed_bytes=%zu,dense_bytes=%zu,"
                "predecode_f16_ms=%.9g,raw_ms=",
                seqlen,
                kKvHeads,
                kHeadDim,
                k_bytes + v_bytes,
                dense_values * sizeof(__half) * 2,
                median(samples));
    for(size_t index = 0; index < samples.size(); ++index)
    {
        std::printf("%s%.9g", index == 0 ? "" : ";", samples[index]);
    }
    std::putchar('\n');
    compare("dense_k", actual_k, expected_k);
    compare("dense_v", actual_v, expected_v);

    check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");
    check_hip(hipFree(packed_k_device), "hipFree(packed K)");
    check_hip(hipFree(packed_v_device), "hipFree(packed V)");
    check_hip(hipFree(dense_k_device), "hipFree(dense K)");
    check_hip(hipFree(dense_v_device), "hipFree(dense V)");
}

#endif

} // namespace

#ifndef HIPFIRE_PREDECODE_LIBRARY_ONLY

int main()
{
    for(const int seqlen : {73, 2048, 4096, 6144, 8192})
    {
        run_case(seqlen);
    }
    return 0;
}

#endif
