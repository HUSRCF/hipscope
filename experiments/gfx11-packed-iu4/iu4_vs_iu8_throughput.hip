// SPDX-License-Identifier: MIT OR Apache-2.0
// Latency-hidden instruction-throughput comparison for gfx11 IU4 and IU8 WMMA.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x4_t = int __attribute__((ext_vector_type(4)));
using int32x8_t = int __attribute__((ext_vector_type(8)));

extern "C" __global__ void iu4_chain(int* output, int iterations) {
    const int lane = threadIdx.x;
    const int32x2_t a = {0x12345670 + lane, 0x23456701 + lane};
    const int32x2_t b = {0x76543210 - lane, 0x65432107 - lane};
    int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t c1 = c0;
    int32x8_t c2 = c0;
    int32x8_t c3 = c0;
    for (int i = 0; i < iterations; i += 4) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c3, false);
    }
    output[blockIdx.x * blockDim.x + lane] = c0[0] + c1[1] + c2[2] + c3[3];
}

extern "C" __global__ void iu8_chain(int* output, int iterations) {
    const int lane = threadIdx.x;
    const int32x4_t a = {
        0x01020304 + lane, 0x02030401 + lane,
        0x03040102 + lane, 0x04010203 + lane,
    };
    const int32x4_t b = {
        0x04030201 - lane, 0x03020104 - lane,
        0x02010403 - lane, 0x01040302 - lane,
    };
    int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t c1 = c0;
    int32x8_t c2 = c0;
    int32x8_t c3 = c0;
    for (int i = 0; i < iterations; i += 4) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3, false);
    }
    output[blockIdx.x * blockDim.x + lane] = c0[0] + c1[1] + c2[2] + c3[3];
}

template <typename Launch>
static float median_ms(Launch launch, int trials) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        std::exit(2);
    }
    for (int i = 0; i < 4; ++i) launch();
    if (hipDeviceSynchronize() != hipSuccess) std::exit(3);
    std::vector<float> times;
    times.reserve(trials);
    for (int i = 0; i < trials; ++i) {
        if (hipEventRecord(start) != hipSuccess) std::exit(4);
        launch();
        if (hipEventRecord(stop) != hipSuccess || hipEventSynchronize(stop) != hipSuccess) {
            std::exit(5);
        }
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) std::exit(6);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    if (hipEventDestroy(start) != hipSuccess || hipEventDestroy(stop) != hipSuccess) {
        std::exit(7);
    }
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 21;
    if (iterations <= 0 || iterations % 4 != 0) return 8;
    int* output = nullptr;
    if (hipMalloc(&output, blocks * 32 * sizeof(int)) != hipSuccess) return 9;
    const dim3 grid(blocks);
    const dim3 block(32);
    const float iu8_ms = median_ms(
        [&] { iu8_chain<<<grid, block>>>(output, iterations); }, trials);
    const float iu4_ms = median_ms(
        [&] { iu4_chain<<<grid, block>>>(output, iterations); }, trials);
    if (hipFree(output) != hipSuccess) return 10;
    std::printf("blocks=%d iterations=%d trials=%d\n", blocks, iterations, trials);
    std::printf("iu8_ms=%.6f\n", iu8_ms);
    std::printf("iu4_ms=%.6f\n", iu4_ms);
    std::printf("iu4_instruction_speedup=%.6fx\n", iu8_ms / iu4_ms);
    std::printf("exact_q8_two_pass_estimate=%.6fx\n", iu8_ms / (2.0f * iu4_ms));
    return 0;
}
