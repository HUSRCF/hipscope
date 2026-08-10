// SPDX-License-Identifier: MIT OR Apache-2.0
// Compile/run probe for the gfx11 packed IU4 WMMA lane contract.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x8_t = int __attribute__((ext_vector_type(8)));

static __host__ __device__ inline uint8_t pack_nibbles(int lo, int hi) {
    return static_cast<uint8_t>((lo & 0xf) | ((hi & 0xf) << 4));
}

static __device__ inline int32x2_t make_a_fragment(int row) {
    int32x2_t fragment = {0, 0};
    auto* bytes = reinterpret_cast<uint8_t*>(&fragment);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const int k0 = 2 * pair;
        const int k1 = k0 + 1;
        bytes[pair] = pack_nibbles((row + k0) & 0xf, (row + k1) & 0xf);
    }
    return fragment;
}

static __device__ inline int32x2_t make_b_fragment(int row) {
    int32x2_t fragment = {0, 0};
    auto* bytes = reinterpret_cast<uint8_t*>(&fragment);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const int k0 = 2 * pair;
        const int k1 = k0 + 1;
        const int v0 = ((row + 2 * k0) % 15) - 7;
        const int v1 = ((row + 2 * k1) % 15) - 7;
        bytes[pair] = pack_nibbles(v0, v1);
    }
    return fragment;
}

extern "C" __global__ void iu4_wmma_contract_probe(int* output) {
    const int lane = threadIdx.x;
    const int matrix_row = lane & 15;
    const int half = lane >> 4;
    const int32x2_t a = make_a_fragment(matrix_row);
    const int32x2_t b = make_b_fragment(matrix_row);
    int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};

    c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, a, true, b, c, false);

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int i = 2 * l + half;
        const int j = matrix_row;
        output[i * 16 + j] = c[l];
    }
}

static int a_value(int row, int k) { return (row + k) & 0xf; }

static int b_value(int row, int k) { return ((row + 2 * k) % 15) - 7; }

int main() {
    int* device_output = nullptr;
    if (hipMalloc(&device_output, 256 * sizeof(int)) != hipSuccess) return 2;
    if (hipMemset(device_output, 0, 256 * sizeof(int)) != hipSuccess) return 3;
    iu4_wmma_contract_probe<<<1, 32>>>(device_output);
    if (hipDeviceSynchronize() != hipSuccess) return 4;

    int output[256] = {};
    if (hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost) != hipSuccess) {
        return 5;
    }
    if (hipFree(device_output) != hipSuccess) return 6;

    int mismatches = 0;
    int max_abs = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int expected = 0;
            for (int k = 0; k < 16; ++k) {
                expected += a_value(i, k) * b_value(j, k);
            }
            const int delta = std::abs(output[i * 16 + j] - expected);
            if (delta != 0) {
                if (mismatches < 8) {
                    std::printf("mismatch i=%d j=%d got=%d expected=%d\n",
                                i, j, output[i * 16 + j], expected);
                }
                ++mismatches;
                if (delta > max_abs) max_abs = delta;
            }
        }
    }
    std::printf("mismatches=%d max_abs=%d\n", mismatches, max_abs);
    return mismatches == 0 ? 0 : 1;
}
