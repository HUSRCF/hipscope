// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck_quantized.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kRows = 3;
constexpr int kHeads = 4;
constexpr int kDim = 256;
constexpr int kKHeadBytes = 132;
constexpr int kVHeadBytes = 272;
constexpr float kC4[16] = {
    -0.241565f, -0.182875f, -0.143012f, -0.111016f,
    -0.083262f, -0.057983f, -0.034295f, -0.011225f,
     0.011225f,  0.034295f,  0.057983f,  0.083262f,
     0.111016f,  0.143012f,  0.182875f,  0.241565f,
};

void check(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

uint16_t bits(__half value)
{
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace

int main()
{
    const size_t values = static_cast<size_t>(kRows) * kHeads * kDim;
    std::vector<uint8_t> packed_k(static_cast<size_t>(kRows) * kHeads * kKHeadBytes);
    std::vector<uint8_t> packed_v(static_cast<size_t>(kRows) * kHeads * kVHeadBytes);
    std::vector<__half> expected_k(values);
    std::vector<__half> expected_v(values);

    for(int row = 0; row < kRows; ++row)
    {
        for(int head = 0; head < kHeads; ++head)
        {
            uint8_t* kh = packed_k.data() +
                          (static_cast<size_t>(row) * kHeads + head) * kKHeadBytes;
            const float cnorm = 0.5f + 0.125f * static_cast<float>(row * kHeads + head);
            std::memcpy(kh, &cnorm, sizeof(cnorm));
            uint8_t* vh = packed_v.data() +
                          (static_cast<size_t>(row) * kHeads + head) * kVHeadBytes;
            for(int dim = 0; dim < kDim; ++dim)
            {
                const int code = (row * 7 + head * 3 + dim) & 15;
                uint8_t& packed = kh[4 + dim / 2];
                packed |= static_cast<uint8_t>(code << ((dim & 1) * 4));
                expected_k[(static_cast<size_t>(row) * kHeads + head) * kDim + dim] =
                    __float2half(cnorm * kC4[code]);

                const int block = dim / 32;
                const int lane = dim % 32;
                uint8_t* q8 = vh + block * 34;
                const __half scale = __float2half(0.01f * static_cast<float>(block + 1));
                std::memcpy(q8, &scale, sizeof(scale));
                const int8_t value = static_cast<int8_t>(((row + head + dim) % 63) - 31);
                q8[2 + lane] = static_cast<uint8_t>(value);
                expected_v[(static_cast<size_t>(row) * kHeads + head) * kDim + dim] =
                    __float2half(__half2float(scale) * static_cast<float>(value));
            }
        }
    }

    void *dk = nullptr, *dv = nullptr, *ok = nullptr, *ov = nullptr;
    check(hipMalloc(&dk, packed_k.size()), "hipMalloc packed K");
    check(hipMalloc(&dv, packed_v.size()), "hipMalloc packed V");
    check(hipMalloc(&ok, values * sizeof(__half)), "hipMalloc dense K");
    check(hipMalloc(&ov, values * sizeof(__half)), "hipMalloc dense V");
    check(hipMemcpy(dk, packed_k.data(), packed_k.size(), hipMemcpyHostToDevice), "copy K");
    check(hipMemcpy(dv, packed_v.data(), packed_v.size(), hipMemcpyHostToDevice), "copy V");

    hipfire_flash_attn_ck_asym4_loader_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ASYM4_LOADER_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.packed_k = static_cast<const uint8_t*>(dk);
    params.packed_v = static_cast<const uint8_t*>(dv);
    params.dense_k_f16 = ok;
    params.dense_v_f16 = ov;
    params.seqlen_k = kRows;
    params.nhead_k = kHeads;
    params.head_dim = kDim;
    params.k_row_stride_bytes = kHeads * kKHeadBytes;
    params.v_row_stride_bytes = kHeads * kVHeadBytes;
    char error[256]{};
    const int status = hipfire_flash_attn_ck_asym4_predecode(&params, error, sizeof(error));
    if(status != 0)
    {
        std::fprintf(stderr, "loader failed: %s\n", error);
        return 2;
    }
    check(hipDeviceSynchronize(), "loader sync");

    std::vector<__half> actual_k(values), actual_v(values);
    check(hipMemcpy(actual_k.data(), ok, values * sizeof(__half), hipMemcpyDeviceToHost), "copy dense K");
    check(hipMemcpy(actual_v.data(), ov, values * sizeof(__half), hipMemcpyDeviceToHost), "copy dense V");
    size_t mismatches = 0;
    for(size_t i = 0; i < values; ++i)
    {
        mismatches += bits(actual_k[i]) != bits(expected_k[i]);
        mismatches += bits(actual_v[i]) != bits(expected_v[i]);
    }
    std::printf("Asym4 loader smoke: rows=%d heads=%d values=%zu mismatches=%zu\n",
                kRows, kHeads, values, mismatches);
    check(hipFree(dk), "free packed K");
    check(hipFree(dv), "free packed V");
    check(hipFree(ok), "free dense K");
    check(hipFree(ov), "free dense V");
    return mismatches == 0 ? 0 : 1;
}
