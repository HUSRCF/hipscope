// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck_quantized.h"

#include <cstdio>
#include <cstdlib>

namespace {

void require(bool condition, const char* message)
{
    if(!condition)
    {
        std::fprintf(stderr, "quantized ABI smoke failed: %s\n", message);
        std::exit(2);
    }
}

} // namespace

int main()
{
    require(hipfire_flash_attn_ck_quantized_abi_version() ==
                HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION,
            "ABI version");

    constexpr int q_rows = 128;
    constexpr int q_heads = 24;
    constexpr int kv_heads = 4;
    constexpr int head_dim = 256;
    const size_t workspace =
        hipfire_flash_attn_ck_quantized_prefill_workspace_bytes(q_rows, q_heads, head_dim);
    require(workspace == static_cast<size_t>(q_rows) * q_heads * head_dim * 4,
            "workspace size");

    hipfire_flash_attn_ck_quantized_prefill_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = reinterpret_cast<const float*>(0x1000);
    params.packed_k = reinterpret_cast<const uint8_t*>(0x2000);
    params.packed_v = reinterpret_cast<const uint8_t*>(0x3000);
    params.out = reinterpret_cast<float*>(0x4000);
    params.workspace = reinterpret_cast<void*>(0x5000);
    params.workspace_bytes = workspace;
    params.cos_theta = reinterpret_cast<const float*>(0x6000);
    params.sin_theta = reinterpret_cast<const float*>(0x7000);
    params.softmax_scale = 0.0625f;
    params.seqlen_q = q_rows;
    params.seqlen_k = 2048;
    params.nhead_q = q_heads;
    params.nhead_k = kv_heads;
    params.head_dim = head_dim;
    params.causal = 1;
    params.k_row_stride_bytes = kv_heads * 100;
    params.v_row_stride_bytes = kv_heads * 272;

    char error[256]{};
    require(hipfire_flash_attn_ck_quantized_prefill_supported(
                &params, error, sizeof(error)) == 0,
            "validated shape should be supported");

    params.seqlen_q = 64;
    require(hipfire_flash_attn_ck_quantized_prefill_supported(
                &params, error, sizeof(error)) != 0,
            "Q<128 must be rejected");
    params.seqlen_q = q_rows;
    params.workspace_bytes = workspace - 1;
    require(hipfire_flash_attn_ck_quantized_prefill_supported(
                &params, error, sizeof(error)) != 0,
            "undersized workspace must be rejected");

    std::printf("quantized ABI smoke passed: abi=%u workspace=%zu bytes\n",
                HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION,
                workspace);
    return 0;
}
