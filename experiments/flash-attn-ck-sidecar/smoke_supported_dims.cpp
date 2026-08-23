// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "usage: %s HEAD_DIMS\n", argv[0]);
        return 2;
    }

    std::unordered_set<int> expected;
    std::stringstream dimensions(argv[1]);
    for(std::string value; std::getline(dimensions, value, ',');)
    {
        expected.insert(std::stoi(value));
    }

    for(const int head_dim : std::array{64, 128, 256})
    {
        hipfire_flash_attn_ck_fwd_params params{};
        params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
        params.struct_size = sizeof(params);
        params.q = reinterpret_cast<const void*>(1);
        params.k = reinterpret_cast<const void*>(1);
        params.v = reinterpret_cast<const void*>(1);
        params.out = reinterpret_cast<void*>(1);
        params.dtype = HIPFIRE_FLASH_ATTN_CK_F16;
        params.batch = 1;
        params.seqlen_q = 1;
        params.seqlen_k = 1;
        params.nhead_q = 4;
        params.nhead_k = 1;
        params.head_dim = head_dim;
        params.softmax_scale = 1.0f;
        params.stride_q = 4 * head_dim;
        params.stride_k = head_dim;
        params.stride_v = head_dim;
        params.stride_out = 4 * head_dim;
        params.nhead_stride_q = head_dim;
        params.nhead_stride_k = head_dim;
        params.nhead_stride_v = head_dim;
        params.nhead_stride_out = head_dim;
        params.batch_stride_q = 4 * head_dim;
        params.batch_stride_k = head_dim;
        params.batch_stride_v = head_dim;
        params.batch_stride_out = 4 * head_dim;

        char error[256]{};
        const bool supported =
            hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) == 0;
        if(supported != expected.contains(head_dim))
        {
            std::fprintf(stderr,
                         "head_dim=%d supported=%d expected=%d error=%s\n",
                         head_dim,
                         supported,
                         expected.contains(head_dim),
                         error);
            return 1;
        }
    }

    std::printf("head-dim capability smoke passed: %s\n", argv[1]);
    return 0;
}
