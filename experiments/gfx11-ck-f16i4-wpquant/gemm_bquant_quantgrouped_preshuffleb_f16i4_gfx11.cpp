// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Local feasibility registration against ROCm libraries commit
// c4a1de3928b2c25d988fb06cb41f17baeadbe3cb.
#include "run_gemm_quant_example.inc"

template <typename T>
struct GemmConfigPreshuffleNoPermuteN
    : public GemmConfigPreshuffleB_BQuant_Prefill_Wmma<T>
{
    static constexpr bool TiledMMAPermuteN = false;
};

static auto register_f16i4_gfx11 = []() {
    auto& lut = get_kernel_lut();
    using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp16_t,
                                                    ck_tile::pk_int4_t,
                                                    ck_tile::half_t,
                                                    float>{});
    lut[hash_multiple_strings(
        {"f16i4", "bquant", "preshuffleb", "non-preshufflequant", "1x1x64"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize =
                ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
            return run_gemm_example_prec_type<
                GemmConfigPreshuffleNoPermuteN<ck_tile::fp16_t>,
                TypeConfig,
                QuantGroupSize,
                QuantGroupSize,
                ck_tile::QuantType::BQuantGrouped>(arg_parser);
        };

    return 0;
}();
