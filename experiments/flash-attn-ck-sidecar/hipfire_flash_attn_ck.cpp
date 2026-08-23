// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include "fmha_fwd.hpp"
#include "mask.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

static_assert(sizeof(hipfire_flash_attn_ck_fwd_params) == 184,
              "FlashAttention CK ABI parameter layout changed");
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, q) == 8);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, dtype) == 48);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, softmax_scale) == 80);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, stride_q) == 88);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, batch_stride_out) == 176);

namespace {

bool supports_head_dim(int head_dim)
{
    switch(head_dim)
    {
#if HIPFIRE_CK_HEAD_DIM_64
    case 64: return true;
#endif
#if HIPFIRE_CK_HEAD_DIM_128
    case 128: return true;
#endif
#if HIPFIRE_CK_HEAD_DIM_256
    case 256: return true;
#endif
    default: return false;
    }
}

void set_error(char* error, size_t capacity, const std::string& message)
{
    if(error == nullptr || capacity == 0)
    {
        return;
    }
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

int validate(const hipfire_flash_attn_ck_fwd_params* p, char* error, size_t error_capacity)
{
    if(p == nullptr)
    {
        set_error(error, error_capacity, "params is null");
        return 1;
    }
    if(p->abi_version != HIPFIRE_FLASH_ATTN_CK_ABI_VERSION)
    {
        set_error(error, error_capacity, "unsupported ABI version");
        return 1;
    }
    if(p->struct_size < sizeof(hipfire_flash_attn_ck_fwd_params))
    {
        set_error(error, error_capacity, "parameter struct is too small");
        return 1;
    }
    if(p->q == nullptr || p->k == nullptr || p->v == nullptr || p->out == nullptr)
    {
        set_error(error, error_capacity, "q, k, v, and out must be non-null");
        return 1;
    }
    if(p->dtype != HIPFIRE_FLASH_ATTN_CK_F16)
    {
        set_error(error, error_capacity, "this optional build supports FP16 only");
        return 1;
    }
    if(p->batch <= 0 || p->seqlen_q <= 0 || p->seqlen_k <= 0 ||
       p->nhead_q <= 0 || p->nhead_k <= 0)
    {
        set_error(error, error_capacity, "batch, sequence lengths, and head counts must be positive");
        return 1;
    }
    if(!supports_head_dim(p->head_dim))
    {
        set_error(error, error_capacity, "head_dim is not included in this optional build");
        return 1;
    }
    if(p->nhead_q % p->nhead_k != 0)
    {
        set_error(error, error_capacity, "nhead_k must divide nhead_q");
        return 1;
    }
    if(p->causal != 0 && p->causal != 1)
    {
        set_error(error, error_capacity, "causal must be 0 or 1");
        return 1;
    }
    if(!(p->softmax_scale > 0.0f))
    {
        set_error(error, error_capacity, "softmax_scale must be positive");
        return 1;
    }
    const int64_t strides[] = {
        p->stride_q,
        p->stride_k,
        p->stride_v,
        p->stride_out,
        p->nhead_stride_q,
        p->nhead_stride_k,
        p->nhead_stride_v,
        p->nhead_stride_out,
        p->batch_stride_q,
        p->batch_stride_k,
        p->batch_stride_v,
        p->batch_stride_out,
    };
    for(const int64_t stride : strides)
    {
        if(stride <= 0 || stride > INT32_MAX)
        {
            set_error(error, error_capacity, "all element strides must be in (0, INT32_MAX]");
            return 1;
        }
    }
    set_error(error, error_capacity, "");
    return 0;
}

} // namespace

extern "C" uint32_t hipfire_flash_attn_ck_abi_version(void)
{
    return HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
}

extern "C" int hipfire_flash_attn_ck_fwd_supported(
    const hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity)
{
    return validate(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_fwd(
    const hipfire_flash_attn_ck_fwd_params* p,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate(p, error, error_capacity); status != 0)
    {
        return status;
    }

    try
    {
        const std::string dtype = "fp16";
        const std::string mask_id = p->causal != 0 ? "b:-1,0" : "0";
        const mask_info mask = mask_info::decode(mask_id, p->seqlen_q, p->seqlen_k);

        fmha_fwd_traits traits{
            p->head_dim,
            p->head_dim,
            dtype,
            false,
            true,
            false,
            mask.type,
            bias_enum::no_bias,
            false,
            false,
            quant_scale_enum::no_scale,
            false,
        };

        fmha_fwd_args args{};
        args.q_ptr = p->q;
        args.k_ptr = p->k;
        args.v_ptr = p->v;
        args.o_ptr = p->out;
        args.seqlen_q = p->seqlen_q;
        args.seqlen_k = p->seqlen_k;
        args.batch = p->batch;
        args.max_seqlen_q = p->seqlen_q;
        args.hdim_q = p->head_dim;
        args.hdim_v = p->head_dim;
        args.nhead_q = p->nhead_q;
        args.nhead_k = p->nhead_k;
        args.scale_s = p->softmax_scale;
        args.logits_soft_cap = 0.0f;
        args.stride_q = static_cast<ck_tile::index_t>(p->stride_q);
        args.stride_k = static_cast<ck_tile::index_t>(p->stride_k);
        args.stride_v = static_cast<ck_tile::index_t>(p->stride_v);
        args.stride_o = static_cast<ck_tile::index_t>(p->stride_out);
        args.nhead_stride_q = static_cast<ck_tile::index_t>(p->nhead_stride_q);
        args.nhead_stride_k = static_cast<ck_tile::index_t>(p->nhead_stride_k);
        args.nhead_stride_v = static_cast<ck_tile::index_t>(p->nhead_stride_v);
        args.nhead_stride_o = static_cast<ck_tile::index_t>(p->nhead_stride_out);
        args.batch_stride_q = static_cast<ck_tile::index_t>(p->batch_stride_q);
        args.batch_stride_k = static_cast<ck_tile::index_t>(p->batch_stride_k);
        args.batch_stride_v = static_cast<ck_tile::index_t>(p->batch_stride_v);
        args.batch_stride_o = static_cast<ck_tile::index_t>(p->batch_stride_out);
        args.window_size_left = -1;
        args.window_size_right = p->causal != 0 ? 0 : -1;
        args.mask_type = static_cast<ck_tile::index_t>(mask.type);
        args.min_seqlen_q = 0;
        args.p_drop = 0.0f;
        args.s_randval = false;
        args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

        ck_tile::stream_config stream_config{
            reinterpret_cast<hipStream_t>(p->stream),
        };
        const float result = fmha_fwd(traits, args, stream_config);
        if(result < 0.0f)
        {
            set_error(error, error_capacity, "CK found no matching forward kernel");
            return 2;
        }
        set_error(error, error_capacity, "");
        return 0;
    }
    catch(const std::exception& exception)
    {
        set_error(error, error_capacity, exception.what());
        return 3;
    }
    catch(...)
    {
        set_error(error, error_capacity, "unknown C++ exception");
        return 3;
    }
}
