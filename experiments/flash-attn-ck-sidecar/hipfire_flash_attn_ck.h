// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIPFIRE_FLASH_ATTN_CK_ABI_VERSION 1u

enum hipfire_flash_attn_ck_dtype {
    HIPFIRE_FLASH_ATTN_CK_F16 = 1,
    HIPFIRE_FLASH_ATTN_CK_BF16 = 2,
};

struct hipfire_flash_attn_ck_fwd_params {
    uint32_t abi_version;
    uint32_t struct_size;

    const void* q;
    const void* k;
    const void* v;
    void* out;
    void* stream;

    int32_t dtype;
    int32_t batch;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t nhead_q;
    int32_t nhead_k;
    int32_t head_dim;
    int32_t causal;

    float softmax_scale;

    int64_t stride_q;
    int64_t stride_k;
    int64_t stride_v;
    int64_t stride_out;
    int64_t nhead_stride_q;
    int64_t nhead_stride_k;
    int64_t nhead_stride_v;
    int64_t nhead_stride_out;
    int64_t batch_stride_q;
    int64_t batch_stride_k;
    int64_t batch_stride_v;
    int64_t batch_stride_out;
};

uint32_t hipfire_flash_attn_ck_abi_version(void);

int hipfire_flash_attn_ck_fwd_supported(
    const struct hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity);

int hipfire_flash_attn_ck_fwd(
    const struct hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
