// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION 1u

struct hipfire_flash_attn_ck_quantized_prefill_params {
    uint32_t abi_version;
    uint32_t struct_size;

    const float* q;
    const uint8_t* packed_k;
    const uint8_t* packed_v;
    float* out;
    void* workspace;
    size_t workspace_bytes;
    const float* cos_theta;
    const float* sin_theta;
    void* stream;

    float softmax_scale;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t nhead_q;
    int32_t nhead_k;
    int32_t head_dim;
    int32_t causal;
    int32_t k_row_stride_bytes;
    int32_t v_row_stride_bytes;
};

struct hipfire_flash_attn_ck_quantized_mq_q8_params {
    uint32_t abi_version;
    uint32_t struct_size;
    struct hipfire_flash_attn_ck_quantized_prefill_params prefill;
    const float* gate;
    const float* signs1;
    const float* signs2;
    void* q8_1_out;
};

uint32_t hipfire_flash_attn_ck_quantized_abi_version(void);

size_t hipfire_flash_attn_ck_quantized_prefill_workspace_bytes(
    int32_t seqlen_q,
    int32_t nhead_q,
    int32_t head_dim);

int hipfire_flash_attn_ck_quantized_prefill_supported(
    const struct hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity);

int hipfire_flash_attn_ck_quantized_prefill(
    const struct hipfire_flash_attn_ck_quantized_prefill_params* params,
    char* error,
    size_t error_capacity);

int hipfire_flash_attn_ck_quantized_mq_q8_supported(
    const struct hipfire_flash_attn_ck_quantized_mq_q8_params* params,
    char* error,
    size_t error_capacity);

int hipfire_flash_attn_ck_quantized_prefill_mq_q8(
    const struct hipfire_flash_attn_ck_quantized_mq_q8_params* params,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
