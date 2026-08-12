#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_ROOT="$(cd "${ROOT}/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
CK_BM="${CK_BM:-64}"
CK_BN="${CK_BN:-32}"
CK_ROOT="${CK_ROOT:-${SIDECAR_ROOT}/build/ck-source}"
FMHA_DIR="${CK_ROOT}/example/ck_tile/01_fmha"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
OUT="${OUT:-${BUILD_DIR}/libhipfire_flash_attn_ck_quantized.so}"
DENSE_SIDECAR="${DENSE_SIDECAR:-${SIDECAR_ROOT}/build/libhipfire_flash_attn_ck.so}"
STAGED="${STAGED:-0}"
PACKET_STORE="${PACKET_STORE:-0}"
ASYM3_CODEBOOK="${ASYM3_CODEBOOK:-0}"
ASYM3_LDS_CODEBOOK="${ASYM3_LDS_CODEBOOK:-0}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"

extra_defines=()
extra_links=()
case "${PACKET_STORE}" in
    0) ;;
    1) extra_defines+=("-DHIPFIRE_PREDECODE_PACKET_STORE=1") ;;
    *)
        echo "PACKET_STORE must be 0 or 1" >&2
        exit 2
        ;;
esac
case "${CK_BM}" in
    16|32|64) ;;
    *)
        echo "CK_BM must be 16, 32, or 64" >&2
        exit 2
        ;;
esac
case "${CK_BN}" in
    32|64|128) ;;
    *)
        echo "CK_BN must be 32, 64, or 128" >&2
        exit 2
        ;;
esac
if [[ "${CK_BN}" == 128 && "${CK_BM}" != 64 ]]; then
    echo "CK_BN=128 is validated only with CK_BM=64" >&2
    exit 2
fi
if [[ "${ASYM3_CODEBOOK}" == "1" && "${ASYM3_LDS_CODEBOOK}" == "1" ]]; then
    echo "ASYM3_CODEBOOK and ASYM3_LDS_CODEBOOK are mutually exclusive" >&2
    exit 2
fi
if [[ "${ASYM3_CODEBOOK}" == "1" ]]; then
    extra_defines+=("-DHIPFIRE_CK_ASYM3_CONSTANT_CODEBOOK=1")
fi
if [[ "${ASYM3_LDS_CODEBOOK}" == "1" ]]; then
    extra_defines+=("-DHIPFIRE_CK_ASYM3_LDS_CODEBOOK=1")
fi
case "${STAGED}" in
    0) ;;
    1)
        if [[ ! -f "${DENSE_SIDECAR}" ]]; then
            echo "missing dense CK sidecar ${DENSE_SIDECAR}; run ../build_sidecar.sh first" >&2
            exit 2
        fi
        extra_defines+=("-DHIPFIRE_ENABLE_STAGED_QUANTIZED_CK=1")
        extra_links+=(
            "-L$(dirname "${DENSE_SIDECAR}")"
            "-lhipfire_flash_attn_ck"
            "-Wl,-rpath,$(dirname "${DENSE_SIDECAR}")"
        )
        ;;
    *)
        echo "STAGED must be 0 or 1" >&2
        exit 2
        ;;
esac

if [[ ! -f "${FMHA_DIR}/fmha_fwd.hpp" ]]; then
    echo "missing prepared CK source under ${CK_ROOT}; run ../build_sidecar.sh first" >&2
    exit 2
fi
mkdir -p "$(dirname "${OUT}")"
"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O3 \
    -shared \
    -fPIC \
    --offload-arch="${GPU_ARCH}" \
    -DHIPFIRE_CK_TARGET_GFX11=1 \
    -DHIPFIRE_CK_FMHA_BM="${CK_BM}" \
    -DHIPFIRE_CK_FMHA_BN="${CK_BN}" \
    -DHIPFIRE_CK_FMHA_OUTPUT_F32=0 \
    -DHIPFIRE_QUANTIZED_CK_SIDECAR=1 \
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
    -fgpu-flush-denormals-to-zero \
    -DCK_ENABLE_BF16 \
    -DCK_ENABLE_FP16 \
    -DCK_ENABLE_FP32 \
    -DCK_ENABLE_FP64 \
    -DCK_ENABLE_INT8 \
    -D__HIP_PLATFORM_HCC__=1 \
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 \
    "${extra_defines[@]}" \
    -Wno-pass-failed \
    -mllvm --lsr-drop-solution=1 \
    -fno-offload-uniform-block \
    -mllvm -enable-post-misched=0 \
    -mllvm -amdgpu-early-inline-all=true \
    -mllvm -amdgpu-function-calls=false \
    -I"${ROOT}" \
    -I"${SIDECAR_ROOT}" \
    -I"${FMHA_DIR}" \
    -I"${CK_ROOT}/include" \
    -I"${CK_ROOT}/library/include" \
    "${ROOT}/quantized_ck_pipeline_smoke.hip" \
    "${extra_links[@]}" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -o "${OUT}"

file "${OUT}"
du -h "${OUT}"

SMOKE="$(dirname "${OUT}")/smoke_quantized_abi"
"${CXX:-c++}" \
    -std=c++20 \
    -O2 \
    "${extra_defines[@]}" \
    -I"${ROOT}" \
    "${ROOT}/smoke_quantized_abi.cpp" \
    "${OUT}" \
    -Wl,-rpath,"$(dirname "${OUT}")" \
    -o "${SMOKE}"
"${SMOKE}"

{
    ck_tree_sha256="$({
        find "${CK_ROOT}/include" "${CK_ROOT}/library/include" "${FMHA_DIR}" \
            -type f -print0 \
            | sort -z \
            | xargs -0 sha256sum
    } | sha256sum | cut -d' ' -f1)"
    local_tree_sha256="$({
        find "${ROOT}" "${SIDECAR_ROOT}" -maxdepth 1 -type f -print0 \
            | sort -z \
            | xargs -0 sha256sum
    } | sha256sum | cut -d' ' -f1)"
    variant_sources=(
        "$0"
        "${FMHA_DIR}/fmha_fwd.hpp"
        "${ROOT}/quantized_ck_pipeline_smoke.hip"
        "${ROOT}/quantized_kv_predecode.hpp"
    )
    dense_sidecar_record=none
    if [[ "${STAGED}" == 1 ]]; then
        dense_sidecar_record="${DENSE_SIDECAR}"
        variant_sources+=("${DENSE_SIDECAR}")
    fi
    printf 'gpu_arch=%s\n' "${GPU_ARCH}"
    printf 'ck_bm=%s\n' "${CK_BM}"
    printf 'ck_bn=%s\n' "${CK_BN}"
    printf 'ck_root=%s\n' "${CK_ROOT}"
    printf 'dense_sidecar=%s\n' "${dense_sidecar_record}"
    printf 'ck_tree_sha256=%s\n' "${ck_tree_sha256}"
    printf 'local_tree_sha256=%s\n' "${local_tree_sha256}"
    printf 'staged=%s\n' "${STAGED}"
    printf 'packet_store=%s\n' "${PACKET_STORE}"
    printf 'asym3_codebook=%s\n' "${ASYM3_CODEBOOK}"
    printf 'asym3_lds_codebook=%s\n' "${ASYM3_LDS_CODEBOOK}"
    sha256sum "${variant_sources[@]}"
} > "${OUT}.variant"
