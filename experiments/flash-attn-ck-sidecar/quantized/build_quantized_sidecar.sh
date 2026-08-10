#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_ROOT="$(cd "${ROOT}/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
CK_ROOT="${SIDECAR_ROOT}/build/ck-source"
FMHA_DIR="${CK_ROOT}/example/ck_tile/01_fmha"
BUILD_DIR="${ROOT}/build"
OUT="${OUT:-${BUILD_DIR}/libhipfire_flash_attn_ck_quantized.so}"
ASYM3_CODEBOOK="${ASYM3_CODEBOOK:-0}"
ASYM3_LDS_CODEBOOK="${ASYM3_LDS_CODEBOOK:-0}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"

extra_defines=()
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
    -DHIPFIRE_CK_FMHA_BM=64 \
    -DHIPFIRE_CK_FMHA_BN=32 \
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
    -I"${FMHA_DIR}" \
    -I"${CK_ROOT}/include" \
    -I"${CK_ROOT}/library/include" \
    "${ROOT}/quantized_ck_pipeline_smoke.hip" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -o "${OUT}"

file "${OUT}"
du -h "${OUT}"

SMOKE="$(dirname "${OUT}")/smoke_quantized_abi"
"${CXX:-c++}" \
    -std=c++20 \
    -O2 \
    -I"${ROOT}" \
    "${ROOT}/smoke_quantized_abi.cpp" \
    "${OUT}" \
    -Wl,-rpath,"$(dirname "${OUT}")" \
    -o "${SMOKE}"
"${SMOKE}"
