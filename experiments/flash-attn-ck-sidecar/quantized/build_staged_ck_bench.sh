#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_ROOT="$(cd "${ROOT}/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
BUILD_DIR="${ROOT}/build"
QUANTIZED_SIDECAR="${QUANTIZED_SIDECAR:-${BUILD_DIR}/libhipfire_flash_attn_ck_quantized.so}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"
DENSE_LIB_DIR="${SIDECAR_ROOT}/build"
QUANTIZED_LIB_DIR="$(dirname "${QUANTIZED_SIDECAR}")"
QUANTIZED_LIB_NAME="$(basename "${QUANTIZED_SIDECAR}")"

mkdir -p "${BUILD_DIR}"
"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O3 \
    --offload-arch="${GPU_ARCH}" \
    -fgpu-flush-denormals-to-zero \
    -Wno-pass-failed \
    -I"${ROOT}" \
    -I"${SIDECAR_ROOT}" \
    "${ROOT}/staged_quantized_ck_bench.hip" \
    -L"${QUANTIZED_LIB_DIR}" \
    -L"${DENSE_LIB_DIR}" \
    -l:"${QUANTIZED_LIB_NAME}" \
    -lhipfire_flash_attn_ck \
    -Wl,-rpath,"${QUANTIZED_LIB_DIR}" \
    -Wl,-rpath,"${DENSE_LIB_DIR}" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -o "${BUILD_DIR}/staged_quantized_ck_bench"

file "${BUILD_DIR}/staged_quantized_ck_bench"
