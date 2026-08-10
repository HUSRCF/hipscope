#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
BUILD_DIR="${ROOT}/build"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"

mkdir -p "${BUILD_DIR}"
"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O3 \
    --offload-arch="${GPU_ARCH}" \
    -fgpu-flush-denormals-to-zero \
    -Wno-pass-failed \
    "${ROOT}/quantized_kv_predecode_bench.hip" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -o "${BUILD_DIR}/quantized_kv_predecode_bench"

file "${BUILD_DIR}/quantized_kv_predecode_bench"
