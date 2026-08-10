#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPFIRE_ROOT="$(cd "${ROOT}/../../.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
BUILD_DIR="${ROOT}/build"
BIN="${BUILD_DIR}/quantized_tile_loader_smoke"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"

if [[ ! -f "${HIP_LIB_DIR}/libamdhip64.so" ]]; then
    echo "missing HIP runtime under ${HIP_LIB_DIR}" >&2
    exit 2
fi

mkdir -p "${BUILD_DIR}"
"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O3 \
    --offload-arch="${GPU_ARCH}" \
    -fgpu-flush-denormals-to-zero \
    -Wno-pass-failed \
    "${ROOT}/quantized_tile_loader_smoke.hip" \
    "${HIPFIRE_ROOT}/kernels/src/kv_cache_write_asym_k_givens3_batched.hip" \
    "${HIPFIRE_ROOT}/kernels/src/kv_cache_write_q8_0_batched.hip" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -o "${BIN}"

"${BIN}"
