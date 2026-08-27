#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
LIB="${LIB:-/tmp/libhipfire_flash_attn_ck_quantized_asym4_loader.so}"
BIN="${BIN:-/tmp/smoke_asym4_loader}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"

if [[ ! -f "${LIB}" ]]; then
    echo "missing ${LIB}; build the staged quantized sidecar first" >&2
    exit 2
fi

"${ROCM_PATH}/bin/hipcc" -std=c++20 -O2 --offload-arch="${GPU_ARCH}" \
    -I"${ROOT}" "${ROOT}/smoke_asym4_loader.hip" \
    -L"$(dirname "${LIB}")" -Wl,-l:"$(basename "${LIB}")" \
    -Wl,-rpath,"$(dirname "${LIB}"):${HIP_LIB_DIR}" -o "${BIN}"
LD_LIBRARY_PATH="${HIP_LIB_DIR}:${LD_LIBRARY_PATH:-}" "${BIN}"
