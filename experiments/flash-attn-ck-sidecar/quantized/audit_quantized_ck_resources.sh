#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
TARGET="${TARGET:-gfx1100}"
BIN="${BIN:-${ROOT}/build/quantized_ck_pipeline_smoke}"
LLVM_BIN="${ROCM_PATH}/lib/llvm/bin"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}; run run_quantized_ck_pipeline_smoke.sh first" >&2
    exit 2
fi

"${LLVM_BIN}/llvm-objcopy" \
    --dump-section ".hip_fatbin=${WORK}/kernel.hipfb" \
    "${BIN}"
"${LLVM_BIN}/clang-offload-bundler" \
    --type=o \
    --input="${WORK}/kernel.hipfb" \
    --targets="hipv4-amdgcn-amd-amdhsa--${TARGET}" \
    --output="${WORK}/kernel.hsaco" \
    --unbundle
"${LLVM_BIN}/llvm-readobj" --notes "${WORK}/kernel.hsaco" | \
    sed -n '/amdhsa.kernels:/,/amdhsa.target:/p'
