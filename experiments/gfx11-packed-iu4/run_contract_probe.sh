#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/experiments/gfx11-packed-iu4/build"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ARCH="${GPU_ARCH:-gfx1100}"

mkdir -p "${OUT}"
"${HIPCC}" -O3 -std=c++17 --offload-arch="${ARCH}" \
  "${ROOT}/experiments/gfx11-packed-iu4/iu4_wmma_contract_probe.hip" \
  -o "${OUT}/iu4_wmma_contract_probe"

"${OUT}/iu4_wmma_contract_probe"
