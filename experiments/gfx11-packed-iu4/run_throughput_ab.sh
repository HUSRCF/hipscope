#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/experiments/gfx11-packed-iu4/build"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ARCH="${GPU_ARCH:-gfx1100}"

mkdir -p "${OUT}"
"${HIPCC}" -O3 -std=c++17 --offload-arch="${ARCH}" \
  "${ROOT}/experiments/gfx11-packed-iu4/iu4_vs_iu8_throughput.hip" \
  -o "${OUT}/iu4_vs_iu8_throughput"

"${OUT}/iu4_vs_iu8_throughput" "${BLOCKS:-1024}" "${ITERATIONS:-4096}" "${TRIALS:-21}"
