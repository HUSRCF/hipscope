#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BUILD_DIR="${ROOT}/build"
SIDECAR="${SIDECAR:-${BUILD_DIR}/libhipfire_flash_attn_ck.so}"
OUT="${OUT:-${BUILD_DIR}/bench_vs_native.csv}"
GPU_ID="${GPU_ID:-0}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
WARMUP="${WARMUP:-3}"
TRIALS="${TRIALS:-9}"
ITERATIONS="${ITERATIONS:-20}"
HEAD_DIM="${HEAD_DIM:-64}"
CAUSAL="${CAUSAL:-0}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"
SIDECAR_DIR="$(dirname "${SIDECAR}")"
BENCH_ARCH_DEFINE=()
case "${GPU_ARCH}" in
    gfx12*) BENCH_ARCH_DEFINE=(-DHIPFIRE_BENCH_GFX12=1) ;;
    gfx11*) BENCH_ARCH_DEFINE=(-DHIPFIRE_BENCH_GFX12=0) ;;
    *)
        echo "unsupported GPU_ARCH ${GPU_ARCH}; expected gfx11* or gfx12*" >&2
        exit 2
        ;;
esac

if [[ ! -f "${SIDECAR}" ]]; then
    echo "missing sidecar ${SIDECAR}; run build_sidecar.sh first" >&2
    exit 2
fi

"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O3 \
    --offload-arch="${GPU_ARCH}" \
    "${BENCH_ARCH_DEFINE[@]}" \
    -I"${ROOT}" \
    "${ROOT}/bench_vs_native.cpp" \
    -L"${SIDECAR_DIR}" \
    -Wl,-rpath,'$ORIGIN' \
    -Wl,-rpath,"${SIDECAR_DIR}" \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -lhipfire_flash_attn_ck \
    -o "${BUILD_DIR}/bench_vs_native"

mkdir -p "$(dirname "${OUT}")"
HIP_VISIBLE_DEVICES="${GPU_ID}" \
    "${BUILD_DIR}/bench_vs_native" \
        --warmup "${WARMUP}" \
        --trials "${TRIALS}" \
        --iterations "${ITERATIONS}" \
        --head-dim "${HEAD_DIM}" \
        --causal "${CAUSAL}" \
    | tee "${OUT}"

echo "wrote ${OUT}" >&2
