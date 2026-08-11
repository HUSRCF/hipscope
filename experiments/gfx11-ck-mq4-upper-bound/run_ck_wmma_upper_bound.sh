#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CK_ROOT="${CK_ROOT:-/home/husrcf/Code/ProtBind/flash_attn_ck/flash-attention-fa4-v4.0.0.beta4_20260319c18_release2/csrc/composable_kernel}"
BUILD_DIR="${BUILD_DIR:-/tmp/ck-gfx1100-i4}"
GPU_ID="${GPU_ID:-1}"
JOBS="${JOBS:-16}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-ck-mq4-upper-bound/results/gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S_%N)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm/core-7.14}"

mkdir -p "${OUT_DIR}"

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    cmake -S "${CK_ROOT}" -B "${BUILD_DIR}" -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D GPU_TARGETS=gfx1100 \
        -D CMAKE_CXX_COMPILER="${ROCM_ROOT}/bin/hipcc" \
        -D BUILD_DEV=OFF \
        -D BUILD_TESTING=OFF >"${OUT_DIR}/configure.log" 2>&1
    ninja -C "${BUILD_DIR}" -j"${JOBS}" \
        example_gemm_wmma_fp16_pk_i4_v3_b_scale \
        example_gemm_wmma_fp16_pk_i4_v3 \
        example_gemm_wmma_int8 >"${OUT_DIR}/build.log" 2>&1
fi

run_case() {
    local name="$1"
    shift
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        LD_LIBRARY_PATH="${ROCM_ROOT}/lib:${ROCM_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "$@" 2>&1 | tee "${OUT_DIR}/${name}.log"
}

# The production GEMM names dimensions as output rows M=17408 and token rows
# N=2048. CK uses the conventional activation-M/output-N ordering.
run_case fp16_i4_bscale \
    "${BUILD_DIR}/bin/example_gemm_wmma_fp16_pk_i4_v3_b_scale" \
    0 1 1 2048 17408 5120 -1 -1 -1 1
run_case fp16_i4 \
    "${BUILD_DIR}/bin/example_gemm_wmma_fp16_pk_i4_v3" \
    0 1 1 2048 17408 5120 -1 -1 -1 1
run_case int8 \
    "${BUILD_DIR}/bin/example_gemm_wmma_int8" \
    0 1 1 2048 17408 5120 -1 -1 -1

{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'ck_root=%s\n' "${CK_ROOT}"
    printf 'shape=CK_M2048_N17408_K5120\n'
} >"${OUT_DIR}/manifest.txt"

printf 'out_dir=%s\n' "${OUT_DIR}"
