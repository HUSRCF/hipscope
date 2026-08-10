#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/bench_mq4v2_coarse_scale_int32_accum"
HSACO_BUNDLE="${ROOT_DIR}/mq4v2_coarse_scale_kernels.hsaco"
DEVICE_HSACO="${ROOT_DIR}/mq4v2_coarse_scale_kernels.gfx1100.hsaco"
LOG="${ROOT_DIR}/bench.log"
NOTES="${ROOT_DIR}/kernel_notes.txt"
DEVICE_LOG="${ROOT_DIR}/device_list.txt"
GPU1_PROBE_LOG="${ROOT_DIR}/gpu1_probe.txt"

MODE="${1:-build}"
HIPCC="${HIPCC:-hipcc}"
ARCH="${ARCH:-gfx1100}"
WARMUP="${WARMUP:-3}"
PAIRS="${PAIRS:-7}"
DPM_WARMUP_MS="${DPM_WARMUP_MS:-5000}"
HIP_RUNTIME_LIB="$(find /opt/rocm -name 'libamdhip64.so' 2>/dev/null | head -n 1 || true)"
if [[ -z "${HIP_RUNTIME_LIB}" ]]; then
  echo "Could not locate libamdhip64.so under /opt/rocm" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "${HIP_RUNTIME_LIB}")"

CXXFLAGS=(
  -O3
  -std=c++17
  --offload-arch="${ARCH}"
)

build_artifacts() {
  cd "${ROOT_DIR}"
  "${HIPCC}" "${CXXFLAGS[@]}" \
    bench_mq4v2_coarse_scale_int32_accum.hip \
    -o "${BIN}"
  "${HIPCC}" "${CXXFLAGS[@]}" --genco -x hip \
    mq4v2_coarse_scale_kernels.hip \
    -o "${HSACO_BUNDLE}"
  if command -v clang-offload-bundler >/dev/null 2>&1; then
    clang-offload-bundler --unbundle \
      --type=o \
      --targets="hipv4-amdgcn-amd-amdhsa--${ARCH}" \
      --input="${HSACO_BUNDLE}" \
      --output="${DEVICE_HSACO}"
  fi
}

emit_notes() {
  if command -v llvm-readelf >/dev/null 2>&1; then
    if [[ -f "${DEVICE_HSACO}" ]]; then
      llvm-readelf --notes "${DEVICE_HSACO}" > "${NOTES}"
    else
      llvm-readelf --notes "${HSACO_BUNDLE}" > "${NOTES}"
    fi
  fi
}

run_bench() {
  cd "${ROOT_DIR}"
  "${BIN}" --list-devices | tee "${DEVICE_LOG}"

  local -a run_env=()
  if env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" HIP_VISIBLE_DEVICES=1 \
      "${BIN}" --list-devices > "${GPU1_PROBE_LOG}" 2>&1; then
    cat "${GPU1_PROBE_LOG}"
    run_env=(env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" HIP_VISIBLE_DEVICES=1)
  else
    cat "${GPU1_PROBE_LOG}" || true
    run_env=(env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}")
  fi

  "${run_env[@]}" "${BIN}" \
    --warmup "${WARMUP}" \
    --pairs "${PAIRS}" \
    --dpm-warmup-ms "${DPM_WARMUP_MS}" | tee "${LOG}"
}

case "${MODE}" in
  build)
    build_artifacts
    emit_notes
    ;;
  run)
    run_bench
    emit_notes
    ;;
  *)
    echo "usage: bash run_probe.sh [build|run]" >&2
    exit 1
    ;;
esac
