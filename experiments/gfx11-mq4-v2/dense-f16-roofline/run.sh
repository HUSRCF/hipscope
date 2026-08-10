#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${GPU_ID:-1}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ROCM_LIB=${ROCM_LIB:-/opt/rocm/core-7.14/lib}

"$HIPCC" -O3 -std=c++20 "$ROOT/bench_rocblas_f16.cpp" \
    -L"$ROCM_LIB" -lrocblas -lamdhip64 -o "$ROOT/bench_rocblas_f16"

export HIP_VISIBLE_DEVICES=$GPU_ID
export LD_LIBRARY_PATH="$ROCM_LIB:${ROCM_LIB}64:${LD_LIBRARY_PATH:-}"
"$ROOT/bench_rocblas_f16" 17408 2048 5120 | tee "$ROOT/gate.log"
"$ROOT/bench_rocblas_f16" 5120 2048 17408 | tee "$ROOT/down.log"
