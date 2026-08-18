#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CK_ROOT=${CK_ROOT:-/home/husrcf/Code/ProtBind/flash_attn_ck/flash-attention-fa4-v4.0.0.beta4_20260319c18_release2/csrc/composable_kernel}
GPU_ID=${GPU_ID:-1}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ROCM_LIB=${ROCM_LIB:-/opt/rocm/core-7.14/lib}
BUILD=${BUILD:-1}

build_one() {
    local precision=$1
    local instance_dir=$2
    shift 2
    local defines=("$@")
    (
        cd "$CK_ROOT"
        "$HIPCC" --offload-arch=gfx1100 -O2 -std=c++20 \
            "${defines[@]}" -DCK_USE_WMMA -DCK_TIME_KERNEL=1 \
            -Wno-bit-int-extension -Wno-pass-failed -Wno-switch-default \
            -Wno-unique-object-duplication \
            -I include -I library/include \
            -I "library/src/tensor_operation_instance/gpu/gemm_universal/$instance_dir" \
            "$ROOT/bench_ck_a16_i4_all.cpp" library/src/utility/device_memory.cpp \
            -o "$ROOT/bench_ck_${precision}_i4_all"
    )
}

if [[ "$BUILD" == 1 ]]; then
    build_one bf16 device_gemm_wmma_universal_bf16_i4_bf16 \
        -DCK_ENABLE_BF16 -DCK_A16_BF16
    build_one f16 device_gemm_wmma_universal_f16_i4_f16 -DCK_ENABLE_FP16
fi

export HIP_VISIBLE_DEVICES=$GPU_ID
export LD_LIBRARY_PATH="$ROCM_LIB:${ROCM_LIB}64:${LD_LIBRARY_PATH:-}"

for precision in bf16 f16; do
    "$ROOT/bench_ck_${precision}_i4_all" 2048 17408 5120 \
        | tee "$ROOT/${precision}_gate.log"
    "$ROOT/bench_ck_${precision}_i4_all" 2048 5120 17408 \
        | tee "$ROOT/${precision}_down.log"
done
