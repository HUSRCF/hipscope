#!/usr/bin/env bash
set -euo pipefail

CK_ROOT="${CK_ROOT:-/home/husrcf/Code/ProtBind/flash_attn_ck/flash-attention-fa4-v4.0.0.beta4_20260319c18_release2/csrc/composable_kernel}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-10}"
OUT="${OUT:-/tmp/ck_gemm_wmma_i8_i8_${GPU_ARCH}}"

SOURCE="${CK_ROOT}/example/01_gemm/gemm_wmma_int8.cpp"
DEVICE_MEMORY="${CK_ROOT}/library/src/utility/device_memory.cpp"
HOST_TENSOR="${CK_ROOT}/library/src/utility/host_tensor.cpp"

"${ROCM_PATH}/bin/hipcc" \
    -std=c++17 \
    -O3 \
    --offload-arch="${GPU_ARCH}" \
    -DCK_ENABLE_INT8 \
    -DCK_ENABLE_FP16 \
    -DCK_ENABLE_FP32 \
    -D__HIP_PLATFORM_HCC__=1 \
    -fgpu-flush-denormals-to-zero \
    -Wno-pass-failed \
    -mllvm --lsr-drop-solution=1 \
    -fno-offload-uniform-block \
    -mllvm -enable-post-misched=0 \
    -mllvm -amdgpu-early-inline-all=true \
    -mllvm -amdgpu-function-calls=false \
    -I"${CK_ROOT}/include" \
    -I"${CK_ROOT}/library/include" \
    -I"${CK_ROOT}/example/01_gemm" \
    "${SOURCE}" \
    "${DEVICE_MEMORY}" \
    "${HOST_TENSOR}" \
    -o "${OUT}"

run_shape() {
    local label="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local stride_a="$5"
    local stride_b="$6"
    local stride_c="$7"

    for ((trial = 1; trial <= TRIALS; trial++)); do
        printf 'shape=%s trial=%d\n' "${label}" "${trial}"
        env HIP_VISIBLE_DEVICES="${GPU_ID}" \
            LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
            "${OUT}" 0 1 1 \
            "${m}" "${n}" "${k}" \
            "${stride_a}" "${stride_b}" "${stride_c}"
    done
}

# CK uses A[M,K] row-major, B[K,N] column-major, and C[M,N] row-major.
run_shape gate_up 2048 17408 5120 5120 5120 17408
run_shape down 2048 5120 17408 17408 17408 5120
