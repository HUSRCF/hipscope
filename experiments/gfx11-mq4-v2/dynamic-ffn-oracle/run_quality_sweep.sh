#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODEL="${MODEL:-/home/husrcf/.hipfire/models/qwen3.6-27b.mq4}"
CORPUS="${CORPUS:-/home/husrcf/Code/ProtBind/unidec/docs/testINPUT.md}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-mq4-v2/dynamic-ffn-oracle/results/local}"
GPU_ID="${GPU_ID:-1}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:-4}"
STRIDE="${STRIDE:-4}"
SLEEP_SECS="${SLEEP_SECS:-20}"
CK_LIB="${CK_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"

mkdir -p "${OUT_DIR}"
cargo build --release -p hipfire-runtime --example flash_prefill_quality \
  --features deltanet,flash-attn-ck

run_one() {
  local keep="$1"
  local -a oracle_env=()
  if [[ "${keep}" != "68" ]]; then
    oracle_env+=("HIPFIRE_RDNA3_FFN_ORACLE_KEEP_GROUPS=${keep}")
  fi

  env -u HIPFIRE_RDNA3_FFN_ORACLE_KEEP_GROUPS \
    HIP_VISIBLE_DEVICES="${GPU_ID}" \
    HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
    HIPFIRE_KV_MODE=asym3 \
    HIPFIRE_GRAPH=0 \
    HIPFIRE_PREFILL_REUSE_PBS=1 \
    HIPFIRE_PREFILL_MAX_BATCH=256 \
    HIPFIRE_FLASH_PARTIALS_BATCH=32 \
    HIPFIRE_DPM_WARMUP_SECS=2 \
    HIPFIRE_QKVZA_SPLIT_TAIL=1 \
    HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
    HIPFIRE_RDNA3_Q8_GROUP128=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
    HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
    HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1 \
    HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1 \
    HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
    "${oracle_env[@]}" \
    "${ROOT}/target/release/examples/flash_prefill_quality" \
    "${MODEL}" "${CORPUS}" "${OUT_DIR}/keep-${keep}.bin" \
    --ctx "${CTX}" --chunks "${CHUNKS}" --stride "${STRIDE}" \
    2>&1 | tee "${OUT_DIR}/keep-${keep}.log"
}

run_one 68
for keep in 60 51 41; do
  sleep "${SLEEP_SECS}"
  run_one "${keep}"
done

for keep in 60 51 41; do
  python "${ROOT}/scripts/flash_prefill_quality_compare.py" \
    "${OUT_DIR}/keep-68.bin" "${OUT_DIR}/keep-${keep}.bin" \
    > "${OUT_DIR}/compare-68-vs-${keep}.txt"
done
