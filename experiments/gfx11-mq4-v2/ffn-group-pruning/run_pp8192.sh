#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BIN=${BIN:-$ROOT/target/release/examples/bench_qwen35_mq4}
ORIGINAL_MODEL=${ORIGINAL_MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}
PRUNED_MODEL=${PRUNED_MODEL:-/tmp/qwen3.6-27b-ffn60g.mq4}
GPU_ID=${GPU_ID:-1}
CK_LIB=${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}
MODE=${MODE:-all}

run_case() {
  local label=$1 model=$2 variable_width=$3
  printf '\n===== %s =====\n' "$label"
  timeout --signal=INT --kill-after=10s 900s env \
    HIP_VISIBLE_DEVICES="$GPU_ID" \
    HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="$CK_LIB" \
    HIPFIRE_KV_MODE=asym3 \
    HIPFIRE_GRAPH=0 \
    HIPFIRE_PREFILL_MAX_BATCH=2048 \
    HIPFIRE_FLASH_PARTIALS_BATCH=32 \
    HIPFIRE_DPM_WARMUP_SECS=5 \
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
    HIPFIRE_RDNA3_FFN_VARIABLE_WIDTH="$variable_width" \
    HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1 \
    HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
    "$BIN" "$model" --prefill 8192 --prefill-runs 7 --warmup 0 --gen 1
}

case "$MODE" in
  original) run_case original "$ORIGINAL_MODEL" 0 ;;
  fallback) run_case pruned-fallback "$PRUNED_MODEL" 0 ;;
  pruned) run_case pruned-fast "$PRUNED_MODEL" 1 ;;
  all)
    run_case original "$ORIGINAL_MODEL" 0
    run_case pruned-fallback "$PRUNED_MODEL" 0
    run_case pruned-fast "$PRUNED_MODEL" 1
    ;;
  *) printf 'unknown MODE=%s (original, fallback, pruned, all)\n' "$MODE" >&2; exit 2 ;;
esac
