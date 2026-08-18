#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/examples/bench_qwen35_mq4}
MODEL=${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}
GPU_ID=${GPU_ID:-1}
CK_LIB=${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-$ROOT/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-$ROOT/experiments/gfx11-gate-up-x256y64/results/pp8192_group128_quad_row_profile_gpu1_$STAMP}
mkdir -p "$OUT_DIR"

timeout --signal=INT --kill-after=5s 240s \
  env \
    HIP_VISIBLE_DEVICES="$GPU_ID" \
    HIPFIRE_KV_MODE=asym3 \
    HIPFIRE_GRAPH=0 \
    HIPFIRE_PROFILE=1 \
    HIPFIRE_PREFILL_MAX_BATCH=2048 \
    HIPFIRE_FLASH_PARTIALS_BATCH=32 \
    HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="$CK_LIB" \
    HIPFIRE_DPM_WARMUP_SECS=5 \
    HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
    HIPFIRE_RDNA3_Q8_GROUP128=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_DUAL_ROW_WEIGHT=0 \
    HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
    "$BIN" "$MODEL" \
      --prefill 8192 --prefill-runs 1 --warmup 2 --gen 8 \
      >"$OUT_DIR/profile.log" 2>&1

rg '^(=== PROFILE|  (gemm_|fused_silu|quantize_|flash_|TOTAL)|PREFILL_SUMMARY|SUMMARY )' \
  "$OUT_DIR/profile.log" | tee "$OUT_DIR/summary.txt"
sha256sum "$BIN" "$CK_LIB" "$0" >"$OUT_DIR/artifacts.sha256"
printf 'out_dir=%s\n' "$OUT_DIR"
