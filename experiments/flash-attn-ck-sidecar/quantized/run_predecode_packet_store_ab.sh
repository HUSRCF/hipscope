#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${GPU_ID:-1}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"$ROOT/results/predecode_packet_store_gpu${GPU_ID}_${STAMP}"}
BIN="$ROOT/build/quantized_kv_predecode_bench"

mkdir -p "$OUT_DIR"

bash "$ROOT/build_predecode_bench.sh" 2>&1 | tee "$OUT_DIR/build.txt"

{
    date '+%F %T %Z'
    rocm-smi --showproductname --showuse --showmemuse --showpids
} > "$OUT_DIR/system.txt" 2>&1

env HIP_VISIBLE_DEVICES="$GPU_ID" "$BIN" 2>&1 | tee "$OUT_DIR/results.txt"

printf 'results=%s\n' "$OUT_DIR"
