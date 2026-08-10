#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/examples/bench_hfq4_group256_direct}
GPU_ID=${GPU_ID:-1}
PAIRS=${PAIRS:-15}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/experiments/gfx11-gate-up-x256y64/results/group128_direct_hot_shapes_gpu1_$STAMP}
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "missing benchmark binary: $BIN" >&2
  exit 1
fi

cat >"$OUT/meta.txt" <<EOF
gpu_id=$GPU_ID
pairs=$PAIRS
candidate=group128-direct
EOF
printf 'shape\tmode\tbaseline_ms\tcandidate_ms\tspeedup\tmax_abs\tmean_abs\n' >"$OUT/results.tsv"

run_shape() {
  local name=$1 m=$2 k=$3 mode=$4
  local log="$OUT/$name.log"
  local extra=()
  if [[ "$mode" == add ]]; then extra+=(--add); fi
  env HIP_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
    --m "$m" --k "$k" --n 2048 --pairs "$PAIRS" \
    --group128-direct "${extra[@]}" 2>&1 | tee "$log"
  awk -v shape="$name" -v mode="$mode" '
    /^group128_lds_ms=/ {split($0,a,"="); base=a[2]}
    /^group256_ms=/ {split($0,a,"="); cand=a[2]}
    /^group256_speedup=/ {split($0,a,"="); speed=a[2]}
    /^max_abs=/ {split($0,a,"="); max=a[2]}
    /^mean_abs=/ {split($0,a,"="); mean=a[2]}
    END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", shape,mode,base,cand,speed,max,mean}
  ' "$log" >>"$OUT/results.tsv"
}

run_shape gate_up 17408 5120 set
run_shape qkvza 10240 5120 set
run_shape gdn_out 6144 5120 set
run_shape attn_qkv 12288 5120 set
run_shape ffn_down 5120 17408 add
run_shape aux_down 5120 6144 add

sha256sum "$BIN" "$0" >"$OUT/artifacts.sha256"
cat "$OUT/results.tsv"
