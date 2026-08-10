#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/examples/bench_hfq4_group256_direct}
GPU_ID=${GPU_ID:-1}
PAIRS=${PAIRS:-21}
IDLE_SECS=${IDLE_SECS:-5}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/experiments/gfx11-gate-up-x256y64/results/quad_row_min1_gpu1_$STAMP}
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "missing benchmark binary: $BIN" >&2
  exit 1
fi

cat >"$OUT/meta.txt" <<EOF
gpu_id=$GPU_ID
pairs=$PAIRS
idle_secs=$IDLE_SECS
shape=gate_up
m=17408
k=5120
n=2048
EOF
printf 'variant\treference_ms\tcandidate_ms\treference_speedup\tmax_abs\tmean_abs\n' >"$OUT/results.tsv"

run_variant() {
  local variant=$1 flag=$2 log="$OUT/$1.log"
  env HIP_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
    --m 17408 --k 5120 --n 2048 --pairs "$PAIRS" "$flag" 2>&1 | tee "$log"
  awk -v variant="$variant" '
    /^group128_lds_ms=/ {split($0,a,"="); base=a[2]}
    /^group256_ms=/ {split($0,a,"="); cand=a[2]}
    /^group256_speedup=/ {split($0,a,"="); speed=a[2]}
    /^max_abs=/ {split($0,a,"="); max=a[2]}
    /^mean_abs=/ {split($0,a,"="); mean=a[2]}
    END {printf "%s\t%s\t%s\t%s\t%s\t%s\n", variant,base,cand,speed,max,mean}
  ' "$log" >>"$OUT/results.tsv"
}

run_variant min_blocks_2 --group128-quad-row-u32x2
sleep "$IDLE_SECS"
run_variant min_blocks_1 --group128-quad-row-min1

sha256sum "$BIN" "$0" >"$OUT/artifacts.sha256"
cat "$OUT/results.tsv"
