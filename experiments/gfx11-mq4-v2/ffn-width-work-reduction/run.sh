#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BIN="$ROOT/target/release/examples/bench_hfq4_group128_packed_weight_y128"
OUT=${OUT:-"$ROOT/experiments/gfx11-mq4-v2/ffn-width-work-reduction/results"}
GPU=${GPU:-1}
PAIRS=${PAIRS:-11}

mkdir -p "$OUT"

for width in 17408 15232 13056 10880 10496 10112 9984 9728; do
    HIP_VISIBLE_DEVICES="$GPU" "$BIN" \
        --m "$width" --k 5120 --n 2048 --pairs "$PAIRS" \
        | tee "$OUT/gate_${width}.log"
    sleep 3
done

for width in 17408 15104 13056 10752 10496 9984 9728; do
    HIP_VISIBLE_DEVICES="$GPU" "$BIN" \
        --m 5120 --k "$width" --n 2048 --pairs "$PAIRS" \
        | tee "$OUT/down_${width}.log"
    sleep 3
done
