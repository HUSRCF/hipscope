#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BIN="$ROOT/target/release/examples/bench_hfq4_group128_packed_weight_y128"
GPU=${GPU:-1}
PAIRS=${PAIRS:-5}
INTERNAL_PAIRS=${INTERNAL_PAIRS:-11}
COOL_SECS=${COOL_SECS:-2}
TARGET_WIDTH=${TARGET_WIDTH:-9984}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-"$ROOT/experiments/gfx11-mq4-v2/ffn-width-work-reduction/results/exact-${TARGET_WIDTH}-abba-gpu${GPU}-${STAMP}"}

mkdir -p "$OUT"
printf 'pair\torder\tfull_gate_ms\ttarget_gate_ms\tfull_down_ms\ttarget_down_ms\tweighted_speedup\n' > "$OUT/results.tsv"

run_shape() {
    local label=$1
    local m=$2
    local k=$3
    local pair=$4
    local slot=$5
    local log="$OUT/pair_${pair}_${slot}_${label}.txt"

    HIP_VISIBLE_DEVICES="$GPU" "$BIN" \
        --m "$m" --k "$k" --n 2048 --pairs "$INTERNAL_PAIRS" \
        > "$log" 2>&1
    awk -F= '/^x256_y64_group128_ms=/{print $2}' "$log"
}

# Compile and establish the same DPM state for all four shapes before recording.
HIP_VISIBLE_DEVICES="$GPU" "$BIN" --m 17408 --k 5120 --n 2048 --pairs 3 >/dev/null 2>&1
HIP_VISIBLE_DEVICES="$GPU" "$BIN" --m "$TARGET_WIDTH" --k 5120 --n 2048 --pairs 3 >/dev/null 2>&1
HIP_VISIBLE_DEVICES="$GPU" "$BIN" --m 5120 --k 17408 --n 2048 --pairs 3 >/dev/null 2>&1
HIP_VISIBLE_DEVICES="$GPU" "$BIN" --m 5120 --k "$TARGET_WIDTH" --n 2048 --pairs 3 >/dev/null 2>&1

for ((pair = 0; pair < PAIRS; pair++)); do
    if ((pair % 2 == 0)); then
        order=FT
        full_gate=$(run_shape full_gate 17408 5120 "$pair" 0)
        target_gate=$(run_shape target_gate "$TARGET_WIDTH" 5120 "$pair" 1)
        full_down=$(run_shape full_down 5120 17408 "$pair" 2)
        target_down=$(run_shape target_down 5120 "$TARGET_WIDTH" "$pair" 3)
    else
        order=TF
        target_gate=$(run_shape target_gate "$TARGET_WIDTH" 5120 "$pair" 0)
        full_gate=$(run_shape full_gate 17408 5120 "$pair" 1)
        target_down=$(run_shape target_down 5120 "$TARGET_WIDTH" "$pair" 2)
        full_down=$(run_shape full_down 5120 17408 "$pair" 3)
    fi

    weighted=$(awk -v fg="$full_gate" -v tg="$target_gate" \
        -v fd="$full_down" -v td="$target_down" \
        'BEGIN { printf "%.8f", (2 * fg + fd) / (2 * tg + td) }')
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$pair" "$order" "$full_gate" "$target_gate" \
        "$full_down" "$target_down" "$weighted" | tee -a "$OUT/results.tsv"
    sleep "$COOL_SECS"
done

printf 'results=%s\n' "$OUT/results.tsv"
