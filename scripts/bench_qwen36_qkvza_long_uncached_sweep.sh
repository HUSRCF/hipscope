#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RDNA3 Qwen3.6 QKVZA split-tail long-uncached-prefill crossover sweep.
#
# The expensive measurements are counterbalanced baseline/active fresh-process
# pairs per prompt length. Threshold rows are projected from those measured cells:
# below a threshold the implementation is byte-for-byte the baseline route;
# at or above it the implementation is the measured active route. This avoids
# reloading a 27B model for redundant threshold/length combinations while still
# producing a reviewable threshold policy table.
#
# Example:
#   HIP_VISIBLE_DEVICES=0 \
#   LENGTHS="512 1024 2048 4096 8192 16384" \
#   THRESHOLDS="512 1024 2048 4096 8192" \
#   PREFILL_RUNS=5 \
#   ./scripts/bench_qwen36_qkvza_long_uncached_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-${HIP_VISIBLE_DEVICES:-0}}"
LENGTHS="${LENGTHS:-512 1024 2048 4096 8192 16384}"
THRESHOLDS="${THRESHOLDS:-512 1024 2048 4096 8192 16384}"
PREFILL_RUNS="${PREFILL_RUNS:-5}"
PROCESS_REPEATS="${PROCESS_REPEATS:-1}"
SLEEP_BETWEEN_MODES="${SLEEP_BETWEEN_MODES:-20}"
SLEEP_BETWEEN_PAIRS="${SLEEP_BETWEEN_PAIRS:-30}"
SLEEP_BETWEEN_LENGTHS="${SLEEP_BETWEEN_LENGTHS:-30}"
APPEND_RESULTS_FROM="${APPEND_RESULTS_FROM:-}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/qkvza_long_uncached_sweep_$(date +%Y%m%d_%H%M%S)}"
AB_SCRIPT="${AB_SCRIPT:-./scripts/bench_qwen36_qkvza_split_tail_ab.sh}"

mkdir -p "$RESULT_DIR/cells"

if [[ ! -x "$AB_SCRIPT" ]]; then
    echo "benchmark driver is not executable: $AB_SCRIPT" >&2
    exit 2
fi

case " $LENGTHS " in
    *" 0 "*) echo "LENGTHS must contain positive integers" >&2; exit 2 ;;
esac

raw_header=$'length\tpair\torder\tmode\trun\tprefill_ms\tprefill_tok_s'
route_header=$'length\tpair\torder\tmode\teligible_events\troute_hit_events'
printf '%s\n' "$raw_header" >"$RESULT_DIR/raw.tsv"
printf '%s\n' "$route_header" >"$RESULT_DIR/routes.tsv"

for prior in $APPEND_RESULTS_FROM; do
    if [[ ! -s "$prior/raw.tsv" || ! -s "$prior/routes.tsv" ]]; then
        echo "prior result is missing raw.tsv/routes.tsv: $prior" >&2
        exit 2
    fi
    if [[ "$(head -n 1 "$prior/raw.tsv")" != "$raw_header" || \
          "$(head -n 1 "$prior/routes.tsv")" != "$route_header" ]]; then
        echo "prior result uses a legacy schema without pair identity: $prior" >&2
        exit 2
    fi
    awk 'NR > 1' "$prior/raw.tsv" >>"$RESULT_DIR/raw.tsv"
    awk 'NR > 1' "$prior/routes.tsv" >>"$RESULT_DIR/routes.tsv"
done

{
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo "date=$(date -Is)"
    echo "gpu_id=$GPU_ID"
    echo "lengths=$LENGTHS"
    echo "thresholds=$THRESHOLDS"
    echo "prefill_runs=$PREFILL_RUNS"
    echo "process_repeats=$PROCESS_REPEATS"
    echo "append_results_from=$APPEND_RESULTS_FROM"
    echo "active_route_measurement_threshold=1"
    echo
    echo "rocm_smi_before:"
    rocm-smi --showproductname --showuse --showmemuse --showpids 2>/dev/null || true
} >"$RESULT_DIR/meta.txt"

read -r -a length_values <<<"$LENGTHS"
index=0
for length in "${length_values[@]}"; do
    if ! [[ "$length" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid length: $length" >&2
        exit 2
    fi

    cell="$RESULT_DIR/cells/pp${length}"
    for ((repeat = 0; repeat < PROCESS_REPEATS; repeat++)); do
        # Counterbalance AB/BA both across lengths and fresh-process pairs.
        if (( (index + repeat) % 2 == 0 )); then
            mode_sequence="off on"
        else
            mode_sequence="on off"
        fi
        pair="$cell/pair$(printf '%02d' $((repeat + 1)))"
        pair_number=$((repeat + 1))
        order=${mode_sequence// /-}
        echo "===== prefill=$length pair=$((repeat + 1))/$PROCESS_REPEATS order=[$mode_sequence] ====="
        GPU_ID="$GPU_ID" \
        PREFILL="$length" \
        PREFILL_RUNS="$PREFILL_RUNS" \
        MIN_PREFILL_TOKENS=1 \
        DIAG=1 \
        MODE_SEQUENCE="$mode_sequence" \
        SLEEP_BETWEEN_MODES="$SLEEP_BETWEEN_MODES" \
        RESULT_DIR="$pair" \
            "$AB_SCRIPT"

        awk -F '\t' -v plen="$length" -v pair="$pair_number" -v order="$order" 'NR > 1 {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", plen, pair, order, $2, $3, $4, $5
        }' "$pair/summary.tsv" >>"$RESULT_DIR/raw.tsv"
        awk -v plen="$length" -v pair="$pair_number" -v order="$order" '
            {
                mode = eligible = hits = ""
                for (i = 1; i <= NF; i++) {
                    split($i, kv, "=")
                    if (kv[1] == "mode") mode = kv[2]
                    if (kv[1] == "eligible_events") eligible = kv[2]
                    if (kv[1] == "route_hit_events") hits = kv[2]
                }
                if (mode != "") {
                    printf "%s\t%s\t%s\t%s\t%s\t%s\n", plen, pair, order, mode, eligible, hits
                }
            }
        ' "$pair/route_summary.txt" >>"$RESULT_DIR/routes.tsv"

        if (( repeat + 1 < PROCESS_REPEATS )); then
            sleep "$SLEEP_BETWEEN_PAIRS"
        fi
    done

    index=$((index + 1))
    if (( index < ${#length_values[@]} )); then
        sleep "$SLEEP_BETWEEN_LENGTHS"
    fi
done

python3 scripts/analyze_qkvza_long_uncached.py "$RESULT_DIR" $THRESHOLDS

echo "results: $RESULT_DIR"
