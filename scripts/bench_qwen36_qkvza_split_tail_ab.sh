#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3.6-27B RDNA3 QKVZA split-tail A/B prefill benchmark.
#
# This is intentionally narrow: it exercises the production
# forward_prefill_batch path through bench_qwen35_mq4 while toggling only
# HIPFIRE_QKVZA_SPLIT_TAIL. It is meant to accompany the gfx1100/1101/1102
# opt-in patch and provide reviewer-reproducible evidence.

set -euo pipefail

cd "$(dirname "$0")/.."

EXE="${EXE:-./target/release/examples/bench_qwen35_mq4}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-0}"
PREFILL="${PREFILL:-4096}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
PREFILL_WARMUP_RUNS="${PREFILL_WARMUP_RUNS:-1}"
GEN="${GEN:-1}"
WARMUP="${WARMUP:-0}"
KV_MODE="${KV_MODE:-q8}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-5}"
MMQ_SCREEN="${MMQ_SCREEN:-1}"
MIN_PREFILL_TOKENS="${MIN_PREFILL_TOKENS:-4096}"
DIAG="${DIAG:-1}"
OUTER_CHUNKED="${OUTER_CHUNKED:-0}"
PROMPT_FILE="${PROMPT_FILE:-}"
TIMEOUT_SECS="${TIMEOUT_SECS:-360}"
PROCESS_REPEATS="${PROCESS_REPEATS:-3}"
MODE_SEQUENCE="${MODE_SEQUENCE:-}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/qkvza_split_tail_rdna3_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULT_DIR"

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
    echo "prompt file not found: $PROMPT_FILE" >&2
    exit 2
fi
if [[ ! "$PREFILL_RUNS" =~ ^[0-9]+$ \
    || ! "$PREFILL_WARMUP_RUNS" =~ ^[0-9]+$ \
    || ! "$PROCESS_REPEATS" =~ ^[0-9]+$ ]]; then
    echo "PREFILL_RUNS, PREFILL_WARMUP_RUNS, and PROCESS_REPEATS must be decimal integers" >&2
    exit 2
fi
if (( PREFILL_RUNS < 1 )); then
    echo "PREFILL_RUNS must be >= 1" >&2
    exit 2
fi
if [[ -z "$MODE_SEQUENCE" ]] && (( PROCESS_REPEATS < 1 )); then
    echo "PROCESS_REPEATS must be >= 1 when MODE_SEQUENCE is not set" >&2
    exit 2
fi

if [[ -z "$MODE_SEQUENCE" ]]; then
    mode_parts=()
    for ((pair = 1; pair <= PROCESS_REPEATS; pair++)); do
        if (( pair % 2 == 1 )); then
            mode_parts+=(off on)
        else
            mode_parts+=(on off)
        fi
    done
else
    read -r -a mode_parts <<<"$MODE_SEQUENCE"
fi
if (( ${#mode_parts[@]} == 0 || ${#mode_parts[@]} % 2 != 0 )); then
    echo "MODE_SEQUENCE must contain complete off/on pairs" >&2
    exit 2
fi
for ((i = 0; i < ${#mode_parts[@]}; i += 2)); do
    if [[ "${mode_parts[i]} ${mode_parts[i + 1]}" != "off on" \
        && "${mode_parts[i]} ${mode_parts[i + 1]}" != "on off" ]]; then
        echo "MODE_SEQUENCE pair $((i / 2 + 1)) must be 'off on' or 'on off'" >&2
        exit 2
    fi
done
MODE_SEQUENCE="${mode_parts[*]}"

export HIP_VISIBLE_DEVICES="$GPU_ID"
export HIPFIRE_KV_MODE="$KV_MODE"
export HIPFIRE_DPM_WARMUP_SECS="$DPM_WARMUP_SECS"
export HIPFIRE_MMQ_SCREEN="$MMQ_SCREEN"
export HIPFIRE_QKVZA_SPLIT_TAIL_MIN_PREFILL_TOKENS="$MIN_PREFILL_TOKENS"
export HIPFIRE_QKVZA_SPLIT_TAIL_DIAG="$DIAG"

summary_tsv="$RESULT_DIR/summary.tsv"
meta_txt="$RESULT_DIR/meta.txt"
route_summary_txt="$RESULT_DIR/route_summary.txt"

{
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo "date=$(date -Is)"
    echo "exe=$EXE"
    echo "model=$MODEL"
    echo "gpu_id=$GPU_ID"
    echo "prefill=$PREFILL"
    echo "prefill_runs=$PREFILL_RUNS"
    echo "prefill_warmup_runs=$PREFILL_WARMUP_RUNS"
    echo "gen=$GEN"
    echo "warmup=$WARMUP"
    echo "kv_mode=$KV_MODE"
    echo "dpm_warmup_secs=$DPM_WARMUP_SECS"
    echo "mmq_screen=$MMQ_SCREEN"
    echo "min_prefill_tokens=$MIN_PREFILL_TOKENS"
    echo "diag=$DIAG"
    echo "outer_chunked=$OUTER_CHUNKED"
    echo "prompt_file=${PROMPT_FILE:-synthetic-token-ids}"
    if [[ -n "$PROMPT_FILE" ]]; then
        echo "prompt_file_sha256=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')"
    fi
    echo "timeout_secs=$TIMEOUT_SECS"
    echo "counterbalanced_pairs=$((${#mode_parts[@]} / 2))"
    echo "mode_sequence=$MODE_SEQUENCE"
    echo
    echo "git_status:"
    git status --short 2>/dev/null || true
    echo
    echo "rocm_smi:"
    rocm-smi 2>/dev/null || true
} >"$meta_txt"

printf "seq\tmode\trun\tprefill_ms\tprefill_tok_s\n" >"$summary_tsv"
: >"$route_summary_txt"

run_mode() {
    local seq="$1"
    local mode="$2"
    local log="$RESULT_DIR/$(printf "%02d" "$seq")_${mode}.log"

    if [[ "$mode" == "on" ]]; then
        export HIPFIRE_QKVZA_SPLIT_TAIL=1
    else
        unset HIPFIRE_QKVZA_SPLIT_TAIL
    fi

    echo "===== mode=$mode ====="
    local extra_args=()
    local total_prefill_runs=$((PREFILL_WARMUP_RUNS + PREFILL_RUNS))
    if [[ "$OUTER_CHUNKED" == "1" ]]; then
        extra_args+=(--outer-chunked)
    fi
    if [[ -n "$PROMPT_FILE" ]]; then
        extra_args+=(--prompt-file "$PROMPT_FILE")
    fi

    timeout "$TIMEOUT_SECS" "$EXE" "$MODEL" \
        --prefill "$PREFILL" \
        --prefill-runs "$total_prefill_runs" \
        --gen "$GEN" \
        --warmup "$WARMUP" \
        "${extra_args[@]}" \
        2>&1 | tee "$log"

    awk -v seq="$seq" -v mode="$mode" -v warmup="$PREFILL_WARMUP_RUNS" '
        /run[[:space:]]+[0-9]+:/ {
            run=$2
            sub(":", "", run)
            if (run <= warmup) {
                next
            }
            ms=$3
            sub("ms", "", ms)
            tps=$4
            printf "%s\t%s\t%s\t%s\t%s\n", seq, mode, run - warmup, ms, tps
            printed++
        }
        /^PREFILL_SUMMARY/ && printed == 0 {
            tps = ""
            ms = ""
            for (i = 1; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == "prefill_tok_s") {
                    tps = kv[2]
                } else if (kv[1] == "prefill_wall_ms") {
                    ms = kv[2]
                }
            }
            if (tps != "" && ms != "") {
                printf "%s\t%s\t%s\t%s\t%s\n", seq, mode, 1, ms, tps
                printed++
            }
        }
    ' "$log" >>"$summary_tsv"

    local sample_count
    sample_count=$(awk -F '\t' -v seq="$seq" 'NR > 1 && $1 == seq { count++ } END { print count + 0 }' "$summary_tsv")
    if (( sample_count != PREFILL_RUNS )); then
        echo "mode=$mode process=$seq produced $sample_count measured samples; expected $PREFILL_RUNS" >&2
        exit 1
    fi

    local eligible route_hits
    eligible=$(grep -c 'qkvza_split_tail request eligible=true' "$log" || true)
    route_hits=$(grep -c 'qkvza_split_tail route=hit' "$log" || true)
    printf 'mode=%s eligible_events=%s route_hit_events=%s\n' \
        "$mode" "$eligible" "$route_hits" | tee -a "$route_summary_txt"
    if [[ "$DIAG" == "1" ]]; then
        if [[ "$mode" == "on" && ( "$eligible" -eq 0 || "$route_hits" -eq 0 ) ]]; then
            echo "active process did not report an eligible request and route hit" >&2
            exit 1
        fi
        if [[ "$mode" == "off" && ( "$eligible" -ne 0 || "$route_hits" -ne 0 ) ]]; then
            echo "off process unexpectedly reported an eligible request or route hit" >&2
            exit 1
        fi
    fi
}

seq=1
for mode in "${mode_parts[@]}"; do
    case "$mode" in
        off|on) ;;
        *)
            echo "unknown mode in MODE_SEQUENCE: $mode (expected off/on)" >&2
            exit 2
            ;;
    esac
    if [[ "$seq" -gt 1 ]]; then
        sleep "${SLEEP_BETWEEN_MODES:-10}"
    fi
    run_mode "$seq" "$mode"
    seq=$((seq + 1))
done

python3 - "$summary_tsv" "$PREFILL_WARMUP_RUNS" "${#mode_parts[@]}" "$PREFILL_RUNS" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
warmup_runs = int(sys.argv[2])
expected_processes = int(sys.argv[3])
expected_runs = int(sys.argv[4])
rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
if len(rows) != expected_processes * expected_runs:
    raise SystemExit(
        f"expected {expected_processes * expected_runs} measured samples, got {len(rows)}"
    )
by_mode = {}
by_process = {}
for row in rows:
    value = float(row["prefill_tok_s"])
    by_mode.setdefault(row["mode"], []).append(value)
    by_process.setdefault((int(row["seq"]), row["mode"]), []).append(value)

print("\n===== median summary =====")
print(f"excluded_prefill_warmup_runs_per_process={warmup_runs}")
for mode in ("off", "on"):
    vals = by_mode.get(mode, [])
    if not vals:
        print(f"{mode}\tNA")
        continue
    print(f"{mode}\tmedian_prefill_tok_s={statistics.median(vals):.3f}\truns={len(vals)}")

if by_mode.get("off") and by_mode.get("on"):
    off = statistics.median(by_mode["off"])
    on = statistics.median(by_mode["on"])
    delta = (on / off - 1.0) * 100.0
    print(f"delta_on_vs_off={delta:.2f}%")

processes = sorted((seq, mode, statistics.median(vals)) for (seq, mode), vals in by_process.items())
pair_deltas = []
for pair_start in range(0, len(processes) - 1, 2):
    pair = processes[pair_start:pair_start + 2]
    values = {mode: value for _, mode, value in pair}
    if set(values) == {"off", "on"}:
        delta = (values["on"] / values["off"] - 1.0) * 100.0
        pair_deltas.append(delta)
        print(f"pair_{pair_start // 2 + 1}_delta={delta:+.2f}%")
if pair_deltas:
    print(f"paired_median_delta={statistics.median(pair_deltas):+.2f}%")
PY

echo
echo "results: $RESULT_DIR"
