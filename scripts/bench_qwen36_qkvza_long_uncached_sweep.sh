#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RDNA3 Qwen3.6 QKVZA split-tail long-uncached-prefill crossover sweep.
#
# The expensive measurements are one baseline and one active-route process per
# prompt length. Threshold rows are then projected from those measured cells:
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

printf 'length\tmode\trun\tprefill_ms\tprefill_tok_s\n' >"$RESULT_DIR/raw.tsv"
printf 'length\tmode\teligible_events\troute_hit_events\n' >"$RESULT_DIR/routes.tsv"

for prior in $APPEND_RESULTS_FROM; do
    if [[ ! -s "$prior/raw.tsv" || ! -s "$prior/routes.tsv" ]]; then
        echo "prior result is missing raw.tsv/routes.tsv: $prior" >&2
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

        awk -F '\t' -v plen="$length" 'NR > 1 {
            printf "%s\t%s\t%s\t%s\t%s\n", plen, $2, $3, $4, $5
        }' "$pair/summary.tsv" >>"$RESULT_DIR/raw.tsv"
        awk -v plen="$length" '
            {
                mode = eligible = hits = ""
                for (i = 1; i <= NF; i++) {
                    split($i, kv, "=")
                    if (kv[1] == "mode") mode = kv[2]
                    if (kv[1] == "eligible_events") eligible = kv[2]
                    if (kv[1] == "route_hit_events") hits = kv[2]
                }
                if (mode != "") {
                    printf "%s\t%s\t%s\t%s\n", plen, mode, eligible, hits
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

python3 - "$RESULT_DIR" "$THRESHOLDS" <<'PY'
import csv
import math
import pathlib
import statistics
import sys

result_dir = pathlib.Path(sys.argv[1])
thresholds = [int(v) for v in sys.argv[2].split()]
rows = list(csv.DictReader((result_dir / "raw.tsv").open(), delimiter="\t"))
route_rows = list(csv.DictReader((result_dir / "routes.tsv").open(), delimiter="\t"))
route_by_cell = {}
for row in route_rows:
    key = (int(row["length"]), row["mode"])
    eligible, hits = route_by_cell.get(key, (0, 0))
    route_by_cell[key] = (
        eligible + int(row["eligible_events"]),
        hits + int(row["route_hit_events"]),
    )

by_cell = {}
for row in rows:
    key = (int(row["length"]), row["mode"])
    by_cell.setdefault(key, []).append(float(row["prefill_tok_s"]))

def percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

length_rows = []
for length in sorted({key[0] for key in by_cell}):
    off = by_cell.get((length, "off"), [])
    on = by_cell.get((length, "on"), [])
    if not off or not on:
        raise SystemExit(f"missing off/on samples for prefill={length}")
    off_med = statistics.median(off)
    on_med = statistics.median(on)
    delta = (on_med / off_med - 1.0) * 100.0
    off_eligible, off_hits = route_by_cell.get((length, "off"), (0, 0))
    on_eligible, on_hits = route_by_cell.get((length, "on"), (0, 0))
    if off_eligible != 0 or off_hits != 0:
        raise SystemExit(f"off route unexpectedly eligible/hit at prefill={length}")
    if on_eligible == 0 or on_hits == 0:
        raise SystemExit(f"active route did not report eligibility/hit at prefill={length}")
    length_rows.append({
        "length": length,
        "off": off_med,
        "on": on_med,
        "delta": delta,
        "off_p25": percentile(off, 0.25),
        "off_p75": percentile(off, 0.75),
        "on_p25": percentile(on, 0.25),
        "on_p75": percentile(on, 0.75),
        "samples": min(len(off), len(on)),
        "active_eligible": on_eligible,
        "active_hits": on_hits,
    })

with (result_dir / "length_summary.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "prefill_tokens", "off_median_tok_s", "active_median_tok_s",
        "delta_pct", "off_p25", "off_p75", "active_p25", "active_p75",
        "samples_per_mode", "active_eligible_events", "active_route_hit_events",
    ])
    for r in length_rows:
        w.writerow([
            r["length"], f'{r["off"]:.3f}', f'{r["on"]:.3f}',
            f'{r["delta"]:.3f}', f'{r["off_p25"]:.3f}', f'{r["off_p75"]:.3f}',
            f'{r["on_p25"]:.3f}', f'{r["on_p75"]:.3f}', r["samples"],
            r["active_eligible"], r["active_hits"],
        ])

policy_rows = []
for threshold in thresholds:
    active = [r for r in length_rows if r["length"] >= threshold]
    ratios = [(r["on"] / r["off"]) if r["length"] >= threshold else 1.0
              for r in length_rows]
    geometric = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
    active_deltas = [r["delta"] for r in active]
    policy_rows.append({
        "threshold": threshold,
        "active_lengths": ",".join(str(r["length"]) for r in active) or "none",
        "active_points": len(active),
        "median_active": statistics.median(active_deltas) if active_deltas else 0.0,
        "worst_active": min(active_deltas) if active_deltas else 0.0,
        "regressions": sum(v < 0.0 for v in active_deltas),
        "all_geomean": (geometric - 1.0) * 100.0,
    })

with (result_dir / "threshold_projection.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "threshold", "active_lengths", "active_points", "median_active_delta_pct",
        "worst_active_delta_pct", "regressed_active_points",
        "equal_weight_all_length_geomean_delta_pct",
    ])
    for r in policy_rows:
        w.writerow([
            r["threshold"], r["active_lengths"], r["active_points"],
            f'{r["median_active"]:.3f}', f'{r["worst_active"]:.3f}',
            r["regressions"], f'{r["all_geomean"]:.3f}',
        ])

with (result_dir / "report.md").open("w") as f:
    f.write("# RDNA3 QKVZA long-uncached-prefill crossover\n\n")
    f.write("## Measured length A/B\n\n")
    f.write("| Prefill tokens | Off median tok/s | Active median tok/s | Delta | Off IQR | Active IQR | Samples/mode | Route hit |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
    for r in length_rows:
        f.write(
            f'| {r["length"]} | {r["off"]:.1f} | {r["on"]:.1f} | '
            f'{r["delta"]:+.2f}% | {r["off_p25"]:.1f}-{r["off_p75"]:.1f} | '
            f'{r["on_p25"]:.1f}-{r["on_p75"]:.1f} | {r["samples"]} | '
            f'{r["active_hits"]} |\n'
        )
    f.write("\n## Threshold policy projection\n\n")
    f.write(
        "Rows below each threshold use the measured off route; rows at or above "
        "it use the measured active route. These are policy projections from the "
        "measured cells above, not redundant benchmark reruns.\n\n"
    )
    f.write("| Threshold | Activated tested lengths | Median active delta | Worst active delta | Regressed active points | All-length geomean delta |\n")
    f.write("|---:|:---|---:|---:|---:|---:|\n")
    for r in policy_rows:
        f.write(
            f'| {r["threshold"]} | {r["active_lengths"]} | '
            f'{r["median_active"]:+.2f}% | {r["worst_active"]:+.2f}% | '
            f'{r["regressions"]} | {r["all_geomean"]:+.2f}% |\n'
        )

print((result_dir / "report.md").read_text())
PY

echo "results: $RESULT_DIR"
