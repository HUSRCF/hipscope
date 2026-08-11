#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-1}
PAIRS=${PAIRS:-31}
RUNS=${RUNS:-5}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"$ROOT/experiments/gfx11-gate-up-x256y64/results/interleave_row_wmma_gpu${GPU_ID}_${STAMP}"}
BIN="$ROOT/target/release/examples/bench_hfq4_group256_direct"

mkdir -p "$OUT_DIR"

run_mode() {
    local label=$1
    local m=$2
    local k=$3
    local add=$4
    local mode=$5
    local add_arg=()
    local mode_arg=(--group128-quad-row-u32x2)
    if [[ "$add" == 1 ]]; then
        add_arg+=(--add)
    fi
    if [[ "$mode" == interleaved ]]; then
        mode_arg=(--group128-interleave-row-wmma)
    fi

    for ((trial = 1; trial <= RUNS; ++trial)); do
        env HIP_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
            --m "$m" --k "$k" --n 2048 --pairs "$PAIRS" \
            "${mode_arg[@]}" "${add_arg[@]}" \
            2>&1 | tee "$OUT_DIR/${label}_${mode}_trial${trial}.txt"
        sleep 2
    done
}

run_shape() {
    local label=$1 m=$2 k=$3 add=$4
    run_mode "$label" "$m" "$k" "$add" quad_row
    run_mode "$label" "$m" "$k" "$add" interleaved
}

{
    date '+%F %T %Z'
    rocm-smi --showproductname --showuse --showmemuse
} > "$OUT_DIR/system.txt" 2>&1

run_shape gate_up_set 17408 5120 0
run_shape down_residual_add 5120 17408 1

python3 - "$OUT_DIR" <<'PY'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for label in ("gate_up_set", "down_residual_add"):
    modes = {}
    for mode in ("quad_row", "interleaved"):
        trials = []
        for path in sorted(root.glob(f"{label}_{mode}_trial*.txt")):
            text = path.read_text()
            fields = {}
            for key in ("group128_lds_ms", "group256_ms", "max_abs", "mean_abs"):
                match = re.search(rf"^{key}=([0-9.eE+-]+)x?$", text, re.MULTILINE)
                if not match:
                    raise SystemExit(f"missing {key} in {path}")
                fields[key] = float(match.group(1))
            trials.append(fields)
        modes[mode] = trials
    rows.append((label, modes))

with (root / "summary.tsv").open("w") as out:
    out.write("shape\truns\tquad_row_ms_median\tinterleaved_ms_median\tinterleaved_vs_quad\tmax_abs_max\n")
    for label, modes in rows:
        quad = statistics.median(x["group256_ms"] for x in modes["quad_row"])
        interleaved = statistics.median(x["group256_ms"] for x in modes["interleaved"])
        max_abs = max(x["max_abs"] for trials in modes.values() for x in trials)
        out.write(f"{label}\t{len(modes['quad_row'])}\t{quad:.4f}\t{interleaved:.4f}\t{interleaved / quad:.4f}\t{max_abs:.8e}\n")
PY

cat "$OUT_DIR/summary.tsv"
printf 'results=%s\n' "$OUT_DIR"
