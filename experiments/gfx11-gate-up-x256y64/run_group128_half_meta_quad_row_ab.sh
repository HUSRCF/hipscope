#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-1}
PAIRS=${PAIRS:-31}
RUNS=${RUNS:-5}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"$ROOT/experiments/gfx11-gate-up-x256y64/results/half_meta_quad_row_gpu${GPU_ID}_${STAMP}"}
BIN="$ROOT/target/release/examples/bench_hfq4_group256_direct"

mkdir -p "$OUT_DIR"

run_shape() {
    local label=$1 m=$2 k=$3 add=$4
    local add_arg=()
    if [[ "$add" == 1 ]]; then
        add_arg+=(--add)
    fi
    for ((trial = 1; trial <= RUNS; ++trial)); do
        env \
            -u HIPFIRE_RDNA3_Q8_GROUP128 \
            -u HIPFIRE_RDNA3_Q8_GROUP128_ROW2 \
            -u HIPFIRE_RDNA3_Q8_GROUP128_DUAL_ROW_WEIGHT \
            -u HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT \
            -u HIPFIRE_RDNA3_Q8_GROUP128_K128 \
            -u HIPFIRE_RDNA3_Q8_GROUP128_DIRECT \
            -u HIPFIRE_RDNA3_Q8_GROUP128_DIRECT_X512 \
            HIP_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
            --m "$m" --k "$k" --n 2048 --pairs "$PAIRS" \
            --group128-half-meta-quad-row "${add_arg[@]}" \
            2>&1 | tee "$OUT_DIR/${label}_trial${trial}.txt"
        sleep 2
    done
}

{
    date '+%F %T %Z'
    rocm-smi --showproductname --showuse --showmemuse
} > "$OUT_DIR/system.txt" 2>&1

run_shape gate_up_set 17408 5120 0
run_shape down_residual_add 5120 17408 1

python3 - "$OUT_DIR" "$RUNS" <<'PY'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
expected_runs = int(sys.argv[2])
rows = []
for label in ("gate_up_set", "down_residual_add"):
    trials = []
    for path in sorted(root.glob(f"{label}_trial*.txt")):
        text = path.read_text()
        if "reference_mode=group128-quad-row-u32x2" not in text:
            raise SystemExit(f"unexpected reference mode in {path}")
        fields = {}
        for key in ("group128_lds_ms", "group256_ms", "max_abs", "mean_abs"):
            match = re.search(rf"^{key}=([0-9.eE+-]+)x?$", text, re.MULTILINE)
            if not match:
                raise SystemExit(f"missing {key} in {path}")
            fields[key] = float(match.group(1))
        trials.append(fields)
    if len(trials) != expected_runs:
        raise SystemExit(
            f"expected {expected_runs} trials for {label}, found {len(trials)}"
        )
    rows.append((label, trials))

with (root / "summary.tsv").open("w") as out:
    out.write(
        "shape\truns\tquad_row_ms_median\thalf_meta_quad_row_ms_median\t"
        "paired_time_ratio_median\tmax_abs_max\n"
    )
    for label, trials in rows:
        baseline = statistics.median(x["group128_lds_ms"] for x in trials)
        candidate = statistics.median(x["group256_ms"] for x in trials)
        paired_ratio = statistics.median(
            trial["group256_ms"] / trial["group128_lds_ms"] for trial in trials
        )
        max_abs = max(x["max_abs"] for x in trials)
        out.write(
            f"{label}\t{len(trials)}\t{baseline:.4f}\t{candidate:.4f}\t"
            f"{paired_ratio:.4f}\t{max_abs:.8e}\n"
        )
PY

cat "$OUT_DIR/summary.tsv"
printf 'results=%s\n' "$OUT_DIR"
