#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
RUNS="${RUNS:-5}"
PAIRS="${PAIRS:-15}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/group128_n2_reuse_${STAMP}}"
BIN="${ROOT}/target/release/examples/bench_hfq4_group256_direct"

mkdir -p "${OUT}"

for run in $(seq 1 "${RUNS}"); do
    HIP_VISIBLE_DEVICES="${GPU_ID}" "${BIN}" \
        --group128-n2-reuse --pairs "${PAIRS}" \
        >"${OUT}/gate_up_run_${run}.txt" 2>&1
    sleep 2
    HIP_VISIBLE_DEVICES="${GPU_ID}" "${BIN}" \
        --group128-n2-reuse --m 5120 --k 17408 --n 2048 --add --pairs "${PAIRS}" \
        >"${OUT}/down_residual_run_${run}.txt" 2>&1
    sleep 2
done

python3 - "${OUT}" <<'PY'
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
for stem in ("gate_up", "down_residual"):
    rows = []
    for path in sorted(out.glob(f"{stem}_run_*.txt")):
        parsed = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                parsed[key] = value
        rows.append((
            path.name,
            float(parsed["group128_lds_ms"]),
            float(parsed["group256_ms"]),
            float(parsed["group256_speedup"].removesuffix("x")),
            float(parsed["max_abs"]),
        ))
    speedups = [row[3] for row in rows]
    print(stem)
    for row in rows:
        print(f"  {row[0]} baseline_ms={row[1]:.4f} candidate_ms={row[2]:.4f} speedup={row[3]:.4f}x max_abs={row[4]:.3e}")
    print(f"  process_median_speedup={statistics.median(speedups):.4f}x")
PY
