#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${GPU_ID:-1}
RUNS=${RUNS:-5}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"$ROOT/results/staged_packet_store_gpu${GPU_ID}_${STAMP}"}
REFERENCE_WORKTREE=${REFERENCE_WORKTREE:-"/home/husrcf/Code/ProtBind/unidec/.worktrees/hipfire-flashattn-beta-latest"}
CK_ROOT=${CK_ROOT:-"$REFERENCE_WORKTREE/experiments/flash-attn-ck-sidecar/build/ck-source"}
DENSE_SIDECAR=${DENSE_SIDECAR:-"$REFERENCE_WORKTREE/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so"}

mkdir -p "$OUT_DIR/scalar" "$OUT_DIR/packet"

build_arm() {
    local arm=$1 packet_store=$2
    local arm_dir="$OUT_DIR/$arm"
    local sidecar="$arm_dir/libhipfire_flash_attn_ck_quantized_${arm}.so"
    env \
        CK_ROOT="$CK_ROOT" \
        DENSE_SIDECAR="$DENSE_SIDECAR" \
        BUILD_DIR="$arm_dir" \
        OUT="$sidecar" \
        STAGED=1 \
        PACKET_STORE="$packet_store" \
        bash "$ROOT/build_quantized_sidecar.sh" \
        2>&1 | tee "$arm_dir/build_sidecar.txt"
    env \
        BUILD_DIR="$arm_dir" \
        QUANTIZED_SIDECAR="$sidecar" \
        DENSE_SIDECAR="$DENSE_SIDECAR" \
        bash "$ROOT/build_staged_ck_bench.sh" \
        2>&1 | tee "$arm_dir/build_bench.txt"
}

build_arm scalar 0
build_arm packet 1

{
    date '+%F %T %Z'
    rocm-smi --showproductname --showuse --showmemuse --showpids
} > "$OUT_DIR/system.txt" 2>&1

for ((trial = 1; trial <= RUNS; ++trial)); do
    if (( trial % 2 )); then
        arms=(scalar packet)
    else
        arms=(packet scalar)
    fi
    for arm in "${arms[@]}"; do
        env HIP_VISIBLE_DEVICES="$GPU_ID" \
            "$OUT_DIR/$arm/staged_quantized_ck_bench" \
            2>&1 | tee "$OUT_DIR/${arm}_trial${trial}.txt"
        sleep 2
    done
done

python3 - "$OUT_DIR" "$RUNS" <<'PY'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
expected_runs = int(sys.argv[2])
rows = []
for seqlen_k in (2048, 4096, 6144, 8192):
    arms = {}
    for arm in ("scalar", "packet"):
        values = []
        for path in sorted(root.glob(f"{arm}_trial*.txt")):
            text = path.read_text()
            match = re.search(
                rf"seqlen_q=2048,seqlen_k={seqlen_k},.*?staged_ck_ms=([0-9.eE+-]+)",
                text,
            )
            if not match:
                raise SystemExit(f"missing K={seqlen_k} staged result in {path}")
            values.append(float(match.group(1)))
        if len(values) != expected_runs:
            raise SystemExit(
                f"expected {expected_runs} runs for K={seqlen_k}/{arm}, found {len(values)}"
            )
        arms[arm] = values
    scalar = statistics.median(arms["scalar"])
    packet = statistics.median(arms["packet"])
    ratios = [p / s for s, p in zip(arms["scalar"], arms["packet"])]
    rows.append((seqlen_k, scalar, packet, statistics.median(ratios)))

with (root / "summary.tsv").open("w") as out:
    out.write("seqlen_q\tseqlen_k\truns\tscalar_ms\tpacket_ms\tpacket_time_ratio\n")
    for seqlen_k, scalar, packet, ratio in rows:
        out.write(f"2048\t{seqlen_k}\t{expected_runs}\t{scalar:.6f}\t{packet:.6f}\t{ratio:.6f}\n")
PY

cat "$OUT_DIR/summary.tsv"
printf 'results=%s\n' "$OUT_DIR"
