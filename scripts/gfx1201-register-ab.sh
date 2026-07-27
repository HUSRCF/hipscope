#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Reproduce the gfx1201 retained-PM4 register-map A/B on one GPU.
#
# A and B are built from detached worktrees.  Their Rust binaries are isolated,
# while HIPFIRE_KERNEL_CACHE is shared and hashed before and after B so that a
# difference in generated GPU code cannot masquerade as a PM4 register result.
#
# The runtime-defect claim is supported only when A fails the bit-exact PM4/HIP
# shadow gate and B passes it under the same environment and code objects.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel)"
cd "$ROOT"
A_REF="origin/beta"
B_REF="HEAD"
MODEL="$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r"
DEVICE=0
PERF_PASSES=1
OUT=""

usage() {
    cat <<'EOF'
Reproduce the gfx1201 retained-PM4 register-map A/B on one GPU.

The two Rust daemons are isolated, while their generated GPU kernel cache is
shared and hashed. The runtime-defect claim is supported only when A fails the
bit-exact PM4/HIP shadow gate and B passes it.

Usage:
  scripts/gfx1201-register-ab.sh [options]

Options:
  --a-ref REF          unmodified baseline (default: origin/beta)
  --b-ref REF          register-map candidate (default: HEAD)
  --model PATH         exact MQ4R model
  --device N           physical ROCr device (default: 0)
  --perf-passes N      stationary product passes per arm (default: 1)
  --out DIR            new artifact directory (default: /tmp with UTC stamp)

PERF_PASSES=2 runs A,B,B,A to reduce order bias. Each product pass already
contains 10 measured rows after the stationarity gate.

Exit status describes experiment execution, not whether the hypothesis won.
Read summary.json: supported is true only for A-fail/B-pass bit-exact parity.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --a-ref) A_REF="${2:?--a-ref requires a ref}"; shift 2 ;;
        --b-ref) B_REF="${2:?--b-ref requires a ref}"; shift 2 ;;
        --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
        --device) DEVICE="${2:?--device requires an index}"; shift 2 ;;
        --perf-passes) PERF_PASSES="${2:?--perf-passes requires a count}"; shift 2 ;;
        --out) OUT="${2:?--out requires a directory}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PERF_PASSES" in
    ''|*[!0-9]*) echo "--perf-passes must be a positive integer" >&2; exit 2 ;;
esac
[ "$PERF_PASSES" -ge 1 ] || {
    echo "--perf-passes must be at least 1" >&2
    exit 2
}

MODEL="$(readlink -f "$MODEL")"
[ -r "$MODEL" ] || {
    echo "model is not readable: $MODEL" >&2
    exit 2
}
command -v cargo >/dev/null || {
    echo "cargo is not on PATH" >&2
    exit 2
}
command -v rocminfo >/dev/null || {
    echo "rocminfo is required" >&2
    exit 2
}

A_SHA="$(git -C "$ROOT" rev-parse --verify "${A_REF}^{commit}")"
B_SHA="$(git -C "$ROOT" rev-parse --verify "${B_REF}^{commit}")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/tmp/hipfire-gfx1201-register-ab-$STAMP}"
[ ! -e "$OUT" ] || {
    echo "refusing to reuse artifact directory: $OUT" >&2
    exit 2
}
mkdir -p "$OUT"
OUT="$(readlink -f "$OUT")"

A_TREE="$OUT/source-a"
B_TREE="$OUT/source-b"
CACHE="$OUT/kernel-cache"
mkdir -p "$CACHE" "$OUT/environment" "$OUT/correctness" "$OUT/performance"

git -C "$ROOT" worktree add --detach "$A_TREE" "$A_SHA"
git -C "$ROOT" worktree add --detach "$B_TREE" "$B_SHA"

ARCH="$(
    ROCR_VISIBLE_DEVICES="$DEVICE" HIP_VISIBLE_DEVICES=0 \
        rocminfo 2>/dev/null |
        sed -nE 's/.*Name:[[:space:]]*(gfx[0-9]+).*/\1/p' |
        sort -u
)"
[ "$ARCH" = "gfx1201" ] || {
    echo "physical device $DEVICE resolved to '$ARCH', expected gfx1201" >&2
    exit 2
}

# The author reference uses automatic clocks. On R9700, "high" can pin a
# lower DPM state and depress both HIP and PM4 by roughly the same amount.
rocm-smi --setperflevel auto >"$OUT/environment/set-perf-auto.txt" 2>&1
rocm-smi --showperflevel >"$OUT/environment/perf-level.txt" 2>&1
grep -qi 'auto' "$OUT/environment/perf-level.txt" || {
    echo "GPU performance level did not report auto" >&2
    exit 2
}

{
    echo "created_at_utc=$STAMP"
    echo "host=$(hostname)"
    echo "kernel=$(uname -r)"
    echo "a_ref=$A_REF"
    echo "a_sha=$A_SHA"
    echo "b_ref=$B_REF"
    echo "b_sha=$B_SHA"
    echo "model=$MODEL"
    echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
    echo "device=$DEVICE"
    echo "arch=$ARCH"
    echo "kernel_cache=$CACHE"
    echo "perf_passes=$PERF_PASSES"
} >"$OUT/environment/contract.txt"

hipcc --version >"$OUT/environment/hipcc.txt" 2>&1 || true
rocminfo >"$OUT/environment/rocminfo.txt" 2>&1
uname -a >"$OUT/environment/uname.txt"
ldconfig -p >"$OUT/environment/ldconfig.txt"
modinfo amdgpu >"$OUT/environment/amdgpu-modinfo.txt" 2>&1 || true
dkms status >"$OUT/environment/dkms.txt" 2>&1 || true
amd-smi static --vbios --ras --limit >"$OUT/environment/amd-smi-static.txt" 2>&1 || true
rocm-smi --showallinfo >"$OUT/environment/rocm-smi.txt" 2>&1 || true

# Confirm the libraries selected by bare dlopen, which is how redline-rocr's
# first candidates are resolved. This catches stale ld.so cache entries after
# a side-by-side ROCm upgrade.
python3 - "$OUT/environment/runtime-libraries.json" <<'PY'
import ctypes
import json
import os
import pathlib
import sys

libs = {}
handles = []
for soname in ("libamdhip64.so", "libhsa-runtime64.so"):
    handle = ctypes.CDLL(soname)
    handles.append(handle)
    libs[soname] = []
maps = pathlib.Path("/proc/self/maps").read_text().splitlines()
for soname in libs:
    libs[soname] = sorted({
        line.split()[-1]
        for line in maps
        if soname.split(".so")[0] in line and line.split()[-1].startswith("/")
    })
pathlib.Path(sys.argv[1]).write_text(json.dumps(libs, indent=2) + "\n")
PY

build_arm() {
    local label="$1" tree="$2"
    echo "building $label ($tree)"
    CARGO_TARGET_DIR="$OUT/target-$label" \
        cargo build \
        --manifest-path "$tree/Cargo.toml" \
        --release \
        --example daemon \
        -p hipfire-runtime
    cp "$OUT/target-$label/release/examples/daemon" "$OUT/daemon-$label"
    sha256sum "$OUT/daemon-$label" >"$OUT/daemon-$label.sha256"
}

build_arm a "$A_TREE"
build_arm b "$B_TREE"

PM4_VARS=(
    HIPFIRE_REPLAY_PM4_QUEUES
    HIPFIRE_REPLAY_PM4_STATEFUL
    HIPFIRE_REPLAY_PM4_WAIT_POLICY
    HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY
    HIPFIRE_REPLAY_PM4_GCR_TRIM
    HIPFIRE_REPLAY_PM4_NATIVE_PHASES
    HIPFIRE_REPLAY_PM4_DYNAMIC_GRID
)

run_clean() {
    local tree="$1"
    shift
    env \
        -u ROCM_PATH \
        -u HIP_PATH \
        -u LD_LIBRARY_PATH \
        -u "${PM4_VARS[0]}" \
        -u "${PM4_VARS[1]}" \
        -u "${PM4_VARS[2]}" \
        -u "${PM4_VARS[3]}" \
        -u "${PM4_VARS[4]}" \
        -u "${PM4_VARS[5]}" \
        -u "${PM4_VARS[6]}" \
        ROCR_VISIBLE_DEVICES="$DEVICE" \
        HIP_VISIBLE_DEVICES=0 \
        HIPFIRE_KERNEL_CACHE="$CACHE" \
        "$@"
}

kernel_manifest() {
    local output="$1"
    (
        cd "$CACHE"
        find "$ARCH" -type f \( -name '*.hsaco' -o -name '*.hash' \) -print0 |
            sort -z |
            xargs -0 sha256sum
    ) >"$output"
}

run_correctness() {
    local label="$1" tree="$2"
    local rc=0
    echo "correctness $label: 128 consecutive token positions"
    run_clean "$tree" \
        python3 "$tree/scripts/redline_daemon_harness.py" \
        --model "$MODEL" \
        --daemon "$OUT/daemon-$label" \
        --skip-prefill \
        --decode-context 128 \
        --decode-iterations 32 \
        --capture-repeats 2 \
        --measure-repeats 3 \
        --shadow-iterations 128 \
        --max-seq 2048 \
        --pm4 \
        --timeout 1200 \
        --out "$OUT/correctness/$label.json" \
        --log "$OUT/correctness/$label.log" || rc=$?
    echo "$rc" >"$OUT/correctness/$label.exit"
}

run_correctness a "$A_TREE"
kernel_manifest "$OUT/kernel-manifest-after-a.sha256"
[ -s "$OUT/kernel-manifest-after-a.sha256" ] || {
    echo "A produced no gfx1201 GPU code objects in $CACHE" >&2
    exit 2
}
run_correctness b "$B_TREE"
kernel_manifest "$OUT/kernel-manifest-after-b.sha256"

cmp "$OUT/kernel-manifest-after-a.sha256" \
    "$OUT/kernel-manifest-after-b.sha256" \
    >"$OUT/kernel-manifest.diff" || {
        echo "GPU kernel cache changed between A and B; refusing to interpret A/B" >&2
        exit 2
    }

run_product() {
    local label="$1" tree="$2" pass="$3"
    local rc=0
    echo "performance $label pass $pass"
    run_clean "$tree" \
        python3 "$tree/scripts/redline_product_bench.py" \
        --model "$MODEL" \
        --daemon "$OUT/daemon-$label" \
        --context 128 \
        --iterations 128 \
        --warmups 10 \
        --warmup-iterations 32 \
        --runs 10 \
        --settle-window 10 \
        --settle-min-runs 10 \
        --settle-confirmation-runs 10 \
        --settle-max-runs 120 \
        --settle-max-slope-pct 0.05 \
        --settle-max-spread-pct 1.0 \
        --settle-max-median-drift-pct 0.5 \
        --transport pm4 \
        --kv-mode q8 \
        --max-seq 2048 \
        --timeout 1200 \
        --work-dir "$OUT/performance/$label-$pass-work" \
        --out "$OUT/performance/$label-$pass.json" || rc=$?
    echo "$rc" >"$OUT/performance/$label-$pass.exit"
}

for pass in $(seq 1 "$PERF_PASSES"); do
    if [ $((pass % 2)) -eq 1 ]; then
        run_product a "$A_TREE" "$pass"
        run_product b "$B_TREE" "$pass"
    else
        run_product b "$B_TREE" "$pass"
        run_product a "$A_TREE" "$pass"
    fi
done

python3 - "$OUT" "$PERF_PASSES" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
passes = int(sys.argv[2])

def load_optional(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

correctness = {
    label: load_optional(root / "correctness" / f"{label}.json")
    for label in ("a", "b")
}

def correctness_pass(label):
    row = correctness[label]
    if not row:
        return False
    shadow = row.get("aql_shadow") or {}
    return bool(row.get("pass")) and shadow.get("bit_exact") is True

perf = {"a": [], "b": []}
for label in perf:
    for index in range(1, passes + 1):
        row = load_optional(root / "performance" / f"{label}-{index}.json")
        if row:
            perf[label].append(row)

def perf_summary(rows):
    hip = [row["hip"]["tok_s"]["median"] for row in rows if row.get("valid")]
    pm4 = [row["auto"]["tok_s"]["median"] for row in rows if row.get("valid")]
    ratios = [row["speedup"] for row in rows if row.get("valid")]
    routes = [
        row["auto"]["route_proof"]
        for row in rows
        if row.get("auto", {}).get("route_proof")
    ]
    return {
        "valid_passes": len(ratios),
        "requested_passes": passes,
        "hip_median_tok_s": statistics.median(hip) if hip else None,
        "pm4_median_tok_s": statistics.median(pm4) if pm4 else None,
        "speedup": statistics.median(ratios) if ratios else None,
        "routes": routes,
    }

a_ok = correctness_pass("a")
b_ok = correctness_pass("b")
if not a_ok and b_ok:
    verdict = "A fails and B passes bit-exact parity: runtime defect hypothesis supported"
    supported = True
elif a_ok and b_ok:
    verdict = "A and B both pass bit-exact parity: no runtime correctness defect demonstrated"
    supported = False
elif a_ok and not b_ok:
    verdict = "A passes and B fails bit-exact parity: candidate is a correctness regression"
    supported = False
else:
    verdict = "A and B both fail or lack reports: experiment is inconclusive"
    supported = False

summary = {
    "schema": "hipfire.gfx1201-register-ab.v1",
    "hypothesis": (
        "the origin/beta gfx1201 register program causes a runtime correctness "
        "failure which the candidate fixes"
    ),
    "supported": supported,
    "verdict": verdict,
    "same_gpu_code_objects": (
        (root / "kernel-manifest-after-a.sha256").read_bytes()
        == (root / "kernel-manifest-after-b.sha256").read_bytes()
    ),
    "correctness": {
        "a": {"pass": a_ok, "report": "correctness/a.json"},
        "b": {"pass": b_ok, "report": "correctness/b.json"},
    },
    "performance": {
        "a": perf_summary(perf["a"]),
        "b": perf_summary(perf["b"]),
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "A/B artifacts: $OUT"
