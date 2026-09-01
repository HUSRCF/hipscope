#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Same-binary A/B for the opt-in gfx11 HFQ4/GDN prefill execution stack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
BIN="${BIN:-$ROOT/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${CK_LIB:-$ROOT/experiments/flash-attn-ck-sidecar/build-asym4-gfx1100/libhipfire_flash_attn_ck.so}"
WORKSPACE_BYTES="${WORKSPACE_BYTES:-536870912}"
PREFILL="${PREFILL:-8192}"
PREFILL_RUNS="${PREFILL_RUNS:-5}"
SLEEP_SECS="${SLEEP_SECS:-10}"
OUT_DIR="${OUT_DIR:-$ROOT/target/validation/gfx11-hfq4-full-stack-$(date +%Y%m%d_%H%M%S)}"
KERNEL_CACHE="${KERNEL_CACHE:-$OUT_DIR/kernel-cache}"

for path in "$MODEL" "$BIN" "$CK_LIB"; do
    [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
(( PREFILL >= 2048 && PREFILL_RUNS >= 3 )) || {
    echo "require PREFILL>=2048 and PREFILL_RUNS>=3" >&2
    exit 2
}

mkdir -p "$OUT_DIR"
printf 'order\tmode\tsample\tprefill_tok_s\tnext_token\tgen_tok_s\n' >"$OUT_DIR/results.tsv"
{
    echo "git_head=$(git -C "$ROOT" rev-parse HEAD)"
    echo "gpu_id=$GPU_ID"
    echo "prefill=$PREFILL"
    echo "prefill_runs=$PREFILL_RUNS"
    echo "workspace_bytes=$WORKSPACE_BYTES"
    sha256sum "$MODEL" "$BIN" "$CK_LIB" "$0"
} >"$OUT_DIR/manifest.txt"

run_one() {
    local order="$1" mode="$2" enabled=0
    [[ "$mode" == full ]] && enabled=1
    local log="$OUT_DIR/${order}_${mode}.log"
    env HIP_VISIBLE_DEVICES="$GPU_ID" \
        HIPFIRE_KERNEL_CACHE="$KERNEL_CACHE" \
        HIPFIRE_FLASH_ATTN_CK_LIB="$CK_LIB" \
        HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES="$WORKSPACE_BYTES" \
        HIPFIRE_KV_MODE=asym4 HIPFIRE_ASYM4_WMMA=1 \
        HIPFIRE_GRAPH=0 HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=64 HIPFIRE_DPM_WARMUP_SECS=3 \
        HIPFIRE_QKVZA_SPLIT_TAIL="$enabled" \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64="$enabled" \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64="$enabled" \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64="$enabled" \
        HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE="$enabled" \
        HIPFIRE_RDNA3_Q8_GROUP128="$enabled" \
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2="$enabled" \
        HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT="$enabled" \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128="$enabled" \
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE="$enabled" \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW="$enabled" \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_RDNA3_GDN_CONV_TOKEN_PARALLEL="$enabled" \
        HIPFIRE_GATED_NORM_MQ_ROTATE_BATCHED="$enabled" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        timeout --signal=INT --kill-after=5s 900s \
        "$BIN" "$MODEL" --prefill "$PREFILL" --prefill-runs "$PREFILL_RUNS" \
        --warmup 2 --gen 8 >"$log" 2>&1

    rg -q '^optional CK attention route: selected_asym4_givens_d256$' "$log"
    local next gen sample=0
    next="$(awk -F= '/^PREFILL_NEXT_TOKEN/{print $2}' "$log" | tail -1)"
    gen="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' "$log" | tail -1)"
    while read -r value; do
        sample=$((sample + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$order" "$mode" "$sample" "$value" "$next" "$gen" >>"$OUT_DIR/results.tsv"
    done < <(awk '/^  run/{print $4}' "$log")
}

orders=(full baseline baseline full)
for i in 0 1 2 3; do
    run_one "$((i + 1))" "${orders[$i]}"
    [[ "$i" == 3 ]] || sleep "$SLEEP_SECS"
done

python3 - "$OUT_DIR/results.tsv" <<'PY' | tee "$OUT_DIR/summary.txt"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
groups = {mode: [float(r["prefill_tok_s"]) for r in rows if r["mode"] == mode]
          for mode in ("baseline", "full")}
for mode, values in groups.items():
    print(f"{mode}: median={statistics.median(values):.3f} raw={values}")
print(f"speedup={statistics.median(groups['full']) / statistics.median(groups['baseline']):.6f}x")
print(f"next_tokens={sorted(set(r['next_token'] for r in rows))}")
for mode in ("baseline", "full"):
    values = [float(r["gen_tok_s"]) for r in rows if r["mode"] == mode]
    print(f"decode_{mode}={statistics.median(values):.3f}")
PY

echo "results: $OUT_DIR"
