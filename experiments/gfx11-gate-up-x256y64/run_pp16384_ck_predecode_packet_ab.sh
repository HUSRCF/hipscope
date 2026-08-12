#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-1}
PREFILL_TOKENS=${PREFILL_TOKENS:-16384}
PREFILL_RUNS=${PREFILL_RUNS:-3}
TRIALS=${TRIALS:-3}
SLEEP_SECS=${SLEEP_SECS:-20}
MODEL=${MODEL:-"${HOME}/.hipfire/models/qwen3.6-27b.mq4"}
BIN=${BIN:-"$ROOT/target/release/examples/bench_qwen35_mq4"}
QUANT_ROOT="$ROOT/experiments/flash-attn-ck-sidecar/quantized"
REFERENCE_WORKTREE=${REFERENCE_WORKTREE:-"/home/husrcf/Code/ProtBind/unidec/.worktrees/hipfire-flashattn-beta-latest"}
CK_ROOT=${CK_ROOT:-"$REFERENCE_WORKTREE/experiments/flash-attn-ck-sidecar/build/ck-source"}
DENSE_SIDECAR=${DENSE_SIDECAR:-"$REFERENCE_WORKTREE/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so"}
SCALAR_LIB=${SCALAR_LIB:-}
PACKET_LIB=${PACKET_LIB:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S_%N)}
OUT_DIR=${OUT_DIR:-"$ROOT/experiments/gfx11-gate-up-x256y64/results/pp${PREFILL_TOKENS}_ck_packet_store_gpu${GPU_ID}_${STAMP}"}

(( PREFILL_TOKENS >= 16384 && PREFILL_RUNS >= 2 && TRIALS >= 2 )) || {
    echo "production gate requires PREFILL_TOKENS>=16384, PREFILL_RUNS>=2, TRIALS>=2" >&2
    exit 1
}
for path in "$MODEL" "$BIN"; do
    [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
if [[ -z "$SCALAR_LIB" && -z "$PACKET_LIB" ]]; then
    for path in "$CK_ROOT" "$DENSE_SIDECAR"; do
        [[ -e "$path" ]] || { echo "missing build input: $path" >&2; exit 1; }
    done
    SCALAR_LIB="$OUT_DIR/sidecars/scalar/libhipfire_flash_attn_ck_quantized_scalar.so"
    PACKET_LIB="$OUT_DIR/sidecars/packet/libhipfire_flash_attn_ck_quantized_packet.so"
    for spec in "scalar:0:$SCALAR_LIB" "packet:1:$PACKET_LIB"; do
        IFS=: read -r arm packet_store sidecar <<< "$spec"
        arm_dir=$(dirname "$sidecar")
        mkdir -p "$arm_dir"
        env \
            CK_ROOT="$CK_ROOT" \
            DENSE_SIDECAR="$DENSE_SIDECAR" \
            BUILD_DIR="$arm_dir" \
            OUT="$sidecar" \
            STAGED=1 \
            PACKET_STORE="$packet_store" \
            bash "$QUANT_ROOT/build_quantized_sidecar.sh" \
            > "$arm_dir/build.txt" 2>&1
    done
elif [[ -z "$SCALAR_LIB" || -z "$PACKET_LIB" ]]; then
    echo "set both SCALAR_LIB and PACKET_LIB, or leave both unset" >&2
    exit 1
fi
for path in "$SCALAR_LIB" "$PACKET_LIB"; do
    [[ -f "$path" ]] || { echo "missing sidecar: $path" >&2; exit 1; }
done
for path in "${SCALAR_LIB}.variant" "${PACKET_LIB}.variant"; do
    [[ -f "$path" ]] || { echo "missing sidecar provenance: $path" >&2; exit 1; }
done
rg -q '^staged=1$' "${SCALAR_LIB}.variant"
rg -q '^packet_store=0$' "${SCALAR_LIB}.variant"
rg -q '^staged=1$' "${PACKET_LIB}.variant"
rg -q '^packet_store=1$' "${PACKET_LIB}.variant"

printf 'pair\torder\tmode\tprocess_median_prefill_tok_s\tsummary_last_prefill_tok_s\tgen_tok_s\ttoken_ids\n' > "$OUT_DIR/results.tsv"
sha256sum \
    "$BIN" "$MODEL" \
    "$SCALAR_LIB" "${SCALAR_LIB}.variant" \
    "$PACKET_LIB" "${PACKET_LIB}.variant" \
    "$0" > "$OUT_DIR/artifacts.sha256"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
    if git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet; then
        printf 'git_tracked_dirty=0\n'
    else
        printf 'git_tracked_dirty=1\n'
    fi
    printf 'gpu_id=%s\n' "$GPU_ID"
    printf 'prefill_tokens=%s\n' "$PREFILL_TOKENS"
    printf 'prefill_runs=%s\n' "$PREFILL_RUNS"
    printf 'trials=%s\n' "$TRIALS"
    printf 'sleep_secs=%s\n' "$SLEEP_SECS"
    printf 'scalar_lib=%s\n' "$SCALAR_LIB"
    printf 'packet_lib=%s\n' "$PACKET_LIB"
    printf 'contract=group128, F32 FFN intermediate, accepted batched GDN norm/rotate\n'
    rocm-smi --showproductname --showuse --showmemuse --showpids
} > "$OUT_DIR/manifest.txt" 2>&1

run_one() {
    local pair=$1 order=$2 mode=$3 sidecar
    case "$mode" in
        scalar) sidecar="$SCALAR_LIB" ;;
        packet) sidecar="$PACKET_LIB" ;;
        *) echo "unknown mode: $mode" >&2; return 1 ;;
    esac
    local log="$OUT_DIR/pair_${pair}_${order}_${mode}.log"

    timeout --signal=INT --kill-after=5s 900s env \
        HIP_VISIBLE_DEVICES="$GPU_ID" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="$sidecar" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=32 \
        HIPFIRE_DPM_WARMUP_SECS=5 \
        HIPFIRE_QKVZA_SPLIT_TAIL=1 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
        HIPFIRE_RDNA3_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=0 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=0 \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_GATED_NORM_MQ_ROTATE_BATCHED=1 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "$BIN" "$MODEL" \
        --prefill "$PREFILL_TOKENS" --prefill-runs "$PREFILL_RUNS" \
        --warmup 2 --gen 32 > "$log" 2>&1

    rg -q '^staged quantized FlashAttention CK prefill active:' "$log"
    local summary prefill summary_prefill gen token_ids
    summary=$(rg '^SUMMARY ' "$log" | tail -n 1)
    prefill=$(awk '/^  median:/ {value=$3} END {print value}' "$log")
    summary_prefill=$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<< "$summary")
    gen=$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<< "$summary")
    token_ids=$(rg '^TOKEN_IDS ' "$log" | tail -n 1 | sed 's/^TOKEN_IDS //')
    [[ -n "$prefill" && -n "$summary_prefill" && -n "$gen" && -n "$token_ids" ]]
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$pair" "$order" "$mode" "$prefill" "$summary_prefill" "$gen" "$token_ids" \
        | tee -a "$OUT_DIR/results.tsv"
}

for ((pair = 1; pair <= TRIALS; ++pair)); do
    if (( pair % 2 )); then modes=(scalar packet); else modes=(packet scalar); fi
    for order in 0 1; do
        run_one "$pair" "$order" "${modes[$order]}"
        sleep "$SLEEP_SECS"
    done
done

python3 - "$OUT_DIR/results.tsv" <<'PY' | tee "$OUT_DIR/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
for mode in ("scalar", "packet"):
    selected = [row for row in rows if row["mode"] == mode]
    prefill = [float(row["process_median_prefill_tok_s"]) for row in selected]
    summary_prefill = [float(row["summary_last_prefill_tok_s"]) for row in selected]
    decode = [float(row["gen_tok_s"]) for row in selected]
    print(f"{mode}: prefill_median={statistics.median(prefill):.3f} "
          f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill} "
          f"summary_last_raw={summary_prefill}")

base = statistics.median(float(row["process_median_prefill_tok_s"])
                         for row in rows if row["mode"] == "scalar")
candidate = statistics.median(float(row["process_median_prefill_tok_s"])
                              for row in rows if row["mode"] == "packet")
tokens = {mode: {row["token_ids"] for row in rows if row["mode"] == mode}
          for mode in ("scalar", "packet")}
by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = float(
        row["process_median_prefill_tok_s"]
    )
pair_ratios = [modes["packet"] / modes["scalar"] for modes in by_pair.values()]
print(f"packet_vs_scalar={candidate / base:.4f}x ({(candidate / base - 1) * 100:+.2f}%)")
print(f"paired_ratio_median={statistics.median(pair_ratios):.4f}x "
      f"positive_pairs={sum(ratio > 1.0 for ratio in pair_ratios)}/{len(pair_ratios)} "
      f"raw_pair_ratios={[round(ratio, 4) for ratio in pair_ratios]}")
print(f"token_ids_match={tokens['scalar'] == tokens['packet']} "
      f"scalar_variants={len(tokens['scalar'])} packet_variants={len(tokens['packet'])}")
PY

printf 'out_dir=%s\n' "$OUT_DIR"
