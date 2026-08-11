#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp16384_gdn_norm_rotate_batched_gpu1_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-16384}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
TRIALS="${TRIALS:-3}"
SLEEP_SECS="${SLEEP_SECS:-20}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"

(( PREFILL_TOKENS >= 16384 && PREFILL_RUNS >= 2 && TRIALS >= 2 )) || {
    echo "production gate requires PREFILL_TOKENS>=16384, PREFILL_RUNS>=2, TRIALS>=2" >&2
    exit 1
}
for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"
sha256sum "${BIN}" "${CK_LIB}" "${MODEL}" "$0" >"${OUT_DIR}/artifacts.sha256"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'trials=%s\n' "${TRIALS}"
    printf 'sleep_secs=%s\n' "${SLEEP_SECS}"
    printf 'required_build_features=deltanet,flash-attn-ck\n'
    printf 'candidate=batched gated_norm + MQ rotation, exact arithmetic order\n'
} >"${OUT_DIR}/manifest.txt"

run_one() {
    local pair="$1" order="$2" mode="$3" enabled=0
    [[ "${mode}" == fused ]] && enabled=1
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"

    timeout --signal=INT --kill-after=5s 900s env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
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
        HIPFIRE_GATED_NORM_MQ_ROTATE_BATCHED="${enabled}" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill "${PREFILL_TOKENS}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 32 >"${log}" 2>&1

    rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"
    local summary prefill gen token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    gen="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    [[ -n "${prefill}" && -n "${gen}" && -n "${token_ids}" ]]
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if ((pair % 2)); then modes=(baseline fused); else modes=(fused baseline); fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${OUT_DIR}/results.tsv" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
for mode in ("baseline", "fused"):
    selected = [row for row in rows if row["mode"] == mode]
    prefill = [float(row["prefill_tok_s"]) for row in selected]
    decode = [float(row["gen_tok_s"]) for row in selected]
    print(f"{mode}: prefill_median={statistics.median(prefill):.3f} "
          f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}")

base = statistics.median(float(row["prefill_tok_s"]) for row in rows if row["mode"] == "baseline")
fused = statistics.median(float(row["prefill_tok_s"]) for row in rows if row["mode"] == "fused")
tokens = {mode: {row["token_ids"] for row in rows if row["mode"] == mode}
          for mode in ("baseline", "fused")}
by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = float(row["prefill_tok_s"])
pair_ratios = [modes["fused"] / modes["baseline"] for modes in by_pair.values()]
print(f"fused_vs_baseline={fused / base:.4f}x ({(fused / base - 1) * 100:+.2f}%)")
print(f"paired_ratio_median={statistics.median(pair_ratios):.4f}x "
      f"positive_pairs={sum(ratio > 1.0 for ratio in pair_ratios)}/{len(pair_ratios)} "
      f"raw_pair_ratios={[round(ratio, 4) for ratio in pair_ratios]}")
print(f"token_ids_match={tokens['baseline'] == tokens['fused']} "
      f"baseline_variants={len(tokens['baseline'])} fused_variants={len(tokens['fused'])}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
