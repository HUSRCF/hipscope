#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_ffn_f16_intermediate_ck_gpu1_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
SLEEP_SECS="${SLEEP_SECS:-5}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"
sha256sum "${BIN}" "${CK_LIB}" "${MODEL}" >"${OUT_DIR}/artifacts.sha256"

run_one() {
    local pair="$1" order="$2" mode="$3" f16=0
    [[ "${mode}" == f16 ]] && f16=1
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"

    timeout --signal=INT --kill-after=5s 240s env \
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
        HIPFIRE_RDNA3_Q8_GROUP128_DUAL_ROW_WEIGHT=0 \
        HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE="${f16}" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs 3 --warmup 2 --gen 32 \
        >"${log}" 2>&1

    rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}"
    rg -q '^quantized FlashAttention CK prefill active:' "${log}"
    local summary prefill gen token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    gen="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if ((pair % 2)); then modes=(f32 f16); else modes=(f16 f32); fi
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
for mode in ("f32", "f16"):
    selected = [row for row in rows if row["mode"] == mode]
    prefill = [float(row["prefill_tok_s"]) for row in selected]
    decode = [float(row["gen_tok_s"]) for row in selected]
    print(f"{mode}: prefill_median={statistics.median(prefill):.3f} "
          f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}")

f32 = statistics.median(float(row["prefill_tok_s"]) for row in rows if row["mode"] == "f32")
f16 = statistics.median(float(row["prefill_tok_s"]) for row in rows if row["mode"] == "f16")
tokens = {mode: {row["token_ids"] for row in rows if row["mode"] == mode}
          for mode in ("f32", "f16")}
by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = float(row["prefill_tok_s"])
pair_ratios = [modes["f16"] / modes["f32"] for modes in by_pair.values()]
print(f"f16_vs_f32={f16 / f32:.4f}x ({(f16 / f32 - 1) * 100:+.2f}%)")
print(f"paired_ratio_median={statistics.median(pair_ratios):.4f}x "
      f"positive_pairs={sum(ratio > 1.0 for ratio in pair_ratios)}/{len(pair_ratios)} "
      f"raw_pair_ratios={[round(ratio, 4) for ratio in pair_ratios]}")
print(f"token_ids_match={tokens['f32'] == tokens['f16']} "
      f"f32_variants={len(tokens['f32'])} f16_variants={len(tokens['f16'])}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
