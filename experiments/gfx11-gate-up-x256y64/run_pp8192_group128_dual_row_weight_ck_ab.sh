#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_group128_dual_row_weight_ck_gpu1_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-1}"
SLEEP_SECS="${SLEEP_SECS:-5}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

# Build the benchmark with `--features flash-attn-ck`; the script fails closed
# below unless both the dynamic loader and quantized CK route report active.

for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"
sha256sum "${BIN}" "${CK_LIB}" "${MODEL}" >"${OUT_DIR}/artifacts.sha256"

run_command() {
    local log="$1" dual_row="$2"
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
        HIPFIRE_RDNA3_Q8_GROUP128_DUAL_ROW_WEIGHT="${dual_row}" \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_DIRECT=0 \
        HIPFIRE_RDNA3_Q8_GROUP128_DIRECT_X512=0 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=0 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs "${PREFILL_RUNS}" --warmup 2 --gen 32 \
        >"${log}" 2>&1

    rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}"
    rg -q '^quantized FlashAttention CK prefill active:' "${log}"
}

run_one() {
    local pair="$1" order="$2" mode="$3" dual_row=0
    [[ "${mode}" == dual_row ]] && dual_row=1
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    run_command "${log}" "${dual_row}"
    local summary prefill gen token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    gen="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

run_command "${OUT_DIR}/prewarm_baseline.log" 0
sleep "${SLEEP_SECS}"
run_command "${OUT_DIR}/prewarm_dual_row.log" 1
sleep "${SLEEP_SECS}"

for ((pair=1; pair<=TRIALS; pair++)); do
    if ((pair % 2)); then modes=(baseline dual_row); else modes=(dual_row baseline); fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${OUT_DIR}/results.tsv" "${TRIM_EACH_SIDE}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trim = int(sys.argv[2])

def values(mode, field, trimmed=False):
    out = [float(row[field]) for row in rows if row["mode"] == mode]
    if trimmed and trim:
        out = sorted(out)[trim:-trim]
    return out

for mode in ("baseline", "dual_row"):
    prefill = values(mode, "prefill_tok_s")
    decode = values(mode, "gen_tok_s")
    print(f"{mode}: prefill_median={statistics.median(prefill):.3f} "
          f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}")

baseline = statistics.median(values("baseline", "prefill_tok_s", True))
dual_row = statistics.median(values("dual_row", "prefill_tok_s", True))
tokens = {mode: {row["token_ids"] for row in rows if row["mode"] == mode}
          for mode in ("baseline", "dual_row")}
print(f"trim_each_side={trim} baseline_trimmed_median={baseline:.3f} "
      f"dual_row_trimmed_median={dual_row:.3f}")
print(f"dual_row_vs_baseline={dual_row / baseline:.4f}x "
      f"({(dual_row / baseline - 1) * 100:+.2f}%)")
print(f"token_ids_match={tokens['baseline'] == tokens['dual_row']} "
      f"baseline_variants={len(tokens['baseline'])} "
      f"dual_row_variants={len(tokens['dual_row'])}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
