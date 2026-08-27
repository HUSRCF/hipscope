#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-8192}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
TRIALS="${TRIALS:-3}"
SLEEP_SECS="${SLEEP_SECS:-20}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${CK_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so}"
KV_MODE="${KV_MODE:-asym3}"
PARTIALS_BATCH="${PARTIALS_BATCH:-$([[ "${KV_MODE}" == asym4 ]] && printf 64 || printf 32)}"
OUT_DIR="${OUT_DIR:-${ROOT}/.redline-work/gfx11-ck-prefill-ab-$(date +%Y%m%d_%H%M%S)}"

(( PREFILL_TOKENS >= 2048 && PREFILL_RUNS >= 3 && PREFILL_RUNS % 2 == 1 && TRIALS >= 3 )) || {
    echo "require PREFILL_TOKENS>=2048, odd PREFILL_RUNS>=3, TRIALS>=3" >&2
    exit 2
}
[[ "${KV_MODE}" == "asym3" || "${KV_MODE}" == "asym4" ]] || {
    echo "KV_MODE must be asym3 or asym4" >&2
    exit 2
}
for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tdecode_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'trials=%s\n' "${TRIALS}"
    printf 'kv_mode=%s\n' "${KV_MODE}"
    printf 'partials_batch=%s\n' "${PARTIALS_BATCH}"
    printf 'model_sha256=%s\n' "$(sha256sum "${MODEL}" | awk '{print $1}')"
    printf 'binary_sha256=%s\n' "$(sha256sum "${BIN}" | awk '{print $1}')"
    printf 'ck_lib_sha256=%s\n' "$(sha256sum "${CK_LIB}" | awk '{print $1}')"
    printf 'harness_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
} >"${OUT_DIR}/manifest.txt"

run_one() {
    local pair="$1" order="$2" mode="$3"
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    local -a sidecar=()
    local asym4_wmma=1
    if [[ "${mode}" == ck ]]; then
        sidecar=(HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}")
    elif [[ "${KV_MODE}" == asym4 ]]; then
        asym4_wmma=0
    fi

    timeout --signal=INT --kill-after=5s 900s env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        "${sidecar[@]}" \
        HIPFIRE_KV_MODE="${KV_MODE}" \
        HIPFIRE_ASYM4_WMMA="${asym4_wmma}" \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH="${PARTIALS_BATCH}" \
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
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill "${PREFILL_TOKENS}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 8 >"${log}" 2>&1

    local route='^staged quantized FlashAttention CK prefill active:'
    [[ "${KV_MODE}" == asym4 ]] && route='^Asym4 staged FlashAttention CK prefill active:'
    if [[ "${mode}" == ck ]]; then
        rg -q "${route}" "${log}"
    elif rg -q "${route}" "${log}"; then
        echo "native arm unexpectedly used CK" >&2
        return 1
    fi

    local summary prefill decode token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(rg '^  median:' "${log}" | tail -n 1 | awk '{print $(NF-1)}')"
    decode="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    [[ -n "${prefill}" && -n "${decode}" && -n "${token_ids}" ]]
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${decode}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if ((pair % 2)); then modes=(native ck); else modes=(ck native); fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        if ((pair < TRIALS || order == 0)); then sleep "${SLEEP_SECS}"; fi
    done
done

python3 - "${OUT_DIR}/results.tsv" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
for mode in ("native", "ck"):
    values = [float(row["prefill_tok_s"]) for row in rows if row["mode"] == mode]
    print(f"{mode}: median={statistics.median(values):.3f} raw={values}")
by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = row
ratios = [float(v["ck"]["prefill_tok_s"]) / float(v["native"]["prefill_tok_s"])
          for v in by_pair.values()]
bad = [pair for pair, v in by_pair.items() if v["native"]["token_ids"] != v["ck"]["token_ids"]]
print(f"paired_ratio_median={statistics.median(ratios):.6f}x "
      f"positive_pairs={sum(r > 1 for r in ratios)}/{len(ratios)} raw={ratios}")
print(f"token_ids_match={not bad} bad_pairs={bad}")
if bad:
    raise SystemExit(1)
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
