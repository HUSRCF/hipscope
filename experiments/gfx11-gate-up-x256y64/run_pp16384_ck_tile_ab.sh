#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-16384}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
TRIALS="${TRIALS:-3}"
SLEEP_SECS="${SLEEP_SECS:-20}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
BASELINE_LIB="${BASELINE_LIB:?set BASELINE_LIB to the retained CK sidecar}"
CANDIDATE_LIB="${CANDIDATE_LIB:?set CANDIDATE_LIB to the candidate CK sidecar}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-candidate}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp${PREFILL_TOKENS}_ck_tile_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S_%N)}"

(( PREFILL_TOKENS >= 16384 && PREFILL_RUNS >= 3 && PREFILL_RUNS % 2 == 1 && TRIALS >= 3 )) || {
    echo "production gate requires PREFILL_TOKENS>=16384, odd PREFILL_RUNS>=3, TRIALS>=3" >&2
    exit 1
}
for label in "${BASELINE_LABEL}" "${CANDIDATE_LABEL}"; do
    [[ "${label}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "labels may contain only letters, digits, dot, underscore, and dash" >&2
        exit 1
    }
done
for path in "${MODEL}" "${BIN}" "${BASELINE_LIB}" "${CANDIDATE_LIB}" \
            "${BASELINE_LIB}.variant" "${CANDIDATE_LIB}.variant"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprocess_median_prefill_tok_s\tsummary_last_prefill_tok_s\tdecode_tok_s\ttoken_ids\n' \
    >"${OUT_DIR}/results.tsv"
for entry in "${BASELINE_LABEL}:${BASELINE_LIB}" "${CANDIDATE_LABEL}:${CANDIDATE_LIB}"; do
    label="${entry%%:*}"
    lib="${entry#*:}"
    cp "${lib}.variant" "${OUT_DIR}/${label}.variant"
done
(
    cd "${OUT_DIR}"
    sha256sum "${BASELINE_LABEL}.variant" "${CANDIDATE_LABEL}.variant" >artifacts.sha256
)
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'trials=%s\n' "${TRIALS}"
    printf 'sleep_secs=%s\n' "${SLEEP_SECS}"
    printf 'baseline_label=%s\n' "${BASELINE_LABEL}"
    printf 'candidate_label=%s\n' "${CANDIDATE_LABEL}"
    printf 'baseline_lib=%s\n' "${BASELINE_LIB}"
    printf 'candidate_lib=%s\n' "${CANDIDATE_LIB}"
    printf 'model_sha256=%s\n' "$(sha256sum "${MODEL}" | awk '{print $1}')"
    printf 'binary_sha256=%s\n' "$(sha256sum "${BIN}" | awk '{print $1}')"
    printf 'baseline_lib_sha256=%s\n' "$(sha256sum "${BASELINE_LIB}" | awk '{print $1}')"
    printf 'candidate_lib_sha256=%s\n' "$(sha256sum "${CANDIDATE_LIB}" | awk '{print $1}')"
    printf 'harness_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
} >"${OUT_DIR}/manifest.txt"

run_one() {
    local pair="$1" order="$2" mode="$3" lib="$4"
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"

    timeout --signal=INT --kill-after=5s 900s env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${lib}" \
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
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill "${PREFILL_TOKENS}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 8 >"${log}" 2>&1

    rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"
    local process_median summary last_prefill decode token_ids
    process_median="$(awk '/^  median:/ {value=$3} END {print value}' "${log}")"
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    last_prefill="$(awk '{for(i=1;i<=NF;i++) if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    decode="$(awk '{for(i=1;i<=NF;i++) if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    [[ -n "${process_median}" && -n "${last_prefill}" && -n "${decode}" && -n "${token_ids}" ]]
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${process_median}" "${last_prefill}" "${decode}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if ((pair % 2)); then
        modes=("${BASELINE_LABEL}" "${CANDIDATE_LABEL}")
        libs=("${BASELINE_LIB}" "${CANDIDATE_LIB}")
    else
        modes=("${CANDIDATE_LABEL}" "${BASELINE_LABEL}")
        libs=("${CANDIDATE_LIB}" "${BASELINE_LIB}")
    fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}" "${libs[$order]}"
        if ((pair < TRIALS || order == 0)); then sleep "${SLEEP_SECS}"; fi
    done
done

python3 - "${OUT_DIR}/results.tsv" "${BASELINE_LABEL}" "${CANDIDATE_LABEL}" <<'PY' \
    | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
baseline, candidate = sys.argv[2:4]
for mode in (baseline, candidate):
    selected = [row for row in rows if row["mode"] == mode]
    values = [float(row["process_median_prefill_tok_s"]) for row in selected]
    print(f"{mode}: process_median={statistics.median(values):.3f} raw={values}")

by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = float(row["process_median_prefill_tok_s"])
ratios = [modes[candidate] / modes[baseline] for modes in by_pair.values()]
tokens = {mode: {row["token_ids"] for row in rows if row["mode"] == mode}
          for mode in (baseline, candidate)}
print(f"paired_ratio_median={statistics.median(ratios):.6f}x "
      f"positive_pairs={sum(ratio > 1 for ratio in ratios)}/{len(ratios)} "
      f"raw_pair_ratios={[round(ratio, 6) for ratio in ratios]}")
bad_pairs = [pair for pair, modes in by_pair.items()
             if next(row["token_ids"] for row in rows
                     if row["pair"] == pair and row["mode"] == baseline)
             != next(row["token_ids"] for row in rows
                     if row["pair"] == pair and row["mode"] == candidate)]
token_ids_match = (not bad_pairs and len(tokens[baseline]) == 1
                   and len(tokens[candidate]) == 1
                   and tokens[baseline] == tokens[candidate])
print(f"token_ids_match={token_ids_match} bad_pairs={bad_pairs} "
      f"{baseline}_variants={len(tokens[baseline])} {candidate}_variants={len(tokens[candidate])}")
if not token_ids_match:
    raise SystemExit(f"token mismatch: bad_pairs={bad_pairs}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
