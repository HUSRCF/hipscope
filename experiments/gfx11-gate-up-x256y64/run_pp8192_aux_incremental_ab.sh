#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_aux_incremental_ab}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
SLEEP_SECS="${SLEEP_SECS:-5}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_target750.so}"
BIN="${ROOT}/target/release/examples/bench_qwen35_mq4"

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\n' > "${OUT_DIR}/results.tsv"

run_one() {
    local pair="$1"
    local order="$2"
    local mode="$3"
    local aux=0
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    [[ "${mode}" == "all_x256y64" ]] && aux=1

    timeout --signal=INT --kill-after=5s 180s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=32 \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
        HIPFIRE_DPM_WARMUP_SECS=5 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64="${aux}" \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs 1 --warmup 2 --gen 8 \
        > "${log}" 2>&1

    local summary prefill gen
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if (( pair % 2 == 1 )); then
        modes=(gate_residual all_x256y64)
    else
        modes=(all_x256y64 gate_residual)
    fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        sleep "${SLEEP_SECS}"
    done
done

awk -F '\t' 'NR > 1 {n[$3]++; pre[$3]+=$4; gen[$3]+=$5} END {for (m in n) printf "%s\tn=%d\tprefill_mean=%.3f\tgen_mean=%.3f\n", m,n[m],pre[m]/n[m],gen[m]/n[m]}' \
    "${OUT_DIR}/results.tsv" | sort | tee "${OUT_DIR}/means.tsv"
