#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PAIRS="${PAIRS:-3}"
COOL_SECS="${COOL_SECS:-20}"
MODEL="${MODEL:-/home/husrcf/.hipfire/models/qwen3.6-27b.mq4}"
LEGACY_SIDECAR="${LEGACY_SIDECAR:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"
STAGED_SIDECAR="${STAGED_SIDECAR:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/results/staged_model_ab_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

run_one() {
    local pair="$1"
    local label="$2"
    local sidecar="$3"
    local expected_log="$4"
    local log="${OUT_DIR}/pair${pair}_${label}.log"

    env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${sidecar}" \
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
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1 \
        "${ROOT}/target/release/examples/bench_qwen35_mq4" \
        "${MODEL}" \
        --prefill 8192 \
        --prefill-runs 1 \
        --warmup 0 \
        --gen 0 >"${log}" 2>&1

    grep -q "${expected_log}" "${log}"
    awk -v pair="${pair}" -v label="${label}" '
        /^PREFILL_SUMMARY/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^prefill_tok_s=/) {
                    split($i, value, "=")
                    print pair "\t" label "\t" value[2]
                    found = 1
                }
            }
        }
        END { if (!found) exit 1 }
    ' "${log}" >>"${OUT_DIR}/summary.tsv"
}

printf 'pair\tmode\tprefill_tok_s\n' >"${OUT_DIR}/summary.tsv"
for ((pair = 1; pair <= PAIRS; pair++)); do
    if ((pair % 2 == 1)); then
        order=(legacy staged)
    else
        order=(staged legacy)
    fi
    for label in "${order[@]}"; do
        if [[ "${label}" == "staged" ]]; then
            run_one "${pair}" "${label}" "${STAGED_SIDECAR}" \
                "staged quantized FlashAttention CK prefill active"
        else
            run_one "${pair}" "${label}" "${LEGACY_SIDECAR}" \
                "quantized FlashAttention CK prefill active"
        fi
        sleep "${COOL_SECS}"
    done
done

cat "${OUT_DIR}/summary.tsv"
