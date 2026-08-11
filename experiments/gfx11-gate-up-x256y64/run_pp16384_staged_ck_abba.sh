#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-16384}"
PREFILL_RUNS="${PREFILL_RUNS:-2}"
PREFILL_BATCH="${PREFILL_BATCH:-2048}"
IDLE_SECONDS="${IDLE_SECONDS:-30}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp${PREFILL_TOKENS}_staged_ck_abba_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S_%N)}"
ORDER=(off on on off)

(( PREFILL_TOKENS > 0 && PREFILL_RUNS >= 2 && PREFILL_BATCH > 0 )) || {
    echo "PREFILL_TOKENS and PREFILL_BATCH must be positive; PREFILL_RUNS must be at least 2" >&2
    exit 1
}
for path in "${MODEL}" "${SIDECAR}" "${BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
sha256sum "${MODEL}" "${SIDECAR}" "${BIN}" >"${OUT_DIR}/artifacts.sha256"
printf 'sample\tmode\tprefill_tok_s\tdecode_tok_s\n' >"${OUT_DIR}/results.tsv"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'prefill_batch=%s\n' "${PREFILL_BATCH}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'idle_seconds=%s\n' "${IDLE_SECONDS}"
    printf 'order=%s\n' "${ORDER[*]}"
    printf 'required_cargo_feature=flash-attn-ck\n'
} >"${OUT_DIR}/manifest.txt"

run_one() {
    local sample="$1"
    local mode="$2"
    local log="${OUT_DIR}/${sample}_${mode}.log"
    local ck_env=()
    if [[ "${mode}" == on ]]; then
        ck_env+=(HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${SIDECAR}")
    fi

    timeout --signal=INT --kill-after=5s 900s \
        env -u HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB \
        "${ck_env[@]}" \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH="${PREFILL_BATCH}" \
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
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill "${PREFILL_TOKENS}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 8 >"${log}" 2>&1

    if [[ "${mode}" == on ]]; then
        if ! rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"; then
            echo "CK sidecar did not activate in on sample ${sample}" >&2
            return 1
        fi
    elif rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"; then
        echo "CK sidecar unexpectedly active in off sample ${sample}" >&2
        return 1
    fi

    local prefill decode
    prefill="$(awk '/^  median:/ {value=$3} END {print value}' "${log}")"
    decode="$(awk -F'gen_tok_s=' '/^SUMMARY / {split($2, fields, " "); value=fields[1]} END {print value}' "${log}")"
    [[ -n "${prefill}" && -n "${decode}" ]] || {
        echo "failed to parse ${log}" >&2
        return 1
    }
    printf '%s\t%s\t%s\t%s\n' "${sample}" "${mode}" "${prefill}" "${decode}" | tee -a "${OUT_DIR}/results.tsv"
}

for index in "${!ORDER[@]}"; do
    sample="$(printf '%02d' "$((index + 1))")"
    run_one "${sample}" "${ORDER[index]}"
    if (( index + 1 < ${#ORDER[@]} )); then
        sleep "${IDLE_SECONDS}"
    fi
done

awk -F'\t' '
    NR > 1 {sum[$2] += $3; count[$2]++}
    END {
        off = sum["off"] / count["off"];
        on = sum["on"] / count["on"];
        printf "off_mean_tok_s=%.3f\n", off;
        printf "on_mean_tok_s=%.3f\n", on;
        printf "speedup=%.6f\n", on / off;
        printf "delta_pct=%.3f\n", 100.0 * (on / off - 1.0);
    }
' "${OUT_DIR}/results.tsv" | tee "${OUT_DIR}/summary.txt"
printf 'out_dir=%s\n' "${OUT_DIR}"
