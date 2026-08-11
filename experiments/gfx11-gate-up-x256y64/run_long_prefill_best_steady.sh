#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-16384}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
PREFILL_BATCH="${PREFILL_BATCH:-2048}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp${PREFILL_TOKENS}_best_steady_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S_%N)}"

(( PREFILL_TOKENS > 0 && PREFILL_RUNS >= 2 && PREFILL_BATCH > 0 )) || {
    echo "PREFILL_TOKENS and PREFILL_BATCH must be positive; PREFILL_RUNS must be at least 2" >&2
    exit 1
}

for path in "${MODEL}" "${SIDECAR}" "${BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir "${OUT_DIR}"
sha256sum "${MODEL}" "${SIDECAR}" "${BIN}" >"${OUT_DIR}/artifacts.sha256"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'prefill_batch=%s\n' "${PREFILL_BATCH}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'kv_mode=asym3\n'
    printf 'graph=0\n'
} >"${OUT_DIR}/manifest.txt"

timeout --signal=INT --kill-after=5s 900s \
    env \
    HIP_VISIBLE_DEVICES="${GPU_ID}" \
    HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${SIDECAR}" \
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
    "${BIN}" "${MODEL}" \
        --prefill "${PREFILL_TOKENS}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 8 \
    >"${OUT_DIR}/bench.log" 2>&1

rg -q '^staged quantized FlashAttention CK prefill active:' "${OUT_DIR}/bench.log" || {
    echo "staged CK route was not active" >&2
    exit 1
}
rg '^PREFILL_SUMMARY |^SUMMARY |^TOKEN_IDS ' "${OUT_DIR}/bench.log" | tee "${OUT_DIR}/summary.txt"
printf 'out_dir=%s\n' "${OUT_DIR}"
