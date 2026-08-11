#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-16384}"
TRIALS="${TRIALS:-3}"
SLEEP_SECS="${SLEEP_SECS:-8}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp${PREFILL_TOKENS}_chunk2048_4096_ab_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S_%N)}"

for path in "${MODEL}" "${SIDECAR}" "${BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done
mkdir -p "${OUT_DIR}"
printf 'pair\torder\tchunk\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"

run_one() {
    local pair="$1" order="$2" chunk="$3"
    local log="${OUT_DIR}/pair_${pair}_${order}_chunk${chunk}.log"

    timeout --signal=INT --kill-after=5s 900s \
        env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${SIDECAR}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH="${chunk}" \
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
        --prefill "${PREFILL_TOKENS}" --prefill-runs 1 \
        --warmup 2 --gen 32 >"${log}" 2>&1

    rg -q '^staged quantized FlashAttention CK prefill active:' "${log}" || {
        echo "CK sidecar did not activate: ${log}" >&2
        return 1
    }

    local summary prefill gen token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<<"${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${chunk}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair = 1; pair <= TRIALS; pair++)); do
    if ((pair % 2 == 1)); then
        chunks=(2048 4096)
    else
        chunks=(4096 2048)
    fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${chunks[$order]}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${OUT_DIR}/results.tsv" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
medians = {}
for chunk in ("2048", "4096"):
    selected = [row for row in rows if row["chunk"] == chunk]
    prefill = [float(row["prefill_tok_s"]) for row in selected]
    decode = [float(row["gen_tok_s"]) for row in selected]
    medians[chunk] = statistics.median(prefill)
    print(
        f"chunk{chunk}: prefill_median={medians[chunk]:.3f} "
        f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}"
    )

token_sets = {
    chunk: {row["token_ids"] for row in rows if row["chunk"] == chunk}
    for chunk in ("2048", "4096")
}
ratio = medians["4096"] / medians["2048"]
print(f"chunk4096_vs_2048={ratio:.4f}x ({(ratio - 1) * 100:+.2f}%)")
print(
    f"token_ids_match={token_sets['2048'] == token_sets['4096']} "
    f"chunk2048_variants={len(token_sets['2048'])} "
    f"chunk4096_variants={len(token_sets['4096'])}"
)
PY

sha256sum "${MODEL}" "${SIDECAR}" "${BIN}" "$0" >"${OUT_DIR}/artifacts.sha256"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'prefill_tokens=%s\n' "${PREFILL_TOKENS}"
    printf 'trials=%s\n' "${TRIALS}"
    printf 'activation_contract=group128\n'
    printf 'ffn_intermediate=F32\n'
    printf 'kv_mode=asym3\n'
    printf 'ck_sidecar=on\n'
} >"${OUT_DIR}/manifest.txt"

printf 'out_dir=%s\n' "${OUT_DIR}"
