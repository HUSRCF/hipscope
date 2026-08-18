#!/usr/bin/env bash
set -euo pipefail

QUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${QUANT_ROOT}/../../.." && pwd)"
EXE="${EXE:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
CK_LIB="${CK_LIB:-${QUANT_ROOT}/build/libhipfire_flash_attn_ck_quantized.so}"
GPU_ID="${GPU_ID:-0}"
PREFILL="${PREFILL:-8192}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
PREFILL_RUNS="${PREFILL_RUNS:-1}"
TRIALS="${TRIALS:-3}"
WARMUP="${WARMUP:-2}"
GEN="${GEN:-8}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-5}"
SLEEP_SECS="${SLEEP_SECS:-5}"
TIMEOUT_SECS="${TIMEOUT_SECS:-150}"
RESULT_DIR="${RESULT_DIR:-${QUANT_ROOT}/results/all_chunk_model_ab_$(date +%Y%m%d_%H%M%S)}"

for path in "${EXE}" "${MODEL}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${RESULT_DIR}"
printf 'trial\torder\tmode\tprefill_ms\tprefill_tok_s\tgen_tok_s\tck_active\n' \
    >"${RESULT_DIR}/results.tsv"

{
    echo "date=$(date -Is)"
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "exe=${EXE}"
    echo "model=${MODEL}"
    echo "ck_lib=${CK_LIB}"
    echo "gpu_id=${GPU_ID}"
    echo "prefill=${PREFILL}"
    echo "prefill_chunk=${PREFILL_CHUNK}"
    echo "prefill_runs=${PREFILL_RUNS}"
    echo "trials=${TRIALS}"
    sha256sum "${EXE}" "${MODEL}" "${CK_LIB}"
} >"${RESULT_DIR}/meta.txt"

run_one() {
    local trial="$1" order="$2" mode="$3"
    local log="${RESULT_DIR}/trial_${trial}_${order}_${mode}.log"
    local -a sidecar_env=()
    if [[ "${mode}" == "ck" ]]; then
        sidecar_env+=("HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=${CK_LIB}")
    fi

    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH="${PREFILL_CHUNK}" \
        HIPFIRE_DPM_WARMUP_SECS="${DPM_WARMUP_SECS}" \
        "${sidecar_env[@]}" \
        timeout --signal=INT --kill-after=5s "${TIMEOUT_SECS}s" \
        "${EXE}" "${MODEL}" --prefill "${PREFILL}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup "${WARMUP}" --gen "${GEN}" >"${log}" 2>&1

    local prefill_ms prefill_tok_s gen_tok_s ck_active
    prefill_ms="$(sed -nE 's/.*prefill_wall_ms=([0-9.]+).*/\1/p' "${log}" | tail -1)"
    prefill_tok_s="$(sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p' "${log}" | tail -1)"
    gen_tok_s="$(sed -nE 's/^SUMMARY  gen_tok_s=([0-9.]+).*/\1/p' "${log}" | tail -1)"
    ck_active=0
    grep -q 'quantized FlashAttention CK prefill active' "${log}" && ck_active=1
    [[ -n "${prefill_ms}" && -n "${prefill_tok_s}" && -n "${gen_tok_s}" ]] || {
        echo "incomplete run: ${log}" >&2
        return 3
    }
    if [[ "${mode}" == "ck" && "${ck_active}" != 1 ]] || \
       [[ "${mode}" == "native" && "${ck_active}" != 0 ]]; then
        echo "unexpected CK route state for mode=${mode}: ${log}" >&2
        return 3
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${trial}" "${order}" "${mode}" "${prefill_ms}" "${prefill_tok_s}" \
        "${gen_tok_s}" "${ck_active}" | tee -a "${RESULT_DIR}/results.tsv"
}

for trial in $(seq 1 "${TRIALS}"); do
    if (( trial % 2 == 1 )); then
        modes=(native ck)
    else
        modes=(ck native)
    fi
    order=0
    for mode in "${modes[@]}"; do
        order=$((order + 1))
        run_one "${trial}" "${order}" "${mode}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${RESULT_DIR}/results.tsv" "${TRIALS}" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trials = int(sys.argv[2])
if len(rows) != 2 * trials:
    raise SystemExit(f"incomplete table: got {len(rows)}, expected {2 * trials}")
for mode, expected_active in (("native", "0"), ("ck", "1")):
    selected = [row for row in rows if row["mode"] == mode]
    if len(selected) != trials or any(row["ck_active"] != expected_active for row in selected):
        raise SystemExit(f"invalid rows for mode={mode}")
    values = [float(row["prefill_tok_s"]) for row in selected]
    print(f"{mode}: median={statistics.median(values):.3f} tok/s raw={values}")
native = statistics.median(float(r["prefill_tok_s"]) for r in rows if r["mode"] == "native")
ck = statistics.median(float(r["prefill_tok_s"]) for r in rows if r["mode"] == "ck")
print(f"ck_vs_native={ck / native:.4f}x ({(ck / native - 1) * 100:+.2f}%)")
PY

echo "results=${RESULT_DIR}"
