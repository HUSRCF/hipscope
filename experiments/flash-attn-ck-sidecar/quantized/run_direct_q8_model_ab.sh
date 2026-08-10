#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
EXE="${EXE:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
BASELINE_LIB="${BASELINE_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_final_20260808.so}"
CANDIDATE_LIB="${CANDIDATE_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"
GPU_ID="${GPU_ID:-0}"
PREFILL="${PREFILL:-8192}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
TRIALS="${TRIALS:-5}"
GEN="${GEN:-8}"
WARMUP="${WARMUP:-2}"
SLEEP_SECS="${SLEEP_SECS:-5}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-3}"
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/results/direct_q8_model_ab_$(date +%Y%m%d_%H%M%S)}"

for path in "${EXE}" "${MODEL}" "${BASELINE_LIB}" "${CANDIDATE_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${RESULT_DIR}"
printf 'trial\torder\tmode\tprefill_ms\tprefill_tok_s\tgen_tok_s\tbridge_active\n' >"${RESULT_DIR}/results.tsv"

{
    echo "date=$(date -Is)"
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "exe=${EXE}"
    echo "model=${MODEL}"
    echo "baseline_lib=${BASELINE_LIB}"
    echo "candidate_lib=${CANDIDATE_LIB}"
    echo "gpu_id=${GPU_ID}"
    echo "prefill=${PREFILL}"
    echo "prefill_runs=${PREFILL_RUNS}"
    echo "trials=${TRIALS}"
} >"${RESULT_DIR}/meta.txt"

run_one() {
    local trial="$1" order="$2" mode="$3" lib="$4"
    local log="${RESULT_DIR}/trial_${trial}_${order}_${mode}.log"
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_DPM_WARMUP_SECS="${DPM_WARMUP_SECS}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${lib}" \
        timeout --signal=INT --kill-after=10s "${TIMEOUT_SECS}s" \
        "${EXE}" "${MODEL}" --prefill "${PREFILL}" \
        --prefill-runs "${PREFILL_RUNS}" --warmup "${WARMUP}" --gen "${GEN}" \
        >"${log}" 2>&1

    local summary prefill_ms prefill_tok_s gen_tok_s bridge_active
    summary="$(grep '^SUMMARY' "${log}" | tail -1)"
    prefill_ms="$(sed -nE 's/.*prefill_wall_ms=([0-9.]+).*/\1/p' "${log}" | tail -1)"
    prefill_tok_s="$(sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p' <<<"${summary}")"
    gen_tok_s="$(sed -nE 's/.*gen_tok_s=([0-9.]+).*/\1/p' <<<"${summary}")"
    bridge_active=0
    grep -q 'direct MQ-Q8 projection bridge active' "${log}" && bridge_active=1
    [[ -n "${summary}" && -n "${prefill_ms}" && -n "${prefill_tok_s}" && -n "${gen_tok_s}" ]] || {
        echo "incomplete benchmark output in ${log}" >&2
        exit 3
    }
    if [[ "${mode}" == "candidate" && "${bridge_active}" != "1" ]]; then
        echo "candidate MQ-Q8 bridge did not activate in ${log}" >&2
        exit 3
    fi
    if [[ "${mode}" == "baseline" && "${bridge_active}" != "0" ]]; then
        echo "baseline unexpectedly activated MQ-Q8 bridge in ${log}" >&2
        exit 3
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${trial}" "${order}" "${mode}" "${prefill_ms}" "${prefill_tok_s}" \
        "${gen_tok_s}" "${bridge_active}" | tee -a "${RESULT_DIR}/results.tsv"
}

for trial in $(seq 1 "${TRIALS}"); do
    if (( trial % 2 == 1 )); then
        modes=(baseline candidate)
    else
        modes=(candidate baseline)
    fi
    order=0
    for mode in "${modes[@]}"; do
        order=$((order + 1))
        if [[ "${mode}" == baseline ]]; then lib="${BASELINE_LIB}"; else lib="${CANDIDATE_LIB}"; fi
        run_one "${trial}" "${order}" "${mode}" "${lib}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${RESULT_DIR}/results.tsv" "${TRIALS}" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trials = int(sys.argv[2])
by_mode = {}
for row in rows:
    by_mode.setdefault(row["mode"], []).append(float(row["prefill_tok_s"]))
for mode in ("baseline", "candidate"):
    values = by_mode.get(mode, [])
    if len(values) != trials:
        raise SystemExit(f"expected {trials} {mode} rows, found {len(values)}")
    print(f"{mode}: median={statistics.median(values):.3f} tok/s raw={values}")
base = statistics.median(by_mode["baseline"])
candidate = statistics.median(by_mode["candidate"])
print(f"candidate_vs_baseline={candidate / base:.4f}x ({(candidate / base - 1) * 100:+.2f}%)")
PY

echo "results=${RESULT_DIR}"
