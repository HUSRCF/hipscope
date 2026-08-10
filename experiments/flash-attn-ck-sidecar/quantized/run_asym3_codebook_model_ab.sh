#!/usr/bin/env bash
set -euo pipefail

QUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${QUANT_ROOT}/../../.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
RUNTIME_LD_LIBRARY_PATH="${HIP_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EXE="${EXE:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-0}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
PREFILL="${PREFILL:-8192}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
TRIALS="${TRIALS:-5}"
GEN="${GEN:-8}"
WARMUP="${WARMUP:-2}"
SLEEP_SECS="${SLEEP_SECS:-5}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-3}"
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"
CANDIDATE_STORAGE="${CANDIDATE_STORAGE:-lds}"
RESULT_DIR="${RESULT_DIR:-${QUANT_ROOT}/results/asym3_codebook_model_ab_$(date +%Y%m%d_%H%M%S)}"
BASELINE_LIB="${BASELINE_LIB:-${RESULT_DIR}/lib/libhipfire_flash_attn_ck_switch.so}"
CANDIDATE_LIB="${CANDIDATE_LIB:-${RESULT_DIR}/lib/libhipfire_flash_attn_ck_${CANDIDATE_STORAGE}.so}"

if [[ "${CANDIDATE_STORAGE}" != "constant" && "${CANDIDATE_STORAGE}" != "lds" ]]; then
    echo "CANDIDATE_STORAGE must be constant or lds" >&2
    exit 2
fi
for path in "${EXE}" "${MODEL}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${RESULT_DIR}/lib"
printf 'trial\torder\tmode\tprefill_ms\tprefill_tok_s\tgen_tok_s\tck_active\n' \
    >"${RESULT_DIR}/results.tsv"

build_sidecar() {
    local output="$1" constant_codebook="$2" lds_codebook="$3"
    env ROCM_PATH="${ROCM_PATH}" GPU_ARCH="${GPU_ARCH}" \
        ASYM3_CODEBOOK="${constant_codebook}" ASYM3_LDS_CODEBOOK="${lds_codebook}" \
        OUT="${output}" bash "${QUANT_ROOT}/build_quantized_sidecar.sh"
}

if [[ "${BUILD_SIDECARS:-1}" == "1" ]]; then
    build_sidecar "${BASELINE_LIB}" 0 0 >"${RESULT_DIR}/build_switch.log" 2>&1
    if [[ "${CANDIDATE_STORAGE}" == "constant" ]]; then
        build_sidecar "${CANDIDATE_LIB}" 1 0 >"${RESULT_DIR}/build_candidate.log" 2>&1
    else
        build_sidecar "${CANDIDATE_LIB}" 0 1 >"${RESULT_DIR}/build_candidate.log" 2>&1
    fi
fi
for path in "${BASELINE_LIB}" "${CANDIDATE_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing sidecar: ${path}" >&2; exit 2; }
done

{
    echo "date=$(date -Is)"
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "exe=${EXE}"
    echo "model=${MODEL}"
    echo "model_size_bytes=$(stat -c %s "${MODEL}")"
    echo "model_mtime=$(stat -c %y "${MODEL}")"
    echo "gpu_id=${GPU_ID}"
    echo "gpu_arch=${GPU_ARCH}"
    echo "rocm_path=${ROCM_PATH}"
    echo "hip_root=${HIP_ROOT}"
    echo "runtime_ld_library_path=${RUNTIME_LD_LIBRARY_PATH}"
    echo "prefill=${PREFILL}"
    echo "prefill_runs=${PREFILL_RUNS}"
    echo "gen=${GEN}"
    echo "warmup=${WARMUP}"
    echo "trials=${TRIALS}"
    echo "candidate_storage=${CANDIDATE_STORAGE}"
    sha256sum "${EXE}" "${MODEL}" "${BASELINE_LIB}" "${CANDIDATE_LIB}"
} >"${RESULT_DIR}/meta.txt"

run_one() {
    local trial="$1" order="$2" mode="$3" lib="$4"
    local log="${RESULT_DIR}/trial_${trial}_${order}_${mode}.log"
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_DPM_WARMUP_SECS="${DPM_WARMUP_SECS}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${lib}" \
        timeout --signal=INT --kill-after=10s "${TIMEOUT_SECS}s" \
        "${EXE}" "${MODEL}" --prefill "${PREFILL}" \
        --prefill-runs "${PREFILL_RUNS}" --warmup "${WARMUP}" --gen "${GEN}" \
        >"${log}" 2>&1

    local summary prefill_ms prefill_tok_s gen_tok_s ck_active
    summary="$(grep '^SUMMARY' "${log}" | tail -1)"
    prefill_ms="$(sed -nE 's/.*prefill_wall_ms=([0-9.]+).*/\1/p' "${log}" | tail -1)"
    prefill_tok_s="$(sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p' <<<"${summary}")"
    gen_tok_s="$(sed -nE 's/.*gen_tok_s=([0-9.]+).*/\1/p' <<<"${summary}")"
    ck_active=0
    grep -q 'quantized FlashAttention CK prefill active' "${log}" && ck_active=1
    [[ -n "${summary}" && -n "${prefill_ms}" && "${ck_active}" == "1" ]] || {
        echo "incomplete or inactive run: ${log}" >&2
        return 3
    }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${trial}" "${order}" "${mode}" "${prefill_ms}" "${prefill_tok_s}" \
        "${gen_tok_s}" "${ck_active}" | tee -a "${RESULT_DIR}/results.tsv"
}

for trial in $(seq 1 "${TRIALS}"); do
    if (( trial % 2 == 1 )); then
        modes=(switch candidate)
    else
        modes=(candidate switch)
    fi
    order=0
    for mode in "${modes[@]}"; do
        order=$((order + 1))
        if [[ "${mode}" == "switch" ]]; then
            lib="${BASELINE_LIB}"
        else
            lib="${CANDIDATE_LIB}"
        fi
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
if len(rows) != trials * 2:
    raise SystemExit(f"incomplete result table: got {len(rows)}, expected {trials * 2}")
by_mode = {}
for mode in ("switch", "candidate"):
    selected = [row for row in rows if row["mode"] == mode]
    if len(selected) != trials or any(row["ck_active"] != "1" for row in selected):
        raise SystemExit(f"mode={mode}: incomplete or inactive trials")
    by_mode[mode] = [float(row["prefill_tok_s"]) for row in selected]
    print(f"{mode}: median={statistics.median(by_mode[mode]):.3f} tok/s raw={by_mode[mode]}")
ratio = statistics.median(by_mode["candidate"]) / statistics.median(by_mode["switch"])
print(f"candidate_vs_switch={ratio:.4f}x ({(ratio - 1) * 100:+.2f}%)")
PY

echo "results=${RESULT_DIR}"
