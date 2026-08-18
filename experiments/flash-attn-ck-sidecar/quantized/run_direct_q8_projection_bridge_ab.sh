#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
TRIALS="${TRIALS:-7}"
ROWS="${ROWS:-128 512 2048 8192}"
SLEEP_SECS="${SLEEP_SECS:-2}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/direct_q8_projection_bridge_ab_$(date +%Y%m%d_%H%M%S)}"
BIN="${ROOT}/build/direct_q8_projection_bridge_smoke"
read -r -a row_values <<<"${ROWS}"

mkdir -p "${ROOT}/build" "${RESULT_DIR}"
"${ROCM_PATH}/bin/hipcc" -O3 --offload-arch="${GPU_ARCH}" -std=c++17 \
    "${ROOT}/direct_q8_projection_bridge_smoke.hip" -o "${BIN}" \
    >"${RESULT_DIR}/build.log" 2>&1

for rows in "${row_values[@]}"; do
    for trial in $(seq 1 "${TRIALS}"); do
        printf 'trial=%s ' "${trial}" | tee -a "${RESULT_DIR}/raw.log"
        "${BIN}" "${rows}" | tee -a "${RESULT_DIR}/raw.log"
        sleep "${SLEEP_SECS}"
    done
done

awk '
{
    trial=""; rows=""; baseline=""; fused=""; speedup=""; mismatches=""; maxd=""; maxsum="";
    for (i=1; i<=NF; ++i) {
        split($i, kv, "=");
        if (kv[1] == "trial") trial=kv[2];
        else if (kv[1] == "rows") rows=kv[2];
        else if (kv[1] == "baseline_ms") baseline=kv[2];
        else if (kv[1] == "fused_ms") fused=kv[2];
        else if (kv[1] == "speedup") speedup=kv[2];
        else if (kv[1] == "q_mismatches") mismatches=kv[2];
        else if (kv[1] == "max_d_abs") maxd=kv[2];
        else if (kv[1] == "max_sum_abs") maxsum=kv[2];
    }
    if (trial == "" || rows == "" || baseline == "" || fused == "" || speedup == "" ||
        mismatches == "" || maxd == "" || maxsum == "") {
        print "incomplete direct-Q8 bridge record" > "/dev/stderr";
        exit 3;
    }
    print trial, rows, baseline, fused, speedup, mismatches, maxd, maxsum;
}
' OFS='\t' "${RESULT_DIR}/raw.log" \
    | { printf 'trial\trows\tbaseline_ms\tfused_ms\tspeedup\tq_mismatches\tmax_d_abs\tmax_sum_abs\n'; cat; } \
    >"${RESULT_DIR}/results.tsv"

expected=$(( TRIALS * ${#row_values[@]} ))
actual=$(( $(wc -l <"${RESULT_DIR}/results.tsv") - 1 ))
[[ "${actual}" -eq "${expected}" ]] || {
    echo "expected ${expected} result rows, found ${actual}" >&2
    exit 3
}

echo "results=${RESULT_DIR}/results.tsv"
