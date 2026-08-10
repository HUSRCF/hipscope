#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
TRIALS="${TRIALS:-5}"
SLEEP_SECS="${SLEEP_SECS:-2}"
CONTEXTS="${CONTEXTS:-2048 8192 16384}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/native_ab_$(date +%Y%m%d_%H%M%S)}"
BIN="${ROOT}/build/quantized_ck_pipeline_smoke"
read -r -a context_values <<<"${CONTEXTS}"

mkdir -p "${RESULT_DIR}"
BUILD_ONLY=1 GPU_ARCH="${GPU_ARCH}" \
    bash "${ROOT}/run_quantized_ck_pipeline_smoke.sh" \
    >"${RESULT_DIR}/build.log" 2>&1

for query_rows in 1 16; do
    mode="--native-ab"
    if [[ "${query_rows}" == "16" ]]; then
        mode="--native-ab-q16"
    fi
    for trial in $(seq 1 "${TRIALS}"); do
        printf 'trial=%s query_rows=%s\n' "${trial}" "${query_rows}" \
            | tee -a "${RESULT_DIR}/raw.log"
        "${BIN}" "${mode}" "${context_values[@]}" | tee -a "${RESULT_DIR}/raw.log"
        sleep "${SLEEP_SECS}"
    done
done

awk '
/^trial=/ {
    split($1, t, "="); trial=t[2];
    split($2, q, "="); query_rows=q[2];
}
/^case=native-ab/ {
    seq=""; rotate=""; ck=""; total=""; native=""; speedup=""; maxabs="";
    for (i=1; i<=NF; ++i) {
        split($i, kv, "=");
        if (kv[1] == "seqlen_k") seq=kv[2];
        else if (kv[1] == "rotate_ms") rotate=kv[2];
        else if (kv[1] == "ck_ms") ck=kv[2];
        else if (kv[1] == "ck_total_ms") total=kv[2];
        else if (kv[1] == "native_ms") native=kv[2];
        else if (kv[1] == "speedup") speedup=kv[2];
        else if (kv[1] == "max_abs") maxabs=kv[2];
    }
    if (trial == "" || query_rows == "" || seq == "" || rotate == "" || ck == "" ||
        total == "" || native == "" || speedup == "" || maxabs == "") {
        print "incomplete native-ab record" > "/dev/stderr";
        exit 3;
    }
    print trial, query_rows, seq, rotate, ck, total, native, speedup, maxabs;
}
' OFS='\t' "${RESULT_DIR}/raw.log" \
    | { printf 'trial\tquery_rows\tseqlen_k\trotate_ms\tck_ms\tck_total_ms\tnative_ms\tnative_over_ck\tmax_abs\n'; cat; } \
    >"${RESULT_DIR}/results.tsv"

expected=$(( 2 * TRIALS * ${#context_values[@]} ))
actual=$(( $(wc -l <"${RESULT_DIR}/results.tsv") - 1 ))
[[ "${actual}" -eq "${expected}" ]] || {
    echo "expected ${expected} result rows, found ${actual}" >&2
    exit 3
}

echo "results=${RESULT_DIR}/results.tsv"
