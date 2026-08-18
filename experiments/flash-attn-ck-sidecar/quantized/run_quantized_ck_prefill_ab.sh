#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
TRIALS="${TRIALS:-3}"
SLEEP_SECS="${SLEEP_SECS:-2}"
TILES="${TILES:-64x64 64x32}"
Q_ROWS="${Q_ROWS:-64 128 512}"
CONTEXTS="${CONTEXTS:-2048 8192}"
CK_OUTPUT_F32="${CK_OUTPUT_F32:-0}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/prefill_ab_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULT_DIR}/bin"
: >"${RESULT_DIR}/raw.log"
read -r -a tile_values <<<"${TILES}"
read -r -a query_values <<<"${Q_ROWS}"
read -r -a context_values <<<"${CONTEXTS}"

for tile in "${tile_values[@]}"; do
    bm="${tile%x*}"
    bn="${tile#*x}"
    bin="${RESULT_DIR}/bin/quantized_ck_bm${bm}_bn${bn}"
    env BUILD_ONLY=1 GPU_ARCH="${GPU_ARCH}" CK_BM="${bm}" CK_BN="${bn}" \
        CK_OUTPUT_F32="${CK_OUTPUT_F32}" BIN="${bin}" \
        bash "${ROOT}/run_quantized_ck_pipeline_smoke.sh" \
        >"${RESULT_DIR}/build_bm${bm}_bn${bn}.log" 2>&1

    for trial in $(seq 1 "${TRIALS}"); do
        for query_rows in "${query_values[@]}"; do
            printf 'tile=%s trial=%s query_rows=%s\n' "${tile}" "${trial}" "${query_rows}" \
                | tee -a "${RESULT_DIR}/raw.log"
            "${bin}" --native-ab-qrows "${query_rows}" "${context_values[@]}" \
                | tee -a "${RESULT_DIR}/raw.log"
        done
        sleep "${SLEEP_SECS}"
    done
done

awk '
/^tile=/ {
    split($1, a, "="); tile=a[2];
    split($2, b, "="); trial=b[2];
    split($3, c, "="); query_rows=c[2];
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
    if (tile == "" || trial == "" || query_rows == "" || seq == "" || rotate == "" ||
        ck == "" || total == "" || native == "" || speedup == "" || maxabs == "") {
        print "incomplete native-ab record" > "/dev/stderr";
        exit 3;
    }
    print tile, trial, query_rows, seq, rotate, ck, total, native, speedup, maxabs;
}
' OFS='\t' "${RESULT_DIR}/raw.log" \
    | { printf 'tile\ttrial\tquery_rows\tseqlen_k\trotate_ms\tck_ms\tck_total_ms\tnative_ms\tnative_over_ck\tmax_abs\n'; cat; } \
    >"${RESULT_DIR}/results.tsv"

expected=$(( ${#tile_values[@]} * TRIALS * ${#query_values[@]} * ${#context_values[@]} ))
actual=$(( $(wc -l <"${RESULT_DIR}/results.tsv") - 1 ))
[[ "${actual}" -eq "${expected}" ]] || {
    echo "expected ${expected} result rows, found ${actual}" >&2
    exit 3
}

echo "results=${RESULT_DIR}/results.tsv"
