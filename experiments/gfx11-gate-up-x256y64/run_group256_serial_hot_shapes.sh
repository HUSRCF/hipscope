#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PAIRS="${PAIRS:-11}"
SLEEP_SECS="${SLEEP_SECS:-2}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/group256_serial_hot_shapes_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"
BIN="${ROOT}/target/release/examples/bench_hfq4_group256_direct"

mkdir -p "${OUT_DIR}"
cargo build --release -p rdna-compute --example bench_hfq4_group256_direct
printf 'label\tm\tk\tn\tlog\n' >"${OUT_DIR}/manifest.tsv"

run_shape() {
    local label="$1" m="$2" k="$3" n="$4"
    local log="${OUT_DIR}/${label}.log"
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
        "${BIN}" --m "${m}" --k "${k}" --n "${n}" \
        --pairs "${PAIRS}" --serial-row | tee "${log}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${m}" "${k}" "${n}" "${log}" >>"${OUT_DIR}/manifest.tsv"
    sleep "${SLEEP_SECS}"
}

run_shape gate_up 17408 5120 2048
run_shape ffn_down 5120 17408 2048
run_shape qkv_large 10240 5120 2048
run_shape gdn_mid 6144 5120 2048
run_shape qkvza_large 12288 5120 2048
run_shape residual_aux 5120 6144 2048

printf 'label\tgroup128_ms\tgroup256_serial_ms\tspeedup\tmax_abs\tmean_abs\n' \
    >"${OUT_DIR}/results.tsv"
while IFS=$'\t' read -r label _m _k _n log; do
    [[ "${label}" == "label" ]] && continue
    awk -F= -v label="${label}" '
        /^group128_lds_ms=/ { base=$2 }
        /^group256_ms=/ { candidate=$2 }
        /^group256_speedup=/ { speed=$2 }
        /^max_abs=/ { max_abs=$2 }
        /^mean_abs=/ { mean_abs=$2 }
        END { printf "%s\t%s\t%s\t%s\t%s\t%s\n", label, base, candidate, speed, max_abs, mean_abs }
    ' "${log}" >>"${OUT_DIR}/results.tsv"
done <"${OUT_DIR}/manifest.tsv"

cat "${OUT_DIR}/results.tsv"
echo "results: ${OUT_DIR}"
