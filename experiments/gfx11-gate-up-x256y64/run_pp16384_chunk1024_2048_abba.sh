#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PAUSE_SECS="${PAUSE_SECS:-60}"
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp16384_chunk1024_2048_abba_gpu${GPU_ID}_${STAMP}}"
RUNNER="${ROOT}/experiments/gfx11-gate-up-x256y64/run_long_prefill_best_steady.sh"

mkdir "${OUT_DIR}"
printf 'order\tchunk\tmedian_tok_s\tmedian_ms\n' >"${OUT_DIR}/results.tsv"

run_one() {
    local order="$1"
    local chunk="$2"
    local run_dir="${OUT_DIR}/${order}_chunk${chunk}"

    GPU_ID="${GPU_ID}" \
    PREFILL_TOKENS=16384 \
    PREFILL_RUNS=2 \
    PREFILL_BATCH="${chunk}" \
    OUT_DIR="${run_dir}" \
    "${RUNNER}"

    awk -v order="${order}" -v chunk="${chunk}" '
        /^  median:/ {
            ms = $2
            sub(/ms$/, "", ms)
            rate = $3
            print order "\t" chunk "\t" rate "\t" ms
        }
    ' "${run_dir}/bench.log" >>"${OUT_DIR}/results.tsv"
}

order=(2048 1024 1024 2048)
for index in "${!order[@]}"; do
    if (( index > 0 )); then
        sleep "${PAUSE_SECS}"
    fi
    run_one "$((index + 1))" "${order[index]}"
done

printf 'chunk\trun1_tok_s\trun2_tok_s\tmean_tok_s\n' >"${OUT_DIR}/summary.tsv"
awk -F '\t' '
    NR > 1 {
        count[$2]++
        rate[$2, count[$2]] = $3
    }
    END {
        for (chunk in count) {
            mean = (rate[chunk, 1] + rate[chunk, 2]) / 2.0
            printf "%s\t%.1f\t%.1f\t%.1f\n", chunk, rate[chunk, 1], rate[chunk, 2], mean
        }
    }
' "${OUT_DIR}/results.tsv" | sort -n >>"${OUT_DIR}/summary.tsv"

cat "${OUT_DIR}/results.tsv"
cat "${OUT_DIR}/summary.tsv"
printf 'out_dir=%s\n' "${OUT_DIR}"
