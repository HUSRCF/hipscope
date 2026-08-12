#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_ROOT="$(cd "${ROOT}/.." && pwd)"
BENCH="${BENCH:-${ROOT}/build/staged_quantized_ck_bench}"
BASE_LIB_DIR="${BASE_LIB_DIR:-${SIDECAR_ROOT}/build}"
CANDIDATE_LIB_DIR="${CANDIDATE_LIB_DIR:-${SIDECAR_ROOT}/build-fa4-gfx11-d256}"
QUANTIZED_LIB_DIR="${QUANTIZED_LIB_DIR:-${CANDIDATE_LIB_DIR}}"
QUANTIZED_LIB="${QUANTIZED_LIB_DIR}/libhipfire_flash_attn_ck_quantized_staged.so"
GPU_ID="${GPU_ID:-1}"
PAIRS="${PAIRS:-10}"
SLEEP_SECS="${SLEEP_SECS:-2}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/staged_fa4_gfx11_d256_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"
RAW="${OUT_DIR}/raw.csv"
MANIFEST="${OUT_DIR}/manifest.txt"

mkdir -p "${OUT_DIR}"
for path in \
    "${BENCH}" \
    "${BASE_LIB_DIR}/libhipfire_flash_attn_ck.so" \
    "${CANDIDATE_LIB_DIR}/libhipfire_flash_attn_ck.so" \
    "${QUANTIZED_LIB}"; do
    [[ -f "${path}" ]] || { printf 'missing required file: %s\n' "${path}" >&2; exit 1; }
done
printf 'pair,label,seqlen_q,seqlen_k,packed_ck_ms,staged_ck_ms,staged_speedup,max_abs,mean_abs,staged_scratch_bytes\n' > "${RAW}"
{
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "${SIDECAR_ROOT}/../.." rev-parse HEAD)"
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'pairs=%s\n' "${PAIRS}"
    printf 'sleep_secs=%s\n' "${SLEEP_SECS}"
    printf 'baseline_lib=%s\n' "${BASE_LIB_DIR}/libhipfire_flash_attn_ck.so"
    printf 'candidate_lib=%s\n' "${CANDIDATE_LIB_DIR}/libhipfire_flash_attn_ck.so"
    printf 'quantized_lib=%s\n' "${QUANTIZED_LIB}"
    printf 'baseline_sha256=%s\n' "$(sha256sum "${BASE_LIB_DIR}/libhipfire_flash_attn_ck.so" | cut -d' ' -f1)"
    printf 'candidate_sha256=%s\n' "$(sha256sum "${CANDIDATE_LIB_DIR}/libhipfire_flash_attn_ck.so" | cut -d' ' -f1)"
    printf 'quantized_sha256=%s\n' "$(sha256sum "${QUANTIZED_LIB}" | cut -d' ' -f1)"
    printf 'bench_sha256=%s\n' "$(sha256sum "${BENCH}" | cut -d' ' -f1)"
} > "${MANIFEST}"

run_one() {
    local pair="$1"
    local label="$2"
    local lib_dir="$3"
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        LD_LIBRARY_PATH="${lib_dir}:${QUANTIZED_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${BENCH}" |
        sed "s/^/pair=${pair},label=${label},/" |
        awk -F, '
            {
                for (i = 1; i <= NF; ++i) {
                    split($i, kv, "=")
                    value[kv[1]] = kv[2]
                }
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                    value["pair"], value["label"], value["seqlen_q"], value["seqlen_k"],
                    value["packed_ck_ms"], value["staged_ck_ms"], value["staged_speedup"],
                    value["max_abs"], value["mean_abs"], value["staged_scratch_bytes"]
                delete value
            }
        ' >> "${RAW}"
}

for ((pair = 0; pair < PAIRS; ++pair)); do
    if ((pair % 2 == 0)); then
        run_one "${pair}" baseline "${BASE_LIB_DIR}"
        sleep "${SLEEP_SECS}"
        run_one "${pair}" candidate "${CANDIDATE_LIB_DIR}"
    else
        run_one "${pair}" candidate "${CANDIDATE_LIB_DIR}"
        sleep "${SLEEP_SECS}"
        run_one "${pair}" baseline "${BASE_LIB_DIR}"
    fi
    sleep "${SLEEP_SECS}"
done

printf '%s\n' "${RAW}"
