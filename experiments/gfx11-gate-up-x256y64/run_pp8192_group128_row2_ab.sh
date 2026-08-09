#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_group128_row2_ab}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-10}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-2}"
SLEEP_SECS="${SLEEP_SECS:-5}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
USE_CK="${USE_CK:-0}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_target750.so}"

for path in "${MODEL}" "${BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done
if [[ "${USE_CK}" == "1" && ! -e "${CK_LIB}" ]]; then
    echo "missing requested quantized CK sidecar: ${CK_LIB}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' > "${OUT_DIR}/results.tsv"
sha256sum "${BIN}" > "${OUT_DIR}/artifacts.sha256"
if [[ "${USE_CK}" == "1" ]]; then
    sha256sum "${CK_LIB}" >> "${OUT_DIR}/artifacts.sha256"
fi

run_command() {
    local bin="$1" log="$2" row2="$3"
    local -a ck_env=()
    if [[ "${USE_CK}" == "1" ]]; then
        ck_env=(HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}")
    fi
    timeout --signal=INT --kill-after=5s 180s \
        env \
        "${ck_env[@]}" \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=32 \
        HIPFIRE_DPM_WARMUP_SECS=5 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
        HIPFIRE_RDNA3_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2="${row2}" \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${bin}" "${MODEL}" \
        --prefill 8192 --prefill-runs "${PREFILL_RUNS}" --warmup 2 --gen 32 \
        > "${log}" 2>&1

    if [[ "${USE_CK}" == "1" ]] &&
       { ! rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}" ||
         ! rg -q '^quantized FlashAttention CK prefill active:' "${log}"; }; then
        echo "quantized CK sidecar was not active; refusing an invalid A/B: ${log}" >&2
        return 1
    fi
}

run_one() {
    local pair="$1" order="$2" mode="$3" bin log summary prefill gen token_ids
    bin="${BIN}"
    log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    if [[ "${mode}" == "row1" ]]; then
        run_command "${bin}" "${log}" 0
    else
        run_command "${bin}" "${log}" 1
    fi

    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

echo "Prewarming row1 and row2 binaries on HIP device ${GPU_ID}"
run_command "${BIN}" "${OUT_DIR}/prewarm_row1.log" 0
sleep "${SLEEP_SECS}"
run_command "${BIN}" "${OUT_DIR}/prewarm_row2.log" 1
sleep "${SLEEP_SECS}"

for ((pair=1; pair<=TRIALS; pair++)); do
    if (( pair % 2 == 1 )); then
        modes=(row1 row2)
    else
        modes=(row2 row1)
    fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        sleep "${SLEEP_SECS}"
    done
done

python3 - "${OUT_DIR}/results.tsv" "${TRIM_EACH_SIDE}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trim = int(sys.argv[2])

def samples(mode, field, trim_values=False):
    values = [float(r[field]) for r in rows if r["mode"] == mode]
    if trim_values and trim:
        values = sorted(values)
        if len(values) <= 2 * trim:
            raise SystemExit(f"cannot trim {trim} samples from each side of {len(values)} values")
        values = values[trim:-trim]
    return values

for mode in ("row1", "row2"):
    prefill = samples(mode, "prefill_tok_s")
    decode = samples(mode, "gen_tok_s")
    print(f"{mode}: prefill_median={statistics.median(prefill):.3f} "
          f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}")

row1 = statistics.median(samples("row1", "prefill_tok_s", trim_values=True))
row2 = statistics.median(samples("row2", "prefill_tok_s", trim_values=True))
token_sets = {
    mode: {r["token_ids"] for r in rows if r["mode"] == mode}
    for mode in ("row1", "row2")
}
print(f"trim_each_side={trim} row1_trimmed_median={row1:.3f} row2_trimmed_median={row2:.3f}")
print(f"row2_vs_row1={row2 / row1:.4f}x ({(row2 / row1 - 1) * 100:+.2f}%)")
print(f"token_ids_match={token_sets['row1'] == token_sets['row2']} "
      f"row1_variants={len(token_sets['row1'])} row2_variants={len(token_sets['row2'])}")
PY
