#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_group256_serial_ck_gpu1_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-1}"
SLEEP_SECS="${SLEEP_SECS:-5}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
GROUP256_SCOPE="${GROUP256_SCOPE:-all}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' > "${OUT_DIR}/results.tsv"
sha256sum "${BIN}" "${CK_LIB}" "${MODEL}" > "${OUT_DIR}/artifacts.sha256"

run_command() {
    local log="$1" group256="$2" group256_all=0 group256_gate_up=0
    if [[ "${group256}" == "1" ]]; then
        case "${GROUP256_SCOPE}" in
            all) group256_all=1 ;;
            gate_up) group256_gate_up=1 ;;
            *) echo "invalid GROUP256_SCOPE=${GROUP256_SCOPE}; expected all or gate_up" >&2; return 1 ;;
        esac
    fi
    timeout --signal=INT --kill-after=5s 240s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=32 \
        HIPFIRE_DPM_WARMUP_SECS=5 \
        HIPFIRE_QKVZA_SPLIT_TAIL=1 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
        HIPFIRE_RDNA3_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW="${group256_all}" \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP="${group256_gate_up}" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs "${PREFILL_RUNS}" --warmup 2 --gen 32 \
        > "${log}" 2>&1

    if ! rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}" ||
       ! rg -q '^quantized FlashAttention CK prefill active:' "${log}"; then
        echo "quantized CK sidecar was not active: ${log}" >&2
        return 1
    fi
    if [[ "${group256}" == "1" ]] &&
       ! rg -q '^RDNA3 Q8 group256 gate/up prefill active:' "${log}"; then
        echo "group256 serial-row route was not active: ${log}" >&2
        return 1
    fi
}

run_one() {
    local pair="$1" order="$2" mode="$3" group256=0
    local log summary prefill gen token_ids
    [[ "${mode}" == "group256" ]] && group256=1
    log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    run_command "${log}" "${group256}"
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

echo "Prewarming both routes on HIP device ${GPU_ID}"
echo "Group256 scope: ${GROUP256_SCOPE}"
run_command "${OUT_DIR}/prewarm_group128.log" 0
sleep "${SLEEP_SECS}"
run_command "${OUT_DIR}/prewarm_group256.log" 1
sleep "${SLEEP_SECS}"

for ((pair=1; pair<=TRIALS; pair++)); do
    if (( pair % 2 == 1 )); then
        modes=(group128 group256)
    else
        modes=(group256 group128)
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
            raise SystemExit(f"cannot trim {trim} samples from each side of {len(values)}")
        values = values[trim:-trim]
    return values

for mode in ("group128", "group256"):
    p = samples(mode, "prefill_tok_s")
    d = samples(mode, "gen_tok_s")
    print(f"{mode}: prefill_median={statistics.median(p):.3f} "
          f"decode_median={statistics.median(d):.3f} raw_prefill={p}")

base = statistics.median(samples("group128", "prefill_tok_s", True))
candidate = statistics.median(samples("group256", "prefill_tok_s", True))
token_sets = {
    mode: {r["token_ids"] for r in rows if r["mode"] == mode}
    for mode in ("group128", "group256")
}
print(f"trim_each_side={trim} group128_trimmed_median={base:.3f} "
      f"group256_trimmed_median={candidate:.3f}")
print(f"group256_vs_group128={candidate / base:.4f}x "
      f"({(candidate / base - 1) * 100:+.2f}%)")
print(f"token_ids_match={token_sets['group128'] == token_sets['group256']} "
      f"group128_variants={len(token_sets['group128'])} "
      f"group256_variants={len(token_sets['group256'])}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
