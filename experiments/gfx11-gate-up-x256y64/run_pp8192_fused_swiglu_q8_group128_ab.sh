#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_fused_swiglu_q8_group128_ab}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-3}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-0}"
SLEEP_SECS="${SLEEP_SECS:-5}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_target750.so}"
BIN="${ROOT}/target/release/examples/bench_qwen35_mq4"

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' > "${OUT_DIR}/results.tsv"

run_one() {
    local pair="$1" order="$2" mode="$3" fused=0
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    [[ "${mode}" == "fused" ]] && fused=1

    timeout --signal=INT --kill-after=5s 180s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=0 \
        HIPFIRE_PREFILL_MAX_BATCH=2048 \
        HIPFIRE_FLASH_PARTIALS_BATCH=32 \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
        HIPFIRE_DPM_WARMUP_SECS=5 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1 \
        HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
        HIPFIRE_RDNA3_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128="${fused}" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs 1 --warmup 2 --gen 32 \
        > "${log}" 2>&1

    local summary prefill gen token_ids
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

for ((pair=1; pair<=TRIALS; pair++)); do
    if (( pair % 2 == 1 )); then
        modes=(baseline fused)
    else
        modes=(fused baseline)
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

def samples(mode, field):
    values = sorted(float(r[field]) for r in rows if r["mode"] == mode)
    if trim:
        if len(values) <= 2 * trim:
            raise SystemExit(f"cannot trim {trim} samples from each side of {len(values)} values")
        values = values[trim:-trim]
    return values

for mode in ("baseline", "fused"):
    selected = [r for r in rows if r["mode"] == mode]
    p = [float(r["prefill_tok_s"]) for r in selected]
    d = [float(r["gen_tok_s"]) for r in selected]
    print(f"{mode}: prefill_median={statistics.median(p):.3f} decode_median={statistics.median(d):.3f} raw_prefill={p}")
base = statistics.median(samples("baseline", "prefill_tok_s"))
fused = statistics.median(samples("fused", "prefill_tok_s"))
token_sets = {mode: {r["token_ids"] for r in rows if r["mode"] == mode} for mode in ("baseline", "fused")}
print(f"trim_each_side={trim} baseline_trimmed_median={base:.3f} fused_trimmed_median={fused:.3f}")
print(f"fused_vs_baseline={fused / base:.4f}x ({(fused / base - 1) * 100:+.2f}%)")
print(f"token_ids_match={token_sets['baseline'] == token_sets['fused']} baseline_variants={len(token_sets['baseline'])} fused_variants={len(token_sets['fused'])}")
PY
