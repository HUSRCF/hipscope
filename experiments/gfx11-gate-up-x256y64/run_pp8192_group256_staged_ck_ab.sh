#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PAIRS="${PAIRS:-5}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-1}"
COOL_SECS="${COOL_SECS:-10}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/pp8192_group256_staged_ck_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"

for path in "${MODEL}" "${SIDECAR}" "${BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"
sha256sum "${MODEL}" "${SIDECAR}" "${BIN}" >"${OUT_DIR}/artifacts.sha256"

run_one() {
    local pair="$1" order="$2" mode="$3" group256=0
    local log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    [[ "${mode}" == "group256" ]] && group256=1

    timeout --signal=INT --kill-after=5s 240s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${SIDECAR}" \
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
        HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW="${group256}" \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs 1 --warmup 0 --gen 8 \
        >"${log}" 2>&1

    rg -q '^staged quantized FlashAttention CK prefill active:' "${log}" || {
        echo "staged CK route was not active: ${log}" >&2
        return 1
    }
    local summary prefill token_ids
    summary="$(rg '^PREFILL_SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<<"${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\n' "${pair}" "${order}" "${mode}" "${prefill}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

echo "Prewarming both routes on HIP device ${GPU_ID}"
run_one 0 0 group128
sleep "${COOL_SECS}"
run_one 0 1 group256
sleep "${COOL_SECS}"
printf 'pair\torder\tmode\tprefill_tok_s\ttoken_ids\n' >"${OUT_DIR}/results.tsv"

for ((pair=1; pair<=PAIRS; pair++)); do
    if ((pair % 2 == 1)); then
        modes=(group128 group256)
    else
        modes=(group256 group128)
    fi
    for order in 0 1; do
        run_one "${pair}" "${order}" "${modes[$order]}"
        sleep "${COOL_SECS}"
    done
done

python3 - "${OUT_DIR}/results.tsv" "${TRIM_EACH_SIDE}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trim = int(sys.argv[2])

def values(mode):
    return [float(row["prefill_tok_s"]) for row in rows if row["mode"] == mode]

def trimmed(samples):
    samples = sorted(samples)
    if trim == 0:
        return samples
    if len(samples) <= 2 * trim:
        raise SystemExit(f"cannot trim {trim} samples per side from {len(samples)}")
    return samples[trim:-trim]

base = values("group128")
candidate = values("group256")
base_median = statistics.median(trimmed(base))
candidate_median = statistics.median(trimmed(candidate))
paired = []
for pair in sorted({row["pair"] for row in rows}, key=int):
    by_mode = {row["mode"]: float(row["prefill_tok_s"]) for row in rows if row["pair"] == pair}
    paired.append(by_mode["group256"] / by_mode["group128"])
token_sets = {
    mode: {row["token_ids"] for row in rows if row["mode"] == mode}
    for mode in ("group128", "group256")
}

print(f"group128_raw={base}")
print(f"group256_raw={candidate}")
print(f"trim_each_side={trim}")
print(f"group128_trimmed_median={base_median:.3f}")
print(f"group256_trimmed_median={candidate_median:.3f}")
print(f"group256_vs_group128={candidate_median / base_median:.4f}x "
      f"({(candidate_median / base_median - 1) * 100:+.2f}%)")
print(f"paired_ratios={paired}")
print(f"paired_ratio_median={statistics.median(paired):.4f}x")
print(f"token_ids_match={token_sets['group128'] == token_sets['group256']} "
      f"group128_variants={len(token_sets['group128'])} "
      f"group256_variants={len(token_sets['group256'])}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
