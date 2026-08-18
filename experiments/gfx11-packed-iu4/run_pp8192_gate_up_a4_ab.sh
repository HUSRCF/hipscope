#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-packed-iu4/results/pp8192_gate_up_a4_ab_gpu1}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-1}"
SLEEP_SECS="${SLEEP_SECS:-5}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

for path in "${MODEL}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
printf 'pair\torder\tmode\tprefill_tok_s\tgen_tok_s\ttoken_ids\n' > "${OUT_DIR}/results.tsv"
sha256sum "${BIN}" "${CK_LIB}" > "${OUT_DIR}/artifacts.sha256"
{
    printf 'gpu_id=%s\n' "${GPU_ID}"
    printf 'trials=%s\n' "${TRIALS}"
    printf 'trim_each_side=%s\n' "${TRIM_EACH_SIDE}"
    printf 'prefill_runs=%s\n' "${PREFILL_RUNS}"
    printf 'sleep_secs=%s\n' "${SLEEP_SECS}"
    printf 'model=%s\n' "${MODEL}"
    printf 'ck_lib=%s\n' "${CK_LIB}"
} > "${OUT_DIR}/meta.txt"

run_command() {
    local log="$1" a4="$2"
    timeout --signal=INT --kill-after=5s 240s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
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
        HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_K128=0 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_IU4_A4="${a4}" \
        HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BIN}" "${MODEL}" \
        --prefill 8192 --prefill-runs "${PREFILL_RUNS}" --warmup 2 --gen 32 \
        > "${log}" 2>&1

    if ! rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}" ||
       ! rg -q '^quantized FlashAttention CK prefill active:' "${log}"; then
        echo "quantized CK sidecar was not active; refusing invalid A/B: ${log}" >&2
        return 1
    fi
    if [[ "${a4}" == "1" ]]; then
        if ! rg -q '^RDNA3 IU4-A4 gate/up prefill active:' "${log}"; then
            echo "IU4-A4 gate/up route was not active; refusing invalid A/B: ${log}" >&2
            return 1
        fi
    elif rg -q '^RDNA3 IU4-A4 gate/up prefill active:' "${log}"; then
        echo "IU4-A4 route unexpectedly active in Q8 control: ${log}" >&2
        return 1
    fi
}

run_one() {
    local pair="$1" order="$2" mode="$3" a4 log summary prefill gen token_ids
    [[ "${mode}" == "a4" ]] && a4=1 || a4=0
    log="${OUT_DIR}/pair_${pair}_${order}_${mode}.log"
    run_command "${log}" "${a4}"
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^prefill_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    gen="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^gen_tok_s=/) {split($i,a,"="); print a[2]}}' <<< "${summary}")"
    token_ids="$(rg '^TOKEN_IDS ' "${log}" | tail -n 1 | sed 's/^TOKEN_IDS //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pair}" "${order}" "${mode}" "${prefill}" "${gen}" "${token_ids}" \
        | tee -a "${OUT_DIR}/results.tsv"
}

echo "Prewarming Q8 and IU4-A4 gate/up paths on HIP device ${GPU_ID}"
run_command "${OUT_DIR}/prewarm_q8.log" 0
sleep "${SLEEP_SECS}"
run_command "${OUT_DIR}/prewarm_a4.log" 1
sleep "${SLEEP_SECS}"

for ((pair=1; pair<=TRIALS; pair++)); do
    if (( pair % 2 == 1 )); then
        modes=(q8 a4)
    else
        modes=(a4 q8)
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
    values = [float(row[field]) for row in rows if row["mode"] == mode]
    if trim_values and trim:
        values = sorted(values)
        if len(values) <= 2 * trim:
            raise SystemExit(
                f"cannot trim {trim} samples from each side of {len(values)} values"
            )
        values = values[trim:-trim]
    return values

for mode in ("q8", "a4"):
    prefill = samples(mode, "prefill_tok_s")
    decode = samples(mode, "gen_tok_s")
    print(
        f"{mode}: prefill_median={statistics.median(prefill):.3f} "
        f"decode_median={statistics.median(decode):.3f} raw_prefill={prefill}"
    )

q8 = statistics.median(samples("q8", "prefill_tok_s", trim_values=True))
a4 = statistics.median(samples("a4", "prefill_tok_s", trim_values=True))
token_sets = {
    mode: {row["token_ids"] for row in rows if row["mode"] == mode}
    for mode in ("q8", "a4")
}
by_pair = {}
for row in rows:
    by_pair.setdefault(row["pair"], {})[row["mode"]] = row
pair_ratios = []
pair_token_matches = []
for pair in sorted(by_pair, key=int):
    arms = by_pair[pair]
    if set(arms) != {"q8", "a4"}:
        raise SystemExit(f"pair {pair} is incomplete: {sorted(arms)}")
    ratio = float(arms["a4"]["prefill_tok_s"]) / float(arms["q8"]["prefill_tok_s"])
    pair_ratios.append(ratio)
    pair_token_matches.append(arms["a4"]["token_ids"] == arms["q8"]["token_ids"])
    print(
        f"pair={pair} a4_order={arms['a4']['order']} ratio={ratio:.4f}x "
        f"tokens_match={pair_token_matches[-1]}"
    )
print(f"trim_each_side={trim} q8_trimmed_median={q8:.3f} a4_trimmed_median={a4:.3f}")
print(f"a4_vs_q8={a4 / q8:.4f}x ({(a4 / q8 - 1) * 100:+.2f}%)")
print(
    f"pairwise_ratio_median={statistics.median(pair_ratios):.4f}x "
    f"raw_pairwise_ratios={[round(value, 4) for value in pair_ratios]}"
)
print(
    f"token_ids_match={token_sets['q8'] == token_sets['a4']} "
    f"pairwise_tokens_match={all(pair_token_matches)} "
    f"q8_variants={len(token_sets['q8'])} a4_variants={len(token_sets['a4'])}"
)
PY
