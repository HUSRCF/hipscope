#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_GIT_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
MAIN_ROOT="$(dirname "${COMMON_GIT_DIR}")"
DEFAULT_PROMPT="${ROOT}/docs/testINPUT.md"
for candidate in "${MAIN_ROOT}/docs/testINPUT.md" "$(dirname "${MAIN_ROOT}")/unidec/docs/testINPUT.md"; do
    if [[ ! -f "${DEFAULT_PROMPT}" && -f "${candidate}" ]]; then
        DEFAULT_PROMPT="${candidate}"
    fi
done
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-packed-iu4/results/gate_up_module_sensitivity_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
PROMPT_FILE="${PROMPT_FILE:-${DEFAULT_PROMPT}}"
GPU_ID="${GPU_ID:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CTX="${CTX:-4096}"
PREFILL="${PREFILL:-8192}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
SLEEP_SECS="${SLEEP_SECS:-5}"
GREEDY_BIN="${GREEDY_BIN:-${ROOT}/target/release/examples/greedy_dump}"
BENCH_BIN="${BENCH_BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"
REQUIRE_CK="${REQUIRE_CK:-0}"

for path in "${MODEL}" "${PROMPT_FILE}" "${GREEDY_BIN}" "${BENCH_BIN}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done
if [[ "${REQUIRE_CK}" == 1 && ! -f "${CK_LIB}" ]]; then
    echo "missing required CK sidecar: ${CK_LIB}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
sha256sum "${GREEDY_BIN}" "${BENCH_BIN}" "${PROMPT_FILE}" > "${OUT_DIR}/artifacts.sha256"
if [[ -f "${CK_LIB}" ]]; then
    sha256sum "${CK_LIB}" >> "${OUT_DIR}/artifacts.sha256"
fi
printf 'mode\tprefill_tok_s\tgen_tok_s\n' > "${OUT_DIR}/performance.tsv"

mode_flags() {
    case "$1" in
        q8)   printf '0 0 0\n' ;;
        gate) printf '0 1 0\n' ;;
        up)   printf '0 0 1\n' ;;
        both) printf '1 0 0\n' ;;
        *) return 1 ;;
    esac
}

common_env() {
    local mode="$1" both gate up
    read -r both gate up <<< "$(mode_flags "${mode}")"
    if [[ -f "${CK_LIB}" ]]; then
        printf '%s\n' "HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=${CK_LIB}"
    fi
    printf '%s\n' \
        "HIP_VISIBLE_DEVICES=${GPU_ID}" \
        "HIPFIRE_KV_MODE=asym3" \
        "HIPFIRE_GRAPH=0" \
        "HIPFIRE_PREFILL_MAX_BATCH=2048" \
        "HIPFIRE_FLASH_PARTIALS_BATCH=32" \
        "HIPFIRE_DPM_WARMUP_SECS=5" \
        "HIPFIRE_QKVZA_SPLIT_TAIL=1" \
        "HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1" \
        "HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1" \
        "HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1" \
        "HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1" \
        "HIPFIRE_RDNA3_Q8_GROUP128=1" \
        "HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1" \
        "HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1" \
        "HIPFIRE_RDNA3_Q8_GROUP128_K128=0" \
        "HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1" \
        "HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1" \
        "HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1" \
        "HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0" \
        "HIPFIRE_RDNA3_HFQ4_GATE_UP_IU4_A4=${both}" \
        "HIPFIRE_RDNA3_HFQ4_GATE_IU4_A4=${gate}" \
        "HIPFIRE_RDNA3_HFQ4_UP_IU4_A4=${up}"
}

check_route() {
    local mode="$1" log="$2"
    case "${mode}" in
        q8)
            ! rg -q '^RDNA3 IU4-A4 .*prefill active:' "${log}" || return 1
            ;;
        gate)
            rg -q '^RDNA3 IU4-A4 projection prefill active: gate=true up=false ' "${log}"
            ;;
        up)
            rg -q '^RDNA3 IU4-A4 projection prefill active: gate=false up=true ' "${log}"
            ;;
        both)
            rg -q '^RDNA3 IU4-A4 gate/up prefill active:' "${log}"
            ;;
    esac
}

check_ck() {
    local log="$1"
    if [[ "${REQUIRE_CK}" == 1 ]]; then
        rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}"
        rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"
    fi
}

run_quality() {
    local mode="$1" log="${OUT_DIR}/${mode}_quality.log" tokens="${OUT_DIR}/${mode}.tokens" prompt
    prompt="$(<"${PROMPT_FILE}")"
    mapfile -t vars < <(common_env "${mode}")
    timeout --signal=INT --kill-after=5s 600s env "${vars[@]}" \
        GREEDY_DUMP_CTX="${CTX}" MAX_TOKENS="${MAX_TOKENS}" PROMPT_MODE=thinking \
        "${GREEDY_BIN}" "${MODEL}" "${tokens}" "${prompt}" > "${log}" 2>&1
    check_ck "${log}"
    check_route "${mode}" "${log}"
}

run_performance() {
    local mode="$1" log="${OUT_DIR}/${mode}_performance.log" summary prefill gen
    mapfile -t vars < <(common_env "${mode}")
    timeout --signal=INT --kill-after=5s 600s env "${vars[@]}" HIPFIRE_BENCH_DUMP_TOKENS=1 \
        "${BENCH_BIN}" "${MODEL}" --prefill "${PREFILL}" --prefill-runs "${PREFILL_RUNS}" \
        --warmup 2 --gen 32 > "${log}" 2>&1
    check_ck "${log}"
    check_route "${mode}" "${log}"
    summary="$(rg '^SUMMARY ' "${log}" | tail -n 1)"
    prefill="$(awk '{for(i=1;i<=NF;i++)if($i~/^prefill_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    gen="$(awk '{for(i=1;i<=NF;i++)if($i~/^gen_tok_s=/){split($i,a,"=");print a[2]}}' <<<"${summary}")"
    printf '%s\t%s\t%s\n' "${mode}" "${prefill}" "${gen}" | tee -a "${OUT_DIR}/performance.tsv"
}

for mode in q8 gate up both; do
    echo "quality mode=${mode}"
    run_quality "${mode}"
    sleep "${SLEEP_SECS}"
done

for mode in q8 gate up both; do
    echo "performance mode=${mode}"
    run_performance "${mode}"
    sleep "${SLEEP_SECS}"
done

python3 - "${OUT_DIR}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
modes = ("q8", "gate", "up", "both")
tokens = {
    mode: [int(line) for line in (root / f"{mode}.tokens").read_text().splitlines() if line]
    for mode in modes
}
ref = tokens["q8"]
for mode in modes:
    candidate = tokens[mode]
    first_diff = next(
        (i for i, pair in enumerate(zip(ref, candidate)) if pair[0] != pair[1]),
        None,
    )
    if first_diff is None and len(ref) != len(candidate):
        first_diff = min(len(ref), len(candidate))
    print(
        f"quality mode={mode} output_tokens={len(candidate)} "
        f"tokens_match={candidate == ref} first_diff={first_diff}"
    )

rows = list(csv.DictReader((root / "performance.tsv").open(), delimiter="\t"))
base = float(next(row["prefill_tok_s"] for row in rows if row["mode"] == "q8"))
for row in rows:
    value = float(row["prefill_tok_s"])
    print(
        f"performance mode={row['mode']} prefill_tok_s={value:.3f} "
        f"vs_q8={value / base:.4f}x gen_tok_s={float(row['gen_tok_s']):.3f}"
    )
PY

echo "results=${OUT_DIR}"
