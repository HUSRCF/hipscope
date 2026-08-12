#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:-/home/husrcf/Code/ProtBind/hipfire/.redline-work/hipfire-flash-prefill-gfx12-beta-20260730/.codeinsight+research/gfx11-fixed-hd-ab/longbench-hard20-pp32k.jsonl}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
BIN="${BIN:-${ROOT}/target/release/examples/greedy_dump}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-/tmp/libhipfire_flash_attn_ck_quantized_staged.so}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/home/husrcf/Code/ProtBind/MTP/data/modelscope_downloads/Qwen/Qwen3.6-27B-FP8/tokenizer.json}"
PYTHON="${PYTHON:-/home/husrcf/anaconda3/envs/UNI/bin/python}"
GPU_ID="${GPU_ID:-0}"
ORDINALS="${ORDINALS:-0 2 4 6 9}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CTX="${CTX:-65536}"
SLEEP_SECS="${SLEEP_SECS:-5}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-packed-iu4/results/longbench_module_quality_$(date +%Y%m%d_%H%M%S)}"

for path in "${DATASET}" "${MODEL}" "${BIN}" "${CK_LIB}" "${TOKENIZER_JSON}" "${PYTHON}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
sha256sum "${DATASET}" "${BIN}" "${CK_LIB}" "${TOKENIZER_JSON}" > "${OUT_DIR}/artifacts.sha256"
printf 'ordinal\treference\tmode\tprediction\tcorrect\tfirst_diff\toutput_tokens\n' > "${OUT_DIR}/results.tsv"

mode_flags() {
    case "$1" in
        q8)   printf '0 0 0\n' ;;
        gate) printf '0 1 0\n' ;;
        up)   printf '0 0 1\n' ;;
        *) return 1 ;;
    esac
}

run_one() {
    local ordinal="$1" mode="$2" both gate up prompt prompt_file log tokens
    read -r both gate up <<< "$(mode_flags "${mode}")"
    prompt="$(jq -r "select(.ordinal == ${ordinal}) | .prompt_no_think" "${DATASET}")"
    [[ -n "${prompt}" && "${prompt}" != null ]] || {
        echo "missing ordinal ${ordinal}" >&2
        return 1
    }
    prompt_file="${OUT_DIR}/q${ordinal}.prompt.txt"
    printf '%s' "${prompt}" > "${prompt_file}"
    log="${OUT_DIR}/q${ordinal}_${mode}.log"
    tokens="${OUT_DIR}/q${ordinal}_${mode}.tokens"
    timeout --signal=INT --kill-after=5s 900s env \
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
        HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
        HIPFIRE_RDNA3_Q8_GROUP128_K128=0 \
        HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1 \
        HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1 \
        HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0 \
        HIPFIRE_RDNA3_HFQ4_GATE_UP_IU4_A4="${both}" \
        HIPFIRE_RDNA3_HFQ4_GATE_IU4_A4="${gate}" \
        HIPFIRE_RDNA3_HFQ4_UP_IU4_A4="${up}" \
        GREEDY_DUMP_CTX="${CTX}" MAX_TOKENS="${MAX_TOKENS}" PROMPT_MODE=nothinking \
        GREEDY_DUMP_PROMPT_FILE="${prompt_file}" \
        "${BIN}" "${MODEL}" "${tokens}" > "${log}" 2>&1

    rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}"
    rg -q '^staged quantized FlashAttention CK prefill active:' "${log}"
    case "${mode}" in
        q8) ! rg -q '^RDNA3 IU4-A4 .*prefill active:' "${log}" ;;
        gate) rg -q '^RDNA3 IU4-A4 projection prefill active: gate=true up=false ' "${log}" ;;
        up) rg -q '^RDNA3 IU4-A4 projection prefill active: gate=false up=true ' "${log}" ;;
    esac
}

for ordinal in ${ORDINALS}; do
    for mode in q8 gate up; do
        echo "quality ordinal=${ordinal} mode=${mode}"
        run_one "${ordinal}" "${mode}"
        sleep "${SLEEP_SECS}"
    done
done

"${PYTHON}" - "${OUT_DIR}" "${DATASET}" "${TOKENIZER_JSON}" ${ORDINALS} <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import json
import pathlib
import re
import sys

from tokenizers import Tokenizer

root = pathlib.Path(sys.argv[1])
dataset = pathlib.Path(sys.argv[2])
tokenizer = Tokenizer.from_file(sys.argv[3])
ordinals = [int(value) for value in sys.argv[4:]]
rows = {row["ordinal"]: row for row in map(json.loads, dataset.read_text().splitlines())}
modes = ("q8", "gate", "up")
scored = []

with (root / "results.tsv").open("a", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    for ordinal in ordinals:
        reference = rows[ordinal]["answer"].strip().upper()
        token_sets = {
            mode: [int(value) for value in (root / f"q{ordinal}_{mode}.tokens").read_text().split()]
            for mode in modes
        }
        for mode in modes:
            tokens = token_sets[mode]
            text = tokenizer.decode(tokens)
            match = re.search(r"correct\s+answer\s+is\s*\(?\s*([A-D])", text, re.I)
            prediction = match.group(1).upper() if match else ""
            first_diff = next(
                (index for index, pair in enumerate(zip(token_sets["q8"], tokens)) if pair[0] != pair[1]),
                None,
            )
            if first_diff is None and len(token_sets["q8"]) != len(tokens):
                first_diff = min(len(token_sets["q8"]), len(tokens))
            correct = prediction == reference
            scored.append((mode, correct))
            writer.writerow((ordinal, reference, mode, prediction, int(correct), first_diff, len(tokens)))
            print(
                f"ordinal={ordinal} reference={reference} mode={mode} prediction={prediction or 'NONE'} "
                f"correct={correct} first_diff={first_diff} output_tokens={len(tokens)} text={text!r}"
            )

for mode in modes:
    values = [correct for candidate, correct in scored if candidate == mode]
    print(f"accuracy mode={mode} correct={sum(values)}/{len(values)} accuracy={sum(values) / len(values):.3f}")
PY

echo "results=${OUT_DIR}"
