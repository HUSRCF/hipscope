#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-5}"
COOL_SECS="${COOL_SECS:-10}"
DECODE_TOKENS="${DECODE_TOKENS:-4096}"
RUN_PREFILL="${RUN_PREFILL:-1}"
RUN_CAPACITY="${RUN_CAPACITY:-1}"
RUN_DECODE="${RUN_DECODE:-1}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.8-27b.mq4}"
BIN="${BIN:-${ROOT}/target/release/examples/bench_qwen35_mq4}"
SIDECAR="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/qwen38-fullopt-ar/results/$(date +%Y%m%d_%H%M%S)}"

read -r -a PREFILL_LENGTHS <<<"${PREFILL_SET:-64 256 1024 2048 4096 8192}"
read -r -a CAPACITIES <<<"${CAPACITY_SET:-65536 131072 196608}"
read -r -a DECODE_CONTEXTS <<<"${DECODE_CONTEXT_SET:-64 65536 131072 196608}"

for path in "${MODEL}" "${BIN}" "${SIDECAR}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done
[[ "${TRIALS}" =~ ^[1-9][0-9]*$ ]] || { echo "TRIALS must be positive" >&2; exit 1; }

mkdir -p "${OUT_DIR}/logs"
if [[ "${RUN_PREFILL}" == "1" ]]; then
    printf 'workload\tprefill_tokens\tkv_seq\tsample\tprefill_ms\tprefill_tok_s\n' >"${OUT_DIR}/prefill.tsv"
elif [[ ! -f "${OUT_DIR}/prefill.tsv" ]]; then
    echo "RUN_PREFILL=0 requires an existing ${OUT_DIR}/prefill.tsv" >&2
    exit 1
fi
if [[ "${RUN_DECODE}" == "1" ]]; then
    printf 'context_tokens\tgen_tokens\tsample\ttotal_ms\tgen_tok_s\tavg_ms\tp50_ms\n' >"${OUT_DIR}/decode.tsv"
elif [[ ! -f "${OUT_DIR}/decode.tsv" ]]; then
    echo "RUN_DECODE=0 requires an existing ${OUT_DIR}/decode.tsv" >&2
    exit 1
fi
sha256sum "${MODEL}" "${BIN}" "${SIDECAR}" >"${OUT_DIR}/artifacts.sha256"

common_env=(
    HIP_VISIBLE_DEVICES="${GPU_ID}"
    HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${SIDECAR}"
    HIPFIRE_KV_MODE=asym3
    HIPFIRE_GRAPH=0
    HIPFIRE_GRAPH_PREFILL=0
    HIPFIRE_PREFILL_MAX_BATCH=2048
    HIPFIRE_FLASH_PARTIALS_BATCH=32
    HIPFIRE_DPM_WARMUP_SECS=5
    HIPFIRE_SPECULATION=off
    HIPFIRE_DFLASH_MODE=off
    HIPFIRE_MTP_MODE=off
    HIPFIRE_QKVZA_SPLIT_TAIL=1
    HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1
    HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1
    HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1
    HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1
    HIPFIRE_RDNA3_Q8_GROUP128=1
    HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1
    HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1
    HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1
    HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1
    HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1
    HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0
)

run_bench() {
    local timeout_s="$1" log="$2"
    shift 2
    timeout --signal=INT --kill-after=10s "${timeout_s}s" \
        env "${common_env[@]}" "${BIN}" "${MODEL}" "$@" >"${log}" 2>&1
}

collect_prefill() {
    local workload="$1" prefill="$2" kv_seq="$3" log="$4"
    python3 - "${workload}" "${prefill}" "${kv_seq}" "${TRIALS}" "${log}" \
        >>"${OUT_DIR}/prefill.tsv" <<'PY'
import re
import sys

workload, prefill, kv_seq, trials, path = sys.argv[1:]
trials = int(trials)
text = open(path, encoding="utf-8", errors="replace").read()
rows = re.findall(r"run\s+(\d+):\s+([0-9.]+)ms\s+([0-9.]+) tok/s", text)
if len(rows) < trials + 1:
    raise SystemExit(f"{path}: expected {trials + 1} prefill runs, found {len(rows)}")
for sample, (_, ms, tok_s) in enumerate(rows[-trials:], 1):
    print(workload, prefill, kv_seq, sample, ms, tok_s, sep="\t")
PY
}

if [[ "${RUN_PREFILL}" == "1" ]]; then
    echo "[1/3] steady-state prefill matrix"
    for prefill in "${PREFILL_LENGTHS[@]}"; do
        log="${OUT_DIR}/logs/prefill_pp${prefill}.log"
        run_bench 600 "${log}" \
            --prefill "${prefill}" --prefill-runs "$((TRIALS + 1))" --warmup 0 --gen 0
        collect_prefill prefill "${prefill}" auto "${log}"
        sleep "${COOL_SECS}"
    done
fi

if [[ "${RUN_CAPACITY}" == "1" ]]; then
    echo "[2/3] PP2048 at fixed KV capacities"
    # Retain ordinary prefill rows while replacing capacity rows on resume.
    awk -F '\t' 'NR == 1 || $1 != "capacity"' "${OUT_DIR}/prefill.tsv" \
        >"${OUT_DIR}/prefill.tsv.tmp"
    mv "${OUT_DIR}/prefill.tsv.tmp" "${OUT_DIR}/prefill.tsv"
    for kv_seq in "${CAPACITIES[@]}"; do
        log="${OUT_DIR}/logs/capacity_pp2048_kv${kv_seq}.log"
        run_bench 600 "${log}" \
            --prefill 2048 --prefill-runs "$((TRIALS + 1))" --warmup 0 --gen 0 --kv-seq "${kv_seq}"
        collect_prefill capacity 2048 "${kv_seq}" "${log}"
        sleep "${COOL_SECS}"
    done
fi

if [[ "${RUN_DECODE}" == "1" ]]; then
    echo "[3/3] long AR decode matrix"
    for context in "${DECODE_CONTEXTS[@]}"; do
        for ((sample=1; sample<=TRIALS; sample++)); do
            log="${OUT_DIR}/logs/decode_ctx${context}_trial${sample}.log"
            timeout_s=$((900 + context / 256 + DECODE_TOKENS))
            run_bench "${timeout_s}" "${log}" \
                --prefill "${context}" --prefill-runs 1 --warmup 8 --gen "${DECODE_TOKENS}"
            python3 - "${context}" "${DECODE_TOKENS}" "${sample}" "${log}" \
                >>"${OUT_DIR}/decode.tsv" <<'PY'
import re
import sys

context, gen_tokens, sample, path = sys.argv[1:]
text = open(path, encoding="utf-8", errors="replace").read()
summary = [line for line in text.splitlines() if line.startswith("SUMMARY  ")]
total = re.findall(r"total:\s+([0-9.]+)ms over ([0-9]+) tokens", text)
if not summary or not total:
    raise SystemExit(f"{path}: missing decode summary")
fields = dict(item.split("=", 1) for item in summary[-1].split()[1:] if "=" in item)
total_ms, emitted = total[-1]
if int(emitted) != int(gen_tokens):
    raise SystemExit(f"{path}: expected {gen_tokens} generated tokens, got {emitted}")
print(context, gen_tokens, sample, total_ms, fields["gen_tok_s"], fields["avg_ms"], fields["p50_ms"], sep="\t")
PY
            sleep "${COOL_SECS}"
        done
    done
fi

python3 - "${OUT_DIR}" <<'PY' | tee "${OUT_DIR}/summary.txt"
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])

def read_tsv(name):
    with (out / name).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

prefill = read_tsv("prefill.tsv")
decode = read_tsv("decode.tsv")
groups = defaultdict(list)
for row in prefill:
    groups[(row["workload"], row["prefill_tokens"], row["kv_seq"])].append(float(row["prefill_tok_s"]))
print("PREFILL_MEDIANS")
for key, values in groups.items():
    print(f"{key[0]} pp={key[1]} kv_seq={key[2]} median_tok_s={statistics.median(values):.2f} raw={values}")
groups.clear()
for row in decode:
    groups[row["context_tokens"]].append(float(row["gen_tok_s"]))
print("DECODE_MEDIANS")
for context, values in groups.items():
    print(f"context={context} median_tok_s={statistics.median(values):.2f} raw={values}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
