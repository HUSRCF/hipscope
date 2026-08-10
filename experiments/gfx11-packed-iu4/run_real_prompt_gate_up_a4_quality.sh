#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-packed-iu4/results/real_prompt_gate_up_a4_quality_gpu1}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
PROMPT_FILE="${PROMPT_FILE:-/home/husrcf/Code/ProtBind/unidec/docs/testINPUT.md}"
GPU_ID="${GPU_ID:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CTX="${CTX:-4096}"
SLEEP_SECS="${SLEEP_SECS:-5}"
BIN="${BIN:-${ROOT}/target/release/examples/greedy_dump}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

for path in "${MODEL}" "${PROMPT_FILE}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
sha256sum "${BIN}" "${CK_LIB}" "${PROMPT_FILE}" > "${OUT_DIR}/artifacts.sha256"

run_one() {
    local mode="$1" a4=0 log tokens prompt
    [[ "${mode}" == "a4" ]] && a4=1
    log="${OUT_DIR}/${mode}.log"
    tokens="${OUT_DIR}/${mode}.tokens"
    prompt="$(<"${PROMPT_FILE}")"
    timeout --signal=INT --kill-after=5s 300s \
        env \
        HIP_VISIBLE_DEVICES="${GPU_ID}" \
        HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${CK_LIB}" \
        HIPFIRE_KV_MODE=asym3 \
        GREEDY_DUMP_CTX="${CTX}" \
        MAX_TOKENS="${MAX_TOKENS}" \
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
        PROMPT_MODE=thinking \
        "${BIN}" "${MODEL}" "${tokens}" "${prompt}" \
        > "${log}" 2>&1

    rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}" || {
        echo "quantized CK sidecar was not loaded: ${log}" >&2
        return 1
    }
    if [[ "${a4}" == "1" ]]; then
        rg -q '^RDNA3 IU4-A4 gate/up prefill active:' "${log}" || {
            echo "IU4-A4 route was not active: ${log}" >&2
            return 1
        }
    elif rg -q '^RDNA3 IU4-A4 gate/up prefill active:' "${log}"; then
        echo "IU4-A4 route unexpectedly active in Q8 control: ${log}" >&2
        return 1
    fi
}

run_one q8
sleep "${SLEEP_SECS}"
run_one a4

python3 - "${OUT_DIR}/q8.log" "${OUT_DIR}/a4.log" "${OUT_DIR}/q8.tokens" "${OUT_DIR}/a4.tokens" <<'PY' | tee "${OUT_DIR}/summary.txt"
import re
import sys

def parse(log_path, token_path):
    text = open(log_path, encoding="utf-8", errors="replace").read()
    tokens = [int(line) for line in open(token_path) if line.strip()]
    prompt = re.search(r"prompt:\s*(\d+) tokens", text)
    return tokens, int(prompt.group(1)) if prompt else None

q8_tokens, q8_prompt = parse(sys.argv[1], sys.argv[3])
a4_tokens, a4_prompt = parse(sys.argv[2], sys.argv[4])
first_diff = next(
    (i for i, (left, right) in enumerate(zip(q8_tokens, a4_tokens)) if left != right),
    None,
)
if first_diff is None and len(q8_tokens) != len(a4_tokens):
    first_diff = min(len(q8_tokens), len(a4_tokens))
print(f"q8_prompt_tokens={q8_prompt} a4_prompt_tokens={a4_prompt}")
print(f"q8_output_tokens={len(q8_tokens)} a4_output_tokens={len(a4_tokens)}")
print(f"tokens_match={q8_tokens == a4_tokens} first_diff={first_diff}")
print(f"q8_tokens={q8_tokens}")
print(f"a4_tokens={a4_tokens}")
PY
