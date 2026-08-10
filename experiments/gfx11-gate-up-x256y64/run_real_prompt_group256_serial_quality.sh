#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-gate-up-x256y64/results/real_prompt_group256_serial_gpu1_${STAMP}}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/docs/testINPUT.md}"
GPU_ID="${GPU_ID:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CTX="${CTX:-8192}"
SLEEP_SECS="${SLEEP_SECS:-5}"
GROUP256_SCOPE="${GROUP256_SCOPE:-all}"
BIN="${BIN:-${ROOT}/target/release/examples/greedy_dump}"
CK_LIB="${HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so}"

for path in "${MODEL}" "${PROMPT_FILE}" "${BIN}" "${CK_LIB}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
sha256sum "${BIN}" "${CK_LIB}" "${PROMPT_FILE}" > "${OUT_DIR}/artifacts.sha256"

run_one() {
    local mode="$1" group256=0 group256_all=0 group256_gate_up=0 log tokens prompt
    if [[ "${mode}" == "group256" ]]; then
        group256=1
        case "${GROUP256_SCOPE}" in
            all) group256_all=1 ;;
            gate_up) group256_gate_up=1 ;;
            *) echo "invalid GROUP256_SCOPE=${GROUP256_SCOPE}; expected all or gate_up" >&2; return 1 ;;
        esac
    fi
    log="${OUT_DIR}/${mode}.log"
    tokens="${OUT_DIR}/${mode}.tokens"
    prompt="$(<"${PROMPT_FILE}")"
    timeout --signal=INT --kill-after=5s 360s \
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
        PROMPT_MODE=thinking \
        "${BIN}" "${MODEL}" "${tokens}" "${prompt}" \
        > "${log}" 2>&1

    rg -q '^loaded optional quantized FlashAttention CK sidecar:' "${log}" || {
        echo "quantized CK sidecar was not loaded: ${log}" >&2
        return 1
    }
    if [[ "${group256}" == "1" ]]; then
        rg -q '^RDNA3 Q8 group256 gate/up prefill active:' "${log}" || {
            echo "group256 serial-row route was not active: ${log}" >&2
            return 1
        }
    fi
}

run_one group128
sleep "${SLEEP_SECS}"
run_one group256

python3 - "${OUT_DIR}/group128.log" "${OUT_DIR}/group256.log" \
    "${OUT_DIR}/group128.tokens" "${OUT_DIR}/group256.tokens" <<'PY' | tee "${OUT_DIR}/summary.txt"
import re
import sys

def parse(log_path, token_path):
    text = open(log_path, encoding="utf-8", errors="replace").read()
    tokens = [int(line) for line in open(token_path) if line.strip()]
    prompt = re.search(r"prompt:\s*(\d+) tokens", text)
    return tokens, int(prompt.group(1)) if prompt else None

base, base_prompt = parse(sys.argv[1], sys.argv[3])
candidate, candidate_prompt = parse(sys.argv[2], sys.argv[4])
first_diff = next(
    (i for i, (left, right) in enumerate(zip(base, candidate)) if left != right),
    None,
)
if first_diff is None and len(base) != len(candidate):
    first_diff = min(len(base), len(candidate))
common_prefix = first_diff if first_diff is not None else min(len(base), len(candidate))
print(f"group128_prompt_tokens={base_prompt} group256_prompt_tokens={candidate_prompt}")
print(f"group128_output_tokens={len(base)} group256_output_tokens={len(candidate)}")
print(f"tokens_match={base == candidate} common_prefix={common_prefix} first_diff={first_diff}")
print(f"group128_tokens={base}")
print(f"group256_tokens={candidate}")
PY

printf 'out_dir=%s\n' "${OUT_DIR}"
