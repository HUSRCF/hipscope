#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-${HOME}/.hipfire/models/qwen3.6-27b.mq4}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/README.md}"
PREFILL="${PREFILL:-256}"
CAPTURE_LAYER="${CAPTURE_LAYER:-0}"
CAPTURE_TOKENS="${CAPTURE_TOKENS:-${PREFILL}}"
OUT_DIR="${OUT_DIR:-/tmp/hipfire-ffn-v2-capture-layer${CAPTURE_LAYER}}"

if [[ ! -f "${MODEL}" ]]; then
  printf 'model not found: %s\n' "${MODEL}" >&2
  exit 2
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  printf 'prompt file not found: %s\n' "${PROMPT_FILE}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  printf 'refusing to overwrite capture output: %s\n' "${OUT_DIR}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
cargo build --release --locked --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime

env \
  HIP_VISIBLE_DEVICES="${GPU_ID}" \
  HIPFIRE_KV_MODE=q8 \
  HIPFIRE_GRAPH_PREFILL=0 \
  HIPFIRE_PROFILE=0 \
  HIPFIRE_RDNA3_FFN_CAPTURE_DIR="${OUT_DIR}" \
  HIPFIRE_RDNA3_FFN_CAPTURE_LAYER="${CAPTURE_LAYER}" \
  HIPFIRE_RDNA3_FFN_CAPTURE_MAX_TOKENS="${CAPTURE_TOKENS}" \
  "${ROOT_DIR}/target/release/examples/bench_qwen35_mq4" \
    "${MODEL}" \
    --prefill "${PREFILL}" \
    --prefill-runs 1 \
    --warmup 0 \
    --gen 1 \
    --prompt-file "${PROMPT_FILE}"

printf 'capture complete: %s\n' "${OUT_DIR}"
