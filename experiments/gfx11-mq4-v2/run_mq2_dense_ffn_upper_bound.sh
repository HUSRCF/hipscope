#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
RUNS="${RUNS:-2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/gfx11-mq4-v2/results/mq2-dense-ffn-upper-bound-gpu${GPU_ID}-${STAMP}}"
BIN="${ROOT}/target/release/examples/bench_mq2g256_lloyd_moe_4w"
BENCH_SOURCE="${ROOT}/crates/rdna-compute/examples/bench_mq2g256_lloyd_moe_4w.rs"

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

cargo build --release -p rdna-compute \
  --example bench_mq2g256_lloyd_moe_4w --features deltanet \
  2>&1 | tee "${OUT_DIR}/build.log"

{
  printf 'gpu_id=%s\n' "${GPU_ID}"
  printf 'runs=%s\n' "${RUNS}"
  printf 'dense_ffn_probe=1\n'
  printf 'head_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'benchmark_sha256=%s\n' "$(sha256sum "${BENCH_SOURCE}" | awk '{print $1}')"
} > "${OUT_DIR}/manifest.txt"

for run in $(seq 1 "${RUNS}"); do
  env HIP_VISIBLE_DEVICES="${GPU_ID}" \
      HIPFIRE_MQ2_DENSE_FFN_PROBE=1 \
      "${BIN}" 2>&1 | tee "${OUT_DIR}/run_${run}.log"
done

printf 'results=%s\n' "${OUT_DIR}"
