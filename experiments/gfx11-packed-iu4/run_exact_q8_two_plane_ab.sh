#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT}/target/release/examples/bench_hfq4_iu4_a4"

cargo build --release -p rdna-compute --example bench_hfq4_iu4_a4 \
  --features deltanet,flash-attn-ck

"${BIN}" --exact-q8 \
  --m "${M:-17408}" \
  --k "${K:-5120}" \
  --n "${N:-2048}" \
  --pairs "${PAIRS:-11}"
