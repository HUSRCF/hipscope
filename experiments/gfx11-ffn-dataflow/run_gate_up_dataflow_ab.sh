#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU_ID:-1}"
PAIRS="${PAIRS:-10}"

cargo build --release -p rdna-compute --example bench_hfq4_gate_up_streams

env \
    HIP_VISIBLE_DEVICES="${GPU_ID}" \
    HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1 \
    HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1 \
    HIPFIRE_RDNA3_Q8_GROUP128=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1 \
    HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1 \
    "${ROOT}/target/release/examples/bench_hfq4_gate_up_streams" \
    --m 17408 \
    --k 5120 \
    --n 2048 \
    --pairs "${PAIRS}"
