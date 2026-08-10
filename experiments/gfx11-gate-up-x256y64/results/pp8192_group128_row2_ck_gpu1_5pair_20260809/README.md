# gfx11 Group128 Row2/Col4 PP8192 A/B

This run compares the existing one-row-fragment/two-column-group topology with an opt-in two-row-fragment/four-column-group topology. Both modes use the same release binary and differ only through `HIPFIRE_RDNA3_Q8_GROUP128_ROW2=0/1`.

## Environment

- GPU: AMD Radeon Pro W7900 Dual Slot (`gfx1100`), selected with `HIP_VISIBLE_DEVICES=1`
- Model: Qwen3.6-27B MQ4
- KV: asym3
- Prefill: 8192 tokens, three passes per process; the last pass is reported
- A/B: five alternating pairs, five-second idle interval, trim one sample from each side
- Quantized FlashAttention CK sidecar enabled

## Result

| Topology | Prefill median | Decode median |
| --- | ---: | ---: |
| row1/col2 | 1042.2 tok/s | 33.0 tok/s |
| row2/col4 | 1056.7 tok/s | 33.0 tok/s |

The trimmed median improvement is **1.0139x (+1.39%)**. Every pair favored row2/col4. Generated token IDs were identical across all runs.

## Resource Audit

Both `full_set` and `full_add` kernels retain wave32, 252 VGPRs, 31 SGPRs, zero VGPR/SGPR spills, and a zero-byte private segment. The output tile, workgroup dimensions, grid dimensions, and dynamic LDS allocation remain unchanged.

## Reproduction

```bash
cargo build --release -p hipfire-runtime \
  --example bench_qwen35_mq4 \
  --features flash-attn-ck

GPU_ID=1 \
TRIALS=5 \
TRIM_EACH_SIDE=1 \
PREFILL_RUNS=3 \
SLEEP_SECS=5 \
OUT_DIR=experiments/gfx11-gate-up-x256y64/results/pp8192_group128_row2_ck_gpu1_5pair_20260809 \
experiments/gfx11-gate-up-x256y64/run_pp8192_group128_row2_ab.sh
```

Raw measurements are in `results.tsv`; `summary.txt` contains the aggregate.
