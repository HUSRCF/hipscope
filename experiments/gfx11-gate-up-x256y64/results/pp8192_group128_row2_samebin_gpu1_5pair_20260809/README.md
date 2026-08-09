# RDNA3 group128 row2 PP8192 A/B

This result isolates the `MMQ_ROW_FRAGS=2` / `MMQ_COL_GROUPS=4` topology from the optional CK attention sidecar.

## Configuration

- GPU: AMD Radeon Pro W7900 Dual Slot (`gfx1100`), selected with `HIP_VISIBLE_DEVICES=1`
- Model: Qwen3.6-27B MQ4
- KV cache: `asym3`
- Prompt: 8192 tokens
- Per-process prefill runs: 3
- Paired trials: 5, alternating row1/row2 order with 5 seconds idle between runs
- CK sidecar: disabled

## Result

| Mode | Prefill median | Decode median |
| --- | ---: | ---: |
| row1 (`ROW_FRAGS=1`, `COL_GROUPS=2`) | 695.8 tok/s | 33.1 tok/s |
| row2 (`ROW_FRAGS=2`, `COL_GROUPS=4`) | 702.2 tok/s | 33.0 tok/s |

The paired prefill improvement is `+0.92%`; all five row2 samples exceeded their row1 pair. Generated token IDs matched exactly. Decode is unchanged within the benchmark's 0.1 tok/s reporting precision.

The compiled row1 and row2 kernels both use wave32, 252 VGPRs, 31 SGPRs, zero scratch, and a zero-byte private segment.

## Reproduce

Build the benchmark without the optional CK feature, then run:

```bash
cargo build --release -p hipfire-runtime --example bench_qwen35_mq4

GPU_ID=1 USE_CK=0 TRIALS=5 TRIM_EACH_SIDE=1 \
  OUT_DIR=/tmp/hipfire-row2-ab \
  ./experiments/gfx11-gate-up-x256y64/run_pp8192_group128_row2_ab.sh
```

`USE_CK=1` is supported only when a compatible sidecar has been built separately and supplied through `HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB`.
