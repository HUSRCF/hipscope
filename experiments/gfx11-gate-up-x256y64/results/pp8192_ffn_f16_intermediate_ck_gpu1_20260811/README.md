# PP8192 FFN F16 Intermediate A/B

This controlled model-level A/B isolates the existing
`HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE` path under the current gfx11 production
configuration. Both modes use the same Qwen3.6-27B MQ4 artifact, Asym3 KV,
2048-token prefill chunks, quantized CK attention sidecar, quad-row packed-MQ4,
and fused SwiGLU-to-Q8 group128 path.

## Result

| Mode | PP8192 median | Decode median | Raw PP8192 tok/s |
|---|---:|---:|---|
| F32 FFN intermediate | 1098.5 tok/s | 33.1 tok/s | 1137.8, 1098.5, 1100.1, 1098.5, 1097.2 |
| F16 FFN intermediate | 1115.4 tok/s | 33.1 tok/s | 1128.5, 1119.2, 1115.4, 1108.7, 1111.6 |

- Ratio of medians: **1.0154x (+1.54%)**
- Median paired ratio: **1.0131x**
- Positive pairs: **4/5**
- Token IDs: identical across every run and mode

The F16 intermediate path is a small, reproducible FFN dataflow improvement in
this configuration. It does not materially change decode speed and is not large
enough to alter the packed-MQ4 backend priority.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=1 \
  TRIALS=5 \
  SLEEP_SECS=2 \
  OUT_DIR="$PWD/experiments/gfx11-gate-up-x256y64/results/pp8192_ffn_f16_intermediate_ck_gpu1_20260811" \
  experiments/gfx11-gate-up-x256y64/run_pp8192_ffn_f16_intermediate_ck_ab.sh
```
