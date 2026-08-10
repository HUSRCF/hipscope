# PP8192 group256 plus staged CK combination A/B

This run measures whether the previously positive group256 activation contract
adds to the retained staged quantized-CK attention and F16 FFN intermediate
configuration. The two modes differ only in the group128 versus group256
activation route used by the gfx1100 packed-MQ4 projections.

```text
GPU: AMD Radeon Pro W7900 Dual Slot (gfx1100), HIP device 1
model: qwen3.6-27b.mq4
workload: PP8192, TG8, prefill batch 2048
KV: Asym3 K + Q8 V
attention: staged quantized CK
common MQ4 options: X256Y64, perm-nibble, quad-row weight
common FFN options: fused SwiGLU, F16 intermediate
pairs: 5, alternating order, 10-second cooling
trim: one sample from each side per mode
```

| Mode | Raw PP8192 tok/s | Trimmed median |
| --- | --- | ---: |
| group128 | 1184.5, 1172.4, 1179.5, 1175.7, 1166.6 | 1175.7 |
| group256 serial-row | 1189.1, 1180.7, 1194.5, 1196.1, 1178.3 | 1189.1 |

The independent trimmed-median gain is **1.0114x (+1.14%)** and the median of
the five paired ratios is **1.0100x**. All five pairs favor group256, and all
runs emit the same token IDs.

The result is a small retainable gain, not an architectural step toward 1.5k
tok/s. It also shows that the historical standalone group256 gain (+4.40%)
mostly overlaps with the newer quad-row/F16/staged-CK configuration and must
not be multiplied into that result.

Reproduce with:

```bash
GPU_ID=1 PAIRS=5 TRIM_EACH_SIDE=1 COOL_SECS=10 \
  ./experiments/gfx11-gate-up-x256y64/run_pp8192_group256_staged_ck_ab.sh
```
