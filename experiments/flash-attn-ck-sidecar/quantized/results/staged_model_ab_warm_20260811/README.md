# Staged quantized CK model A/B

This run compares the legacy packed Asym3-K/Q8-V CK view with an additive staged
route. The staged route decodes each physical KV head once into reusable FP16
scratch and invokes the mature dense CK D256 GQA pipeline. All retained gfx11
MQ4 and F16-FFN options are identical between modes.

```text
GPU: Radeon Pro W7900 Dual Slot, gfx1100
model: qwen3.6-27b.mq4
workload: PP8192, gen=0, max prefill batch=2048
KV: Asym3 K + Q8 V
pairs: 3, alternating order, 20-second inter-run cooling
```

| Pair | Legacy packed CK | Staged CK | Paired gain |
| ---: | ---: | ---: | ---: |
| 1 | 1127.3 tok/s | 1188.1 tok/s | +5.39% |
| 2 | 1106.9 tok/s | 1180.3 tok/s | +6.63% |
| 3 | 1106.5 tok/s | 1173.5 tok/s | +6.05% |

The median paired gain is **+6.05%**. The ratio of independent mode medians is
`1180.3 / 1106.9 = 1.0663x`. Each log contains the expected route activation
message. An earlier six-process run is retained separately because its first
legacy process was a cold outlier (`923.2 tok/s`); it is not used for the main
claim.

Reproduce with:

```bash
GPU_ID=1 PAIRS=3 COOL_SECS=20 \
  ./experiments/flash-attn-ck-sidecar/quantized/run_staged_model_ab.sh
```
