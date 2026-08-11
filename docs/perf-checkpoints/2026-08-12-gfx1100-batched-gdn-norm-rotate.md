# gfx1100 Batched GDN Norm + MQ Rotate

## Scope

This checkpoint promotes a prefill-only sibling of the existing Qwen3.6 GDN decode fusion. On gfx1100, it computes gated normalization for two independent D128 heads, keeps the normalized 256-value MQ group in LDS, and applies the existing MQ rotation order before the output projection. Other architectures, dtypes, dimensions, AWQ-scaled weights, and single-token decode retain their existing paths.

The route is resolved once through `FeatureFlags`. It defaults on only for `gfx1100` and can be disabled with:

```bash
HIPFIRE_GATED_NORM_MQ_ROTATE_BATCHED=0
```

## PP16384 Evidence

The required long-prefill gate used Qwen3.6-27B MQ4, asym3 KV, staged quantized CK attention, three alternating A/B process pairs, three prefill repetitions per process, and 20-second idle intervals.

| Mode | Median prefill tok/s | Raw prefill tok/s |
|---|---:|---:|
| Existing two-dispatch path | 1115.0 | 1114.7, 1115.0, 1115.9 |
| Batched fused path | 1121.0 | 1122.6, 1121.0, 1119.2 |

The candidate is **+0.54%**, with all three paired ratios positive and exact token-ID parity. Decode throughput remains approximately 31.8-31.9 tok/s and is outside this prefill-only route.

The generated gfx1100 object is Wave32 with 27 VGPRs, zero VGPR spills, and no private segment. Full logs and artifact hashes are stored under:

```text
experiments/gfx11-gate-up-x256y64/results/
  pp16384_gdn_norm_rotate_batched_gpu1_20260812_065956_051386494/
```

This is a small production improvement, not a change to the main bottleneck: packed-MQ4 projections still dominate PP16384 wall time.
