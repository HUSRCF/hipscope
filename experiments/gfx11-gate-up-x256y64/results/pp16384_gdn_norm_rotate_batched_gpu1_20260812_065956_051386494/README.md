# PP16384 Batched GDN Norm + MQ Rotate A/B

This run validates the gfx1100 batched GDN gated-normalization and MQ-rotation fusion at the production promotion length. The baseline and candidate use the same Qwen3.6-27B MQ4 model, asym3 KV mode, staged quantized CK attention sidecar, PP16384 input, and three in-process prefill repetitions. Three alternating process pairs were separated by 20 seconds.

## Build

```bash
cargo build --release --locked \
  --features deltanet,flash-attn-ck \
  --example bench_qwen35_mq4 \
  -p hipfire-runtime
```

The benchmark script rejects a run unless the log confirms that staged quantized CK prefill is active. A prior run built without `flash-attn-ck` measured about 500 tok/s and is excluded because it exercised a different backend.

## Result

| Mode | Raw prefill tok/s | Median prefill tok/s | Median decode tok/s |
|---|---:|---:|---:|
| Baseline | 1114.7, 1115.0, 1115.9 | 1115.0 | 31.9 |
| Fused | 1122.6, 1121.0, 1119.2 | 1121.0 | 31.8 |

- Median speedup: **1.0054x (+0.54%)**.
- Paired ratios: `1.0071x`, `1.0054x`, `1.0030x`; all three pairs are positive.
- Token IDs match exactly across all baseline and fused runs.
- Decode is intentionally unchanged because the new route requires `batch_size > 1`; the existing decode fusion remains separate.

## Resource Audit

The generated gfx1100 object reports:

```text
wavefront_size=32
vgpr_count=27
vgpr_spill_count=0
private_segment_fixed_size=0
```

The launch uses 64 threads, one workgroup per two-head MQ group and token row. The shape guard restricts the route to even collections of 128-value GDN heads with MQ4G256 output weights and no AWQ scale.

Raw measurements are in `results.tsv`; `summary.txt`, process logs, artifact hashes, and the run manifest are retained beside this file.
