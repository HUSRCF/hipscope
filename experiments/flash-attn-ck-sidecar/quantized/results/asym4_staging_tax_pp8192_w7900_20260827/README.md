# Asym4 staging-tax profile, W7900, PP8192

This profile tests whether retaining a dense FP16 K/V mirror per attention
layer is justified. It is not: the existing one-pass Asym4/Q8 predecode is a
small part of the production prefill wall time.

## Configuration

- GPU: Radeon Pro W7900 (`gfx1100`), GPU 0
- Model: Qwen3.6-27B MQ4
- KV: Givens-Asym4 K, Q8 V
- Prompt: 8192 tokens, four 2048-token chunks
- Sidecar: staged CK route from commit `160f09c1`
- Profiler: `rocprofv3` kernel trace through `scripts/rocprof-wrap.sh`
- Application wall: `8827.9 ms` (`928.0 tok/s`, profiler-instrumented run)
- Raw kernel-stat CSV SHA-256:
  `bec9fe64ee51e66cc189437f65cf7f1b27bf7bfdf39408c1fb73a605f4b5ece1`

## Relevant kernel totals

| Stage | Calls | Total | Fraction of wall |
| --- | ---: | ---: | ---: |
| Asym4 K + Q8 V predecode | 64 | `23.027 ms` | `0.261%` |
| Q Givens rotation | 64 | `2.081 ms` | `0.024%` |
| Dense CK attention | 64 | `231.290 ms` | `2.620%` |
| FP16-to-FP32 output bridge | 64 | `3.222 ms` | `0.036%` |

Across four equal chunks, full-history staging decodes `2K+4K+6K+8K` rows,
whereas a persistent incremental mirror would decode `8K` rows. Even assuming
perfect 60% removal of predecode work, the upper bound is approximately
`13.8 ms`, or `0.16%` of this wall time.

Keeping dense FP16 K/V for all 16 full-attention layers would instead consume
about 512 MiB at 8192 tokens and grow linearly with context (about 2 GiB at
32K and 12 GiB at 192K). The project therefore rejects a per-layer persistent
dense mirror. The current transient one-pass staging remains the production
route; optimization effort returns to the packed-MQ4 projection families that
account for most of the profile.
