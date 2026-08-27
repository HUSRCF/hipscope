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
- Application wall: `6615.8 ms` (`1238.2 tok/s`, profiler-instrumented run)
- Packed-MQ4 production flags: X256/Y64, permuted nibble, group128 quad-row,
  group256 serial-row tails, fused SwiGLU, and FP16 FFN intermediates
- Raw kernel-stat CSV SHA-256:
  `3006351c4ce2fbd0f84190841816cbad861b05965d1ee7eded1c54d92adb263a`

## Relevant kernel totals

| Stage | Calls | Total | Fraction of wall |
| --- | ---: | ---: | ---: |
| Asym4 K + Q8 V predecode | 64 | `22.986 ms` | `0.347%` |
| Q Givens rotation | 64 | `2.193 ms` | `0.033%` |
| Dense CK attention | 64 | `240.201 ms` | `3.630%` |
| FP16-to-FP32 output bridge | 64 | `3.415 ms` | `0.052%` |

Across four equal chunks, full-history staging decodes `2K+4K+6K+8K` rows,
whereas a persistent incremental mirror would decode `8K` rows. Even assuming
perfect 60% removal of predecode work, the upper bound is approximately
`13.8 ms`, or `0.21%` of this wall time.

Keeping dense FP16 K/V for all 16 full-attention layers would instead consume
about 512 MiB at 8192 tokens and grow linearly with context (about 2 GiB at
32K and 12 GiB at 192K). The project therefore rejects a per-layer persistent
dense mirror. The current transient one-pass staging remains the production
route; optimization effort returns to the packed-MQ4 projection families that
account for most of the profile.
