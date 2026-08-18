# gfx11 X192/Y96 exact-Q8 topology negative result

This standalone probe tests a traffic-motivated topology change rather than a blind tile sweep. At the divisible proxy shape `M=17472, K=5120, N=2304`, X192/Y96 reduces the grid from `273x9=2457` to `182x12=2184` workgroups and reduces repeated activation staging per output row. It retains the MQ4-G256 weights, group128 Q8 activation contract, FP32 accumulation order, and quad-row packed-weight loader.

## Result

| Variant | Candidate median | Relative to scalar quad-row | Max abs | Mean abs |
|---|---:|---:|---:|---:|
| X256/Y64 scalar quad-row | 4.8966 ms | 1.0000x | 0 | 0 |
| X192/Y96 | 6.6948 ms | **0.7314x (-26.86%)** | 0 | 0 |

Each process ran 21 paired baseline/candidate measurements after warmup. The direct percentage compares candidate medians from the two processes; process-local LDS references are controls only.

## ISA/resource audit

| Metric | X256/Y64 | X192/Y96 |
|---|---:|---:|
| Static instructions | 1422 | 1178 |
| Global/buffer loads | 75 | 39 |
| `s_waitcnt` | 117 | 94 |
| LDS instructions | 179 | 145 |
| WMMA instructions | 128 | 96 |
| Barriers | 5 | 5 |
| VGPR | 256 | 256 |
| VGPR spills | 4 | 22 |
| Private segment | 20 B | 92 B |

The candidate performs less static work but exceeds the gfx11 per-thread register envelope and spills substantially more state. The resulting scratch traffic and 12-wave workgroup cost dominate the modeled activation-reuse benefit. Since lowering the launch-bound occupancy cannot raise the 256-VGPR architectural ceiling, no `min_blocks=1` follow-up is justified.

This proxy is sufficiently negative that real-shape tail handling for `M=17408, N=2048` is not implemented. The path remains standalone-only and must not enter serving dispatch.
