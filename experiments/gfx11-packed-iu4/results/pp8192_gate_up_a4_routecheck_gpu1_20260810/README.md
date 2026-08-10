# PP8192 gate/up IU4-A4 route check

This is a guarded performance and plumbing result, not a quality approval for
the approximate signed-A4 activation format.

## Configuration

- GPU: Radeon Pro W7900 / gfx1100, HIP device 1
- model: Qwen3.6-27B MQ4
- workload: synthetic PP8192, prefill chunks of 2048, 32 greedy AR tokens
- KV: Asym3
- attention: quantized CK sidecar
- retained MQ4 path: X256/Y64, group128, row2, fused SwiGLU
- candidate: gate/up only, packed MQ4 x signed-A4 through native IU4 WMMA
- method: separate process per arm, both arms prewarmed, three alternating pairs,
  five seconds idle between processes

The runner requires the A4 arm to print the one-time production route marker
and requires that marker to be absent from the Q8 control arm.

## Result

| Pair | A4 order | Q8 tok/s | A4 tok/s | A4 / Q8 | Tokens match |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 1 | 1062.6 | 1110.6 | 1.0452x | yes |
| 2 | 0 | 1054.3 | 1105.9 | 1.0489x | yes |
| 3 | 1 | 1052.0 | 1105.1 | 1.0505x | yes |

Median throughput is `1054.3 tok/s` for Q8 and `1105.9 tok/s` for A4. The
pairwise speedup median is **1.0489x (+4.89%)**. Decode remains effectively
unchanged at `33.1-33.2 tok/s`. All three pairs emitted the same 35 recorded
token IDs.

A preceding five-pair run before adding the route marker measured `1056.3`
versus `1105.7 tok/s` (`+4.68%`) with identical token IDs. It is retained under
`../pp8192_gate_up_a4_ab_gpu1_20260809/` as corroborating evidence, but this
route-check run is the primary result.

## Boundary

The standalone full-shape comparison measured `relative_l2=0.0370` and
`cosine=0.999548` against the Q8 path. A subsequent 3371-token real-prompt gate
matched only the first 32 generated token IDs before the A4 output diverged and
degraded. See
`../real_prompt_gate_up_a4_quality_gpu1_20260810_v2/README.md`. The candidate is
therefore rejected for production despite the repeatable performance gain; it
remains default-off as a research-only route.

Reproduce with:

```bash
GPU_ID=1 TRIALS=3 TRIM_EACH_SIDE=0 \
  experiments/gfx11-packed-iu4/run_pp8192_gate_up_a4_ab.sh
```
