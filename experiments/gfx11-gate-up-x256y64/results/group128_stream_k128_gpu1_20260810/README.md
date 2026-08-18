# Group128 K128 Weight-Window Streaming Probe

## Scope

- GPU: Radeon Pro W7900 (`gfx1100`), device 1
- Shape: gate/up projection, `M=17408`, `K=5120`, `N=2048`
- Baseline: retained group128 X256/Y64 path
- Candidate: exact X128/Y64 path that streams two K64 weight windows per K128 activation window
- Timing: warmed standalone medians

## Result

| Path | Median (ms) | Relative | Correctness |
|---|---:|---:|---|
| group128 X256/Y64 baseline | 4.6215 | 1.0000x | reference |
| group128 X128/Y64 K128 stream | 6.0918 | 0.7586x | exact match |

The candidate kernel uses 191 VGPRs, 27 SGPRs, no scratch/private spill, and about 27.5 KiB dynamic LDS. The regression is therefore not explained by a resource spill. Halving the X tile duplicates weight/global work and grid-level overhead enough to dominate any benefit from the narrower activation staging window.

## Decision

Do not route production GEMMs through the K128 streaming variant. Keep it as a default-off structural probe and move to projection-family isolation of the faster group256 activation path.
