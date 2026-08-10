# Exact Group128 Serial-Row Probe

This standalone probe serialized the two output-row accumulator fragments while
preserving the production group128 activation scale contract. It tested the
full Qwen3.6 gate/up shape (`M=17408, K=5120, N=2048`) on Radeon Pro W7900
(`gfx1100`), GPU1, using 15 alternating pairs after warmup.

| Path | Median latency | Relative | Max abs | Mean abs |
| --- | ---: | ---: | ---: | ---: |
| Production group128 row2 | 4.9521 ms | 1.0000x | - | - |
| Group128 serial-row | 4.9664 ms | 0.9971x | 0 | 0 |

The exact serial-row form is performance-neutral. Together with the faster but
inexact group256 result, this indicates that the group256 gain comes from
coarser scale sharing and reduced scale application work, not merely from
shortening accumulator live ranges. Do not route this probe into serving.
