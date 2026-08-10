# Group128 K32 Weight-Stationary Probe

This standalone probe changed the exact group128 inner loop from output-column
stationary to K32 weight-stationary. It tested the full Qwen3.6 gate/up shape
(`M=17408, K=5120, N=2048`) on Radeon Pro W7900 (`gfx1100`), GPU1.

| Path | Median latency | Relative | Max abs | Mean abs | VGPR | Spill/private |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production group128 | 4.7080 ms | 1.0000x | - | - | - | - |
| K32 weight-stationary | 5.8262 ms | 0.8081x | 2.86e-6 | 1.56e-7 | 215 | 0 |

The first fully unrolled form spilled heavily (`256 VGPR`, `196` spills,
`488 B` private). Serializing the K32 outer loop eliminated spills, but remained
19.2% slower. Saving repeated LDS weight-fragment loads does not cover the
additional scale/conversion work. Keep this path standalone and default-off.
