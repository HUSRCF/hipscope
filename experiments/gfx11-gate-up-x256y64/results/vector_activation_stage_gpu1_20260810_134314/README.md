# Vectorized Q8 Activation Staging Probe

This standalone gfx1100 experiment changes only the Q8 activation copy into LDS. The quad-row reference copies one `int` per lane and iteration; the candidate copies one aligned `uint4`, reducing 36 scalar iterations per lane and activation half to 9 vector iterations. MQ4 weights, metadata, barriers, WMMA, accumulation, and output are unchanged.

GPU1, gate/up `M=17408`, `K=5120`, `N=2048`, 21 alternating pairs:

| Variant | Reference ms | Candidate ms | Relative | max_abs |
|---|---:|---:|---:|---:|
| Scalar activation stage | 4.6959 | 4.4147 | 1.0637x | 0 |
| `uint4` activation stage | 4.6952 | 4.5058 | 1.0420x | 0 |

The vector candidate is 2.06% slower by absolute median while preserving exact output.

## ISA and Resources

| Metric | Scalar quad | `uint4` candidate |
|---|---:|---:|
| static instructions | 1423 | 1273 |
| `global_load_b32` | 72 | 0 |
| `global_load_b128` | 2 | 20 |
| VGPR | 256 | 237 |
| VGPR spills | 4 | 0 |
| private segment | 20 B | 0 B |

Despite fewer instructions and no spills, widening each lane's activation transaction loses performance. The scalar loads already coalesce at wave level and provide the better global-to-LDS schedule for this kernel. Do not promote the vector path to production.
