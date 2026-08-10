# Quad-Row Launch-Bound Probe

This standalone gfx1100 experiment changes only `MMQ_MIN_BLOCKS_PER_CU` from 2 to 1 for the exact quad-row MQ4 kernel. It tests whether relaxing the launch bound removes VGPR spills or shortens the critical path enough to offset lower occupancy.

GPU1, gate/up `M=17408`, `K=5120`, `N=2048`, 21 alternating pairs:

| Variant | Reference ms | Candidate ms | Relative | max_abs |
|---|---:|---:|---:|---:|
| min blocks 2 | 4.6413 | 4.4104 | 1.0524x | 0 |
| min blocks 1 | 4.7278 | 4.4298 | 1.0673x | 0 |

The min-blocks-1 candidate is 0.44% slower by absolute median. Both compiled full set/add kernels retain the same 256 VGPR, 4 VGPR spills, and 20-byte private segment as the min-blocks-2 quad kernel. Relaxing the launch bound therefore neither removes the spills nor improves performance; do not add a production route.
