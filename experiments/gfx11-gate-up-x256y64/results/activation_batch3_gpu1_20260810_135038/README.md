# Batch-3 Q8 Activation Staging Probe

This standalone gfx1100 experiment changes only Q8 activation staging for the retained quad-row packed-weight kernel. Each source-level loop issues three independent aligned `uint4` loads before storing those vectors to LDS. MQ4 weight loading, metadata, barriers, WMMA, accumulation, and output remain unchanged.

GPU1, gate/up `M=17408`, `K=5120`, `N=2048`, 21 pairs:

| Variant | Group128 LDS reference ms | Candidate ms | Candidate vs LDS | max_abs |
|---|---:|---:|---:|---:|
| Scalar activation staging | 4.6791 | 4.3987 | 1.0638x | 0 |
| Batch-3 `uint4` staging | 4.6700 | 4.5990 | 1.0154x | 0 |

The speedup column compares each candidate with its separately measured group128 LDS reference. Comparing the two candidate medians directly, batch-3 is 4.55% slower than scalar quad-row (`4.5990 / 4.3987`) while preserving exact output.

## ISA and Resources

| Metric | Scalar quad | Batch-3 candidate |
|---|---:|---:|
| static instructions | 1422 | 1272 |
| `global_load_b32` | 72 | 0 |
| `global_load_b128` | 2 | 20 |
| VGPR | 256 | 237 |
| VGPR spills | 4 | 0 |
| private segment | 20 B | 0 B |
| barriers | 5 | 5 |
| WMMA instructions | 128 | 128 |

The source-level three-load batching survives only for the second activation half. For the first half, the compiler serializes each vector load with `vmcnt(0)` before its LDS store. The reduced instruction count and eliminated spill therefore do not translate into a better global-to-LDS schedule. Do not promote this path to production.
