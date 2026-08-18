# RDNA3 wave-specialized staging A/B

This experiment assigns MQ4 payload/metadata loading to waves 0-3 and Q8 activation staging to waves 4-7. All eight waves join the existing workgroup barriers and execute the unchanged WMMA loop.

## Result

| Variant | Candidate time | Speedup vs its run-local LDS reference | Exactness |
|---|---:|---:|---:|
| Scalar quad-row | 4.3953 ms | 1.0696x | exact |
| Wave-specialized | 5.1877 ms | 0.8862x | exact |

The wave-specialized candidate is **18.03% slower** than the retained scalar quad-row candidate by direct comparison of the two candidate medians (`5.1877 / 4.3953 - 1`). The LDS reference was measured independently in each process and is not used for that cross-candidate percentage. The candidate is therefore rejected and was not promoted to a production PP8192 run.

## ISA/resource audit

| Metric | Scalar quad-row | Wave-specialized |
|---|---:|---:|
| Instructions | 1422 | 1669 |
| Global/buffer loads | 75 | 41 |
| `s_waitcnt` | 117 | 124 |
| Branches | 10 | 12 |
| Compares | 5 | 6 |
| LDS instructions | 179 | 187 |
| WMMA instructions | 128 | 128 |
| Barriers | 5 | 5 |
| VGPR | 256 | 228 |
| VGPR spills | 4 | 0 |
| Private segment | 20 B | 0 B |

The candidate removes spills and lowers VGPR usage, but it adds 247 static instructions and seven waits. Concentrating each staging task in four waves did not create a profitable cross-wave pipeline; the additional control/address work and reduced per-task wave parallelism outweighed the resource reduction.

## Scope

- GPU: AMD Radeon Pro W7900, gfx1100
- Shape: `M=17408, K=5120, N=2048`
- 21 timed pairs per candidate
- GPU selected with `HIP_VISIBLE_DEVICES=1`
- Numerical comparison: `max_abs=0`, `mean_abs=0`
