# RDNA3 next-group packed-weight prefetch A/B

This standalone probe keeps the production X256/Y64 group128 WMMA path and compact MQ4 wire format. It fetches the next 256-K group's packed payload and FP32 metadata into registers before computing the current group, then performs the unchanged nibble expansion and LDS commit at the next iteration.

## Result

| Variant | Candidate median | Speedup vs its run-local LDS reference | Exactness |
|---|---:|---:|---:|
| Scalar quad-row | 4.4088 ms | 1.0742x | exact |
| Next-group prefetch | 4.3867 ms | 1.0677x | exact |

Direct comparison of candidate medians gives only **1.0050x (+0.50%)**. The two LDS references were measured in separate processes and are not used for that percentage. This is below the production promotion threshold, so the path remains standalone-only.

## ISA/resource audit

| Metric | Scalar quad-row | Next-group prefetch |
|---|---:|---:|
| Instructions | 1422 | 1462 |
| Global/buffer loads | 75 | 80 |
| `s_waitcnt` | 117 | 119 |
| Branches | 10 | 17 |
| Compares | 5 | 7 |
| LDS instructions | 179 | 181 |
| WMMA instructions | 128 | 128 |
| Barriers | 5 | 5 |
| VGPR | 256 | 247 |
| VGPR spills | 4 | 0 |
| Private segment | 20 B | 0 B |

The compiler preserves a lower-resource, spill-free candidate, but the latency overlap is too small to materially change the full-shape kernel. Combined with the expanded-i8 upper-bound result (`5.2899 ms` versus `5.2317 ms`, -1.10%), this rejects weight-loader/decode latency as the next production lever.

## Scope

- GPU: AMD Radeon Pro W7900, gfx1100
- Shape: `M=17408, K=5120, N=2048`
- 21 timed pairs per candidate
- GPU selected with `HIP_VISIBLE_DEVICES=1`
- Numerical comparison: `max_abs=0`, `mean_abs=0`
