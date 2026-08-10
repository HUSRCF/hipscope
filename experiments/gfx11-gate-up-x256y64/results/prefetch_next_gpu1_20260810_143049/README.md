# RDNA3 next-group packed-weight prefetch A/B

This corrected standalone probe keeps the production X256/Y64 group128 WMMA path and compact MQ4 wire format. It prefetches the next 256-K group's packed payload and FP32 metadata into registers before computing the current group. Metadata is copied only by the two lanes that initialized and consume it.

## Result

| Variant | Candidate median | Speedup vs its run-local LDS reference | Exactness |
|---|---:|---:|---:|
| Scalar quad-row | 4.3930 ms | 1.0629x | exact |
| Next-group prefetch | 4.4340 ms | 1.0633x | exact |

Direct comparison of candidate medians is **0.9908x (-0.93%)**. Each process ran 21 paired baseline/candidate measurements with alternating execution order, so both medians contain the same sample count. The two LDS references are process-local controls and are not used for the direct percentage.

The earlier `prefetch_next_gpu1_20260810_141825` run copied a whole register struct whose metadata members were initialized only on consumer lanes. That run is retained as a debugging artifact but is excluded from evidence. After removing that undefined-state copy, next-group prefetch is a small regression and remains standalone-only.

## ISA/resource audit

| Metric | Scalar quad-row | Next-group prefetch |
|---|---:|---:|
| Static instructions | 1422 | 1458 |
| Global/buffer loads | 75 | 80 |
| `s_waitcnt` | 117 | 119 |
| Branches | 10 | 17 |
| Compares | 5 | 8 |
| LDS instructions | 179 | 181 |
| WMMA instructions | 128 | 128 |
| Barriers | 5 | 5 |
| VGPR | 256 | 247 |
| VGPR spills | 4 | 0 |
| Private segment | 20 B | 0 B |

The compiler removes the scalar path's small spill and lowers VGPR allocation, but adds five global loads, two waits, and seven branches. The extra scheduling work outweighs any overlap at this shape. Together with the expanded-i8 upper-bound result (`5.2899 ms` versus `5.2317 ms`, -1.10%), this closes next-group packed-weight prefetch as a production lever.

## Scope

- GPU: AMD Radeon Pro W7900, gfx1100
- Shape: `M=17408, K=5120, N=2048`
- 21 paired measurements per process, alternating execution order
- GPU selected with `HIP_VISIBLE_DEVICES=1`
- Numerical comparison: `max_abs=0`, `mean_abs=0`
- Source and binary hashes: `artifacts.sha256`
