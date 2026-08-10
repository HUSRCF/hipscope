# Dual-row packed-weight loader probe (W7900 GPU1)

This standalone probe changes only cooperative MQ4 packed-weight staging for the production-shaped X256/Y64, row2/col4, Q8-group128 kernel. Each half-wave stages one weight row and therefore covers two rows per wave. The arithmetic, metadata, output tile, activation format, and WMMA loop remain unchanged.

## Gate/up result

Shape: `M=17408, K=5120, N=2048`, set path, paired medians.

| Candidate | Pairs | Baseline ms | Candidate ms | Speedup | max_abs |
|---|---:|---:|---:|---:|---:|
| aligned `u32x2` | 25 | 4.6643 | 4.4535 | 1.0473x | 0 |
| aligned `u32x2` | 75 | 5.0037 | 4.8157 | 1.0390x | 0 |
| sequential scalar `u32` x2 | 35 | 4.8092 | 4.6363 | 1.0373x | 0 |

The scalar-load ablation does not remove the resource increase: both full kernels use `256 VGPR`, spill 9 VGPRs, and reserve 40 bytes of private storage. The spill therefore follows the dual-row mapping rather than the `uint2` temporary. Despite that cost, the dual-row staging remains faster in paired timing.

## Other `u32x2` hot shapes

| Projection | M | K | N | Baseline ms | Candidate ms | Speedup | max_abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| FFN down residual | 5120 | 17408 | 2048 | 4.8913 | 4.7472 | 1.0304x | 0 |
| GDN QKVZA | 10240 | 5120 | 2048 | 2.8409 | 2.7224 | 1.0436x | 0 |
| attention QKV | 12288 | 5120 | 2048 | 3.3593 | 3.2164 | 1.0444x | 0 |
| GDN output | 6144 | 5120 | 2048 | 1.7571 | 1.6761 | 1.0484x | 0 |
| auxiliary residual | 5120 | 6144 | 2048 | 1.7782 | 1.7137 | 1.0376x | 0 |

## Boundary

This is standalone evidence, not a production-model throughput claim. Promotion requires a guarded full PP8192 A/B because the increased VGPR/private footprint may interact differently across the complete model.
