# gfx11 Stream-K128 Architecture Probes

These standalone probes test whether reducing packed-weight LDS or increasing
activation-tile reuse improves the existing gfx11 Wave32 WMMA MQ4 path. They
are not wired into serving dispatch.

## Environment

- GPU: AMD Radeon Pro W7900 (`gfx1100`), physical GPU 1
- Visibility: `HIP_VISIBLE_DEVICES=1`
- Shape: `M=17408, K=5120, N=2048`
- Reference: production X256/Y64 group128 packed-MQ4 kernel

## Results

| Candidate | Reference median | Candidate median | Relative | Correctness | Key resource result |
|---|---:|---:|---:|---|---|
| X256/Y64, K128 weight window, two N128 phases | 4.4726 ms | 10.7424 ms | 0.4164x | exact (`max_abs=0`) | 27,648 B dynamic LDS |
| X256/Y128, K128 weight window, 512-thread workgroup | 4.6483 ms | 7.4750 ms | 0.6218x | exact (`max_abs=0`) | 216 VGPR, 0 spill, 56,320 B dynamic LDS |
| X256/Y64, FP16 accumulator | 4.7301 ms | 4.9549 ms | 0.9546x | `max_abs=1.014e-2` | lower-precision accumulation did not pay back conversion cost |

The production full-add epilogue was also measured at 4.5548 ms versus
4.4067 ms for full-set, a 3.36% local tax. This is too small to explain the
remaining packed-MQ4 bottleneck.

## Interpretation

The phased X256/Y64 design reduces LDS, but the additional staging phases and
barriers dominate. The X256/Y128 design increases activation reuse and removes
scratch, but its 512-thread workgroup is still substantially slower. Therefore
the negative result is not explained by register spill alone: the larger
workgroup and synchronization topology are the limiting costs in these probes.

These results close three local directions for the current execution contract:

1. Reducing LDS by serializing two N128 weight phases.
2. Doubling the output-row tile with a 512-thread workgroup.
3. Reducing accumulator precision to FP16.

The next meaningful packed-MQ4 step must change the execution representation or
the scale/decode contract; another near-identical tile topology is not justified
by these data.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 5 \
  --stream-k128-phased-x256

HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 5 \
  --stream-k128-x256y128

HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 5 \
  --group128-f16-accum
```
