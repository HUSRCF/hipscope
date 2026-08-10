# Group128 packed-FP16 accumulator probe

Standalone-only RDNA3 experiment on GPU1. The retained production baseline is the exact X256/Y64, row2/col4, Q8-group128 path with FP32 cross-group accumulation.

## Gate/up shape

```text
M=17408 K=5120 N=2048 pairs=15
baseline FP32:       4.6916 ms
packed half2 accum:  4.8580 ms
speedup:             0.9657x
max_abs:             1.01430416e-2
mean_abs:            7.05955200e-4
```

## Resource audit

```text
baseline full_set:   252 VGPR, 0 spills, 0 private bytes
half2 full_set:      256 VGPR, 13 spills, 56 private bytes
```

The first `_Float16[]` spelling was optimized into FP32 registers and was therefore not a valid probe. The reported result uses explicit `half2` packing and conversion on every accumulator update. It genuinely changes numerical results, but conversion/indexing pressure pushes the kernel into scratch and makes it slower. This route is rejected and is not connected to serving dispatch.
