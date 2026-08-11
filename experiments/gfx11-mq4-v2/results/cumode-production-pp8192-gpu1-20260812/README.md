# gfx11 module-exact CU-mode production probe

## Question

Does compiling only the production `gemm_hfq4g256_residual_mmq` code object in
CU mode improve the 57 KiB-LDS, Wave32-WMMA packed-MQ4 path on gfx1100?

The probe did not change kernel arithmetic, tensor layouts, launch geometry, or
model routing. A temporary module-exact compiler flag selected `-mcumode`; the
entry point was removed after measurement.

## Configuration

- GPU: W7900 / gfx1100, GPU1, ROCm runtime 7.14
- model: `qwen3.6-27b.mq4`
- prefill: 8192 tokens, three in-process runs
- KV: asym3
- attention: quantized CK sidecar
- retained MQ4: X256/Y64, group128, row2, quad-row weight, fused SwiGLU
- graph: disabled

The first prefill run includes module JIT and is excluded from the hot-state
comparison. The reported benchmark median therefore corresponds to the slower
of the two subsequent hot runs.

## Result

| Mode | Hot runs (tok/s) | Reported median (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Production WGP mode | 1152.6, 1144.6 | 1144.6 | 33.3 |
| Module-exact CU mode | 1131.9, 1123.7 | 1123.7 | 33.2 |

CU mode is `0.9817x` the production WGP result (`-1.83%`). It is rejected and
the temporary compiler entry point is not retained.
