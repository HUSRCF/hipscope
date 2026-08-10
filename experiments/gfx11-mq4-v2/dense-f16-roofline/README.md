# Dense FP16 roofline

This standalone rocBLAS probe measures the same two Qwen3.6-27B FFN shapes as
the retained packed-MQ4 benchmark after replacing both operands with dense
FP16. It is an optimistic same-math execution ceiling, not a deployable weight
format: a full FP16 checkpoint does not fit the same memory budget.

```text
GPU: AMD Radeon Pro W7900 / gfx1100
ROCm: 7.14
warmup/trials: 1000/51
activation rows: 2048
```

| Shape | rocBLAS FP16 | Effective TFLOP/s | Retained MQ4 | Local ceiling |
|---|---:|---:|---:|---:|
| gate/up M17408 K5120 N2048 | 3.7282 ms | 97.92 | 4.2418 ms | 1.138x |
| down M5120 K17408 N2048 | 3.9090 ms | 93.39 | 4.2675 ms | 1.092x |

Even a dense path with no packed-weight decode, group scale, or affine
correction is only 9-14% faster per projection. If the entire 71.7% packed-MQ4
wall share could reach the more optimistic 1.138x local ceiling, Amdahl's law
would yield only about 1.095x overall, or roughly 1.30k tok/s from the current
1189 tok/s best controlled result. Reaching 1.5k therefore requires reducing
effective model work or a specialized execution mechanism that materially
exceeds the observed rocBLAS/CK dense roofline; execution-format cleanup alone
is not enough.

Reproduce with:

```bash
GPU_ID=1 ./experiments/gfx11-mq4-v2/dense-f16-roofline/run.sh
```
