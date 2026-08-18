# CK A16 x I4 admission screen

This standalone screen enumerates all nine default Composable Kernel gfx11
Wave32-WMMA universal GEMM instances for BF16 x packed-I4 and FP16 x
packed-I4. It uses the production Qwen3.6-27B FFN matrix shapes but does not
change serving dispatch.

```text
GPU: AMD Radeon Pro W7900 / gfx1100
ROCm runtime/compiler: 7.14
CK source: flash-attention-fa4-v4.0.0.beta4_20260319c18_release2
activation rows: 2048
warmup/iterations per instance: 10/30
```

The CK runner uses signed I4 weights without the retained affine group scales,
so this is an optimistic execution-backend admission test rather than an
exact numerical replacement.

| Precision | Shape | Best CK | Retained MQ4 | Local speedup |
|---|---:|---:|---:|---:|
| BF16 x I4 | gate/up 2048x5120x17408 | 4.7422 ms | 4.2418 ms | 0.894x |
| BF16 x I4 | down 2048x17408x5120 | 4.9813 ms | 4.2675 ms | 0.857x |
| FP16 x I4 | gate/up 2048x5120x17408 | 3.9965 ms | 4.2418 ms | 1.061x |
| FP16 x I4 | down 2048x17408x5120 | 4.2058 ms | 4.2675 ms | 1.015x |

Three best results use the 128-thread, 128x128x32, 4x4 wave-map,
intrawave-v1 instance. BF16 down instead selects the 256-thread, 4x2 wave-map,
interwave-v1 instance by a small margin. FP16 improves on the retained timing
slightly, but it is far below the 1.30x per-shape admission threshold and has
a weaker weight contract. BF16 is slower outright. Do not integrate either
path into serving.

Reproduce with:

```bash
GPU_ID=1 ./experiments/gfx11-mq4-v2/ck-a16-i4-admission/run.sh
```

After the one-time build, set `BUILD=0` to rerun only the GPU measurements.
