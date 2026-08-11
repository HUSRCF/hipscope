# gfx11 CK packed-I4/INT8 upper-bound check

This experiment asks whether an existing Composable Kernel WMMA backend offers
a credible replacement boundary for Hipfire's retained gfx1100 Q8/HFQ4-G256
prefill primitive. It uses the same large gate/up matrix dimensions but is an
upper-bound diagnostic, not a correctness-equivalent backend: the stock CK
examples do not implement Hipfire's group128 activation scale, affine HFQ4 zero
correction, FP32 output contract, or exact execution layout.

Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14, CK shape
`M=2048, N=17408, K=5120`:

| CK path | Median kernel time | Effective throughput |
| --- | ---: | ---: |
| FP16 x packed-I4, no scale | 5.6019 ms | 65.17 TFLOP/s |
| INT8 x INT8 WMMA core | 5.1473 ms | 70.93 TFLOP/s |
| FP16 x packed-I4 with B-scale | 35.7224 ms | 10.22 TFLOP/s |

The retained Hipfire exact-Q8/HFQ4 gate/up projection is about `4.4 ms` at the
same logical dimensions. Even the simpler stock CK kernels are slower, while
the CK B-scale path is much slower. Adapting that B-scale implementation cannot
provide the required whole-family speedup; Hipfire's custom IU8-WMMA path is
already ahead of these mature generic CK controls. The attention CK sidecar
remains useful, but CK GEMM is not promoted into the packed-MQ4 serving path.

Reproduce with:

```bash
GPU_ID=1 \
  experiments/gfx11-ck-mq4-upper-bound/run_ck_wmma_upper_bound.sh
```

Set `CK_ROOT` to another Composable Kernel checkout if needed. Reusing an
existing configured build is supported with `SKIP_BUILD=1 BUILD_DIR=...`.
