# gfx11 CK FFN upper-bound probe

This experiment asks a narrow question: does a stock CK WMMA GEMM leave enough
headroom to justify replacing hipfire's packed-MQ4 FFN kernels on gfx1100?
It is an upper bound, not a candidate backend. CK receives unpacked INT8 A/B,
uses an INT32 accumulator, and writes INT8 C. It omits MQ4 codebook decoding,
per-group scale/zero handling, the production output type, and the residual
epilogue.

Run on an idle GPU:

```bash
GPU_ID=1 TRIALS=10 \
  ./experiments/gfx11-ck-ffn-upper-bound/run_ck_int8_upper_bound.sh \
  | tee /tmp/ck-ffn-upper-bound.log
```

The two shapes correspond to Qwen3.6-27B prefill with a 2048-token chunk:

- gate/up projection: `[2048,5120] x [5120,17408]`
- down projection: `[2048,17408] x [17408,5120]`

The W7900 result is in `results/w7900_20260809/results.tsv`. Stock CK is only
2.1%-3.7% faster than the measured production MQ4 calls even after removing
the quantized-weight and epilogue work. Expanding MQ4 weights to INT8 also
approximately doubles their storage and traffic. A generic CK FFN replacement is therefore
rejected. A future CK-based attempt would need a native packed-MQ4 B loader and
must first demonstrate substantially more headroom in an isolated prototype.
