# gfx11 packed IU4 WMMA probes

These probes test the native packed-int4 WMMA contract before any Hipfire production integration. They do not register kernels with the runtime dispatcher.

The initial contract probe uses `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`, unsigned packed weights, signed packed activations, and the gfx11 duplicated-lane/output mapping documented by CK Tile.

```bash
HIP_VISIBLE_DEVICES=1 ./experiments/gfx11-packed-iu4/run_contract_probe.sh
HIP_VISIBLE_DEVICES=1 ./experiments/gfx11-packed-iu4/run_throughput_ab.sh
```

The contract probe must remain bit-exact. The throughput A/B determines whether an exact Q8 decomposition into two IU4 operations has any instruction-level budget; it is not a GEMM speedup claim.

The completed exact two-plane GEMM is also a negative result. It is bit-exact
to the Q8 group128 contract within FP32 accumulation order, but reaches only
`0.462x` on the full `M17408/K5120/N2048` gate/up shape. See
`results/exact_q8_two_plane_gpu1_20260810/README.md`.

## Signed-A4 boundary

The original combined gate/up group128 activation approximation improved an
older Qwen3.6-27B PP8192 path by `4.89%`, but failed the longer real-prompt
quality gate after 32 matching output tokens. Projection-isolated tests on the
current production recipe narrowed the result further:

| Mode | PP8192 paired gain | LongBench accuracy | Token-exact vs Q8 |
| --- | ---: | ---: | ---: |
| Q8 control | baseline | 8/20 | 20/20 |
| gate A4 | not retained | 7/20 | 16/20 |
| up A4 | +2.05% | 8/20 | 15/20 |

The up-only result is a stable five-pair median (`1217.4` vs `1192.9 tok/s`),
but the output trajectory changes on five of twenty long-context cases. The
accuracy tie is too small a quality sample to approve a semantic change for a
2% prefill gain. Both projection-isolation flags therefore remain default-off
research controls.

A finer group32-scale prototype was tested only in the standalone GEMM harness:

| Activation format | Shape | Q8-relative speed | Relative L2 | Cosine |
| --- | --- | ---: | ---: | ---: |
| A4 group128 | M17408/K5120/N2048 | 1.087x | 3.70% | 0.999548 |
| A4 group32 | M17408/K5120/N2048 | 0.341x | 15.42% | 0.989579 |

The group32 decomposition is both slower and less accurate in the current IU4
WMMA organization. No A4 variant is approved for production. The production
flag remains default-off and research-only; further prefill work must preserve
the Q8 activation contract.

## Layer-isolation probe

`HIPFIRE_RDNA3_HFQ4_GATE_IU4_A4_LAYERS` and
`HIPFIRE_RDNA3_HFQ4_UP_IU4_A4_LAYERS` accept comma-separated layer numbers or
ranges (for example `8,16-19`). They are research-only overrides for the
Qwen3.6 FP16-intermediate prefill path. With neither variable set, dispatch is
unchanged and continues to use the startup feature flags.

An eight-layer up-A4 sample on the 3375-token `docs/testINPUT.md` prompt dumped
the hidden state at absolute position 2047 and compared the final layer against
Q8. Each run changed only one layer:

| A4 layer | Final cosine | Final relative L2 | First generated token |
| ---: | ---: | ---: | --- |
| 0 | 0.998842 | 4.81% | match |
| 8 | 0.999144 | 4.17% | match |
| 16 | 0.999039 | 4.39% | match |
| 24 | 0.998735 | 5.80% | match |
| 32 | 0.999226 | 3.97% | match |
| 40 | 0.998992 | 4.51% | match |
| 48 | 0.999273 | 3.82% | match |
| 56 | 0.998665 | 5.20% | match |

No sampled layer is close to numerically lossless, and quantizing one of 64
FFN up projections has only about `2.05% / 64 = 0.03%` overall prefill upside.
Layer masking therefore does not rescue the current group128 A4 contract as a
production optimization. Use `analyze_hidden_layers.py` to parse future
`HIPFIRE_DUMP_HIDDEN` comparisons.
