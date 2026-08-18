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

The group128 activation approximation improved Qwen3.6-27B PP8192 prefill by
`4.89%`, but failed the longer real-prompt quality gate after 32 matching output
tokens. A finer group32-scale prototype was then tested only in the standalone
GEMM harness:

| Activation format | Shape | Q8-relative speed | Relative L2 | Cosine |
| --- | --- | ---: | ---: | ---: |
| A4 group128 | M17408/K5120/N2048 | 1.087x | 3.70% | 0.999548 |
| A4 group32 | M17408/K5120/N2048 | 0.341x | 15.42% | 0.989579 |

The group32 decomposition is both slower and less accurate in the current IU4
WMMA organization. No A4 variant is approved for production. The production
flag remains default-off and research-only; further prefill work must preserve
the Q8 activation contract.
