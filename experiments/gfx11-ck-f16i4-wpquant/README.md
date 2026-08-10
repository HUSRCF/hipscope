# gfx11 CK FP16 x packed-I4 WPQuant probe

This experiment tests whether the generic Composable Kernel weight-preshuffle
pipeline can replace hipfire's specialized packed-MQ4 projection primitive on
gfx1100. It is a feasibility probe, not a serving integration.

## Environment

```text
GPU: AMD Radeon Pro W7900 Dual Slot (gfx1100)
HIP: 7.14.60850
ROCm libraries source: c4a1de3928b2c25d988fb06cb41f17baeadbe3cb
CK example: projects/composablekernel/example/ck_tile/38_block_scale_gemm
pipeline: WPQuant preshuffle-B, FP16 A, packed INT4 B, FP16 C
quant group: 1x1x64
TiledMMAPermuteN: false
```

The stock example type gate was extended to admit FP16 activations, and the
local registration in `gemm_bquant_quantgrouped_preshuffleb_f16i4_gfx11.cpp`
instantiates the existing gfx11-aware WMMA policy. No hipfire serving code was
changed.

## Correctness

The first configuration used the inherited `TiledMMAPermuteN=true` setting and
failed the CPU reference check for about 99.7% of elements. Disabling the N
permutation produced a correct `M=N=128, K=256` result:

```text
M=128 N=128 K=256: 0.0602 ms, CPU reference PASS
```

This validates the load/type plumbing before the production-shape timing.

## Production-shape timing

The table reports the raw CK main GEMM only. It does not include the affine
MQ4 correction `(zero + 8 * scale) * sum(x)`, so it is an optimistic lower
bound for a complete adapter.

| Projection | M tokens | N output | K input | CK raw main | Retained MQ4 | CK / retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate/up | 2048 | 17408 | 5120 | 31.4427 ms | 4.8368 ms | 0.1538x |
| down | 2048 | 5120 | 17408 | 34.9142 ms | 4.8166 ms | 0.1380x |

The effective raw-main throughput was 11.61 TFLOP/s for gate/up and 10.46
TFLOP/s for down. Adding affine correction cannot recover a 6.5-7.2x deficit.

## Decision

**Reject this generic CK FP16 x packed-I4 WPQuant pipeline for hipfire's gfx11
MQ4-v2 backend.** The result does not reject CK or Wave32 WMMA in general. It
only closes this existing generic pipeline/configuration; a useful replacement
still needs a hipfire-specific execution format and dataflow that avoids the
current packed-MQ4 decode/feed costs without losing the specialized kernel's
projection throughput.

