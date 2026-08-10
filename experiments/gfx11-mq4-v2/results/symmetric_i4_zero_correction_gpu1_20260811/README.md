# Symmetric signed-int4 zero-correction probe

This standalone probe tests whether changing the affine MQ4 contract to signed
int4 plus one scale can provide the required backend-level gain. It does not
change serving dispatch or add a checkpoint quantizer.

The synthetic test carrier remains 136 bytes per group for a mathematically
equivalent kernel-plumbing A/B before metadata downcast: reference weights use unsigned nibbles with
`zero=-8*scale`; the candidate recenters each nibble to `[-8,7]` and skips
affine zero correction. The separate FP32 scale/zero values are converted to
FP16 metadata by the retained path, so operation ordering is not bit-exact.
The unused zero field is retained only so both kernels consume the same
allocation and payload layout. The error metrics below compare these generated
carrier tensors; they are not checkpoint-derived model-quality evidence.

Hardware: AMD Radeon Pro W7900 Dual Slot, gfx1100, HIP 7.14. GPU1 was idle and
had zero allocated VRAM before the serial runs. Each result is a ten-pair
alternating median; the two shapes were separated by 20 seconds.

| Shape | Reference | Candidate | Speedup | Max abs | Relative L2 | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| gate/up set, M17408 K5120 N2048 | 4.9252 ms | 4.7937 ms | 1.0274x | 1.4083e-3 | 4.8906e-4 | 0.9999998804 |
| down add, M5120 K17408 N2048 | 4.9849 ms | 4.7997 ms | 1.0386x | 2.1646e-3 | 4.1442e-4 | 0.9999999141 |

The full set/add code objects use wave32, 256 VGPRs, 26 SGPRs, three VGPR
spills, a 16-byte fixed private segment, and 56,320 bytes of launch-time dynamic
LDS. Resident bytes remain 4.25 bits per weight in this plumbing probe; a final
symmetric format could remove four metadata bytes per 256 weights, but the
measured compute-path gain is already far below the 1.30x admission threshold.

Decision: reject. Affine zero correction is measurable but not a structural
bottleneck. Do not add a model quantizer or serving route for this candidate.

Reproduction:

```bash
HIP_VISIBLE_DEVICES=1 target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 10 --symmetric-i4

sleep 20

HIP_VISIBLE_DEVICES=1 target/release/examples/bench_hfq4_group256_direct \
  --m 5120 --k 17408 --n 2048 --pairs 10 --symmetric-i4 --add
```
