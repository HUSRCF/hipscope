# gfx11 fused SwiGLU-to-group128 MMQ input

Qwen3.6-27B MQ4 was measured on a Radeon Pro W7900 (`gfx1100`) at PP8192 with 2048-token chunks, Asym3 KV, and the quantized CK attention sidecar. Both arms used the existing opt-in X256/Y64, permutation, and group128 MMQ routes. The candidate additionally fused dense FFN SwiGLU, FWHT rotation, and group128 Q8 packing into the existing MMQ scratch before the down projection.

## Production A/B

Ten fresh-process pairs alternated execution order and slept eight seconds between runs. The reported result trims the two fastest and two slowest samples from each arm before taking the median; the untrimmed median is identical.

| Path | Raw PP8192 prefill tok/s | Trim-2 median | Decode median |
|---|---|---:|---:|
| Existing group128 | 1048.8, 1023.1, 1027.5, 1027.3, 1025.3, 1025.3, 1026.9, 1027.8, 1027.4, 1026.3 | **1027.10** | 32.9 |
| Fused producer | 1042.7, 1035.9, 1044.3, 1044.0, 1041.2, 1038.9, 1047.1, 1046.9, 1039.2, 1040.0 | **1041.95** | 32.9 |

The production prefill delta is **1.0145x (+1.45%)**. All 20 runs emitted the same 35 greedy token IDs; this is a routing/correctness gate, not a broad quality evaluation of group128 activation quantization.

## Attribution and resources

The internal profiler measured the old `fused_silu_mul_mq_rotate_batched` at 744 us/call and the fused Q8 producer at 610 us/call, an 18% reduction for that stage. The stage was only about 2.7% of serialized GPU time before the change, so the small end-to-end lift is consistent with its Amdahl bound. The group128 MQ4 set/residual kernels remain roughly 82% of serialized time.

The new HIP kernel compiles to wave32 with 82 VGPR, 18 SGPR, zero LDS, zero private storage, and zero VGPR/SGPR spills. A focused review found the scratch lifetime and half-wave group128 packing sound. The route was restricted to `MQ4G256` after review because `HFQ4G256` does not use the MQ FWHT input contract. It also requires the existing residual-X256/Y64 and permutation flags, preventing the new opt-in from bypassing their rollout boundaries.

## Reproduction

```bash
TRIALS=10 TRIM_EACH_SIDE=2 SLEEP_SECS=8 \
OUT_DIR=experiments/gfx11-gate-up-x256y64/results/pp8192_fused_swiglu_q8_group128_ab_10pair_trim2 \
bash experiments/gfx11-gate-up-x256y64/run_pp8192_fused_swiglu_q8_group128_ab.sh

OUT_DIR=experiments/gfx11-gate-up-x256y64/results/pp8192_fused_swiglu_q8_group128_profile_20260809 \
bash experiments/gfx11-gate-up-x256y64/run_pp8192_fused_swiglu_q8_group128_profile.sh
```

This path remains opt-in through `HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1` and additionally requires `HIPFIRE_RDNA3_Q8_GROUP128=1`.

## Chunk-size boundary

A separate three-pair PP8192 A/B kept the fused path enabled and changed only `HIPFIRE_PREFILL_MAX_BATCH`: chunk 2048 measured **1041.7 tok/s**, while chunk 4096 measured **990.1 tok/s** (`0.9505x`, `-4.95%`). Token IDs matched. Reducing the number of chunk boundaries does not offset the larger-workset cost on this W7900 path, so chunk 2048 remains the measured configuration and the chunk-size line is closed.
