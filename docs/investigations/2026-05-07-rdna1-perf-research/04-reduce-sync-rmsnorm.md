# Exp #4: __reduce_*_sync (or __shfl_xor) replacing LDS-tree reduction in rmsnorm.hip

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment)

## Lever

Replace the block-level LDS-tree reduction in `kernels/src/rmsnorm.hip` with intra-wave reduction via either `__reduce_add_sync` (HIP 7.0+) or `__shfl_xor` butterfly, plus a small cross-wave LDS step. This is a drop-in kernel rewrite; no FFI / wrapper changes; the dispatch site (`crates/rdna-compute/src/dispatch.rs::rmsnorm_f32`) is unchanged.

## Why

Per `feedback_hip7_levers_for_gfx1010_2026_05_07.md`, the current `rmsnorm.hip` uses the canonical LDS-tree pattern:

```cpp
sdata[threadIdx.x] = sum_sq;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

For blockDim.x=256, this is 8 iterations of LDS round-trip + barrier. Replacing with intra-wave (wave32 butterfly, 5 DPP/shuffle ops, no LDS) plus a single cross-wave LDS reduction reduces LDS pressure substantially.

## Scenario

- Hardware: hipx, single RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1).
- Models: `qwen3.5-9b.mq4` (RMSNorm fires once per FA layer per token; with 32 layers and 16 FA, ~16 invocations per decode token).
- KV mode: asym3.
- Prompt: `"Why is the sky blue? Answer in two sentences."` (19 tokens).
- max_seq: 4096; max_tokens: 120; temperature: 0.0.
- 3 fresh-process runs per condition.

## Win criterion (pre-registered)

Per the autoresearch contract Exp #4 spec: "no regression on gfx1010 (the explicit target), no regression on gfx1151 (incidental), bonus consideration if RDNA2 throughput improves. Quality must remain bit-stable."

Concretely:
- Decode tok/s on gfx1010 within 0% to +5% of baseline (no regression, optional bonus). Median outside 2σ of baseline only if positive direction.
- Coherence gate PASS — output for the canonical prompt is fluent, on-topic, no token loops, self-EOS or normal stop.
- NRMSE within bf16 ULP (~1e-2 relative) is implied by passing coherence gate; bit-exact equivalence is NOT required (reduction order changes are expected).

## Loss criterion

- Decode tok/s ≥2% below baseline median, with statistical significance.
- Or coherence gate FAIL.

## No-change band

Decode tok/s between -2% and +5%, AND coherence gate PASS.

## Quality gate

`./scripts/coherence-gate.sh` runs after the kernel rebuild but before perf bench. If it fails, the kernel change is reverted regardless of perf data.

## Action on win

If decode improves >5% with quality intact: write up, squash-merge to master, update baseline.

## Action on no-change

Replacement is still cleaner code (less LDS pressure), free cross-arch portability. **Optional merge** — write up, propose if it's a clear stylistic / portability improvement. Default: revert to keep master diff minimal in autoresearch context. Document the negative perf result.

## Action on loss

Revert immediately. Document. Do not retry variants.
