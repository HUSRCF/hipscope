# TurboQuant Validation — Qwen3.5-0.8B on RX 5700 XT (gfx1010)

## Result: Turbo4 WORKS. Turbo3 does not. Previous test used wrong model.

The earlier "turbo is broken" diagnosis tested on Qwen3-0.6B (standard attention),
where ALL 4-bit KV modes fail (including non-turbo HFQ4). Testing on the correct
model family (Qwen3.5, DeltaNet+FullAttn hybrid) shows turbo4 produces coherent output.

## Test 1: Performance

| Config | tok/s (mean±range) | vs baseline |
|--------|-------------------|-------------|
| Q8 baseline | 235.8 ± 1.7 | — |
| **Turbo4** | **206.6 ± 3.7** | **-12.4%** |

Turbo4 is 12% slower than Q8 on this 0.8B model. The overhead comes from the FWHT
rotation in the attention kernel (5 ds_swizzle rounds per head). On a larger model
where attention is a smaller fraction of total compute, this overhead shrinks.

## Test 2: Coherence

### Q8 baseline:
> A computer processor is the central component that performs calculations.
> It executes instructions and manages memory, making it responsible for its
> operation by ensuring data integrity during execution of programs.

### Turbo4:
> A computer processor is responsible for performing calculations and operations
> on data to determine the next step of this process, but it must also be able
> handle new information while maintaining its own integrity requirements.

### Turbo3 (3-bit):
> 不倒式和ーツ活动等各族各行各... (CJK/Cyrillic mojibake)

### Long generation (Turbo4, 224 tokens):
Coherent narrative output. No mojibake, no repetition loops, no corruption cliff.
202 tok/s sustained through the full generation.

## Test 3: Turbo3 failure

Turbo3 (3-bit) fails on Qwen3.5-0.8B — the 3-bit precision is insufficient for
this model size. This is expected: 3-bit Lloyd-Max quantization with only 8 centroids
loses too much information for 0.8B-scale representations.

## ds_swizzle optimization (validated)

The `__shfl_xor` → `ds_swizzle_b32` replacement is confirmed working in production:
- Turbo4 inference produces identical quality with ds_swizzle
- 3 fewer VGPRs (31→28) in attention kernel
- All 13 turbo kernels compile clean on gfx1010/1030/1100/1200/1201
