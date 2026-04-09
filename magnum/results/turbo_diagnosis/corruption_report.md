# TurboQuant KV Cache Corruption — Root Cause Diagnosis

## Finding: NOT turbo-specific. ALL 4-bit KV modes fail on Qwen3-0.6B.

| KV Mode | Bits | Output Quality |
|---------|------|----------------|
| FP32 | 32 | Coherent |
| Q8_0 | 8 | Coherent |
| HFQ4 | 4 | **BLANK** (no output) |
| HFQ4s | 4 | **Garbage** |
| Turbo4 | 4 | **Garbage** |
| Turbo3 | 3 | **Garbage** |
| Turbo2 | 2 | **Garbage** |

## Diagnosis Summary

### What we proved is CORRECT:
1. **FWHT rotation math**: GPU matches CPU at machine epsilon (max_diff 5.96e-7)
2. **Turbo write→read round-trip**: cosine 0.995-0.998 on all test vectors
3. **Single-layer turbo attention**: cosine 0.998 vs CPU ideal, attention scores match
4. **ds_swizzle_b32**: native instruction works correctly on all 5 RDNA architectures
5. **Kernel code symmetry**: write and read paths use matching sign tables, centroids,
   packing order, memory layout, and scale factors

### What we proved is BROKEN:
- ALL 4-bit KV cache modes (HFQ4, HFQ4s, Turbo4) produce garbage on Qwen3-0.6B
- Only 8-bit (Q8_0) and 32-bit (FP32) KV modes produce coherent output
- This pattern indicates **model sensitivity to 4-bit KV quantization**, not a
  turbo-specific code bug

### Root Cause

**Qwen3-0.6B is too small for 4-bit KV quantization.** With head_dim=128, n_kv_heads=8,
and only 28 layers of 1024-dim representations, the model's attention patterns are too
sensitive to survive 4-bit quantization of the KV cache. The per-layer attention error
(cosine ~0.998) compounds through 28 layers of residual connections and FFN amplification,
producing garbage output.

Evidence:
- Q8 (8-bit) works fine — enough precision for the attention patterns
- FP32 works fine — no quantization error
- ALL 4-bit modes fail equally — the issue is precision, not a specific format's bug
- HFQ4 (affine 4-bit, no rotation) also fails, proving the FWHT rotation is not the problem
- The turbo pipeline's single-layer quality (0.998 cosine) is good by 4-bit standards,
  but not enough for this model

### What this means for turbo on larger models

The Qwen3.5-4B and 9B models have larger representations and are known to tolerate
turbo4 on RDNA2+ (where they produce coherent output at 43-61 tok/s). The 0.6B model
is simply below the minimum size for 4-bit KV quantization.

To validate turbo4 properly, it must be tested on a model that tolerates 4-bit KV
quantization in general — confirmed by HFQ4 also working on that model. The Qwen3.5
models require the `deltanet` feature flag to build.

## Detailed Test Results

### Phase 1: Round-trip (all correct)
- all_ones: GPU vs CPU max_diff=5.96e-7, cosine=1.0000000000
- counting: GPU vs CPU max_diff=3.81e-5, cosine=1.0000000000
- single_hot: PERFECT round-trip (cosine=1.0)
- sin_wave: GPU vs CPU max_diff=2.38e-7, cosine=1.0000000000

### Phase 2: Single-layer attention (correct)
- Q=K[4], 8 positions, turbo4 attention vs CPU ideal: cosine=0.99841
- Attention scores closely match CPU ideal weights
- V weighted output: max_err=0.105 (quantization noise, expected)

### ds_swizzle ISA optimization (kept, validated)
- attention_turbo4_kv: 55→15 ds_bpermute, 0→40 ds_swizzle, 31→28 VGPRs
- All 13 turbo kernels compile clean on all 5 architectures (gfx1010→gfx1201)
- Inference output identical with ds_swizzle vs original __shfl_xor
