# MagnumQuant Full Results — gfx1010 (RX 5700 XT)

## Summary

| Phase | Target | Actual | Status |
|-------|--------|--------|--------|
| 1. ISA: VGPRs | <=20 | **6** | PASS |
| 1. ISA: ds_swizzle | 5 instructions | **5** | PASS |
| 1. Correctness | round-trip < 1e-4 | **9.54e-7** | PASS |
| 2. Adaptive modes | scalar branch | **s_cbranch_execz** | PASS |
| 3. Fused VGPRs | <=24 | **10** | PASS |
| 4. Cosine sim >=0.990 (4-bit, Mode 1) | 0.990 | **0.996** | PASS |
| 5. Encoder (group-aware) | mode distribution | **99.5% Mode 0** | OK |
| 6. BW utilization | >=75% | **49.6%** | MISS |
| 6. Rotation overhead | <5% | **~32%** | MISS |

## Phase 1+2: Butterfly Rotation ISA

### magnum_butterfly_rotate_f32
- **VGPRs: 6**, SGPRs: 14, LDS: 0, Scratch: 0
- 5x ds_swizzle_b32 with SWAP patterns confirmed
- 3 VALU per round (cndmask + mul + fmac), total ~15 VALU for 5 rounds
- Compiler loads all params via s_load_dwordx8 (scalar path)

### magnum_butterfly_adaptive
- **VGPRs: 8**, SGPRs: 12
- Mode decode: global_load_ubyte → v_cmp → s_and_saveexec_b32 → s_cbranch_execz
- Uniform across wavefront → effectively scalar branching

### Round-trip correctness
| Config | Max Error | MSE | Cosine Sim |
|--------|-----------|-----|------------|
| 5-round fwd→inv | 9.54e-7 | 4.79e-14 | 1.0000000000 |
| Mode 0 fwd→inv | 3.58e-7 | 5.66e-15 | 1.0000000000 |
| Mode 1 fwd→inv | 5.96e-7 | 1.02e-14 | 1.0000000000 |
| Mode 2 fwd→inv | 9.54e-7 | 4.86e-14 | 1.0000000000 |

## Phase 3: Fused Dequant Kernel

### magnum_dequant_hfq4 (rotation + dequant)
- **VGPRs: 10**, SGPRs: 16, LDS: 0, Scratch: 0
- Rotation adds only **1 VGPR** vs dequant-only baseline (9 VGPRs)
- All 5 ds_swizzle patterns present in ISA

### magnum_dequant_hfq4_norot (dequant only)
- VGPRs: 9, SGPRs: 7

## Phase 4: Quantization Quality

Test data: 131072 floats (sin-based deterministic), HFQ4-G256 format.
Rotation params: arbitrary angles (pi/7, pi/5, pi/11, pi/3, pi/13).

| Method | Max Error | MSE | Cosine Sim |
|--------|-----------|-----|------------|
| **Baseline (no rotation)** | **0.200** | **0.0122** | **0.99868** |
| Mode 0 (2 rounds) | 0.534 | 0.0340 | 0.99626 |
| Mode 1 (3 rounds) | 0.588 | 0.0375 | 0.99590 |
| Mode 2 (5 rounds) | 1.120 | 0.0854 | 0.99066 |

### Analysis
**The rotation HURTS quality with these parameters.** This is expected because:
1. The rotation params are arbitrary, not optimized for quantization equalization
2. Givens rotations with random angles can make distributions less uniform
3. The test data (sin waves) already has relatively uniform distribution per group
4. More rounds = more mixing = larger range within groups = worse per-group quantization

**To make rotation help quality, we need:**
- Optimized rotation angles (learned per-layer or analytically derived)
- Or switch to Hadamard rotation (always equalizes) instead of arbitrary Givens
- Real KV cache data which has outlier-heavy distributions

## Phase 5: Mode Selection

Greedy encoder operating on real HFQ4-G256 groups (256 elements sharing one
scale+zero). Per-sub-block mode selection evaluated within group-level
quantization context. Iteratively upgrades sub-blocks that miss the threshold.

Threshold: cosine >= 0.995

| Mode | Blocks | Percentage |
|------|--------|------------|
| Mode 0 (2 rounds) | 4076 | 99.5% |
| Mode 1 (3 rounds) | 20 | 0.5% |
| Mode 2 (5 rounds) | 0 | 0.0% |

End-to-end quality with selected modes: max_err=0.534, MSE=3.41e-2, cosine=0.9963

Nearly all blocks satisfied with minimum rotation — consistent with Phase 4
showing rotation hurts with arbitrary angles on this data distribution.
With optimized params and real KV data (outlier-heavy), expect more varied distribution.

## Phase 6: Bandwidth

Peak theoretical: 448 GB/s (RX 5700 XT GDDR6)

| Seq Len | Fused (us) | Fused BW | % Peak | NoRot (us) | NoRot BW | Rotation Overhead |
|---------|------------|----------|--------|------------|----------|-------------------|
| 2048 | 347.5 | 218.8 GB/s | 48.8% | 261.2 | 291.1 GB/s | 33.1% |
| 4096 | 684.4 | 222.2 GB/s | 49.6% | 518.8 | 293.1 GB/s | 31.9% |
| 8192 | 1368.8 | 222.1 GB/s | 49.6% | 1035.8 | 293.6 GB/s | 32.2% |

### Analysis
- **Dequant-only achieves 65% peak BW** — reasonable for scattered nibble loads
- **Fused rotation adds ~32% overhead** — misses <5% target significantly
- **Root cause**: ds_swizzle has ~8 cycle latency per round; 5 rounds = 40 cycles serialized
  - In a bandwidth-bound dequant kernel with minimal compute, these 40 cycles dominate
  - The "effectively free" claim would hold in a compute-heavy kernel (e.g., full attention)
- **Bandwidth plateaus at ~222 GB/s** regardless of data size → compute-bound on rotation

## What We Learned About gfx1010 Shuffle Hardware

1. **ds_swizzle_b32 supports all 5 butterfly strides** (1,2,4,8,16) natively via XOR bitmask
2. **No LDS fallback needed** for any stride — all fit in the 5-bit xor_mask field
3. **ds_swizzle does NOT consume VGPRs** — pattern encoded as immediate operand
4. **ds_swizzle has significant latency** (~8 cycles) that serializes butterfly rounds
5. **DPP16 row_xmask could handle strides 1-8** (within 16-lane rows) but NOT stride 16
6. **DPP can fuse with ALU** (source modifier) — potential optimization for strides 1-8

## Recommendations

1. **Quality**: Replace arbitrary Givens angles with FWHT (existing turbo_common.hip
   already has this). Or learn optimal angles via gradient descent on KV cache statistics.

2. **Bandwidth**: For bandwidth-bound dequant, consider:
   - Using fewer rotation rounds (Mode 0/1 only)
   - Fusing rotation into the attention kernel (compute-heavy) instead of dequant
   - Using DPP row_xmask for strides 1-8 (ALU fusion) + ds_swizzle for stride 16

3. **Format**: Current prototype uses separate mode buffer. For production, pack mode
   bits into the HFQ block header (8 extra bytes per 256 elements = 6% overhead).
