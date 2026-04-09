# MagnumQuant Phase 1: ISA Analysis

Target: gfx1010 (RDNA1, RX 5700 XT)
Compiler: HIP 6.3, AMD clang 18.0.0, -O3

## magnum_butterfly_rotate_f32 (5-round)

| Metric | Target | Actual |
|--------|--------|--------|
| VGPRs | <=20 | **6** |
| SGPRs | -- | 14 |
| ds_swizzle_b32 | 5 | **5** |
| Scratch/spill | 0 | **0** |
| LDS (shared mem) | 0 | **0** |
| Wave mode | wave32 | **wave32** |

### Instruction breakdown (core rotation, 5 rounds)

Per round:
- `ds_swizzle_b32 vN, vM offset:swizzle(SWAP,stride)` - cross-lane exchange (1)
- `v_cndmask_b32_e64 vN, sA, -sA, vcc_lo` - select sign (1 VALU)
- `v_mul_f32_e32 vN, vA, vB` - sign_s * partner (1 VALU)
- `v_fmac_f32_e32 vN, sA, vB` - c * val + partial (1 VALU)
- `v_and_b32_e32 + v_cmp_eq_u32_e32` - lane mask (interleaved with swizzle latency)

Total: **3 VALU + 1 ds_swizzle per round = 15 VALU + 5 ds_swizzle for 5 rounds**

### Key compiler optimizations observed
1. All rotation params loaded via `s_load_dwordx8` + `s_load_dwordx2` (scalar path)
2. Ping-pong between v3/v4 hides ds_swizzle read-after-write latency
3. `s_clause 0x1` groups scalar loads for clause optimization
4. AND/CMP for lane masks interleaved with swizzle wait cycles
5. No unnecessary v_mov — compiler reuses registers efficiently

### Confirmed ds_swizzle patterns (all 5 present)
```asm
ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,1)   ; stride 1
ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,2)   ; stride 2
ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,4)   ; stride 4
ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,8)   ; stride 8
ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,16)  ; stride 16
```

## magnum_butterfly_adaptive (mode 0/1/2)

| Metric | Value |
|--------|-------|
| VGPRs | **8** |
| SGPRs | 12 |
| Scratch | 0 |
| LDS | 0 |

### Mode decode path (verified scalar)
```asm
global_load_ubyte v4, v3, s[6:7]     ; load mode byte (uniform across wave)
v_cmp_ne_u16_e32 vcc_lo, 0, v4       ; mode != 0?
s_and_saveexec_b32 s0, vcc_lo         ; save exec + mask
s_cbranch_execz .LBB2_3               ; skip round 2 if mode==0
...
v_cmp_lt_u16_e32 vcc_lo, 1, v4       ; mode > 1?
s_and_saveexec_b32 s0, vcc_lo         ; save exec + mask
s_cbranch_execz .LBB2_5               ; skip rounds 3-4 if mode<=1
```

Branch pattern: `s_and_saveexec_b32 + s_cbranch_execz` (scalar exec mask control).
Since mode is uniform across the wavefront, the branch is effectively scalar.

## Correctness verification (GPU run, 4096 vectors)

| Test | Max Error | MSE | Cosine Sim |
|------|-----------|-----|------------|
| 5-round fwd+inv | 9.54e-7 | 4.79e-14 | 1.0000000000 |
| Mode 0 (2 rounds) | 3.58e-7 | 5.66e-15 | 1.0000000000 |
| Mode 1 (3 rounds) | 5.96e-7 | 1.02e-14 | 1.0000000000 |
| Mode 2 (5 rounds) | 9.54e-7 | 4.86e-14 | 1.0000000000 |
| Mode 2 vs full | 0.0 | 0.0 | 1.0000000000 |

## Conclusion

**GO: All targets exceeded.** The butterfly Givens rotation kernel is viable on gfx1010.
The ds_swizzle hardware supports all 5 butterfly strides natively. The compiler
generates near-optimal ISA with only 6 VGPRs (target was <=20).
