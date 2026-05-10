# Dev log 2026-05-10 — MQ3 batch-tile fanout (`_mb4`), gfx1151

**Branch:** `feat/mq3-batch-fanout` (off `upstream/master` `a4e42c5`).
**Hardware:** gfx1151 (Radeon 8060S, AMD Ryzen AI Max+ 395 / Strix Halo APU,
LPDDR5x ~250 GB/s), ROCm 7.12.
**Sibling work:** MQ4 Phase D — `feat/issue-182-mq4-lloyd` branch landed
the same multi-batch-tile pattern for the MQ4-Lloyd family; this branch
ports the pattern to **HFQ3 (uniform-MQ3) AND MQ3-Lloyd** jointly.

## Motivation

Initial baseline bench (gfx1151, 9B prefill=256, ROCm 7.12):

| Model | prefill tok/s | Lloyd / uniform |
|---|---|---|
| qwen3.5-9b.mq3 (uniform) | 503.7 | — |
| qwen3.5-9b.mq3-lloyd | 494.1 | 98.1 % |

The Lloyd / uniform ratio was already healthy at 98 % — but **both
absolute numbers are roughly half of uniform-MQ4's 1123 tok/s** on the
same hardware. Profile confirmed why: uniform-MQ3 (HFQ3) has no `_mmq`
sibling (unlike HFQ4 which routes batch ≥ 128 to a 128×128 LDS-staged
tile), so HFQ3 stays on the 16×16 WMMA path at all batch sizes —
the same wall MQ4-Lloyd hit pre-Phase-D.

Per-kernel GiB/s on the 9B baseline profile:

| Kernel | HFQ3 GiB/s | MQ3-Lloyd GiB/s |
|---|---|---|
| `gemm_gate_up_*_wmma` | 12.3 | 9.0 |
| `gemm_*_residual_wmma` | 12.8 | 13.4 |
| `gemm_qkvza_*_wmma` | 13.4 | 10.1 |
| `gemm_qkv_*_wmma` | 14.3 | 10.3 |

Both families at 9-14 GiB/s — exactly the regime mb4 fixed for MQ4-Lloyd.

## Approach

8 new kernels following the Phase D `_mb4` template (16×64 output tile
per WG, 4 batch sub-tiles share `a_reg` decode, 4 distinct named B-vars
to defeat the compiler register-recycling trap):

**MQ3-Lloyd `_mb4`** (codebook-based, 256 B LDS):
- `kernels/src/gemm_mq3g256_lloyd_residual_wmma_mb4.hip`
- `kernels/src/gemm_qkvza_mq3g256_lloyd_wmma_mb4.hip`
- `kernels/src/gemm_qkv_mq3g256_lloyd_wmma_mb4.hip`
- `kernels/src/gemm_gate_up_mq3g256_lloyd_wmma_mb4.hip`

**HFQ3 (uniform) `_mb4`** (affine sc/zp, no LDS, no syncs):
- `kernels/src/gemm_hfq3g256_residual_wmma_mb4.hip`
- `kernels/src/gemm_qkvza_hfq3g256_wmma_mb4.hip`
- `kernels/src/gemm_qkv_hfq3g256_wmma_mb4.hip`
- `kernels/src/gemm_gate_up_hfq3g256_wmma_mb4.hip`

HFQ3 mb4 is structurally simpler than the Lloyd siblings — no
cooperative codebook load, no `__syncthreads()`, no LDS. The 4× batch
fanout is the only structural change vs the Phase A WMMA kernel.

Wired through size-gated dispatch on each existing `_wmma` method:
default routes to `_mb4` when `batch ≥ 128 AND total_m ≥ 4096 AND
arch ∈ {gfx1100,1101,1102,1150,1151}`. Env override
`HIPFIRE_MQ3_MB4={0|1}` for A/B benching.

## Parity (gfx1151)

**MQ3-Lloyd `_mb4` vs `_wmma`** (both run with synthetic shapes,
compared against fp64 CPU reference):
- `test_gemm_mq3g256_lloyd_residual_wmma` with `HIPFIRE_MQ3_MB4=1`:
  bit-exact across all 8 residual shapes (max_abs/max_rel/rms identical
  to last digit between `_wmma` and `_mb4`).
- `test_gemm_fused_mq3g256_lloyd_wmma` with `HIPFIRE_MQ3_MB4=1`: bit-exact
  across all 8 fused shapes (qkvza, qkv, gate_up).

**HFQ3 `_mb4` vs `_wmma`** (`test_gemm_hfq3g256_wmma`, GPU-vs-GPU
comparison — both kernels run on identical inputs, output diff
measured directly):

```
GPU: gfx1151
Verifying HFQ3 _mb4 == _wmma at all shapes (TOL=1e-04).
--- residual M=64    K=1024  N=16   max_abs=0.000e0  PASS
--- residual M=64    K=1024  N=64   max_abs=0.000e0  PASS
--- residual M=256   K=4096  N=256  max_abs=0.000e0  PASS
--- residual M=1024  K=4096  N=64   max_abs=0.000e0  PASS
--- residual M=1024  K=12288 N=64   max_abs=0.000e0  PASS
--- qkvza M=(64+16+8+8)     K=1024 N=16   PASS  all=0.000e0
--- qkvza M=(256+32+16+16)  K=4096 N=64   PASS  all=0.000e0
--- qkvza M=(512+64+32+32)  K=4096 N=32   PASS  all=0.000e0
--- qkv   M=(64+64+64)      K=1024 N=16   PASS  all=0.000e0
--- qkv   M=(256+32+32)     K=4096 N=64   PASS  all=0.000e0
--- qkv   M=(512+64+64)     K=4096 N=32   PASS  all=0.000e0
--- gate_up M=(256+256)     K=1024 N=16   PASS  all=0.000e0
--- gate_up M=(1024+1024)   K=4096 N=64   PASS  all=0.000e0
ALL PASS — HFQ3 _mb4 produces output bit-equivalent to _wmma
```

**Bit-exact, zero diff** across all 13 shapes. HFQ3 `_mb4` produces
output literally identical to `_wmma` for the same inputs — no
fp32-reorder envelope opens up, since both kernels use the same
WMMA accumulation order per output (one acc per 16×16 sub-tile;
mb4 multiplexes 4 sub-tiles in time but each acc is independent).
Strongest possible correctness signal — `_mb4` inherits `_wmma`'s
production-validated correctness mechanically.

## Cross-process A/B (gfx1151, 9B prefill=256, asym3, GRAPH=1)

3 fresh process invocations × 2 modes, `--prefill 256 --prefill-runs 3
--gen 30 --warmup 5`.

### qwen3.5-9b.mq3 (uniform / HFQ3)

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen |
|---|---|---|---|---|---|
| `MB4=0` | 521.4 | 516.6 | 516.3 | **518.1 tok/s** | 53.97 |
| `MB4=1` | 907.9 | 920.2 | 916.7 | **914.9 tok/s** | 53.63 |
| Δ      |       |       |       | **+396.8 (+76.6 %)** | -0.3 |

### qwen3.5-9b.mq3-lloyd

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen |
|---|---|---|---|---|---|
| `MB4=0` | 503.7 | 501.8 | 503.1 | **502.9 tok/s** | 49.0 |
| `MB4=1` | 823.4 | 852.8 | 852.1 | **842.8 tok/s** | 48.8 |
| Δ      |       |       |       | **+339.9 (+67.6 %)** | -0.2 |

### qwen3.5-4b.mq3-lloyd

(No 4B uniform-MQ3 model on this host; just the Lloyd self-A/B.)

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen |
|---|---|---|---|---|---|
| `MB4=0` | 980.8  | 997.0  | 985.6  | **987.8 tok/s**  | 70.4 |
| `MB4=1` | 1596.8 | 1596.2 | 1566.3 | **1586.4 tok/s** | 69.9 |
| Δ      |        |        |        | **+598.6 (+60.6 %)** | -0.5 |

Notable: 4B-MQ3-Lloyd post-mb4 (1586 tok/s) is essentially equal to
4B-MQ4-Lloyd post-Phase-D (1607 tok/s) on the same hardware. At 4B
both formats are in the same compute-bound regime; the LPDDR5x
bandwidth ceiling that punishes MQ4-Lloyd's larger format at 9B
isn't fully exposed at 4B.

### Lloyd / uniform-MQ3 ratio

| Phase | HFQ3 prefill | Lloyd prefill | Ratio |
|---|---|---|---|
| Pre-mb4 | 518.1 | 502.9 | 97.1 % |
| Post-mb4 | 914.9 | 842.8 | 92.1 % |

The ratio narrowed slightly (97.1 → 92.1 %) because uniform-MQ3 gained
more than Lloyd (76.6 % vs 67.6 %). Both are still far above the
format-tax floor of ~85 % (Lloyd's 7.7 % weight footprint penalty,
112 B/group vs 104 B/group).

### Comparison with MQ4 (post-Phase-D)

| Model | Prefill tok/s on gfx1151 (post-mb4) |
|---|---|
| qwen3.5-9b.mq4 (uniform) | 1122.9 |
| qwen3.5-9b.mq4-lloyd     |  800.8 |
| qwen3.5-9b.mq3 (uniform) |  914.9 |
| qwen3.5-9b.mq3-lloyd     |  842.8 |

**MQ3-Lloyd post-mb4 (843 tok/s) is now FASTER than MQ4-Lloyd
post-Phase-D (801 tok/s)** on the same hardware — consistent with MQ3's
30 % smaller weight format on bandwidth-bound APUs. MQ3 uniform sits
between MQ3-Lloyd and MQ4 uniform; the latter still has a faster path
because HFQ4 routes through `_mmq` (128×128 LDS-staged tile) at large
batch.

## Size gate validation

Force `MB4=1` at prefill=16 on 9B-MQ3 (uniform) regresses:

| Mode | prefill tok/s |
|---|---|
| `MB4=0` (baseline)             | 479.9 |
| `MB4=1` (force, gate bypassed) | 361.4 (**-24.7 %**) |

Default mode (no env var) at prefill=16 falls through to `_wmma` (batch
< 128) and matches the 479.9 baseline. The 4× WG reduction at small
batch starves the SIMDs — same effect as MQ4 Phase D-C documented.

## Decode regression — none

`gen_tok_s` invariant for both models across MB4 modes:
- 9B-MQ3 (uniform): 53.97 (MB4=0) vs 53.63 (MB4=1) — within ±0.6%
- 9B-MQ3-Lloyd: 49.0 (MB4=0) vs 48.8 (MB4=1) — within ±0.4%

`bw_gib_s` (decode bandwidth) also matches to ±0.5 GiB/s. Confirms
mb4 only affects the prefill batched-LA path; per-token decode goes
through `forward_scratch` + GEMV, untouched.

## What's locked in

- ✅ Bit-exact parity on residual + 3 fused MQ3-Lloyd kernels (16 shapes vs CPU reference)
- ✅ HFQ3 _mb4 == _wmma bit-exact across 13 shapes (`test_gemm_hfq3g256_wmma`, GPU-vs-GPU)
- ✅ Size-gated default routing — production-safe at all prefill sizes
- ✅ Env override `HIPFIRE_MQ3_MB4={0,1}` for A/B benching
- ✅ gfx12 explicitly excluded from `_mb4` selectors
- ✅ 9B: MQ3 +76.6 %, MQ3-Lloyd +67.6 %
- ✅ 4B: MQ3-Lloyd +60.6 % (4B uniform-MQ3 model not on this host)
- ⏳ gfx1100 cross-arch confirmation — bench help-wanted (no local hardware)

## Bench config (reproducer)

```bash
export PATH="/opt/rocm-7.12/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm-7.12/lib:${LD_LIBRARY_PATH:-}"

source scripts/gpu-lock.sh && gpu_acquire "mq3-mb4-bench"

for model in qwen3.5-9b.mq3 qwen3.5-9b.mq3-lloyd; do
  for mb4 in 0 1; do
    for run in 1 2 3; do
      HIPFIRE_MQ3_MB4=$mb4 HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
        ./target/release/examples/bench_qwen35_mq4 \
        ~/.hipfire/models/$model \
        --prefill 256 --prefill-runs 3 --warmup 5 --gen 30
    done
  done
done

gpu_release
```

| Field | Value |
|---|---|
| Bench binary md5 | `5230b8391b4c00659b86b27112947352` |
| Model md5 (9B-MQ3) | `e4412f6dd785e82de2ece629f33ecc14` |
| Model md5 (9B-MQ3-Lloyd) | `ba99df5e20f44a49fe45f2fa3ce05f59` |
| Model md5 (4B-MQ3-Lloyd) | `865a2ab1e7d97ae02b02be95e08cc852` |
| Branch HEAD | `feat/mq3-batch-fanout` (off `upstream/master a4e42c5`) |
| Prompt | N/A — synthetic deterministic token sequence (token ids `0..prefill_len`) |
