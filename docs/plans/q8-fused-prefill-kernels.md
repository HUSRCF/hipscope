# Q8_0 Fused Prefill Kernels — Production Throughput Plan (Tier 3)

**Status:** Draft (rev 2, post combined adversarial review 2026-05-13).
**Targets:** gfx1100 (Navi 31, 7900 XTX) and gfx1151 (Strix Halo APU) primary. gfx1200/1201 (RDNA4) and gfx906 (CDNA1) — Experimental / blind-port, see §Out of scope.
**Base:** `feat/q8-prefill-tier2` (the Tier 2 unfused dispatch wiring) once landed; otherwise `master`.
**Branch:** `feat/q8-fused-prefill-kernels`.
**Estimated effort:** 2–3 weeks for one kernel-experienced contributor (mirrors the original MQ4 batched-family wall time; Q8's mechanically simpler dequant offsets the loss of the FWHT-free preamble shortcuts).
**Companion docs:**
- `docs/plans/hfp4-mfp4-rdna3-accel.md` — sibling plan, same template (port MQ4 batched-prefill family to a new quant format).
- `docs/plans/issue-113-quant-quality-eval.md` — the eval that drives the Q8 noise-floor requirement.
- `docs/plans/qwen35-mq4-quality-gap.md:705,959` — the 0.5735 KLD floor anchor and methodology context.
- `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/` — the raw eval result this plan must stay numerically consistent with.

(A scoping doc covering Tier 1's failure mode and the Tier 2 wiring lives on `feat/mq-v2-quant-format` as `docs/plans/q8_batchable.md`. It's not on master; the relevant bits — the silent-corruption failure mode and the KLD-window validation oracle — are repeated inline below so this plan is self-contained.)

## Why now

Tier 2 (just-landed on `feat/q8-prefill-tier2`) wired `DType::Q8_0` into the qwen35 batched-prefill dispatch via the existing `gemm_q8_0_batched` kernel, sub-batched at `MAX_BATCH=64` (kernel default since 2026-05-13). This **closed the ~7 % per-token-vs-prefill kernel-path bias** that was contaminating `KLD(MQ4_prefill) − KLD(Q8_per-token)` comparisons in the quant-quality eval, but **did not deliver real throughput**:

```
qwen3.5-9b.q8f16 gfx1151 kv=asym3 chunk-1 smoke (2026-05-13):
  per-token mode (existing floor path)         KLD 0.10811   11 tok/s
  prefill mode, Tier 2 batched (new path)      KLD 0.06913   12 tok/s
  prefill mode, HIPFIRE_PREFILL_BATCHED=0      KLD 0.10811   10 tok/s
```

`gemm_q8_0_batched` was authored as a substrate — its own header says it benchmarks "roughly equal to the serial-GEMV-with-staged-output path on a 4B Q8 lm_head", and was kept as a base for "further tuning (smarter X layout, shared-memory weight broadcast)". For Tier 2 (eval methodology fix), that's enough. For Q8 to become a shippable prefill target at MQ4-adjacent speeds, we need the same fused-projection / WMMA / dot4 family that MQ4 already has.

This plan ports that family to Q8_0.

(Earlier Tier-1 work that just flipped `is_batchable_la` to include Q8 without writing per-dtype dispatch arms produced **silent corruption** — HFQ4-stride kernels reading Q8_0-stride bytes, output KLD = 0 and PPL = NaN. The Tier 2 wiring fixed this for the dispatch surface; the same all-or-nothing kernel-vs-stride pitfall applies here and is enumerated in §Risks.)

## Q8_0 format recap (what kernels see)

```
[ fp16 scale (2B) | int8[32] (32B) ]  per 32-element block
  ↑ block stride = 34 B, weights/byte = 1.0625
```

Compared to MQ4G256 / HFQ4G256:

| | MQ4G256 / HFQ4G256 | Q8_0 |
|---|---|---|
| Group size (K elems) | 256 | 32 |
| Bytes/group | 136 / 104 | 34 |
| Bytes/weight | 0.531 / 0.406 | 1.0625 |
| Element format | int4 + scale + zero | int8 + scale (symmetric) |
| FWHT rotation | yes (offline) | no |
| Dequant per element | unpack + scale + offset | `scale * int8` |

Implications for kernel design:

- **No nibble unpacking** — Q8 dequant is mechanically simpler than HFQ4. Inner loop options: `v_dot4_i32_i8` (sdot4) on RDNA / gfx906, `v_wmma_i32_16x16x16_iu8` (INT8 WMMA) on RDNA3+, or int8→fp16 cast + `v_wmma_f32_16x16x16_f16` (FP16 WMMA). The choice is open — see §Element format choice.
- **No FWHT** — Q8 activations are *not* pre-rotated. The dispatch wiring in Tier 2 already handles this via the existing `else { rmsnorm_batched }` / `else { silu_mul_f32 }` HFQ-family arms. No new preamble kernels needed.
- **2 × the bandwidth pressure of MQ4** — Q8 weights/byte is 1.0625 vs MQ4's 0.531. Q8 prefill is inherently more BW-bound than MQ4 on the same model. The structural performance ceiling is lower; see §Performance targets.
- **8 × the scale-load frequency** — 32-elem block vs MQ4's 256-elem group. Each K-step reads a fresh fp16 scale. ALU cost is small per scale, but cumulative — and `s_waitcnt` overhead can show up. The stride is also 34 bytes, **not power-of-2-aligned**; unaligned 16-bit scale loads will hurt unless `s_buffer_load` / `v_buffer_load` is used carefully.

## Goals & non-goals

**Goals.**

1. Q8_0 reaches the batched-prefill path on gfx1100 and gfx1151 with fused kernels matching the HFQ4 family shape.
2. Numerical equivalence to the Tier 2 chunked-GEMV path within fp16 tolerance (`max_abs_rel_diff < 1e-3`). Byte-exact is not a goal — see §Numerical equivalence for why.
3. Coherence-gate PASS on a new Q8 test case (see T3-0 prerequisite).
4. Closes the prefill throughput gap to the BW-bound ceiling. Concrete targets in §Performance targets.

**Non-goals.**

1. **MoE+Q8 batched dispatch.** Tier 2's eligibility predicate at `qwen35.rs:3712–3716` filters Q8 out of `LayerWeights::DeltaNetMoe` and `FullAttnMoe` arms, mirroring the existing `mq3_in_moe` guard. The MoE dispatch arms remain HFQ4-stride-only — adding Q8 arms here would re-introduce the Tier-1 silent-corruption failure mode unless done together with all MoE FFN routing kernels. Out of scope for this plan; gated by user request.
2. **Q8_HFQ / split-metadata variants.** `gemv_q8hfq` is a separate format with 128-B-aligned rows — not in scope here.
3. **Q8 weight quantization quality work.** Q8 vs FP16 is ~0.001–0.005 KLD in literature; this plan is throughput-only. Quality-side work belongs in `qwen35-mq4-quality-gap.md`.
4. **DFlash speculative decoding with Q8 weights.** The Tier 2 substrate (`gemm_q8_0_batched`) preserves single-accumulator FMA order specifically for greedy parity with `gemv_q8_0` (kernel header at `gemm_q8_0_batched.hip:21-23`). The fused kernels in this plan will have different reduction order. DFlash + Q8 is therefore **explicitly out of scope** — if it becomes a target, either the fused kernels need a parity-preserving mode, or the substrate stays as the decode path. See §Constraints.

## Constraints

**Greedy-parity invariant (substrate-only).** The Tier 2 substrate kernel matches `gemv_q8_0`'s single-accumulator reduction order. The fused kernels in this plan **will not** preserve that invariant — WMMA reduction order is hardware-determined. Consequence: the substrate remains the canonical Q8 decode path, and DFlash+Q8 is out of scope (see Non-goal #4). If DFlash+Q8 becomes a target later, the substrate must remain on master as the decode-time kernel even after Tier 3 lands.

**MoE+Q8 silent-corruption barrier.** The Tier 2 filter at `qwen35.rs:3712–3716` is a fragile one-line guard against the Tier-1 failure. Any change touching MoE dispatch arms must preserve it OR add Q8 arms to all of qkvza/qkv/gate_up/residual MoE sites simultaneously. The "all-together corruption-prevention" rule from `docs/plans/mq-lloyd-batched-prefill-followup.md` applies.

## Kernel surface

Mirroring `kernels/src/gemm_*_hfq4g256*.hip`, the Q8 family needs the same four projection-shape kernels, each with two arch variants on the WMMA path (RDNA3 wave32 + RDNA4 gfx12 sibling) plus a non-WMMA dot4 fallback for gfx1010/1030. This is a **minimum viable surface** — the full HFQ4 family has 7+ arch variants per op (fp16, wmma_ldsx, wave64, dot2, mmq-x8-through-x64); we intentionally start with the smallest set that delivers the target throughput on gfx11.

| Op family | Dispatch site (qwen35.rs Q8 arm, post-Tier-2 lines) | Template kernel | New Q8 kernels |
|---|---|---|---|
| 4-way fused LA QKV+z+α | DeltaNet preamble Q8 arm (~4341) | `gemm_qkvza_hfq4g256_wmma.hip` | `gemm_qkvza_q8_0.hip` (dot4 fallback) + `_wmma.hip` (gfx11) + `_wmma.gfx12.hip` (gfx12) |
| 3-way fused FA QKV | FullAttn preamble Q8 arm (~4753) | `gemm_qkv_hfq4g256_wmma.hip` | `gemm_qkv_q8_0.hip` + `_wmma.hip` + `_wmma.gfx12.hip` |
| 2-way fused FFN gate+up | DeltaNet ~4586, FullAttn ~5059 | `gemm_gate_up_hfq4g256_wmma.hip` | `gemm_gate_up_q8_0.hip` + `_wmma.hip` + `_wmma.gfx12.hip` |
| Residual GEMM (wo, w_down) | DeltaNet wo ~4533 / w_down ~4653; FullAttn wo ~5007 / w_down ~5117 | `gemm_hfq4g256_residual_wmma.hip` | `gemm_q8_0_residual.hip` + `_wmma.hip` + `_wmma.gfx12.hip` |

**Total new kernels: 12** (4 ops × 3 variants). Implementors must re-verify the line numbers against current HEAD before editing — every prior touch of `qwen35.rs` shifts these. The Q8 arm itself is the anchor; if the line drifted, search for `qkv_is_q8` / `wo_is_q8` / `ffn_is_q8` / `w_down_is_q8`.

Plus dispatch arms in:
- `crates/rdna-compute/src/dispatch.rs` — one helper fn per kernel, parallel to the existing `gemm_qkv_hfq4g256` family.
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — replace the Tier 2 `gemm_q8_0_batched_chunked` arms with the fused calls. Per-arch routing inside each Q8 arm: WMMA on gfx1100/1101/1102/1150/1151, fallback dot4 elsewhere. Mirror the existing `is_mq3 && arch_has_wmma` selector pattern.
- `crates/hipfire-runtime/src/llama.rs` — same arms in the plain-Llama prefill path.

## Element format choice — FP16 vs INT8 WMMA

RDNA3 offers two relevant WMMA builtins:

- **`v_wmma_f32_16x16x16_f16`** — FP16 inputs, F32 accumulator. Requires int8→fp16 cast in the prologue (3 ops per byte: `v_sext_i8_to_i32` → `v_cvt_f32_i32` → `v_cvt_f16_f32`, or 2 ops via the int16-pack trick).
- **`v_wmma_i32_16x16x16_iu8`** — INT8 inputs, I32 accumulator. No cast needed for weights, but activations must be online-quantized to int8 per WMMA tile (requires a per-tile activation scale that gets folded out at the i32→f32 store).

For Q8 the choice matters because:

- INT8 WMMA: weights stream as int8 directly; per-block fp16 scale folded once at output stage. Activation int8 quant cost is per-WMMA-tile, amortizing across the K dimension.
- FP16 WMMA: weights need the int8→fp16 cast (paid per K-element), but activations don't need online quant — they flow as fp16 (or stay fp32 with a cast).

There is no a-priori winner. The HFQ4 family chose FP16 WMMA because nibble unpack pairs naturally with FP16; Q8 has no such asymmetry.

**T3-1 microbench gate:** before the recipe is locked, build a minimal `gemm_qkv_q8_0_wmma.hip` in *both* variants (FP16-WMMA + INT8-WMMA) and bench in isolation. The faster variant wins the recipe; the loser becomes a stretch follow-up.

## Recipe (mirrors HFQ4G256 WMMA, register-only)

The HFQ4 WMMA template uses **register-redundant dequant — not LDS staging.** Threads in a wave32 redundantly dequantize the weight tiles they need into registers; the wave-broadcast nature of the WMMA op + the small per-thread tile size means LDS is wasted overhead. Q8's even simpler dequant doubles down on this — no reason to add LDS traffic for `scale * int8`.

For the WMMA variant (final recipe pending T3-1 outcome), per WMMA tile (16×16 K-block):

1. **Dequant in registers.** Cooperative wave32 load: 32 lanes × 1 byte = 32 int8 weights from one block. Multiply by the broadcast fp16 scale; cast (FP16 path) or keep as int8 (INT8 path).
2. **WMMA inner loop.** Issue `v_wmma_*` with the per-thread weight tile and the per-thread x tile, accumulating into f32 (FP16 path) or i32 (INT8 path) registers.
3. **Output stage.** Fold per-block scale (INT8 path only — FP16 path already folded it in step 1). Write f32 outputs.

No `__syncthreads()`, no LDS. The HFQ4 WMMA family lives without either; Q8 should too.

For the non-WMMA (dot4) variant — used on gfx1010/1030 and as the universal fallback:

1. **Dequant inline.** Load `i32` (4 int8 weights packed), `v_dot4_i32_i8` against `i32`-packed x. Same trick `gemm_q8_0_batched` already uses.
2. **Per-block scale fold.** Multiply accumulator by f16 scale at block boundary.
3. **Wave reduction.** 32-lane sum to one output row.

**Residual fusion is a requirement, not an open question.** The HFQ4 family fuses residual-add into `gemm_hfq4g256_residual` (`x_batch += W·input` atomic). Tier 2's Q8 path used GEMM-into-scratch + separate `add_inplace_f32`, which is 64 extra kernel launches per chunk (32 layers × 2 residual sites × 1 add-launch each), and at prefill chunk sizes the launch overhead is measurable. The fused `gemm_q8_0_residual` family must fuse the add.

## Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| T3-0 | Extend `coherence-gate.sh` with a 9B q8f16 test case (`--scoring-mode prefill`, kv=asym3). Verify it currently FAILS on the substrate path (or PASSES — establish baseline). | New test case present, baseline behavior recorded. |
| T3-1a | INT8-WMMA vs FP16-WMMA microbench in `crates/rdna-compute/examples/bench_q8_wmma_variants.rs`. | Bench numbers committed; faster variant declared the recipe. |
| T3-1b | `gemm_qkv_q8_0_wmma` (gfx1100/1151), recipe per T3-1a. | Unit-test PASS in `crates/rdna-compute/examples/test_gemm_q8_qkv_wmma.rs`: `max_abs_rel_diff < 1e-3` vs `gemm_q8_0_batched_chunked` reference. **No throughput claim.** |
| T3-2 | `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma`. | Per-op unit tests (4 separate files in `examples/`): each `max_abs_rel_diff < 1e-3` vs Tier 2 substrate composition. |
| T3-3 | Dispatch wiring in qwen35.rs + llama.rs — swap Tier 2 `gemm_q8_0_batched_chunked` arms for fused WMMA calls behind `arch_has_wmma`. | 9B q8f16 prefill ≥ target tok/s on gfx1100 (see §Performance targets); coherence-gate PASS on Q8 test case from T3-0; slice-mean KLD over the full 256-chunk eval lands within 0.04 of the per-token floor (0.5735, sourced from `qwen35-mq4-quality-gap.md:705`). |
| T3-4 | Non-WMMA `gemm_*_q8_0.hip` siblings (for gfx1010/1030 / universal fallback). | Per-op unit tests PASS; dispatch route falls back to dot4 on non-WMMA arches; coherence-gate PASS on a non-WMMA arch if one is available, else env-gated and noted. |

Each phase is a single commit + bench row in `docs/perf-bench/`. Per `docs/methodology/perf-benchmarking.md`, every claimed win is verified across a fresh process via `scripts/probe_commits.sh PARENT HEAD` before commit. Coherence-gate (extended in T3-0) runs automatically via `.githooks/pre-commit`.

Tier 2's `gemm_q8_0_batched_chunked` stays on master after Tier 3 as **(a)** the decode-path Q8 kernel (greedy-parity preserved for any DFlash+Q8 follow-up; see §Constraints) and **(b)** the fallback for any future projection that doesn't get a fused variant.

## Numerical equivalence test

`crates/rdna-compute/examples/test_gemm_q8_qkv_wmma.rs` (new) — and one sibling per op (qkvza, gate_up, residual). Mirrors the existing HFQ4 test layout (5 separate test binaries, not one monolithic file). Each test:

1. Generate a fixed-seed random Q8_0-quantized weight matrix [M, K] and a random f32 activation [N, K].
2. Run the new fused kernel → Y_new [N, M].
3. Run `gemm_q8_0_batched_chunked` (Tier 2 substrate) over the same weights → Y_ref [N, M].
4. Assert `max_abs(Y_new - Y_ref) < 1e-3 * max_abs(Y_ref)` — **fp16 tolerance, not byte-exact**. The substrate uses single-accumulator FMA order (matches `gemv_q8_0`); WMMA reduces differently. Byte-exact is impossible by construction.
5. Sweep N ∈ {1, 4, 16, 32, 64, 128, 256}; M ∈ {dim, 4·dim}; K ∈ {dim, 4·dim}. Same K-sweep matrix the existing HFQ4 tests use.
6. Add an "every-int8-value-once" weight pattern test (one block stores int8 [-128..127] in order) to catch sign-extension regressions on the dot4 path.

## Performance targets

**Pure-BW ceiling derivation (gfx1100, HBM3 960 GB/s, 9B Q8 weights = 9.5 GB):**

- One forward pass reads all weights once → 9.5 GB / 960 GB/s = **9.9 ms / fwd**.
- Prefill chunk size is `PREFILL_MAX_BATCH = 256` (per `qwen35.rs:3309`); each fwd processes one chunk.
- 256 tokens / 9.9 ms = **25 900 tok/s** pure-BW ceiling on gfx1100.

**MQ4-relative-efficiency anchor:**

- MQ4 on the same hardware achieves 1 134 tok/s (per `hfp4-mfp4-rdna3-accel.md`).
- MQ4 weights = 5.3 GB → MQ4 pure-BW ceiling at chunk=256 = 256 / (5.3/960e9) ≈ 46 400 tok/s.
- MQ4 BW utilization = 1134 / 46400 ≈ **2.4%** of pure-BW ceiling. (MQ4 is *not* BW-bound on gfx1100 at this chunk size — dispatch + ALU + LDS dominate.)
- Q8 at the same 2.4% efficiency: 0.024 × 25 900 ≈ **620 tok/s**.

**However:** Q8 has structural overheads MQ4 doesn't — 8 × more fp16 scale loads, unaligned 34-byte stride, no FWHT-fused preamble shortcut. A reasonable expectation is Q8 falls to ~70-80% of MQ4's BW efficiency, giving **440–560 tok/s** as the realistic operating range.

The simpler "2× BW pressure → halve MQ4's measured rate" rule (per Gemini's review) gives 1134 / 2 = **567 tok/s** as a clean back-of-envelope. The two derivations converge in the 500–600 range; either anchor is reasonable.

**Acceptance criteria for T3-3:**

- gfx1100: 9B q8f16 prefill ≥ **500 tok/s** (~42× over Tier-2's 12 tok/s). 540 tok/s is target; 600 tok/s is stretch.
- gfx1151: 9B q8f16 prefill ≥ **40 tok/s** (~3.3× over Tier-2's 12 tok/s; LPDDR5x-bound on Strix Halo's ~256 GB/s shared bus). Remeasure MQ4 baseline on this hardware before locking the target — the 199 tok/s number in CLAUDE.md is on gfx1100.

If we miss the gfx1100 floor, the failure mode is likely **register pressure spilling to scratch** in the WMMA variant (mitigation: per-tile register reuse tuning, see HFQ4's `_wmma_k4` / `_wmma_ksplit` precedents) OR **non-amortized scale broadcast cost** (mitigation: prefetch scales one block ahead of the WMMA issue).

## Tuning the substrate kernel (`gemm_q8_0_batched`)

The Tier 2 work bumped `MAX_BATCH` from 16 to 64 (kernel default since 2026-05-13). Register metadata on gfx1151 (per the `gfx-kernel-metadata` skill recipe):

| | MAX_BATCH=16 | MAX_BATCH=64 |
|---|---|---|
| VGPRs/wave | 26 | 76 |
| SGPRs/wave | 60 | 107 |
| VGPR spill count | 0 | **0** |
| Private (scratch) bytes | 0 | **0** |
| SGPR values relocated to VGPR lanes | 0 | 98 (via `v_writelane`; no scratch memory traffic) |
| Waves/SIMD on RDNA3 (1024 VGPR/SIMD) | 16 (VGPR-bound at max) | **SGPR-bound** — exact figure depends on RDNA3 SGPR allocation granule, but the VGPR-only 12 waves/SIMD figure originally claimed in this plan is too optimistic. Lower bound from a 16-byte granule estimate: ≤ 9 waves/SIMD. Verify against the AMD ISA reference before defaulting either way. |

MAX_BATCH=64 is safe register-wise (zero spill-to-memory). The occupancy hit is materially larger than 25% once SGPR pressure is accounted for. Whether the 4× weight-load amortization wins net is **still an empirical question**, blocked on a GPU-clean bench (Tier 2 didn't get one — llama.cpp held the GPU during the original smoke). Run the bench as part of T3-0 prerequisites and revert MAX_BATCH if needed.

**Cleanup item (passing):** `crates/hipfire-arch-qwen35/src/speculative.rs:2095` has a stale comment claiming "MAX_BATCH=16 in the kernel" and a local `Q8_LM_MAX: usize = 16`. Pre-existing bug, not introduced by this plan, but fix in passing when touching adjacent code.

## Risks

1. **Register pressure on the WMMA path.** Q8's 32-elem block stride means the WMMA tile dequant prologue runs 8× more often per K-step than HFQ4's. Per-tile registers add up. Mitigation: precedent in the HFQ4 family's `_k4` / `_ksplit` variants (split the K dimension to reduce per-tile live ranges).
2. **Unaligned scale loads at 34-byte stride.** The fp16 scale at offset 0 of each 34-byte block doesn't sit on a 4-byte boundary across all blocks. Mitigation: use `s_buffer_load_short` / `v_buffer_load_short` with explicit offsets, OR reorder the kernel to load scales contiguously via a parallel scale array (would need a Q8 storage-layout follow-up — not Tier 3 scope).
3. **Reduction-order divergence from substrate.** The WMMA path reduces in a different order than `gemm_q8_0_batched`. Acceptable per §Constraints (greedy-parity invariant scoped to substrate / decode); flagged here so future readers don't accidentally re-couple the two paths.
4. **MoE+Q8 silent corruption regression.** Tier 2 filter (`qwen35.rs:3712–3716`) is fragile. Any MoE dispatch refactor must preserve it or fail loudly. Mitigation: add a runtime assertion in any new MoE dispatch path; treat as `mq3_in_moe`-class barrier.
5. **Compile time / JIT cache growth.** 12 new kernels enter the cache. Should be fine relative to the HFQ4 family's 30+, but worth noting if iteration time visibly degrades.
6. **lm_head fan-out cost.** Today the eval path issues per-row `weight_gemv` (`eval_hipfire.rs:417`) for the 1023-token-per-chunk scored region. With vocab=248K and chunk=256, that's a lot of GEMV calls. If T3-3 lands and prefill is fast but eval throughput is still mediocre, lm_head fan-out is the next bottleneck. Profile before declaring out of scope.

## Open questions (resolve before T3-1b)

1. **FP16-WMMA vs INT8-WMMA recipe.** Resolved by T3-1a microbench (see §Element format choice).
2. **gfx1151 MQ4 prefill baseline.** The 1134 tok/s anchor is gfx1100. We need a gfx1151 MQ4 prefill number on this hardware (clean bench, no llama.cpp contention) to set the gfx1151 Q8 target floor. Per `feedback_rebaseline_before_cross_arch_compare.md`.
3. **Scale-broadcast prefetch.** Does prefetching the next block's fp16 scale into a register one iteration ahead help, or does the existing memory pipeline absorb the load latency? Bench in T3-1b once the recipe is fixed.

## Out of scope

- **gfx12 (RDNA4) WMMA siblings.** Originally listed as a primary target. RDNA4 lacks verified hardware in this project's CI loop, and `v_wmma_*_w32_gfx12` has different lane layout than RDNA3. Shipping kernels that never touched silicon risks silent corruption or driver hangs. **Reclassified to Experimental / Blind Port:** when a gfx12 user with the hardware shows up, port via the `gemm_*_hfq4g256_wmma.gfx12.hip` precedent (`gemm_qkv_hfq4g256_wmma.gfx12.hip`, `gemm_hfq4g256_residual_wmma.gfx12.hip`), guard behind `HIPFIRE_LLOYD_GFX12`-style env gate, and require explicit hardware-verified coherence-gate before promoting to default.
- **gfx906 (CDNA1, MI50) wave64 + dp4a Q8 path.** Same precedent as `hfp4-mfp4-rdna3-accel.md`'s gfx906 carve-out. Reference kernel for this family would be the residual variant `gemm_hfq4g256_residual_wave64_dp4a.hip` (note: the gfx906 MMQ x8–x64 sweep family lives in `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip`). Defer until a gfx906 user materializes.
- **MoE+Q8 batched dispatch.** Per §Non-goals. Needs an additional 4+ fused kernels for the MoE FFN routing path, plus relaxation of the Tier 2 eligibility filter.
- **lm_head Q8 batched dedicated kernel.** Today fans out via per-row `weight_gemv`. If T3-3 ships and lm_head becomes the eval bottleneck, write `gemm_lm_head_q8_0` (M = vocab = 248K is the difficult case). Likely not needed; profile first.
- **Q8 KV cache + Q8 weights joint optimization.** Current eval uses asym3 KV. Memory-budget-constrained devices may want both — separate workstream.
- **Storage-layout follow-up to remove the 34-byte stride.** A `Q8_0_packed` variant with scales in a separate row-major array and weights contiguous would eliminate the unaligned-load issue. Format-change work, not kernel work; out of scope.

## Memory + bench discipline

- Per `feedback_rebaseline_before_cross_arch_compare.md`: every perf claim must be measured **on this hardware** in a fresh process. No transplanting historical numbers across gfx targets — the 1134 tok/s MQ4 reference was on gfx1100, not gfx1151.
- Canonical bench config: 9B Qwen3.5, `.q8f16`, `--kv-mode asym3`, `--scoring-mode prefill`, full 256 chunks (≈ 261 888 scored positions) for KLD-bound claims; smoke uses `--max-chunks 1` for throughput-only checks. PREFILL_MAX_BATCH=256 is the default chunk size; if a downstream caller uses a smaller chunk, document that explicitly in the bench row.
- Multi-agent GPU coordination via `gpu-lock.sh` (auto-acquired by hooks). Llama.cpp / external GPU consumers must be confirmed idle via `rocm-smi` or `gpu_status` before throughput numbers count.
- The `0.5735 ± 0.04` KLD window from the canonical Q8 floor cohort (`qwen35-mq4-quality-gap.md:705,959`, results at `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/`) is the validation oracle. Tier 3 must stay inside it on a full 256-chunk run.
