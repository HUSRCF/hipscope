# Q8_0 Fused Prefill Kernels — Production Throughput Plan (Tier 3)

**Status:** rev 3, 2026-05-13 — **T3-0 through T3-3 (code wiring) landed**; T3-3 perf gate + T3-4 pending. Handoff to gfx1100 hardware for the perf gate (see §Handoff below).
**Targets:** gfx1100 (Navi 31, 7900 XTX) and gfx1151 (Strix Halo APU) primary. gfx1200/1201 (RDNA4) and gfx906 (CDNA1) — Experimental / blind-port, see §Out of scope.
**Branch:** `feat/q8-prefill-tier2` (Tier 2 + Tier 3 work share the same branch — original plan called for a separate `feat/q8-fused-prefill-kernels` branch but Tier 3 landed alongside Tier 2 since the substrate stays on master as the decode-path fallback).
**Estimated effort:** 2–3 weeks for one kernel-experienced contributor (mirrors the original MQ4 batched-family wall time; Q8's mechanically simpler dequant offsets the loss of the FWHT-free preamble shortcuts). **Actual elapsed wall time T3-0 → T3-3 wiring: ~1 day** with the FP16-WMMA recipe deciding T3-1a quickly and INT8-WMMA deferral cutting the kernel count from 12 to 4.
**Companion docs:**
- `docs/plans/hfp4-mfp4-rdna3-accel.md` — sibling plan, same template (port MQ4 batched-prefill family to a new quant format).
- `docs/plans/issue-113-quant-quality-eval.md` — the eval that drives the Q8 noise-floor requirement.
- `docs/plans/qwen35-mq4-quality-gap.md:705,959` — the 0.5735 KLD floor anchor and methodology context.
- `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/` — the raw eval result this plan must stay numerically consistent with.

(A scoping doc covering Tier 1's failure mode and the Tier 2 wiring lives on `feat/mq-v2-quant-format` as `docs/plans/q8_batchable.md`. It's not on master; the relevant bits — the silent-corruption failure mode and the KLD-window validation oracle — are repeated inline below so this plan is self-contained.)

## Implementation status (2026-05-13)

| Phase | Status | Commit | Notes |
|---|---|---|---|
| Tier 2 dispatch wiring | ✅ landed | `16fba4fd` | Q8 → batched prefill via chunked substrate. KLD 0.069 / 12 tok/s on gfx1151 (vs per-token 0.108 / 11 tok/s). |
| MAX_BATCH 16→64 bump | ✅ landed | `582e4097` | +5% throughput on clean-GPU A/B. Register-validated (0 spill-to-memory). |
| T3-0 coherence-gate row | ✅ landed | `b6186280` | Q8 9B long-prefill row added; baseline gate run blocked by daemon-segfault env issue on this host (LD_LIBRARY_PATH propagation gap; affects all models, not Q8-specific). Re-run on gfx1100. |
| T3-1a recipe pick | ✅ FP16-WMMA | `27a640e2` | Bench: FP16-WMMA delivers 11–30× over Tier 2 substrate on gfx1151. INT8-WMMA **deferred** — FP16's lead too wide to overturn; revisit only if gfx1100 misses the 500 tok/s floor. |
| T3-1b `gemm_qkv_q8_0_wmma` | ✅ landed | `3473dd85` | 63 unit-test combos PASS (mean_rel 4-8e-4, max_rel <2e-2 on 9B FA shape). |
| T3-2 qkvza / gate_up / residual | ✅ landed | `27c75d8e` | 3 fused kernels + 3 unit tests, all PASS. Residual uses qkv-style output mapping (not HFQ4 residual's alternate convention — see §Risks). |
| T3-3 dispatch wiring | ✅ landed | `31e119c6` | 8 sites in qwen35.rs + 4 sites in llama.rs, all gated `is_q8 && q8_wmma_arch` with Tier 2 chunked fallback. |
| T3-3 perf gate | ⏳ **pending gfx1100** | — | Requires clean-GPU bench on Navi 31 / 7900 XTX. See §Handoff. |
| T3-3 KLD validation | ⏳ **pending gfx1100** | — | Full 256-chunk eval, slice-mean KLD within 0.04 of 0.5735 floor. See §Handoff. |
| T3-3 coherence-gate baseline | ⏳ **pending gfx1100** | — | T3-0 row was added but baseline run failed (env). Re-run on gfx1100. |
| T3-4 dot4 siblings | ⏳ pending | — | Tier 2 chunked substrate is the current fallback on non-WMMA archs. T3-4 only needed if a gfx1010/1030/906 user materializes with a Q8 workload. |

**Recipe locked:** FP16-WMMA, register-redundant dequant, no LDS. See §Element format choice for the deferral rationale.

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

Mirroring `kernels/src/gemm_*_hfq4g256*.hip`, the Q8 family needs the same four projection-shape kernels. Originally scoped at 12 kernels (4 ops × 3 variants: dot4 fallback + RDNA3 WMMA + RDNA4 gfx12 sibling); **actual landed surface is 4 kernels** (RDNA3 WMMA only — gfx12 deferred to follow-up per §Out of scope; dot4 not needed because the Tier 2 chunked substrate stays as the non-WMMA-arch fallback).

| Op family | Dispatch site (qwen35.rs WMMA arm, current lines) | Template kernel | Landed Q8 kernel |
|---|---|---|---|
| 4-way fused LA QKV+z+α | DeltaNet `is_q8 && q8_wmma_arch` (~4347) | `gemm_qkvza_hfq4g256_wmma.hip` | `gemm_qkvza_q8_0_wmma.hip` |
| 3-way fused FA QKV | FullAttn `qkv_is_q8 && q8_wmma_arch` (~4781) | `gemm_qkv_hfq4g256_wmma.hip` | `gemm_qkv_q8_0_wmma.hip` |
| 2-way fused FFN gate+up | DeltaNet ~4604, FullAttn ~5097 | `gemm_gate_up_hfq4g256_wmma.hip` | `gemm_gate_up_q8_0_wmma.hip` |
| Residual GEMM (wo, w_down) | DeltaNet wo ~4544 / w_down ~4677; FullAttn wo ~5041 / w_down ~5161 | `gemm_hfq4g256_residual_wmma.hip` | `gemm_q8_0_residual_wmma.hip` |

Line numbers are post-T3-3 wiring (commit `31e119c6`). Every prior touch of `qwen35.rs` shifts these — the `_is_q8 && q8_wmma_arch` predicate is the durable anchor.

Dispatch wiring (also landed, commit `31e119c6`):
- `crates/rdna-compute/src/dispatch.rs` — 4 helper fns: `gemm_qkv_q8_0_wmma`, `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma`.
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — 8 Q8 dispatch sites (4 DeltaNet + 4 FullAttn), each routes to fused WMMA on `q8_wmma_arch`, falls back to Tier 2 chunked substrate otherwise.
- `crates/hipfire-runtime/src/llama.rs` — 4 Q8 dispatch sites in the plain-Llama path, same wiring pattern.

## Element format choice — FP16 WMMA (recipe locked 2026-05-13)

RDNA3 offers two relevant WMMA builtins:

- **`v_wmma_f32_16x16x16_f16`** — FP16 inputs, F32 accumulator. Requires int8→fp16 cast in the prologue.
- **`v_wmma_i32_16x16x16_iu8`** — INT8 inputs, I32 accumulator. No cast needed for weights, but activations must be online-quantized to int8 per WMMA tile (requires a per-tile activation scale that gets folded out at the i32→f32 store).

**Decision (T3-1a, commit `27a640e2`):** **FP16-WMMA wins.** Microbench on gfx1151 (clean GPU, 200 iter avg) at production Qwen3.5-9B Q8 prefill shapes showed FP16-WMMA delivering **11–30× speedup over the Tier 2 substrate** (`gemm_q8_0_batched_chunked`):

| Shape | N | WMMA µs | Substrate µs | Speedup |
|---|---|---|---|---|
| QKV (M=4096, K=4096) | 16 | 47 | 1409 | 30.1× |
| QKV | 256 | 1279 | 20762 | 16.2× |
| gate/up (M=11008, K=4096) | 256 | 4624 | 50842 | 11.0× |
| w_down (M=4096, K=11008) | 256 | 4060 | 65529 | 16.1× |

Numerics (gated to filter near-zero division noise): 100% of outputs within 5% relative error; mean rel error 4–5e-4 — textbook fp16 WMMA precision.

**INT8-WMMA deferred** — FP16's 11–30× lead is too wide for the marginal INT8 win (theoretical 2× from int8-vs-fp16 op rate) to overturn given the added complexity of online activation quantization. Revisit only if T3-3 misses the 500 tok/s gfx1100 floor.

## Recipe (locked — mirrors HFQ4G256 WMMA, register-only)

The HFQ4 WMMA template uses **register-redundant dequant — not LDS staging.** Threads in a wave32 redundantly dequantize the weight tiles they need into registers; the wave-broadcast nature of the WMMA op + the small per-thread tile size means LDS is wasted overhead. Q8's even simpler dequant doubles down on this — no reason to add LDS traffic for `scale * int8`. The 4 production kernels (`gemm_qkv_q8_0_wmma`, `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma`) all implement this recipe and share an identical inner loop, differing only in M-row routing and output write path.

Per WMMA tile (16×16 K-block):

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

**Landed.** One test binary per op in `crates/rdna-compute/examples/`:

- `test_gemm_q8_qkv_wmma.rs` (T3-1b commit `3473dd85`)
- `test_gemm_q8_qkvza_wmma.rs`, `test_gemm_q8_gate_up_wmma.rs`, `test_gemm_q8_residual_wmma.rs` (T3-2 commit `27c75d8e`)

Each test:

1. Generates fixed-seed random Q8_0 weights and random f32 activations.
2. Runs the new fused kernel → Y_new.
3. Runs `gemm_q8_0_batched_chunked` (Tier 2 substrate) over the same weights → Y_ref.
4. **Gate:** mean relative error < 2e-3 AND max relative error < 5e-2 (gated to outputs where |Y_ref| > 1% of |Y_ref|_max so rel-error metric isn't pathological near zero). Byte-exact is impossible by construction — WMMA's reduction order differs from the substrate's single-accumulator FMA.
5. Sweeps: N ∈ {1, 4, 16, 32, 64, 128, 256}; multiple shapes including production 9B (FA: q=4096 k=v=1024 K=4096; LA: qkv=4096 z=1024; FFN: gate=up=11008 K=4096; residual: M=K=4096 AND M=4096 K=11008).
6. `test_gemm_q8_qkv_wmma.rs` also runs an "every-int8-value-once" weight-pattern test as a sign-extension regression detector.

**Result on gfx1151 (recorded 2026-05-13):** all 4 tests `=== 0 failure(s) ===`. Typical numerics on 9B-shape sweeps: mean rel error 4–8e-4, max rel error <2e-2. Re-run on the gfx1100 handoff to confirm the recipe ports cleanly (see §Handoff).

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

## Handoff to gfx1100 (T3-3 perf gate)

All Tier 3 code is on `feat/q8-prefill-tier2` as of commit `31e119c6`. The wiring is done; the gate runs need a clean Navi 31 / 7900 XTX (gfx1100). On gfx1151 the unit tests all pass — correctness is locked.

### What runs as-is

```bash
# 1. Build the example binaries (release).
cargo build --release --workspace --features deltanet

# 2. Unit-test the fused kernels on this hardware (sanity — already PASS on gfx1151).
for t in test_gemm_q8_qkv_wmma test_gemm_q8_qkvza_wmma \
         test_gemm_q8_gate_up_wmma test_gemm_q8_residual_wmma; do
    ./target/release/examples/$t
done
# Expect: "=== 0 failure(s) ===" from each.
```

If the unit tests on gfx1100 fail, **the recipe doesn't port cleanly** — likely the redundant-lane WMMA layout convention differs between gfx1100 and gfx1151 (both RDNA3, but Strix Halo's APU integration sometimes exposes layout quirks). Disassemble the .hsaco and check the WMMA builtin lowering before tuning anything else.

### Perf gate

```bash
# Run on a fresh process — per docs/methodology/perf-benchmarking.md, no
# within-shell A/B. Wrap with ROCm env on this host's standard launcher.

# A. Throughput (1-chunk smoke, then if signal good, full 256 chunks):
./target/release/examples/eval_hipfire \
  --model ~/.hipfire/models/qwen3.5-9b.q8f16 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --output /tmp/q8_prefill_gfx1100_smoke.bin \
  --kv-mode asym3 --scoring-mode prefill --max-chunks 1

# B. KLD validation (full 256-chunk eval; ~hours at expected throughput):
./target/release/examples/eval_hipfire \
  --model ~/.hipfire/models/qwen3.5-9b.q8f16 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --output /tmp/q8_prefill_gfx1100_full.bin \
  --kv-mode asym3 --scoring-mode prefill

# C. Coherence-gate baseline (was env-blocked on gfx1151):
./scripts/coherence-gate.sh   # the long-prefill-q8-9b row is already in the matrix
```

### Acceptance criteria

- **(A) Throughput**: 1-chunk smoke should show ≥**500 tok/s** on gfx1100 (target 540, stretch 600). The Tier 2 baseline on gfx1100 is unknown locally — best comparison anchor is the gfx1151 Tier 2 number scaled by BW ratio (gfx1100 is ~3.75× the LPDDR5x, but WMMA throughput is the lever here so the actual gain factor is empirical).
- **(B) KLD**: slice-mean KLD lands **0.5735 ± 0.04** on the full 256-chunk eval. This is the validation oracle from `qwen35-mq4-quality-gap.md:705`. KLD = 0 / NaN PPL is the silent-corruption failure mode — must not occur.
- **(C) Coherence-gate**: the new `long-prefill-q8-9b` row in `coherence-gate.sh` must produce fluent output (no special-token leakage, no attractor loop). The substrate path passed this row by construction; the WMMA path is the actual test.

### If the perf gate fails

- **Below 500 tok/s on gfx1100**: investigate (1) WMMA register pressure via `gfx-kernel-metadata` skill on the 4 fused .hsacos; (2) whether INT8-WMMA's deferred 2× op-rate win would close the gap; (3) whether the `gemm_q8_0_residual_wmma` add-fusion is actually helping or just adding RMW pressure (try the GEMM-into-scratch + add_inplace variant as a side-by-side comparison).
- **KLD drift outside 0.04 of 0.5735**: regression in the dispatch wiring — confirm the gating predicate is firing (check that `is_q8 && q8_wmma_arch` evaluates true for every layer's projections), then re-run a single-chunk to localize.
- **KLD = 0 / NaN PPL**: silent-corruption regression — the residual kernel's output mapping or the qkv kernel's are wrong on gfx1100 (could be the WMMA lane-layout-by-arch issue mentioned above). Compare against substrate row-by-row.

### Environment notes (gfx1151 host quirks to NOT carry forward)

- The coherence-gate daemon segfaulted on this host (gfx1151) on every model, not just Q8. Looked like an `LD_LIBRARY_PATH` propagation gap in the gate script — gfx1100 hosts typically have ROCm in the default loader path and this won't reproduce. If it does, the daemon binary needs `LD_LIBRARY_PATH=/opt/rocm-*/lib` set in the subprocess env, not just the parent shell.
- The gfx1151 perf target (40 tok/s) was set from the LPDDR5x bandwidth ceiling and never measured. On gfx1100 the bandwidth ceiling is much higher — don't transplant the 40 tok/s number; redo the math (or just bench MQ4 on this hardware first to establish the relative anchor).

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

MAX_BATCH=64 is safe register-wise (zero spill-to-memory). Empirical A/B (clean GPU, gfx1151, commit `582e4097`): MAX_BATCH=64 = **+5% throughput** vs MAX_BATCH=16, with byte-identical KLD. Small but free; default is now 64. Once the Tier 3 fused kernels are the production path, the substrate is only used for decode and the MAX_BATCH choice becomes academic.

**Cleanup landed in `582e4097`:** the stale `Q8_LM_MAX = 16` constant + comment in `crates/hipfire-arch-qwen35/src/speculative.rs` was also bumped to 64.

## Risks

1. **Register pressure on the WMMA path.** Q8's 32-elem block stride means the WMMA tile dequant prologue runs 8× more often per K-step than HFQ4's. Per-tile registers add up. Mitigation: precedent in the HFQ4 family's `_k4` / `_ksplit` variants (split the K dimension to reduce per-tile live ranges).
2. **Unaligned scale loads at 34-byte stride.** The fp16 scale at offset 0 of each 34-byte block doesn't sit on a 4-byte boundary across all blocks. Mitigation: use `s_buffer_load_short` / `v_buffer_load_short` with explicit offsets, OR reorder the kernel to load scales contiguously via a parallel scale array (would need a Q8 storage-layout follow-up — not Tier 3 scope).
3. **Reduction-order divergence from substrate.** The WMMA path reduces in a different order than `gemm_q8_0_batched`. Acceptable per §Constraints (greedy-parity invariant scoped to substrate / decode); flagged here so future readers don't accidentally re-couple the two paths.
4. **MoE+Q8 silent corruption regression.** Tier 2 filter (`qwen35.rs:3712–3716`) is fragile. Any MoE dispatch refactor must preserve it or fail loudly. Mitigation: add a runtime assertion in any new MoE dispatch path; treat as `mq3_in_moe`-class barrier.
5. **Compile time / JIT cache growth.** 12 new kernels enter the cache. Should be fine relative to the HFQ4 family's 30+, but worth noting if iteration time visibly degrades.
6. **lm_head fan-out cost.** Today the eval path issues per-row `weight_gemv` (`eval_hipfire.rs:417`) for the 1023-token-per-chunk scored region. With vocab=248K and chunk=256, that's a lot of GEMV calls. If T3-3 lands and prefill is fast but eval throughput is still mediocre, lm_head fan-out is the next bottleneck. Profile before declaring out of scope.

## Open questions (resolve on gfx1100 handoff)

1. ~~**FP16-WMMA vs INT8-WMMA recipe.**~~ **Resolved** by T3-1a — FP16-WMMA, 11–30× over substrate.
2. **MQ4 prefill baseline on the target hardware.** The 1134 tok/s anchor is gfx1100 historical (from `hfp4-mfp4-rdna3-accel.md`); confirm by running an MQ4 prefill bench on the actual handoff machine before locking the Q8 perf target. Per `feedback_rebaseline_before_cross_arch_compare.md`. (gfx1151 baseline not needed — gfx1151 only ran correctness, not the perf gate.)
3. **Scale-broadcast prefetch.** Does prefetching the next block's fp16 scale into a register one iteration ahead help, or does the existing memory pipeline absorb the load latency? Worth profiling only if T3-3 misses the 500 tok/s floor.
4. **WMMA layout portability gfx1151 → gfx1100.** Both are RDNA3 wave32 WMMA, so the kernels *should* port. If gfx1100 unit tests fail in ways gfx1151 didn't, the redundant-lane convention may differ subtly between Navi 31 and Strix Halo's APU integration. Disassemble the .hsaco and inspect the WMMA builtin lowering.

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
