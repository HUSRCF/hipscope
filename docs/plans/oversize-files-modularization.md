# Oversize files — modularization plan

**Status:** plan rev-2 (2026-05-09). Combined-review findings (`oversize-plan-rev-combined.md`) folded; bucket layouts grounded in actual prefix counts; effort estimate revised; `coherence-gate-dflash.sh` adopted as canonical gate per AGENTS.md:30.
**Tracking:** TBD (open issue once this plan lands).
**Companion plan:** `size-limit-ci-gates.md` — defines the ≤1500-line target this plan targets.
**Scope:** mechanical splits of oversized `.rs` files into sibling/child modules. **No logic changes, no public-API renames.** Pure file moves with `mod` declarations, plus the unavoidable `pub(crate)` visibility-promotion of cross-module helpers (see §"Internal API surface" below).

## Why now (eventually)

Total workspace src is **94K lines** (including examples); the 4 Tier-1 files sum to **31,257 lines = 33% of the total**:

```
15,137  crates/rdna-compute/src/dispatch.rs       — impl Gpu with 311 pub + 13 private = 324 methods,
                                                    + 11 private free fns + 1 thread_local
 7,202  crates/hipfire-arch-qwen35/src/qwen35.rs   — config / weights / MoE / forward variants /
                                                    graph-capture branches; 4 inline-scoped macros
 4,889  crates/hipfire-arch-qwen35/src/speculative.rs — 6 concerns: oracle, slot-state, DeltaNet
                                                    tape, scratch, draft strategies, spec-step entry
 4,029  crates/hipfire-runtime/src/llama.rs        — CPU dequant + GPU dispatch wrappers + KvCache
                                                    + prefill_forward + the only file with #[cfg(test)]
```

Plus a Tier-1.5 file:

```
 3,253  crates/hipfire-quantize/src/main.rs        — safetensors I/O + 19 quantize variants
                                                    (3 Q4, 2 Q8, 7 MQ, 7 HFQ) + FWHT + CLI
```

The size concentration produces three concrete frictions: full-file rebuilds on incremental compile, merge-conflict clusters on every PR touching the area, and `git blame` flooded with mechanical edits. None is fatal alone; together they tax every change.

## Non-goals

1. **No clever abstractions.** Don't introduce traits or "interface modules". The goal is to move existing code, not to redesign it.
2. **No public API renames.** Public types and functions keep their full path. `pub use` re-exports in the crate root preserve `crate::dispatch::Gpu` style imports. Consumers (other crates, examples) should not need a single edit. **Caveat:** internal API surface DOES grow — see §"Internal API surface" below.
3. **No clippy fix-ups in scope.** Don't take this opportunity to clean up `clippy::` warnings. Each PR is "files moved + visibility-promoted only".
4. **No new features.** Refactoring while landing a feature is the standard way to ship the bug.

## What stays unsplit (deliberately)

- `kernels.rs` (1,291 lines) — flat list of `pub const FOO_SRC: &str = include_str!(...)`. Data, not code.
- `pflash.rs` (1,690), `triattn.rs` (1,073), `dflash.rs` (1,011), `ddtree.rs` (1,121) — single-feature files with internal coupling.
- `hip-bridge/src/ffi.rs` (1,102) — FFI bindings; flat-list-by-API-call is conventional.
- Examples > 1000 lines (`daemon.rs` 3,813; `dflash_spec_demo.rs` 1,647) — examples get more leeway than library code; reading top-to-bottom is part of their value.

## Internal API surface (rev-2 acknowledgment)

The plan rev-1 claim "no API change" was technically true for *public* surface but obscured a real cost: **internal visibility grows** when private helpers cross module boundaries. Specific cases empirically validated:

- **`dispatch.rs`**: 11 private free functions at lines 35–236 (`gemv_rows_override`, `gemv_dp4a_enabled`, `gemv_prefetch_enabled`, `gemv_rows_default`, `has_dot2_f32_f16`, `has_wmma_f16`, `has_wmma_f16_gfx12`, `is_gcn5_wave64`, `has_wave64_native`, `has_mmq_dp4a_or_wmma`, `should_use_mmq`) plus `thread_local! { static LAST_BOUND_DEVICE }` at line 24. Called from methods across all proposed module boundaries.
- **`speculative.rs`**: `argmax_u32` (line 1352) called from 9+ sites across at least 3 proposed module boundaries (`verify_dflash_block_inner`, `spec_step_dflash`, `spec_step_ddtree`, `spec_step_ddtree_batched`, `spec_step_ddtree_path_c`, `sample_residual`).
- **`hipfire-quantize`**: 5 cross-module types (`HFQ_MAGIC`, `HFQ_VERSION`, `QuantType` (24-variant enum at line 1291), `QuantLevel`, `HfqTensor`, `TensorSpill`) needed by every quant-format module.

These items must be promoted to `pub(crate)` (not exposed externally) and given a stable home (a shared `arch_helpers.rs` / `types.rs` / similar). Each PR's checklist requires an explicit "list of visibility promotions" so reviewers can sign off.

## Size target: ≤1500 lines per resulting file

The companion `size-limit-ci-gates.md` enforces a 2500-line cap. **This plan targets ≤1500 lines per resulting file (sub-cap)** so that natural growth post-split doesn't immediately re-violate the gate. The 1500 target also forces an honest split: if a single proposed module would land at 2000+ lines, sub-split it.

## Tier 1 splits (rev-2 — bucket layouts grounded in actual prefix counts)

### 1. `hipfire-quantize/main.rs` (3,253 → 9 files)

**First — sequenced first as a mechanical-machinery rehearsal**, with the honest caveat that this PR's risk surface (CLI tool, offline-only) does NOT match the dispatch.rs PR (which exercises coherence-gate + perf-gate). The rehearsal value is real but bounded — extra vigilance applies on PR #2.

19 quantize functions verified: 3 Q4 + 2 Q8 + 7 MQ + 7 HFQ.

```
crates/hipfire-quantize/src/
├── main.rs                 # CLI parse, dispatch to format-specific quantizers
├── types.rs                # NEW: HFQ_MAGIC, HFQ_VERSION, QuantType (24-variant enum),
│                           #      QuantLevel, HfqTensor, TensorSpill — shared by every
│                           #      quant module. Without this 9th file, the split won't
│                           #      compile (circular deps between q4/q8/mq/hfq).
├── safetensors.rs          # SafetensorsFile, SafetensorsMeta, TensorMeta
├── conversions.rs          # f16/f32/bf16 conversions, to_f32 dispatch
├── fwht.rs                 # cpu_fwht_256, gen_fwht_signs
├── q4.rs                   # 3 fns: quantize_q4f16_g64, quantize_q4k, quantize_q4_as_q8
├── q8.rs                   # 2 fns: quantize_q8f16, quantize_q8hfq
├── mq.rs                   # 7 fns: mq3g256, mq4g256, mq6g256, mq8g256 + Lloyd variants
└── hfq.rs                  # 7 fns: hfq4g256, hfq2g256 + variants
```

**Estimated effort:** 1 day (was 0.5d in rev-1; revised for shared-types module + import replication).

### 2. `dispatch.rs` (15,137 → ~9 files at ≤1500 lines each)

**Worst offender, biggest win.** Empirical method-name-prefix counts (sampled 311 `pub fn` declarations):

```
73  gemm                              ← largest bucket; sub-split required (see below)
52  gemv
24  attention
21  kv (cache reads/writes)
21  graph-capture (verify 6 + replay 6 + begin 3 + end 3 + ensure 3)
19  fused                             ← cross-family; tie-break rule below
 7  embedding
 7  gated
 5  triattn
 5  conv1d
 4  rope
 4  moe
 3  rotate
 …  (~30 misc)
```

Layout:

```
crates/rdna-compute/src/dispatch/
├── mod.rs              # struct Gpu, init/drop, scratch helpers; pub use re-exports.
│                       # Hard cap: ≤500 lines (see §"mod.rs gravity well" below).
├── arch_helpers.rs     # NEW (preliminary): 11 private arch-feature fns (lines 35–236) +
│                       # thread_local LAST_BOUND_DEVICE. Promoted to pub(crate) so
│                       # all sibling impl-blocks can call them. THIS IS THE FIRST PR
│                       # IN THE SPLIT — done before any impl-block split.
├── gemv.rs             # ≈52 methods
├── gemm/               # 73 methods — sub-split by data-type axis to fit ≤1500/file
│   ├── mod.rs          # impl Gpu glue; weight-format dispatch
│   ├── gemm_q4.rs      # q4_0 / q4_k variants
│   ├── gemm_q8.rs      # q8_0 / q8hfq variants
│   ├── gemm_mq.rs      # mq3 / mq4 / mq6 / mq8 (FWHT-based)
│   └── gemm_hfq.rs     # hfq4g128 / hfq2g256 (no FWHT)
├── attn.rs             # attention(24) + kv(21) + rope(4) + rotate(3) + triattn(5) ≈ 57
├── moe.rs              # moe(4) + gated(7) + embedding(7) + conv1d(5) ≈ 23
├── fused.rs            # fused(19) — see tie-break rule below
└── graph.rs            # verify(6) + replay(6) + begin(3) + end(3) + ensure(3) ≈ 21
```

**Cross-family (fused) tie-break rule:** Methods whose name spans families ("fused_X_Y") are filed by the **leftmost-named operation**:
- `fused_gemv_residual` → `gemv.rs`
- `fused_rmsnorm_rotate_for_mq` → `attn.rs` (rmsnorm groups under attn here since there's no separate norm.rs)
- `fused_gate_up_mq3g256_lloyd` → `moe.rs` (gate_up is a gated FFN op)

This is a tie-break, not a perfect classification. Accept ~5% of methods as reasonably-disputable; lock in the rule rather than re-litigating per-method.

**Pre-split sequence:**

1. **PR 2a:** extract `arch_helpers.rs` from `dispatch.rs` (still as one file). Tiny PR; loud-fail; validates the visibility-promotion mechanic in isolation.
2. **PR 2b:** the actual impl-block split, post-`arch_helpers.rs` landing.

**Estimated effort:** 3 days for PR 2b (was 1d in rev-1; revised for: gemm sub-split, feature-gate audit, coherence-gate iteration, perf bench, review cycles). PR 2a adds ~0.5d.

### 3. `llama.rs` (4,029 → 4 files; partial extraction sufficient)

**Highest layering value.** The pure-CPU dequantization functions (`dequantize_q4_0`, `dequantize_q8_0`, `dequantize_q4_k`, `dequantize_q6_k`, `f16_to_f32`, `f32_to_f16`, `convert_q4k_to_q4f16_g64/g32`) sit in a file otherwise full of GPU coupling. Extracting them lets downstream tooling consume dequant logic without pulling in `rdna-compute`.

```
crates/hipfire-runtime/src/llama/
├── mod.rs              # ModelArch, LlamaConfig, prefill_forward (GPU coupling stays)
├── dequantize.rs       # all CPU-only dequant + dtype conversion fns
├── types.rs            # WeightTensor, EmbeddingFormat, LlamaWeights, LayerWeights
└── kv_cache.rs         # KvCache + KvMode types
```

**`load_tensor_f32` decision:** stays in `mod.rs`. It's the GgmlType-dispatcher for the dequant path; it depends on `crate::gguf` (`GgmlType`, `GgufFile`) which is GPU-coupled. Calling cross-module *into* `dequantize.rs` is the right shape — `dequantize.rs` stays pure CPU, the dispatcher stays in the loader.

**`#[cfg(test)] mod tests` block:** llama.rs is the only Tier-1 file with inline tests. The split must keep the `mod tests` block intact (likely in `mod.rs` or moved to whichever new file the tests touch most). PR checklist explicitly verifies tests pass post-split.

`pub use llama::dequantize::*;` in the crate root preserves callers.

**Estimated effort:** 1 day.

### 4. `qwen35.rs` (7,202 → 5+ files)

```
crates/hipfire-arch-qwen35/src/qwen35/
├── mod.rs              # Qwen35Config, Qwen35Weights, Qwen35Scratch types; re-exports
├── config.rs           # config_from_hfq + parsing
├── weights.rs          # load_weights, load_weights_multi, load_*_into helpers (~1000 lines)
├── moe.rs              # MoeFfnWeights handling, free_moe_ffn, moe_ffn_decode variants
├── forward.rs          # forward_scratch (the hot decode path) + graph-capture branches
└── prefill.rs          # forward_prefill_batch + helpers
```

**Pre-split consolidation opportunity:** `forward_scratch_layers` (line 5604) and `forward_scratch_layers_multi` (line 6332) are ~95% identical near-duplicates. Splitting these into separate modules would freeze the duplication into module boundaries. Before opening this PR, decide:
- (a) Unify the two via a generic over single/multi-GPU dispatch (separate prior PR), or
- (b) Document the duplication as intentional (with rationale) so reviewers don't argue.

**Macro handling:** qwen35.rs has 4 `macro_rules!` declarations (`givens_cos_view!` 4022, `givens_sin_view!` 4025, `ct!` 6375, `st!` 6380). All are **inline-scoped to function bodies** — they travel automatically with the function. No module-level macro handling required.

**Defer until in-flight kernel work settles.** This file is touched by every new arch + spec-decode change.

**Estimated effort:** 2 days (was 1d; revised for pre-split consolidation decision + reviewer cycles in volatile area).

### 5. `speculative.rs` (4,889 → 6 files)

**6 concerns identified** (rev-1 missed `draft_strategies`):

```
crates/hipfire-arch-qwen35/src/speculative/
├── mod.rs              # spec_step entry + re-exports
├── oracle.rs           # SeedOracleStats, DdtreeMetaStats, ddtree_logw_cutoff
├── state.rs            # KvMode, ModelSlot, ModelSlotConfig, SpecPair, SpecStepResult
├── deltanet_tape.rs    # DeltaNetSnapshot, GdnTape, DeltaNetTape
├── draft_strategies.rs # NEW: NgramCache (line 1437), PldMatcher (1502), PldMatch (1524)
│                       # — pure-CPU draft prediction helpers, don't fit any other concern
└── scratch.rs          # DdtreeScratch + related allocators; dflash_extract_layer_ids (line 872)
                        # is a 10-line geometry helper called by HiddenStateRingBuffer::new()
                        # — co-locate with scratch.
```

**Visibility promotion:** `argmax_u32` (line 1352) is currently private but called from 9+ sites across at least 3 proposed modules. Must be promoted to `pub(crate)` — recommended location: `mod.rs` (small enough to live there; called from the spec_step entry as well as cross-module callers).

**Defer with qwen35.rs.** Same reasoning — touched by every spec-decode iteration.

**Estimated effort:** 2 days (was 1d; revised for 6th concern + visibility promotion + reviewer cycles).

## Sequencing

PR-per-file (with PR 2 split into 2a + 2b for arch_helpers extraction). No megacommit. Each PR is bisectable in isolation.

| # | Target | When | Cost | Risk |
|---|---|---|---|---|
| 1 | `hipfire-quantize/main.rs` | first — mechanical-machinery rehearsal | 1d | low (CLI, no runtime path) |
| 2a | `dispatch.rs`: extract arch_helpers.rs | after #1 | 0.5d | low (loud-fail; tiny scope) |
| 2b | `dispatch.rs`: impl-block split | after 2a + bake | 3d | low–medium (compile-loud; perf bench guards regression) |
| 3 | `llama.rs` (dequant + types + kv_cache extraction) | after dispatch lands + bakes | 1d | low |
| 4 | `qwen35.rs` | after #113 cohort + any pending arch PRs land | 2d | medium (touched often) |
| 5 | `speculative.rs` | after qwen35.rs lands and bakes | 2d | medium (touched often) |

**Total budget: ~9 person-days across 6 PRs**, spread across 3–4 wall weeks to let each split soak before the next.

## Pre-flight bucket validation (new in rev-2)

Before opening any Tier-1 split PR, share the proposed bucket layout as a one-page proposal for sign-off:

1. **Sample method-name distribution** (or struct/concern distribution for non-impl files): `grep -E "^    pub fn " <file>.rs | awk -F_ '{print $1}' | sort | uniq -c | sort -rn`.
2. **Translate to bucket assignments** in a table: each method/struct → proposed file.
3. **Estimate per-file line count** assuming average method body length. Verify all proposed files land ≤1500 lines.
4. **Reviewer signs off on the layout** (not the code yet).
5. **Then mechanical moves happen.**

This step costs ~1 hour. It prevents the "we got the axis wrong" failure mode that requires re-doing the split.

## Per-PR checklist (rev-2)

For every split PR:

1. **Two-step commit pattern (per Gemini):** preserves `git log --follow`.
   - Commit 1a: `git mv <file>.rs <file>/mod.rs` (pure rename; perfect rename detection).
   - Commit 1b: move chunks from `mod.rs` → siblings; add `mod` declarations.
2. **Compile must pass at every commit** in the series (each commit independently bisectable).
3. **Coherence-gate:** Run `./scripts/coherence-gate-dflash.sh` post-split. Per AGENTS.md:30, this is the canonical correctness gate (`coherence-gate.sh` is deprecated). Required for PRs 2, 3, 4, 5; not required for PR 1 (offline CLI; no kernel path).
4. **Per-family smoke matrix** (new in rev-2): coherence-gate uses a fixed prompt set and does NOT exercise every kernel. After PRs that touch dispatch / qwen35 / speculative, run a representative smoke prompt for any kernel family without coverage in the gate (MoE smoke, tri-attn smoke, vision-language smoke if applicable). Document in PR description which smokes ran.
5. **Perf bench:** `scripts/probe_commits.sh HEAD~1 HEAD` cross-process (per CLAUDE.md "Perf benchmarking" methodology). **Threshold: ≤5% regression allowed.** Greater than 5% blocks the PR until investigated.
6. **Feature-gate audit:** post-split, count `#[cfg(feature = "..."]` items in the moved files; verify count matches pre-split. dispatch.rs has 25 such items today; if post-split count ≠ 25, gate dropped somewhere.
7. **Visibility-promotion list:** PR description enumerates every `fn` whose visibility was raised (e.g., private → `pub(crate)`). Reviewer signs off on each promotion explicitly.
8. **Size delta:** `tokei` before/after. PR description shows max line count of any resulting file is ≤1500.
9. **`mod.rs` size cap:** ≤500 lines per `mod.rs`. If split forces mod.rs over the cap, factor again — don't dump into mod.rs.
10. **No new content.** Reviewer's job: confirm `git log --follow` per moved chunk shows clean history. If any non-moved hunk appears, send back.

## Risks and mitigations

- **Internal API surface grows when private items become `pub(crate)`.** Once promoted, items can be imported anywhere in the crate — there's no automatic enforcement that they're only used by the module family that needed them. Mitigation: PR review of the visibility-promotion list (checklist #7); follow-up audit pass narrowing visibility back to `pub(super)` where possible.
- **Intra-crate inlining shifts at module boundaries.** Empirical risk is bounded: dispatch.rs has 2 `#[inline]` annotations out of 311+ methods; workspace `Cargo.toml` has `lto = false` (default), so cross-crate inlining wasn't happening before. Expect ≤2% perf delta on the canonical bench. Anything ≥5% is a real regression — investigate before landing (checklist #5).
- **Hidden cyclic dependencies** — splitting occasionally surfaces a cycle the compiler had been silently flattening. Mitigation: build at every commit. Resolution **must NOT** be "dump it in mod.rs" (that re-creates the god-file). Instead: extract to a shared sibling module (the `arch_helpers.rs` / `types.rs` pattern). The ≤500-line mod.rs cap (checklist #9) enforces this.
- **Coherence-gate noise** — if a split changes inlining and floats compute in a slightly different order, the dflash gate may flag. This is a *real* behavior change (not a refactor) and means the split should be reverted, not the gate threshold relaxed.
- **Coherence-gate coverage gap** — gate uses a fixed prompt set; doesn't exercise every kernel. The C-symbol-class bug from PR #195 (gfx12 suffix mismatch) is the canonical example of this latent risk. Mitigation: per-family smoke matrix (checklist #4).
- **Mid-flight branches conflict** — announce each split PR ahead of time; give in-flight authors a chance to land first.

## Rollback policy (new in rev-2)

If a split lands, bakes for a week, then a regression is found 3+ weeks later when subsequent work has built on the new module structure:

- **Default: fix-forward.** Revert just the buggy method to its pre-split file location via `git mv`. Visually odd post-split (one method back in its old home), but cheap and bisectable.
- **NOT a cascading revert.** Reverting a split + all subsequent work that touched the new structure is the nuclear option; only justified if the regression is unfixable in-place.
- **Implication for reviewers:** know that fix-forward is the policy; don't gate the initial split PR on "what if we have to undo this in 6 weeks" — the answer is "we patch the specific item, we don't unwind the structure."

## Out of scope (for follow-up issues)

- `daemon.rs` extract of `generate_vl` (small win, not worth a separate PR cycle right now).
- `tokenizer.rs` extract of prompt-normalization helpers (~200 lines).
- `kernels.rs` family-grouping (data-only, low value).
- `hip-bridge/src/ffi.rs` (FFI flat-list is conventional).
- Visibility narrowing pass: after the splits land, an audit PR can move some `pub(crate)` items back to `pub(super)` once their callers are confirmed to live in the same module subtree.

These can be opportunistic — done as drive-by cleanups during unrelated PRs that touch those files anyway.

## Definition of done

When all 5 Tier-1 PRs (or 6 with PR 2a) have landed, baked for one week each without:
- coherence-gate-dflash regressions, OR
- ≥5% perf regression on the canonical cross-process bench (per CLAUDE.md "Perf benchmarking"),

AND the size-limit CI gate (companion plan: `size-limit-ci-gates.md`) is enforcing the 2500-line cap, AND every resulting file is ≤1500 lines, AND no `// NOLINT: file-size` exemption headers remain — the work is done.
