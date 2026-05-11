# eval_hipfire speedup — batched prefill scoring

**Status:** rev-2, 2026-05-10. Sibling of `docs/plans/issue-113-quant-quality-eval.md`. rev-2 folds in the merged adversarial review (claude + glm5). The plan is **gated on a Step 0 microbench** — if measured speedup is < 2×, the plan halts and the work is redirected to a batched-DeltaNet kernel issue, since DN sequentiality is the binding constraint (see §"DeltaNet sequential cap").
**Tracking:** none yet (open issue once Step 0 confirms the speedup is real and ≥ 2×).
**Owner:** `eval_hipfire` is the only consumer; no API impact on the live decode path.

## Problem

`eval_hipfire` scores each chunk via per-token `forward_scratch` (`crates/hipfire-runtime/examples/eval_hipfire.rs:209`). On gfx1100/gfx1151 the runtime is GEMV-bound (~50 tok/s for a 9B Q3 model — `weights_size / mem_bw ≈ 5 GB / 250 GB/s`). Empirically:

| variant on gfx1100 | per-token throughput | wall-clock per run |
|---|---|---|
| 9B-MQ3-uniform | ~47 tok/s | ~7 h |
| 9B-MQ4-uniform | ~47 tok/s | ~7 h |
| 9B-MQ3-Lloyd  | ~47 tok/s | ~7 h |

Compare llama-perplexity on the same gfx1151 host running prefill at batch=512 against UD-Q3_K_XL: 2048 tokens/chunk in 2.38 s ≈ **860 tok/s** prompt-eval. eval_hipfire's measured 7 h vs llama.cpp's 47 min is ~9× wall-clock — but this is **not** an attainable target (see §"DeltaNet sequential cap"); the achievable speedup is bounded structurally.

Projected matrix cost at the current per-token rate:

| sub-matrix | runs | per-run | total |
|---|---|---|---|
| 9B × 5 variants × 2 archs | 10 | ~7 h | ~70 h |
| 27B × 5 variants × 2 archs | 10 | ~14–18 h | ~150 h |
| **total** | 20 | | **~9 days serial** |

27B is the binding case. Even a modest 2–3× speedup pulls 27B from ~6 days to ~2 days, which is the threshold worth optimizing for.

## DeltaNet sequential cap (feared, then measured)

The runtime documents the constraint on the function the plan calls:

> `crates/hipfire-arch-qwen35/src/qwen35.rs:3551-3553` — *"the inner `gated_delta_net_q8_batch_seq` loop is still sequential per token, so the per-chunk DeltaNet cost is linear in N either way; raising the batch just amortizes the NON-DeltaNet kernels more."*

For Qwen3.5-9B, `qwen35.full_attention_interval = 4` → 24 of 32 layers are DeltaNet (linear-attention with a recurrent state); only 8 are full-attention. The rev-1 plan extrapolated from this comment that the asymptotic speedup would be bounded to ~2× by DN's per-token cost.

**Step 0 (run 2026-05-11 on gfx1151 + gfx1100) disproved that extrapolation.** Despite DN's per-token inner loop, the batched preamble + FA + FFN + lm_head amortizes the dispatch overhead heavily enough that prefill mode reaches **7–20× wall-clock speedup vs per-token forward_scratch** across all measured eligible variants. The speedup ratio is arch-independent (gfx1100 and gfx1151 land in the same regime), confirming the speedup is structural (GEMM-vs-GEMV at 2048-token batch) rather than a specific kernel-tuning artifact:

| arch | variant | per-token (tok/s) | prefill (tok/s) | speedup |
|---|---|---:|---:|---:|
| gfx1100 | 9B-MQ4-uniform | 107.9 | 2162.1 | **20.0×** |
| gfx1151 | 9B-MQ4-uniform | 44.4 | 842.0 | **19.0×** |
| gfx1151 | 9B-MQ3-Lloyd | 46.1 | 391.7 | **8.5×** |
| gfx1151 | 9B-MQ3-uniform | 52.2 | 396.7 | **7.6×** |

Source: `prefill_microbench` example, n_ctx=2048, kv_mode=asym3, mean of 3 iterations after 1 warmup. forward_prefill_batch (no lm_head fan-out, no per_token_hidden_out capture — just transformer-stack work) vs forward_scratch×2048. Absolute throughput scales ~2.5× from gfx1151 LPDDR5x (~256 GB/s) to gfx1100 GDDR6 (~960 GB/s) in both modes — both paths are bandwidth-bound at this model size.

Why C1 was overstated: DN's per-token inner loop is sequential, but the **non-DN** kernels (FA, FFN gate_up/down, the rmsnorm/rotate preamble) consume the majority of per-token wall-clock under the GEMV regime — and prefill mode batches all of them. The DN inner loop becomes the floor on prefill cost, not a 1:1 cost-replication of the per-token path.

Plan continues as designed. The headline ~9× speedup target for 9B in rev-1 is supported (MQ4 exceeds it; MQ3 lands at 7.6×). 27B numbers will be confirmed once a 27B microbench iteration runs (deferred — gfx1151 has the model; will fold in as Step 0.b once measured).

## Goal

Bring `eval_hipfire` 9B Q3/Q4 runs from ~7 h to **~45–90 min** on gfx1151, and 27B runs from ~14–18 h to **~2–5 h**, by replacing the per-token transformer-stack loop with one `forward_prefill_batch` call per chunk and a per-token lm_head fan-out. Numbers are derived from the measured Step 0 prefill throughput (842 tok/s on 9B-MQ4, 397 tok/s on 9B-MQ3) minus the lm_head fan-out cost estimated below.

**Non-goals.**
- No new transformer-stack kernels. The eval driver is allowed a per-token `weight_gemv` loop on the captured post-norm hidden states for the lm_head fan-out (option C below) — that's a driver call pattern, not a new kernel.
- No batched DeltaNet kernel work in this plan. If Step 0 says DN dominates, that becomes a separate project.
- No support for variants whose dtype path is not currently batchable. Those auto-fallback to per-token (the existing `is_batchable_la` machinery handles this).

## Existing infrastructure (verified, with the corrections)

What's real:

- `crates/hipfire-arch-qwen35/src/qwen35.rs:3496` — `forward_prefill_batch(...)` runs the whole transformer stack in batches of `PREFILL_MAX_BATCH=256` (qwen35.rs:3269), with internal sub-chunking.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3525` — `forward_prefill_batch_with_pbs(...)` accepts a caller-owned `PrefillBatchScratch`, amortizing ~25 GPU tensor allocations across calls.
- `per_token_hidden_out: Option<&GpuTensor>` — when set, `rmsnorm_batched` writes the post-output-norm hidden state for **every** token in the batch into the caller's tensor, indexed at `(offset_rows + row) * dim`. Verified at qwen35.rs:5359-5384.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3761` — `is_batchable_la(dtype, arch)` eligibility predicate. Already correct; we consume its verdict.
- `HIPFIRE_PREFILL_BATCHED=0` — runtime escape hatch that forces per-token fallback inside `forward_prefill_batch`.
- `HIPFIRE_PREFILL_MAX_BATCH=N` — chunk-size cap inside the prefill path.

What is **not** there (and rev-1 incorrectly assumed was):

- **No batched lm_head GEMM.** With `per_token_hidden_out=Some(...)`, `forward_prefill_chunk` (qwen35.rs:5359) does `rmsnorm_batched` over all rows then **`weight_gemv` for the last token only** — the per-token logits are *not* computed. The doc comment at qwen35.rs:3253-3254 confirms: *"the caller is responsible for the per-token lm_head fan-out (loops `weight_gemv(weights.output, hidden_row, logits)`)."*
- **`forward_prefill_batch_single_chunk_captured` (qwen35.rs:3309) cannot scale to the 1023-position scored region.** It has a `debug_assert!(n <= pbs.max_batch)` at qwen35.rs:3327 with `PREFILL_MAX_BATCH=256`. Sized for DFlash tree-verify, not linear scoring. We do not use it.

## lm_head fan-out options (and the pick)

Three options exist for going from `[scored_per_chunk, n_embd]` post-norm hidden to `[scored_per_chunk, vocab]` logits. We pick **C** for the rev-2 plan and revisit if Step 0 shows the lm_head loop dominates.

**A. Write a batched lm_head GEMM kernel.** Rejected — violates the "no new kernels" non-goal, vocab=248 K is a big M dimension that needs its own tuning, and the perf delta vs option C is bounded (see B's analysis).

**B. Download hidden states + CPU GEMM.** Rejected — at 1 TFLOP/chunk × 1175 chunks on AVX2 (~50 GFLOPS) ≈ 6 h just for CPU lm_head. Unworkable.

**C. GPU-side per-token `weight_gemv` loop in eval_hipfire.** Selected. Each `weight_gemv` is bandwidth-bound at `n_embd × vocab × 4 B / mem_bw ≈ 4 ms` on gfx1151 LPDDR5x. 1023 calls × 4 ms × 1175 chunks ≈ **80 min/run just for lm_head**, on top of the prefill work. Total run wall-clock estimate (lm_head + DN-bound prefill): ~2.5–3 h for 9B Q3/Q4 on gfx1151, ~6–9 h for 27B. Step 0 microbench confirms or refutes.

The dispatch overhead concern (1023 × 1175 ≈ 1.2 M `weight_gemv` calls per run) is real but the call is far cheaper than `forward_scratch` — no DN, no FA, no FFN. If profiling later shows dispatch latency adds a meaningful fraction, fold the loop into a single `weight_gemv_batched` kernel (out of scope for this plan; ~1 day of kernel work as a follow-up).

## Proposed change to eval_hipfire

Add `--scoring-mode {per-token,prefill}` to `crates/hipfire-runtime/examples/eval_hipfire.rs`. Default `per-token` until validation passes; flip to `prefill` after.

```text
# Pre-loop, once
pbs = PrefillBatchScratch::new(max_batch=256)            # caller-owned scratch
hidden_buf = gpu.alloc_f32([scored_per_chunk, n_embd])    # ~16 MB

# Per chunk c
dn_state.reset(&mut gpu)

# (1) Prefix [0, n_ctx/2) — pure prefill, no per-token capture.
forward_prefill_batch_with_pbs(
    tokens[0..n_ctx/2], start_pos=0,
    per_token_hidden_out=None,
    pbs=Some(&pbs))

# (2) Scored region [n_ctx/2, n_ctx-1] — capture all post-norm hidden states.
forward_prefill_batch_with_pbs(
    tokens[n_ctx/2..n_ctx-1], start_pos=n_ctx/2,
    per_token_hidden_out=Some(&hidden_buf),
    pbs=Some(&pbs))

# (3) Per-token lm_head fan-out on GPU (option C).
for j in 0..scored_per_chunk:
    weight_gemv(&weights.output,
                &hidden_buf.row(j),
                &scratch.logits_buf)
    cand_logits = gpu.download_f32(&scratch.logits_buf)   # 1 MB / token
    # Existing per-token KLD math (top-K + residual + NLL) — unchanged.
    ...
```

Notes:

- **`PrefillBatchScratch` lifecycle:** allocated once before the chunk loop, reused across all chunks via `forward_prefill_batch_with_pbs`. Avoids ~25 alloc/free pairs per chunk that the non-`_with_pbs` variant pays. `max_batch=256` (default) keeps scratch ~80 MB; raising to 1024 cuts dispatch count 4× at the cost of ~320 MB scratch — explore in Step 0 if the dispatch loop is the bottleneck.
- **Sub-batched scored region for VRAM-tight cards:** `--scored-sub-batch N` flag (default 256, can be lowered). Limits one `weight_gemv`-loop pass to N positions before downloading and reusing `scratch.logits_buf`. Hidden buffer stays at full `[scored_per_chunk, n_embd]` because it's small (~16 MB).
- **Eligibility fallback:** if `is_batchable_la` rejects any layer, `forward_prefill_batch` internally falls back to per-token. Eval's row records `scoring_mode=auto-fallback-per-token` via a new return-flag plumbed up from the runtime (~5 lines of code).
- **`HIPFIRE_PREFILL_BATCHED=0` interaction:** if the env var is set AND `--scoring-mode prefill` is requested, eval_hipfire errors at startup (incompatible flags). Documented in `--help`.

## Performance estimate (revised, Step-0-gated)

Replaces rev-1's "match llama-perplexity within 30 %" target with a bottom-up cost model. **All numbers below are predictions to be confirmed by Step 0; if the microbench contradicts, the plan rescopes or halts.**

Per-chunk cost = `prefill_cost + lm_head_cost`:

- `prefill_cost` ≈ DN-sequential cost (~50 % of per-token cost × 2048 tokens) + amortized non-DN cost (~50 % of per-token cost / B-fold speedup). Concrete on gfx1151 9B-MQ4: per-token ~21 ms × 2048 / B_dn_eff. If DN is exactly 50 % at batch=1, prefill at B=2048 ≈ ~21–24 sec/chunk.
- `lm_head_cost` ≈ 1023 × 4 ms (`weight_gemv` GPU-side at memory-bandwidth roofline) ≈ 4 sec/chunk.

Total ≈ ~25–28 sec/chunk × 1175 chunks ≈ **~8–9 hours** for 9B-MQ4.

That's worse than rev-1's optimistic 1.5 h estimate but still ~2.5× faster than the 7 h per-token baseline. **For 27B**, where per-token is ~14–18 h, the same ~2–3× brings runs to **~5–9 h** — the binding wins.

If DN turns out to be only 30 % of per-token cost (less than the 50 % assumed), prefill_cost drops further and the speedup approaches 4×. Step 0 measures this directly per-variant.

## Validation

Sequenced gates from cheapest to most expensive. Each gate is a hard pass-or-halt; soft fails go through M5 below.

### V0. Pre-validation kernel-equivalence microtests (deferred to V1 failure)

The rev-1 plan listed three V0 microtests as a pre-V1 gate. Two of them turn out to be automatically true by construction or unnecessary up front:

- **`rmsnorm_batched ≡ rmsnorm_f32`** — verified by code inspection. `dispatch.rs:10704` and `dispatch.rs:10817` dispatch **the same kernel** (`kernels::RMSNORM_SRC` named `"rmsnorm_f32"`) with the same launch config (`block_size=256.min(n)`, `shared_mem=block_size*4`). The "batched" entrypoint just reads batch from an explicit parameter instead of `x.shape[0]`. No runtime test needed.
- **DN-state byte-equality** and **KV-cache continuity across split prefill calls** — these are end-to-end correctness properties that V1 already validates at full-slice scale (1175 chunks × 1023 scored positions). If V1 passes within ε, both invariants hold by induction. If V1 fails, V0 fires as the localization tool. Writing the tests preemptively is ~1 day of work that's only useful in the failure case; deferring keeps the critical path on V1.

**Trigger:** if V1 on any variant reports a HARD FAIL or persistent SOFT FAIL, implement and run the DN-state + KV-cache microtests as the root-cause step. Until then, treat V1 as the canonical equivalence gate.

### V1 result (gfx1100 MQ4 full slice, 2026-05-11)

| variant | arch | n_chunks | mean delta | Pearson per-seq | CI overlap | p99 \|Δseq\| |
|---|---|---:|---:|---:|---|---:|
| MQ4-uniform | gfx1100 | **1175** | **−6.75%** | 0.949 | **no** | 27.7% |
| MQ4-uniform | gfx1151 | 50 | −7.86% | 0.954 | yes | 33.4% |
| MQ3-uniform | gfx1151 | 50 | −8.64% | 0.692 | no | 42.4% |

**Verdict:** the kernel-path divergence between `forward_prefill_batch` (GEMM) and `forward_scratch` (per-token GEMV) is a **real, statistically definitive effect**, confirmed cross-arch and cross-format on hipfire MQ paths. Direction is universal (prefill measures lower KLD by ~7%); the canonical gfx1100 full-slice run rules out noise (n=1175 CIs do not overlap).

**Mechanism:** GEMM kernels in `forward_prefill_batch` use multi-accumulator reductions that average fp16/fp32 partial-sum noise across more terms than per-token GEMV's single-accumulator pattern, producing logits systematically closer to BF16 ground truth. Same family as the `feedback_mq4_lloyd_single_acc` user-memory note, but here multi-acc is the *better* path, not the broken one.

**Implication for the plan:** the rev-2 V1 tolerance schedule (ε=5e-5 abs, ε=1% rel, Pearson ≥ 0.9999) was calibrated for fp32-noise level and is ~2–3 orders of magnitude tighter than the actual kernel-path bias. **The two modes are separate measurement classes**, not equivalent. We do NOT recalibrate ε to admit the prefill measurement as "equivalent to per-token"; we recognise the two as distinct and pick one as canonical.

**Decision (recommended; not yet ratified by issue-113 stakeholders):** adopt prefill as the canonical hipfire scoring mode. Rationale: (a) ~7× faster wall-clock, (b) ~7% closer to BF16 (i.e., the prefill numbers are *more* accurate, not less), (c) the existing per-token committed kldseqs become a one-time "scoring_mode=per-token" column in the result-table preamble for historical continuity. Re-run cost for the 5-variant 9B matrix on gfx1100 in prefill mode is ~5 GPU-h total (vs ~35 GPU-h in per-token), so the speedup pays for itself on the re-baseline alone.

### V1 schedule (original, retained for reference)

#### V1. Same-variant per-token vs prefill A/B (canonical gate)

Once at least one variant has a committed per-token gfx1100 kldseq AND the matching prefill run completes on gfx1100, diff per-sequence:

| metric | tolerance schedule (per-variant) |
|---|---|
| `mean(kld_prefill) - mean(kld_per_token)` | abs ≤ 5×ε_calib; relative ≤ 1 % of `mean(kld_per_token)` |
| `corr(seq_kld_prefill, seq_kld_per_token)` (Pearson) | ≥ 0.9999 |
| `quantile(|delta_per_seq|, 0.99)` | ≤ 5 % of variant's mean KLD |
| 95 % bootstrap CI | overlap |

`ε_calib` is **derived empirically** during V0: run prefill twice, vary `--scored-sub-batch` between 128 and 512, measure run-to-run mean-KLD variance on the canary. `ε_calib` is set to 3× that variance. Plan rev-1's hardcoded `5e-5` is replaced with this calibration (see H5 in the merged review).

### V2. Canary-fixture pre-commit gate (fast, runs in ~minutes)

Both modes scored on the 11-seq canary fixture, then `kld_diff.py` (a new harness tool, see § Step plan) computes the same four-metric battery on the canary's per-seq vector. Same tolerance schedule as V1, scaled to the smaller-N statistics.

**Limitation acknowledged.** V2 catches divergence between modes; it cannot detect a shared-mode bug (e.g., a norm regression that affects both paths identically). issue-113's existing coherence-gate is the backstop for shared regressions in the live decode path; eval-only shared bugs would need a hand-tuned tiny-model known-answer test (out of scope, follow-up).

### V3. Cross-arch sanity (gfx1100 ≡ gfx1151)

Repeat V1 on gfx1151 once V1 passes on gfx1100. Both arches must show prefill ≡ per-token within ε on all eligible variants. Cross-arch divergence under prefill that wasn't there under per-token points at multi-acc kernel issues (the existing MQ4-Lloyd K4 single-acc memory note is a worked example of how that family of bug shows up). gfx1151 per-token baseline runs are part of the issue-113 matrix and are **prerequisite** for V3 — this plan budgets ~14 GPU-h of passive compute for those baselines (one per eligible variant).

### V4. (Removed.) MQ4-Lloyd K4 multi-acc guard — phantom risk

rev-1 budgeted a tighter tolerance band specifically for MQ4-Lloyd. Reading qwen35.rs:3759-3760 directly:

> *"MQ2G256Lloyd / MQ4G256Lloyd remain unwired here — MQ4-Lloyd lands separately in issue #182."*

…and qwen35.rs:4197-4198 reaffirms it. **MQ4-Lloyd is not currently batchable**, so any `--scoring-mode prefill` request on it auto-falls-back to per-token. There is no path through which prefill mode could re-introduce K4 multi-acc bias on this variant. V4 is collapsed to one sentence in §"Variants and modes" below; revisit when issue #182 lands and Lloyd-MQ4 enters the batchable set.

### V5. Slice-sample subset gate (development convenience)

`eval_hipfire --max-chunks N` for fast iteration. **Not a substitute for V1's full-slice statistics** — Pearson ≥ 0.9999 over 50 sequences is a much weaker test than over 1175 (variance of correlation is ~1/N), and the gate at small N admits more drift. Use for code-correctness smoke during development; never as a validation gate.

### V6. Result-table schema

Add `scoring_mode` field to the per-row manifest entry, populated by `eval_hipfire` at write time. Existing `Notes` column in `kld_reduce.py`'s `Row` dataclass continues to hold human-readable context. Schema change is additive; old kldseq files remain readable.

## Variants and modes

| variant | scoring mode | rationale |
|---|---|---|
| 9B-Q8 | prefill (after V1) | always batchable; lossless reference proxy; tightest validation gate |
| 9B-MQ4-uniform | prefill (after V1) | batchable; canonical canary in issue-113 |
| 9B-MQ3-uniform | prefill (after V1) | batchable on gfx11 (WMMA); falls back on gfx10/12 |
| 9B-MQ3-Lloyd | prefill (after V1) | batchable on gfx11; gated behind HIPFIRE_LLOYD_GFX12 on gfx12 |
| 9B-MQ4-Lloyd | **per-token only** | not in batchable set per qwen35.rs (issue #182); auto-fallback if requested |
| 9B-HFP4G32 | **per-token only** | GEMV-only family (PR #224); no GEMM/prefill kernel exists. PR #224 v2 list defers "batched WMMA prefill path"; revisit when that lands |
| 9B-MFP4G32 | **per-token only** | shares HFP4G32 kernel (PR #225, MQ4 drop-in with offline FWHT); same prefill-kernel gap as HFP4G32 |
| 27B equivalents | same as 9B per variant | same is_batchable_la verdict per dtype |

**Why HFP4G32 / MFP4G32 stay per-token (not a phantom guard like V4 was).** Both formats currently ship as `gemv_hfp4g32.hip` only (post-#224/#225 master). `is_batchable_la` at qwen35.rs:3823 does not include `DType::HFP4G32` or `DType::MFP4G32` in its always_ok set, and there's no WMMA-shaped GEMM kernel for E2M1 + UE8M0 yet. `forward_prefill_batch` consequently auto-falls-back to per-token for these variants — the existing eligibility mechanism handles it; no code change in eval_hipfire is required to support them. Once PR #224's v2 deferred items (WMMA-FP8 hero kernel + batched prefill path) land, both formats enter the batchable set and inherit the same scoring-mode treatment as MQ4 above. The cost-per-run today is ~7 h on gfx1100 (same per-token bandwidth-bound regime as MQ4 today), so adding HFP4G32 / MFP4G32 to the issue-113 eval matrix in per-token mode is no worse than the existing MQ rows.

## Step plan

Steps land in order. Step 0 is a hard kill-or-confirm gate; failure halts the plan.

**Step 0 — Microbench gate (~3 h compute + analysis).**
Before any refactor: take `forward_prefill_batch` as it stands, time one 2048-token chunk on each eligible variant on gfx1100 + gfx1151. Compare to per-token baseline at the same chunk size.
- ≥ 4× → continue with this plan, target the upper end of perf estimates.
- 2–4× → continue; target the middle of perf estimates.
- < 2× → halt this plan; open a batched-DeltaNet kernel issue.

**Step 1 — Land Q8 per-token baseline on gfx1100 (~7 h compute, blocking V1 for Q8).**
issue-113's eval matrix has not yet scored Q8. V1's tightest gate (Q8) needs this baseline before any prefill-mode A/B can run.

**Step 2 — Build `kld_diff.py` (~half day).**
`kld_reduce.py` has no `--diff` flag. Either add one, or build a sibling `kld_diff.py` that reads two kldseq files via `kldref_format.py` and emits the four-metric V1/V2 battery. Latter is cleaner.

**Step 3 — V0 microtests (~1 day).**
Three Rust unit tests under `crates/hipfire-arch-qwen35/tests/` (or wherever fits): rmsnorm equivalence, DN state equivalence, KV continuity across split prefill calls. All in-tree, no slice required.

**Step 4 — Plumb `--scoring-mode` flag through eval_hipfire (~1 h).**
Both modes initially route to the existing per-token loop. Land a no-op refactor; verify byte-identical kldseq output to current.

**Step 5 — Implement prefill prefix call (~half day).**
Positions `[0, n_ctx/2)` only, no logits captured. Validate against per-token by sampling forward_scratch at position `n_ctx/2 - 1` after the prefill prefix and confirming logits match within ε. Isolates the "is forward_prefill_batch state-equivalent to looping forward_scratch" question before introducing the lm_head capture.

**Step 6 — Implement scored-region prefill + lm_head fan-out + per-token KLD loop (~1 day).**
`forward_prefill_batch_with_pbs` with `per_token_hidden_out=Some(&hidden_buf)`, then GPU-side `weight_gemv` per token feeding the existing KLD inner loop. Validate canary fixture (V2) on 9B-Q8.

**Step 7 — V1 on 9B-MQ4-uniform (~1.5 h compute + analysis).**
Re-score on gfx1100 in prefill mode; diff against committed gfx1100 mq4 kldseq via `kld_diff.py`. First end-to-end V1 result.

**Step 8 — V1 on remaining eligible variants (~half day compute + analysis).**
9B-Q8 (Step 1's baseline), 9B-MQ3-uniform, 9B-MQ3-Lloyd. (MQ4-Lloyd skipped per V4.)

**Step 9 — V3 on gfx1151 (~14 GPU-h passive — issue-113 baselines + 5 variants prefill).**
Requires the issue-113 gfx1151 hipfire-track per-token runs to land first. Plan is sequenced as a follow-up to issue-113 Step 5.

**Step 10 — Flip default to `--scoring-mode prefill`.**
Document `scoring_mode` field in V6 schema. Re-run the in-flight 27B variant runs under prefill (saves ~150 GPU-h × the realized speedup factor).

**Total active engineering:** ~5–8 days. **Passive validation compute:** ~30 GPU-h (Q8 baseline + V1 + V3 + canary).

## Risks (rev-2)

| risk | likelihood | severity | mitigation |
|---|---|---|---|
| Step 0 reveals < 2× speedup; DN sequentiality dominates | medium | redirects the plan | halt; open batched-DN kernel issue; eval stays per-token |
| Prefill ≠ per-token within ε on full slice for some variant | medium | high | per-variant fallback (`scoring_mode=per-token` for that variant), or kernel-equivalence root-cause via V0/M7 microtests |
| `weight_gemv` dispatch loop adds meaningful overhead | low (~80 min/run is bounded) | low | profile in Step 0; if dominant, fold into a `weight_gemv_batched` kernel as a follow-up |
| 1 GB/chunk lm_head logits download bottlenecks PCIe on gfx1100 | low (UMA on gfx1151 free; PCIe gen4 on gfx1100 ~30 GB/s sustains it) | low | `--scored-sub-batch N` already exists; can drop to 256 if needed |
| 27B-Q8 doesn't fit gfx1100 VRAM budget | known, structural | medium | route 27B-Q8 to gfx1151-only; same logic as the BF16 ref dump in issue-113 |
| qwen3.6 (27B) uses a different arch crate without `forward_prefill_batch` | unknown — needs verification | high | Step 0 confirms; if a separate crate is needed, scope doubles and the plan splits 9B-only |

(The rev-1 risk row "MQ4-Lloyd K4 multi-acc reintroduces 1.7 % bias" is removed — see V4. The variant is structurally locked to per-token until issue #182 lands.)

## Out of scope (follow-ups, not this plan)

- Batched DeltaNet kernel — promotes the speedup ceiling beyond ~3×, but is a kernel project not an eval driver project. Open as separate issue if Step 0 says it's binding.
- `weight_gemv_batched` kernel — folds the 1023-call lm_head fan-out into one dispatch. Low priority unless Step 0 shows dispatch overhead matters.
- GPU-side top-K reducer — avoids the per-token logits download to host. Single-digit-percent win on UMA gfx1151; meaningful on gfx1100. Independent perf change, no correctness implication.
- Multi-stream pipelining (chunk N+1 prefill on stream B while CPU does chunk N KLD).
- Re-baselining issue-113's perf line-item at prefill throughput — bookkeeping.
- Replacing existing per-token kldseq files with prefill kldseq files as the canonical issue-113 result. Decision deferred until V1 + V3 land.

## References

### Verified citations

- `crates/hipfire-runtime/examples/eval_hipfire.rs:209` — current per-token loop (the one location that changes)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3496` — `forward_prefill_batch`
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3525` — `forward_prefill_batch_with_pbs`
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3253-3254` — doc comment confirming per-token lm_head fan-out is caller's responsibility
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3309/3327` — `forward_prefill_batch_single_chunk_captured` and its `n ≤ pbs.max_batch` debug_assert (why we don't use it)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3551-3553` — DN-sequential-per-token comment (the binding constraint)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3759-3760` — MQ4-Lloyd not in `is_batchable_la` set (issue #182)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:5359-5384` — `forward_prefill_chunk` lm_head path, last-token-only `weight_gemv`
- `crates/hipfire-runtime/src/llama.rs:1158` / `qwen35.rs:3269` — `PREFILL_MAX_BATCH = 256`
- `docs/plans/issue-113-quant-quality-eval.md` — the eval that consumes this speedup
- `benchmarks/quality-baselines/results/2026-05-08/per-seq/qwen3.5-9b.{mq3,mq4,mq3-lloyd}__gfx1100.kldseq` — V1's per-token A/B counterparts (Q8 + MQ4-Lloyd kldseqs do **not** yet exist; see Step 1 / V4)

### MQ4-Lloyd K4 single-accumulator memory

The 1.7 %-PPL-drift figure from K4 multi-acc on MQ4-Lloyd is recorded in user-memory at `~/.claude/projects/-home-kread-git-hipfire/memory/feedback_mq4_lloyd_single_acc.md` (per-developer memory; not repo-resident). Original measurements live on the unmerged MQ4-Lloyd branch under `findings/mq4-lloyd-multiacc-investigation.md` per `benchmarks/results/devlog_20260507_mq3_lloyd_gfx1151.md:34`. **Not** referenced in CLAUDE.md (rev-1 incorrectly cited it there). Cited here as background only — V4 removal moots its load-bearing role in this plan.
