# Quant quality eval — KLD-primary harness for MQ3/MQ4 (uniform + Lloyd) + Q8 reference proxy + GGUF anchors

**Status:** plan rev-3 (2026-05-08). Folds the multi-reviewer adversarial review (Claude + GLM-5 + Gemini consolidated; 3 blockers / 8 serious / 9 medium / 10 minor) into the plan body. The standalone review files were dropped after fold-in.
**Tracking:** #113 (uniform), #116 (Lloyd, mirror sub-section).

## Goal & non-goal

**Goal:** produce a **hipfire-internal Pareto curve** (mean-KLD vs size, per (model, variant, arch)) plus a `DFlash τ` correlate per row. The combined table answers two questions:

1. **User-facing positioning** — "where does each hipfire MQ-family quant sit on a (size, KLD) plot?" The plot is the deliverable.
2. **Product decisions** — "is MQ4-Lloyd worth keeping shipped after PR #197 lands?", "should the MQ3 ship gate be recalibrated?" KLD is one input among many to those decisions.

**Secondary goal (degradable):** anchor the hipfire Pareto curve against llama.cpp's GGUF Q-family Pareto via a single shared corpus run on llama.cpp. **This is degradable** — it depends on a tokenizer-parity check (step 2 below) and possibly a token-input bridge to llama-perplexity. If those costs explode, drop the GGUF anchor track and ship the hipfire-internal-only Pareto. The plan's primary value does not depend on the GGUF anchor.

**Non-goal:** **KLD is not an implementation ship gate.** Coherence-gate + perf-gate remain the gating signals for shipping a kernel/format. KLD informs **editorial decisions** (promotion to default, deprecation note, recalibrated thresholds) — it does not automatically revert or block any PR.

This non-goal is the resolution of the previous "decides ship-or-not" framing for MQ4-Lloyd: **PR #197 ships on its own merits (kernel correctness + perf gate). The KLD result for MQ4-Lloyd lands as a comment recommending whether to *promote it as default* over MQ4-uniform.** No automatic revert path. (S7)

## Why KLD over PPL

PPL collapses the full output distribution to one scalar per token (probability of the actual next token). A quant can preserve top-1 perfectly while scrambling the tail; PPL won't see it, but DFlash speculative acceptance, RAG retrieval, factual recall, and sampling diversity all will. KLD against a BF16 reference measures exactly the perturbation the quant introduces.

llama.cpp's `llama-perplexity --kl-divergence` mode emits both PPL and mean/median/p99 KLD in one pass. PPL stays as a secondary sanity column.

**Caveat on top-K=256 truncation (M1):** Qwen 3.5/3.6 has ~151K-token vocab; top-K=256 covers ~0.17%. `sum_exp_residual` corrects mean-KLD in expectation but provides **zero information about the structure of the truncated tail** — exactly the regime DFlash speculative acceptance and high-temperature sampling are most sensitive to. KLD on top-K=256 is a **necessary but not sufficient** signal for those use cases. Step 1.6 of sequencing measures the residual mass fraction; if it exceeds 2%, raise K to 512.

## The Pareto-positioning outcome

**Deliverable:** a 2-axis plot (size GB on x, mean-KLD on y) with hipfire MQ-family points + GGUF Q-family anchor points (if anchor track survives) + bootstrap-CI error bars + p99-KLD shading. The plot is the headline.

The result **table** below is the reproducibility artifact, not the user-facing message. (S1 / m6)

```
       (size GB) →
    | Q8_0  ●
    | Q6_K   ●
    | MQ4-lloyd      ★  ★ has CI error bars
    | Q5_K_M       ●
    | MQ4-uniform     ★
    | Q5_K_S        ●
    | MQ3-lloyd        ★
    | Q4_K_M         ●
    | MQ3-uniform     ★
    | Q3_K_M           ●
    +-------------- mean KLD →
```

A single representative GGUF callout ("MQ3-Lloyd sits between Q4_K_M and Q5_K_S at equivalent size") replaces the rev-2 "Mapped to" column. Reader looks at the plot for the rest.

## Gate structure (revised, locked)

| Gate | Type | Decision rule |
|---|---|---|
| Coherence-gate (existing) | hard ship gate | unchanged |
| Perf-gate (existing) | hard ship gate | unchanged |
| KLD eval | **editorial signal** | informs comments on #113, #116, #197; does not gate PRs |

(S7) — single narrative: KLD is editorial. No automatic-revert language anywhere in the plan.

**For #113 (uniform MQ3/MQ4):** result is a positioning comment on the issue. The original 5%-MQ3-vs-MQ4 framing is superseded.

**For #116 (Lloyd):** result is a positioning comment + recalibration input.

**For #182 / PR #197 (MQ4-Lloyd):** result feeds an *editorial* decision on whether to promote MQ4-Lloyd as the default over MQ4-uniform, or to mark it "supported but not recommended" in the registry. PR #197 ships on coherence + perf alone; KLD doesn't gate the merge.

## Eval matrix

Per model: 9B / 27B / 27B-3.6.

| Track | Variants |
|---|---|
| Hipfire (all evaluated together) | Q8-uniform, MQ4-uniform, MQ3-uniform, MQ4-lloyd, MQ3-lloyd |
| GGUF anchors (secondary, conditional) | Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q3_K_M, Q3_K_S — 27B-3.6 only |

Total: 5 hipfire variants × 3 models × 2 archs = 30 hipfire runs. GGUF anchors (7 runs) on 27B-3.6 only, gfx1151 only. 3 BF16 reference dumps (one-time per model, gfx1151 only — see B3 below).

Q8-uniform plays a dual role: it's the **reference proxy** (closest hipfire can get to BF16, since hipfire's runtime doesn't support BF16 — see B3) AND a hipfire-side sanity check that the eval harness reproduces "near-zero KLD" for a known-good quant. Title and table reflect this.

## Arch coverage

Each hipfire variant evaluated on **two archs**: gfx1100 (canonical ship target, GDDR6) and gfx1151 (deployment, Strix Halo APU, LPDDR5x).

| Why gfx1100 | Why gfx1151 |
|---|---|
| canonical ship-gate target, matches CI / production deploy | the bench/dev host that produces BF16 refs (only host with enough RAM for 27B BF16 — 137 GB UMA) |

BF16 reference dumps are **produced on gfx1151** via `llama-perplexity` (not via hipfire's runtime — see B3). All quant evals score against that same reference regardless of where the quant was scored — references are arch-independent (just bytes).

Result table grows a per-arch column. If gfx1100 and gfx1151 KLDs diverge by more than canary tolerance, that's a finding (decode-path multi-acc drift surfaces as a quant-quality artifact — m4).

**Light cross-host BF16 sanity (m3):** as part of step 4, produce a 9B-only BF16 reference on CPU (~30 min) and compare to gfx1151 BF16 ref via mean-KLD on 100 random tokens. If <0.001, gfx1151 is a neutral observer for the BF16 path. If >0.01, escalate. Not extended to 27B (CPU BF16 27B is days).

## Tokenizer alignment + bridge investigation

llama.cpp tokenizes via per-GGUF tokenizer; hipfire uses the upstream Qwen tokenizer through hipfire's loader. **If these tokenize the slice differently, the entire GGUF-anchor track is invalid.**

**Step 1.5 (mandatory before any reference dump):** byte-identical tokenizer-parity check.

```python
# benchmarks/quality-baselines/harness/tokenizer_parity.py — concrete sketch
import subprocess, struct
# hipfire side: wraps hipfire's Qwen tokenizer (HF tokenizers lib for now;
# later via eval_hipfire.rs --tokenize-only mode if added).
hf_ids = hf_tokenizer.encode(open('slice.txt').read())
# llama.cpp side: invokes llama-tokenize with the GGUF's bundled tokenizer.
gguf_ids = subprocess.check_output([
    'llama-tokenize', '-m', 'qwen3.5-9b-q8_0.gguf',
    '-f', 'slice.txt', '--ids'
]).decode().split()
gguf_ids = [int(x) for x in gguf_ids]
assert len(hf_ids) == len(gguf_ids), f"len mismatch {len(hf_ids)} vs {len(gguf_ids)}"
for i, (a, b) in enumerate(zip(hf_ids, gguf_ids)):
    assert a == b, f"token {i}: hf={a} gguf={b}"
```

**Step 2 (bridge feasibility investigation, ~30 min):** read `examples/perplexity/perplexity.cpp` on the pinned llama.cpp commit. Verify whether `llama-perplexity` accepts pre-tokenized input via any flag. If yes, document the exact invocation. If no:

- **2a-bridge:** estimate the cost of patching `llama-perplexity` to add `--input-tokens`. If <2 days (most likely), commit to the patch and either upstream or vendor the binary. Update §"Storage and cost" with the patch cost.
- **2b-drop:** drop the GGUF anchor track entirely. The hipfire-internal Pareto + bootstrap CIs is the deliverable. Plan still ships.

**Step 2 sequencing branches:**

```
step 1.5 tokenizer-parity result:
  PASS → continue (steps 3, 4, 5, ...)
  FAIL → step 2 bridge feasibility:
    bridge feasible (≤ 2 days) → patch llama-perplexity, continue
    bridge not feasible       → drop GGUF anchor track, continue
                                 with hipfire-internal Pareto only
```

(M7)

## Repo layout (committed)

```
benchmarks/quality-baselines/
├── slice/
│   ├── wikitext2-1024s-2048ctx.txt        # frozen prompt bytes, ~1 MB
│   ├── slice.md5                          # checksum
│   └── tokens.bin                         # u32 token IDs (post-bridge or absent if bridge skipped)
├── harness/
│   ├── manifest.json                      # {sha256, hf_url, producer_cmd, llamacpp_commit, host_arch}
│   ├── eval_gguf.sh                       # llama.cpp --kl-divergence wrapper
│   ├── eval_hipfire.rs                    # binary: dumps logits in llama.cpp's KLD format
│   ├── kld_reduce.py                      # reducer + bootstrap CI
│   ├── tokenizer_parity.py                # step 1.5
│   ├── canary.md                          # 11-seq fixture (10 short + 1 long-ctx)
│   └── README.md                          # how to add a quant
├── refs/
│   └── (downloaded blobs, .gitignored — fetched via fetch-eval-refs.sh)
└── results/
    └── 2026-05-XX-quant-pareto.md          # output table + plot
```

(m9 — moved manifest.json under harness/ for dependency clarity.)

## Binary format pinning

`eval_hipfire.rs` emits **llama.cpp's exact `--kl-divergence-base` binary layout**. This makes `llama-perplexity --kl-divergence-base <hipfire.bin>` work as the reducer for both tracks; `kld_reduce.py` is a sanity wrapper / cross-check.

**Prerequisite (S6 — 1 hour, before sequencing step 1):** read `examples/perplexity/perplexity.cpp` on the pinned llama.cpp commit. Replace the layout below with the actual layout, citing line numbers in this section. The layout below is **inferred and unverified**; treat as a sketch until step 0 confirms.

```
header (inferred — verify):
  u32 magic
  u32 version
  u32 n_tokens
  u32 top_k
per-token block:
  fp32 logprob_target
  fp32 logprobs_topk[top_k]
  u32  topk_indices[top_k]
  fp32 sum_exp_residual
```

**Numerical strategy (S4):** `sum_exp_residual` over Qwen's 151K vocab requires:
- **log-sum-exp** formulation to avoid overflow on extreme logits.
- **double-precision (fp64) accumulation** to avoid bias on long sums of small values.

llama.cpp uses both in `softmax_f32` and the perplexity-mode reducer. `eval_hipfire.rs` MUST match. Verify by computing top-K + residual on 10 random tokens of the 9B canary fixture and comparing bit-for-bit (within fp32 tolerance) against `llama-perplexity`'s output for the same inputs.

**MUST top-K-reduce in-flight (M9 / Gemini 5.2):** for Qwen's 151K vocab × 2M tokens × fp32, dumping full-vocab logits would cost **~1.2 TB**. `eval_hipfire.rs` MUST compute top-K + residual per token *before* writing to disk. Never dump full-vocab logits as an intermediate. The plan's storage estimate (3-8 GB per reference) assumes top-K reduction has happened.

**Top-K choice (M1 sub-step 1.6):** before bulk eval, run a residual-mass sanity check on 10 random tokens from the 9B canary fixture. Compute fraction of probability mass in the truncated tail (i.e. `sum_exp_residual / (sum_exp_residual + sum_exp_topk)`). If the median is <0.5%, top-K=256 is fine. If >2%, raise to top-K=512 (storage doubles, still within budget). Record the actual K chosen in `manifest.json`.

## Reference dump → HF Hub

Per-model BF16 reference logit dumps live at **`hipfire-models/hipfire-eval-refs`** (m1).

**On the org choice:** `hipfire-models` is a new HF org created 2026-05-08 (see project memory; we already use it for the dev-stage Lloyd quants at `hipfire-models/qwen3.5-{4b-dev,9b-dev,9b}` and `hipfire-models/qwen3.6-27b-dev`). The legacy registry (`cli/registry.json`) still references `schuttdev/hipfire-*` for non-Lloyd quants where the canonical owner has write access. **Eval refs go to `hipfire-models` because Kevin (org admin) has write access; `schuttdev` is read-only.** This will look like an inconsistency in committed state at PR review time — flag it in the PR description.

Files:

```
qwen3.5-9b-bf16.kldref.bin     (~3-5 GB, top-K=256 + residual)
qwen3.5-27b-bf16.kldref.bin    (~5-8 GB)
qwen3.6-27b-bf16.kldref.bin    (~5-8 GB)
README.md                       (producer commands, slice md5, llama.cpp commit pin, host_arch=gfx1151, model SHA256)
```

In-tree `manifest.json` carries `{sha256, hf_url, producer_cmd, model_sha256, llamacpp_commit, slice_md5, host_arch, top_k}` per reference. `scripts/fetch-eval-refs.sh` reads manifest, pulls via `hf` CLI into `~/.cache/hipfire-eval-refs/`, verifies SHA256.

## Reference dump methodology (one-time per model, on gfx1151)

```
llama-perplexity \
  -m qwen3.5-9b-bf16.gguf \
  -f benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
  --kl-divergence-base refs/qwen3.5-9b-bf16.kldref.bin \
  -c 2048 -b 512
```

**BF16 only**, produced by **llama.cpp only** (B3). The hipfire runtime does not support BF16 inference — see `crates/hipfire-runtime/src/llama.rs:481`, which panics on any `GgmlType` outside `{F32, F16, Q4_0, Q8_0, Q4K, Q6K}`. `eval_hipfire.rs` runs hipfire's quantized variants only (Q8, MQ3-uniform, MQ4-uniform, MQ3-Lloyd, MQ4-Lloyd) and scores them against the externally-produced BF16 reference.

gfx1151 has 137 GB UMA — 27B BF16 fits with huge headroom.

**Producer command + SHA recorded** alongside the reference SHA in `manifest.json` so divergence-debugging 6 months from now is trivial.

## Reference-drift canary (clarified)

The canary serves a different purpose than the SHA256 manifest check (S5):

- **SHA256 in `manifest.json`** = **reference identity guard.** Detects re-uploads or file-corruption.
- **Canary fixture** = **harness output reproducibility guard.** Detects if `eval_hipfire.rs` regresses (kernel changes, fp accumulator changes) such that the same model + same reference yields different KLD.

Q8 is so close to BF16 that small reference *content* changes don't move the canary KLD much, so the canary is poorly suited to catch silent reference replacement — that's SHA256's job. Both checks run before each eval; they validate independent failure modes.

**Canary fixture (M8):** 11 wikitext sequences:
- 10 short (≤500 tokens each) — fast harness check.
- **1 near-max ctx (1800-2000 tokens)** — catches RoPE / KV-cache drift that only emerges late in the context window.

Expected KLD per sequence committed in `canary.md`. Tolerance per sequence stated explicitly.

## Eval-mode hipfire flags

`eval_hipfire.rs` always sets:

```
HIPFIRE_NORMALIZE_PROMPT=0     # raw byte-stream-through; eval is byte-deterministic
HIPFIRE_GRAPH=0                # capture-mode adds capture-illegal paths; eval doesn't need it
HIPFIRE_KV_MODE=asym3          # canonical KV-mode for ship benches (M6)
HIPFIRE_LLOYD_GFX12=1          # if running on gfx1200/1201 — see PR #195 hardening commit
```

All four recorded per-row in the result table so future replays match.

**On NORMALIZE_PROMPT off (m2):** Gemini's adversarial review argued raw-byte-stream might push the model into garbage-in-garbage-out logit regions. We disagree — wikitext-2 is well-formed text without weird whitespace runs; normalize OFF doesn't push the model into a degenerate input regime. As a hedge, **the canary fixture runs with both `NORMALIZE_PROMPT=0` and `=1`**. If KLD diverges meaningfully between the two, that's a finding (and we revisit the choice). If they're identical, confirms eval-mode-OFF is a safe choice for the bulk run.

**Historical-baseline drift caveat (M4 + M5):** `HIPFIRE_NORMALIZE_PROMPT` was flipped to ON by default on 2026-04-26. The Lloyd findings doc (`lloyd_max_findings_20260501.md`) is from after that date, so its baselines were likely run with normalize ON. The new eval forces OFF. **PPLs in the new table will not reproduce historical baselines on the same corpus** — and that mismatch will look like a regression unless flagged. Same caveat applies to `HIPFIRE_GRAPH`. Document in §"Result table format" below.

## Result table format (rev-3 schema)

```markdown
| Model     | Variant       | Arch    | Size GB | Mean KLD ± 95% CI | p99 KLD | PPL    | DFlash τ | Notes |
|-----------|---------------|---------|---------|--------------------|---------|--------|----------|-------|
| 9B        | Q8-uniform    | gfx1100 | 9.4     | 0.0008 ± 0.0001    | 0.012   | 9.81   | n/a      | reference proxy |
| 9B        | Q8-uniform    | gfx1151 | 9.4     | 0.0008 ± 0.0001    | 0.012   | 9.81   | n/a      | within canary tol |
| 9B        | MQ4-uniform   | gfx1100 | 5.2     | 0.014 ± 0.002      | 0.18    | 10.34  | 8.05     | |
| 9B        | MQ4-uniform   | gfx1151 | 5.2     | 0.014 ± 0.002      | 0.18    | 10.34  | 8.10     | |
| 9B        | MQ3-uniform   | gfx1100 | 4.1     | 0.087 ± 0.009      | 1.44    | 42.0   | 6.2      | |
| 9B        | MQ3-uniform   | gfx1151 | 4.1     | 0.091 ± 0.010      | 1.51    | 42.8   | 6.1      | +5% mean vs gfx1100 — multi-acc drift surfacing |
| 9B        | MQ4-lloyd     | gfx1100 | 6.5     | 0.012 ± 0.002      | 0.16    | 10.20  | 8.20     | |
| 9B        | MQ3-lloyd     | gfx1100 | 4.4     | 0.039 ± 0.005      | 0.71    | 18.5   | 7.6      | tail worse than mean |
... (per model, per arch)
```

Numbers above are illustrative — real numbers come from the eval runs.

**`DFlash τ` column (S8):** for each variant where a draft model exists in the hipfire registry, the τ value comes from the canonical `merge_sort` prompt (per CLAUDE.md "Prompt-structure τ sensitivity"; byte-identical prompt with md5 recorded in the result file's preamble). `n/a` for variants with no compatible draft (Q8-uniform, sometimes MQ4-Lloyd before draft variants exist). The τ row anchors the editorial decision: "MQ4-Lloyd has lower mean-KLD than MQ4-uniform but lower τ — don't promote".

**`Mean KLD ± 95% CI` (M2):** bootstrap CI from 10,000 resamples of per-sequence KLDs. Per-sequence KLDs persisted by default (1024 × 4 B = 4 KB per row) so CI is reproducible without re-running.

**Mandatory notes section** at the top of the result file:

```
## Caveats for direct comparison

- PPL column was measured with HIPFIRE_NORMALIZE_PROMPT=0 and
  HIPFIRE_GRAPH=0. Historical baselines in
  benchmarks/results/lloyd_max_findings_20260501.md were likely run
  with normalize ON (default flipped 2026-04-26) and may not match
  the values in this table even on the same logical corpus.
- The eval slice (wikitext2-1024s-2048ctx.txt) is a frozen committed
  fixture introduced by this harness; PR #115's corpus
  (dev/bench/data/wikitext2-test.txt) is not committed and was a
  single contiguous window, not multi-sequence. Cross-PR PPL
  reproduction is not attempted.
- Both archs use HIPFIRE_KV_MODE=asym3.
```

## Storage and cost (rewritten with realistic numbers, S3)

**Bench-host commitment is much larger than rev-2 estimated.** Pre-reserve before starting.

- **gfx1100 subset:** 5 hipfire variants × 3 models × ~67 min/run = **~17 GPU-hours**.
- **gfx1151 subset:** 5 hipfire variants × 3 models × ~3 hours/run = **~45 GPU-hours**.
- **GGUF anchors:** 7 runs on gfx1151 × ~1.5 hr = **~10 GPU-hours**.
- **BF16 reference dumps:** 3 models × ~1.5 hr (gfx1151) = **~5 GPU-hours**.
- **9B CPU BF16 sanity (m3):** ~30 min.
- **Total: ~75 GPU-hours wall-clock.** Add ~20% padding for retries, tokenizer-parity / bridge investigation, and binary-format source-read (S6) → **~90 GPU-hours.**

Storage:
- Slice text: ~1 MB in git.
- Slice tokens.bin (post-bridge): ~8 MB if 1024 × 2048 × 4 B, in git.
- Per-model BF16 reference: ~3-8 GB on HF Hub. Three models = ~10-25 GB external. Acceptable on hipfire-models org's HF storage.
- Per-quant per-sequence KLDs (CI reproducibility): 1024 × 4 B per row = 4 KB; total across 30+7 rows = ~150 KB; in git or attached to the result file.
- Optional `--keep-logs` for per-token logits (debugging only): ~3-8 GB per variant per arch, not persisted by default.

## Open questions resolved (rev-2 → rev-3)

| # | Q | rev-3 resolution |
|---|---|---|
| 1 | Threshold recalibration | Outcome reframed: not a threshold. Pareto-positioning + editorial decisions replace pass/fail gate. |
| 2 | BF16 vs Q8 reference | BF16 only, produced by llama.cpp on gfx1151 (137 GB UMA). hipfire's runtime cannot run BF16 — see B3. |
| 3 | HF org for refs | `hipfire-models/hipfire-eval-refs` (justification in §Reference dump → HF Hub). |
| 4 | MQ4-Lloyd inclusion | Yes, full row × both archs. Result feeds editorial decision on promotion-to-default; does NOT gate PR #197. |
| 5 | Tokenizer divergence | Step 1.5 byte-identical check. Step 2 explicit branches: bridge if feasible, drop GGUF anchor track if not. |
| 6 | Coherence × KLD × perf | Coherence + perf are hard ship gates (unchanged). KLD is editorial signal only. (S7) |
| 7 | Arch coverage | Both gfx1100 and gfx1151. BF16 refs on gfx1151 only. 9B CPU sanity check (m3). |
| 8 | Prompt-normalize default | Force OFF in eval. Canary runs both ON and OFF as a hedge. (m2) |
| 9 | p99 as gate | No, column-only. p99 used in plot's tail-shading and per-row narrative ("tail worse than mean"). |
| 10 | Bootstrap CI | Column in result table. Per-sequence KLDs persisted by default for reproducibility. |
| 11 (new) | Top-K choice for 151K vocab | Step 1.6 sanity check on residual mass. Default 256, raise to 512 if residual >2%. |
| 12 (new) | sum_exp_residual numerics | log-sum-exp + fp64 accumulation, verified against llama.cpp on canary. |
| 13 (new) | Historical baselines comparable? | No. Document in result-file preamble. |
| 14 (new) | KV mode for eval | `asym3`, recorded per-row. |
| 15 (new) | DFlash τ included? | Yes, as a column on rows where a draft exists. Canonical merge_sort prompt + md5. |

## Sequencing (rev-3, with explicit prerequisites)

**Step 0 (PRE-WORK, ~1 hour, before any harness code is written):** read `examples/perplexity/perplexity.cpp` on the pinned llama.cpp commit. Verify the `--kl-divergence-base` binary layout, the `sum_exp_residual` numerics (log-sum-exp + fp64), and whether any tokens-as-input flag exists. Replace the inferred §"Binary format pinning" layout with the actual layout, citing line numbers. (S6, S4 prerequisites; informs B1 bridge-feasibility.)

**Step 1:** land harness skeleton — slice + `kld_reduce.py` + `eval_gguf.sh` + `tokenizer_parity.py` + canary fixture (11 sequences, 10 short + 1 long-ctx) + `manifest.json` schema.

**Step 1.5:** tokenizer-parity check (M3). Run `tokenizer_parity.py`. Pass → continue. Fail → Step 2.

**Step 1.6:** top-K residual-mass sanity check (M1). Dump full-vocab logits for 10 random tokens from canary, compute residual fraction. If <0.5%, top-K=256 confirmed. If >2%, raise to 512 and update §"Binary format pinning" + manifest schema.

**Step 2 (only if Step 1.5 fails):** bridge feasibility investigation (B1). 30-min llama-perplexity source read. If patch ≤ 2 days, commit to it and budget. If not, drop GGUF anchor track entirely and skip future GGUF steps.

**Step 3:** write `eval_hipfire.rs`. Top-K-reduce in-flight (M9). Output llama.cpp KLD-base format (S6 prereq). log-sum-exp + fp64 (S4). Verify bit-for-bit against llama.cpp on 10 canary tokens.

**Step 4:** dump 9B BF16 reference on gfx1151 → upload to `hipfire-models/hipfire-eval-refs` → manifest. Verify canary passes (both NORMALIZE_PROMPT settings — m2). 9B CPU BF16 sanity check (m3).

**Step 5:** run hipfire track on 9B × {gfx1100, gfx1151} (5 variants × 2 archs = 10 runs). **Validate via canary, NOT against PR #115 PPLs** (B2 / S2 — historical reproduction is impossible). The canary's expected per-sequence KLDs are the harness validation; PPL plausibility ranges (within ~30% of historical) are a sanity check, not a reproduction.

**Step 6:** repeat steps 4–5 for 27B and 27B-3.6.

**Step 7:** GGUF anchor track on 27B-3.6 (gfx1151 only, conditional on Step 1.5/2 outcome).

**Step 8:** measure DFlash τ (canonical merge_sort prompt, md5-pinned) for each variant where a draft model exists. Add as the `DFlash τ` column (S8).

**Step 9:** write up `results/2026-05-XX-quant-pareto.md` with the full table + Pareto plot + caveats preamble. Post comments on #113 (positioning), #116 (positioning + recalibration input), #197 (MQ4-Lloyd promotion-or-not editorial recommendation).

**Step 10 (optional):** Unsloth corroboration plot if same-model dense data exists.

**Step 11 (follow-up, not this PR):** p99-gate calibration once mean-KLD CI variance is understood empirically. GEMV single-acc port (universal multi-acc drift fix) if the gfx1100-vs-gfx1151 KLD delta is >10%.

## References

- Issue #113 — uniform MQ3 quality (this doc supersedes the PPL-only / 5%-gate scoping).
- Issue #116 — Lloyd-MQ3 ship gates (this doc adds *positioning*, not a third gate).
- Issue #182 / PR #197 — MQ4-Lloyd WMMA prefill (this doc decides promotion-to-default *post-merge*, not whether PR #197 itself merges).
- PR #195 — MQ3-Lloyd WMMA prefill (merged 2026-05-08; phase C devlog establishes ship gates that this eval *informs* but does not replace).
- `docs/plans/mq-sub4bit-prd.md` Section 3 — quality validation requirement.
- `docs/plans/mq-sub4bit-research-queue.md` Q1 — Lloyd-Max codebook research item.
- `benchmarks/results/lloyd_max_findings_20260501.md` — empirical Lloyd writeup. **Note: corpus referenced (`dev/bench/data/wikitext2-test.txt`) is not committed; PPL numbers are not reproducible.**
- CLAUDE.md "Prompt-structure τ sensitivity" — byte-identical prompt rule for the DFlash τ column.
- llama.cpp `llama-perplexity` `--kl-divergence` / `--kl-divergence-base` modes.
- Unsloth Qwen3.6 GGUF Pareto plot (https://unsloth.ai/docs/models/qwen3.6).
