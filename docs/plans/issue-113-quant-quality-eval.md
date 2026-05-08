# Quant quality eval — KLD-primary harness for MQ3/MQ4 (uniform + Lloyd)

Tracking: #113 (uniform gate), #116 (Lloyd gate, new mirror sub-section).

## Why KLD over PPL

PPL collapses the full output distribution to one scalar per token (the
probability of the actual next token). A quant can preserve top-1
perfectly while scrambling the tail, and PPL won't see it — but DFlash
acceptance and sampling diversity will. KLD against a BF16 reference
measures exactly the perturbation the quant introduces, which is the
question #113 is actually asking ("MQ3 is X% worse than MQ4"). It's
also much more sensitive at the low-distortion end where these quants
live, so the "within 5%" gate becomes meaningful instead of fighting
noise.

llama.cpp's `llama-perplexity --kl-divergence` mode emits both PPL and
mean/median KLD in one pass. Cheap to keep PPL as a secondary sanity
column. Unsloth's published GGUF Pareto plots (qwen3.6-35-A3B, etc.)
also use mean-KLD, so this is the same axis the comparison data lives
on if/when external corroboration is wanted.

## Gate structure (uniform and Lloyd treated as separate quant families)

Per consensus, uniform and Lloyd are distinct quant families with
distinct ship gates. The original 5% threshold in #113 was written
before Lloyd-MQ existed.

- **#113 (uniform):** MQ3-uniform vs MQ4-uniform vs Q8 reference.
  Alpha→beta gate: MQ3-uniform mean-KLD within 5% of MQ4-uniform on
  each of 9B / 27B / 27B-3.6.
- **#116 (Lloyd, new sub-section):** MQ3-lloyd vs MQ4-lloyd vs Q8
  reference. Mirror gate, same threshold structure. Add as a third
  gate alongside the existing K4-unroll perf gate and coherence
  eyeball gate.
- **Cross-cut row in both tables:** Lloyd-vs-uniform delta per
  bit-width per model. Reports the "is Lloyd worth it" question
  independently of the ship gates. PR #115 already shows 9B MQ3-lloyd
  = 18.5 vs MQ3-uniform = 42.0 (2.27× ppl reduction); this formalizes
  that as KLD across all three models.

The 5% threshold itself is likely too strict for KLD ratios —
empirically MQ3 is nowhere near 5% of MQ4 even with Lloyd. The
threshold needs recalibrating once data exists; that's a #113 / #116
conversation, not a harness-design one. Track it as a follow-up after
the first eval lands.

## Eval matrix

Per model: 9B / 27B / 27B-3.6.

| Track | Variants |
|---|---|
| Uniform (gate #113) | Q8-uniform, MQ4-uniform, MQ3-uniform |
| Lloyd (gate #116) | MQ4-lloyd, MQ3-lloyd |
| GGUF anchors (Pareto, 27B-3.6 only) | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S |

Total: 5 hipfire variants × 3 models + 5 GGUF on 1 model = **20 eval
runs** + 3 reference dumps (one-time).

GGUF anchors run on 27B-3.6 only — one model is enough to anchor the
external Pareto curve and validate that hipfire's local llama.cpp
numbers match published GGUF numbers within noise. No need to
replicate the GGUF curve on every model.

Lloyd-MQ4 is "out of scope" for shipping per #116 ("9B MQ4 ppl=10.34
already has narrow quant-loss room; +24% bandwidth for ~10% quality
reclaim isn't worth it") but the PR exists. Measuring it once
empirically anchors that deprioritization decision.

## Repo layout (committed)

```
tests/quality-baselines/
├── slice/
│   ├── wikitext2-1024s-2048ctx.txt        # frozen prompt bytes, ~1 MB
│   └── slice.md5                          # checksum
├── harness/
│   ├── eval_gguf.sh                       # llama.cpp --kl-divergence wrapper
│   ├── eval_hipfire.rs                    # binary: dumps per-token top-K logits
│   ├── kld_reduce.py                      # canonical KLD scoring code
│   ├── canary.md                          # 10-seq fixture + expected KLD
│   └── README.md                          # how to add a quant
├── refs/
│   └── manifest.json                      # {sha256, hf_url, producer_cmd, llamacpp_commit}
└── results/
    └── 2026-05-XX-uniform-vs-lloyd.md     # output table
```

The slice and the harness scripts live in git. KLD math lives in one
place (`kld_reduce.py`), version-pinned. Both runtime tracks
(`eval_gguf.sh`, `eval_hipfire.rs`) just dump per-token top-K logits
in a shared binary format and feed the reducer — so the metric
implementation is shared, not duplicated across two languages.

The slice itself (wikitext-2 raw, 1024 sequences at 2048 context) is
~1 MB and committed directly. Don't reference HF "wikitext-2-raw-v1"
by dataset name; HF datasets get re-uploaded and quietly change. The
md5 of the slice is committed alongside it as a tripwire.

## Reference dump → HF Hub

Per-model BF16 reference logit dumps live external. Suggest one repo
`huggingface.co/<org>/hipfire-eval-refs`:

```
qwen3.5-9b-bf16.kldref.bin     (~2-3 GB, top-K=256 + residual)
qwen3.5-27b-bf16.kldref.bin    (~3-5 GB)
qwen3.6-27b-bf16.kldref.bin    (~3-5 GB)
README.md                       (producer commands, slice md5, llama.cpp commit pin)
```

In-tree `manifest.json` carries
`{sha256, hf_url, producer_cmd, model_sha256, llamacpp_commit, slice_md5}`
per reference. `scripts/fetch-eval-refs.sh` reads manifest, pulls via
`hf` CLI into `~/.cache/hipfire-eval-refs/`, verifies SHA256 — fails
loudly on mismatch.

New reference adoption = add a row to `manifest.json` and upload one
blob. The reference is content-addressed by SHA256, so the manifest is
the source of truth; if someone re-uploads the blob with the same name
but different bytes, the SHA mismatch catches it.

## Reference dump methodology (one-time per model)

```
llama-perplexity \
  -m qwen3.5-9b-bf16.gguf \
  -f tests/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
  --kl-divergence-base refs/qwen3.5-9b-bf16.kldref.bin \
  -c 2048 -b 512
```

Pin the llama.cpp commit in `manifest.json`. Q8_0 used as reference
proxy if BF16 doesn't fit on a single 24 GB card — verify Q8/BF16 KLD
< 0.001 on a small sanity slice and document the substitution in the
manifest entry.

The reference dump path is the single point of failure for
reproducibility. If someone six months from now runs the eval and
gets numbers that don't match the committed table, the first
hypothesis is *"did the reference get re-dumped?"* — and tracking
that down is awful. Commit the producing command (exact llama.cpp
args + commit SHA) alongside the reference SHA in `manifest.json`.

## Reference-drift canary

Tiny separate fixture: 10 wikitext sequences + a frozen
`qwen3.5-9b-q8_0.gguf` GGUF. Expected KLD against the reference is
committed in-tree (`canary.md`). Eval harness runs canary first; if
KLD diverges from committed value beyond tight tolerance, it aborts
with "reference drift detected — rerun reference dump or fetch correct
one." Catches the worst failure mode (silent reference replacement) in
seconds, not after a 20-run eval has already burned an evening.

## Result table format (per gate)

```markdown
| Model     | Variant       | Size GB | mean KLD | p99 KLD | PPL    | Δ vs Q8 |
|-----------|---------------|---------|----------|---------|--------|---------|
| 9B        | Q8-uniform    | 9.4     | 0.0008   | 0.012   | 9.81   | —       |
| 9B        | MQ4-uniform   | 5.2     | 0.014    | 0.18    | 10.34  | +0.013  |
| 9B        | MQ3-uniform   | 4.1     | 0.087    | 1.44    | 42.03  | +0.086  |
| 9B        | MQ4-lloyd     | 6.5     | 0.012    | 0.16    | 10.20  | +0.011  |
| 9B        | MQ3-lloyd     | 4.4     | 0.039    | 0.71    | 18.52  | +0.038  |
... etc per model
```

(Numbers above are illustrative — 9B PPLs come from PR #115 figures;
KLD numbers are placeholders until measured.)

Gate cell: bold the comparison that gates ship (e.g., MQ3-uniform /
MQ4-uniform = 6.2× → fails 5% gate; MQ3-lloyd / MQ4-lloyd = 3.3× →
also fails the strict 5% gate but is meaningful enough to revisit
threshold).

## DFlash τ table (separate, since metric differs)

Reuse the merge_sort canonical prompt per CLAUDE.md "Prompt-structure
τ sensitivity" rules — byte-identical prompt with md5 recorded
alongside the table. 27B target × 9B/27B draft × {uniform, lloyd}
cross matrix. Already partially measured during MQ3 ship work; this
run formalizes it.

## Unsloth corroboration (optional)

Once the local Pareto exists, if Unsloth has dense Qwen3.5-27B or
Qwen3.6-27B numbers (not just the MoE 35-A3B), append their points to
the Pareto plot as a separate line. If shapes match, that's
confidence; if not, a footnote on methodology delta.

Strictly cherry-on-top — no dependency in the eval pipeline. The
local llama.cpp Pareto from the GGUF anchors is the source of truth
for hipfire-vs-GGUF positioning; Unsloth is external sanity check
only.

Caveat on co-plotting: Unsloth's GGUF Q-family points and hipfire's
MQ family use different quant schemes. We can plot on the same
KLD-vs-size axes and compare the *frontiers*, but cannot claim "MQ3 ≈
Q3_K_M" — only "MQ3 hits a similar Pareto position to Q3_K_S at
roughly equal size."

## Extending later

Adding a new quant variant (same model):

1. (No manifest change.)
2. `cargo run --example eval_kld --release -- --variant <name>
   --model <path>` (or `eval_gguf.sh <gguf>` for GGUF).
3. Append row to `results/<date>-<gate>.md`.

Adding a new model family:

1. Dump BF16 reference (~30 min GPU time for 27B-class), upload to HF
   Hub, add row to `manifest.json` with SHA256 + producer command.
2. Run all variants of the matrix.
3. New results file dated for the cohort.

Changing slice or n_ctx starts a new comparable cohort — old numbers
don't carry over. Treat as a new baseline file with its own date
stamp; don't try to reconcile.

## Storage and cost

- Slice: ~1 MB in git.
- Per-model BF16 reference: ~2–5 GB on HF Hub (top-K=256 + residual at
  fp16). Three models = ~6–15 GB total external.
- Per-quant candidate dump: not persisted by default — KLD scalar is
  the artifact, raw logit dump is intermediate. Optional `--keep-logs`
  flag for debugging.
- Reference dump cost: ~30 min GPU time per 27B-class model
  (one-time).
- Per-quant eval cost: ~5–10 min per variant (1024 seqs × 2048 ctx,
  batch 512). 20 runs ≈ 2–3 GPU hours total.

## Open questions to resolve before starting

1. **Threshold recalibration:** the 5% MQ3-vs-MQ4 gate is almost
   certainly wrong for KLD ratios. Decide whether to leave the issue
   gates as "MQ3 within 5% KLD of MQ4" (probably fails everything) or
   restate as "MQ3 KLD within Nx of MQ4" with N picked from the data.
   Defer until first numbers land.
2. **BF16 vs Q8 reference:** if 27B BF16 doesn't fit on the dev card,
   does the team accept Q8_0 as reference proxy with a documented
   Q8/BF16 KLD < 0.001 sanity check? Alternative: dump on a multi-GPU
   host once and never re-dump.
3. **HF org/repo for refs:** which HF org owns
   `hipfire-eval-refs`? Personal account vs project org affects
   whether re-uploads can be SHA-pinned externally.
4. **MQ4-lloyd inclusion:** confirm we want one row even though #116
   marks it out of scope. Cost is one extra eval run; benefit is
   empirical anchor for the deprioritization rationale.

## Sequencing

1. Land the harness skeleton: slice + `kld_reduce.py` +
   `eval_gguf.sh` + canary fixture + `manifest.json` schema. No
   reference dumps yet.
2. Dump 9B BF16 reference, upload to HF, fill manifest. Verify canary
   passes.
3. Run uniform track on 9B (3 variants: Q8, MQ4, MQ3). Validate the
   numbers reproduce PR #115's PPLs within noise. This is the
   methodology-validation step.
4. Add `eval_hipfire.rs` if not already in step 1; run Lloyd track on
   9B (MQ4-lloyd, MQ3-lloyd). Check Lloyd-vs-uniform delta matches
   PR #115's expectations.
5. Repeat steps 2–4 for 27B and 27B-3.6.
6. Run GGUF anchor track on 27B-3.6.
7. Write up `results/2026-05-XX-uniform-vs-lloyd.md` with the full
   tables. Post comments on #113 (uniform table + gate decision) and
   #116 (Lloyd table + new gate decision).
8. (Optional) Unsloth corroboration plot if same-model data is found.

## References

- Issue #113 — uniform MQ3 quality gate (this doc supersedes the
  PPL-only scoping)
- Issue #116 — Lloyd-MQ3 ship gates (this doc adds the quality gate
  as a third sub-section)
- PR #115 — Lloyd-Max codebooks for MQ3 implementation; source of
  preliminary 9B PPL numbers
- `docs/plans/mq-sub4bit-prd.md` Section 3 — quality validation
  requirement
- `docs/plans/mq-sub4bit-research-queue.md` Q1 — Lloyd-Max codebook
  research item, blocked on this eval
- `benchmarks/results/lloyd_max_findings_20260501.md` — empirical
  Lloyd writeup
- CLAUDE.md "Prompt-structure τ sensitivity" — byte-identical prompt
  rule for the DFlash τ table
- llama.cpp `llama-perplexity` `--kl-divergence` /
  `--kl-divergence-base` modes — reference dump and KLD computation
- Unsloth Qwen3.6 GGUF Pareto plot
  (https://unsloth.ai/docs/models/qwen3.6) — external KLD-vs-size
  data, MoE only as of writing
