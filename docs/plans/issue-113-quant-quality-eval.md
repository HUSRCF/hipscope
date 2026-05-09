# Quant quality eval — KLD-primary harness for MQ3/MQ4 (uniform + Lloyd) + Q8 reference proxy + GGUF anchors

**Status:** plan rev-3.3 (2026-05-09 ~20:30). Steps 0–4, 6 part-A, 7.A done (both BF16 refs produced); Step 5 (canary candidate) in progress on gfx1100. Format β locked; qwen3.5-27B dropped (superseded by qwen3.6-27B); GGUF anchor architecture locked (no FIFO tee / no llama.cpp-native cached ref needed — both tracks use the same hipfire-β ref, candidate-side KLD math is identical, only the cand-logit source differs). Step 1.5 + 1.6 verdicts in §"Step 1.5 verdict" and §"Open questions resolved" row 11.
**Tracking:** #113 (uniform), #116 (Lloyd, mirror sub-section).
**Pinned llama.cpp commit:** `9dcf83552887bb898b4a98a5761361e504e31fc3` (master, 2026-05-08).

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

**Caveat on top-K=256 truncation (M1):** Qwen 3.5/3.6 has **248,320-token vocab** (corrected from earlier "~151K" estimate after Step 1.6 measured the actual reference); top-K=256 covers ~0.103%. `sum_p_residual` corrects mean-KLD in expectation but provides **zero information about the structure of the truncated tail** — exactly the regime DFlash speculative acceptance and high-temperature sampling are most sensitive to. KLD on top-K=256 is a **necessary but not sufficient** signal for those use cases. Step 1.6 of sequencing measures the residual mass fraction empirically; the 9B BF16 reference at top_k=256 has median residual 0.41% (under the 0.5% gate), p99 17.9%, max 72.0% — passing the gate, with a long-tail caveat to flag in the result write-up.

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

Per model: 9B (qwen3.5) / 27B (qwen3.6).

(Dropped qwen3.5-27B from rev-3.1: superseded by qwen3.6-27B which has the same parameter budget and a newer training corpus. Carrying both adds eval time without an editorially-actionable signal.)

| Track | Variants |
|---|---|
| Hipfire (all evaluated together) | Q8-uniform, MQ4-uniform, MQ3-uniform, MQ4-lloyd, MQ3-lloyd |
| GGUF anchors (secondary, conditional) | Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q3_K_M, Q3_K_S — qwen3.6-27B only |

Total: 5 hipfire variants × 2 models × 2 archs = 20 hipfire runs. GGUF anchors (7 runs) on qwen3.6-27B only, gfx1151 only. 2 BF16 reference dumps (one-time per model, gfx1151 only — see B3 below).

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

**Resolved 2026-05-08 (Step 1.5 ran on 9B BF16 GGUF):** the parity check failed structurally — hipfire and llama.cpp diverge on ~46% of token positions on the slice. **However, the divergence does not block the eval pipeline.** See §"Step 1.5 verdict" below for the full reasoning. tl;dr: `eval_hipfire.rs` reads token IDs from the reference file (written by llama-perplexity); it never re-tokenizes the slice. So whichever tokenizer produced the IDs in the reference is the only tokenizer that matters; the candidate model just consumes valid IDs from its vocabulary.

The original concern (below) was over-cautious. Keeping the §"Tokenizer alignment + bridge investigation" framing for context, but the bridge work / drop-anchor decision tree is moot: we proceed with llama-perplexity for both BF16 reference dump AND GGUF candidate evaluations, eval_hipfire reads token IDs from reference, no re-tokenization anywhere in the pipeline.

### Original concern (pre-Step-1.5)

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

### Step 1.5 verdict (2026-05-08, 9B BF16 GGUF + slice md5 `83b0205a`)

Empirically:
- hipfire produced 2,407,713 tokens
- llama.cpp produced 2,407,712 tokens (Δ = 1)
- 1,104,899 of 2,407,712 positions differ (45.9%)
- 11,638 contiguous diff runs, mostly length 2
- **Every divergence is the same pattern**: hipfire emits `[2071, 110]` where llama.cpp emits `[220, 28495]` — same merge work, different pair chosen by the BPE merge-priority tiebreaker on a specific byte sequence. Streams realign within 2 tokens.

This is a known behavior difference between llama.cpp's GGUF-bundled BPE encoder and the upstream HF Qwen tokenizer. PR #201's encoder has been verified byte-identical to HF tokenizers; the divergence is on llama.cpp's side, not hipfire's.

**Why this doesn't block the eval pipeline:**

`eval_hipfire.rs` reads the per-token reference's `tokens` field (written by llama-perplexity during the BF16 ref dump) and feeds those IDs directly into `forward_scratch`. It **never re-tokenizes the slice**. So the candidate model just consumes the reference's token IDs — same vocabulary, valid input regardless of which BPE produced them.

| | producer | consumer | tokenizer agreement needed? |
|---|---|---|---|
| BF16 reference | llama-perplexity → ref file | (none) | — |
| hipfire candidate | hipfire forward_scratch | KLD vs ref | **No** (reads token IDs from ref) |
| GGUF candidate | llama-perplexity on same slice | KLD vs ref | only that llama-perplexity is deterministic on its own slice (which it is) |

**Implication for §"GGUF anchor track is viable":** the anchor track is viable. We can run Q3_K_S etc. against the same BF16 reference and compare KLDs.

**The only place tokenizer parity would matter:** if a USER ran hipfire on the same slice TEXT and expected to get the same per-token NLL/PPL as llama.cpp — they wouldn't, because the tokenization differs. But that's a user-facing comparison, not the eval pipeline. The Pareto plot's KLD axis is internally consistent (all measurements use llama-perplexity's tokenization).

**Decision: proceed with both tracks (hipfire candidates + GGUF anchors) using llama-perplexity tokenization throughout.** No bridge work, no anchor-track drop. The plan rev-3.2 already permits this — we're just observing that the pessimistic fallback (drop anchor on parity fail) was unnecessary.

## Repo layout (committed)

```
benchmarks/quality-baselines/
├── slice/
│   ├── wikitext2-1024s-2048ctx.txt        # frozen prompt bytes, ~10 MB
│   ├── slice.md5                          # checksum (md5 tripwire)
│   └── make_slice.sh                      # deterministic regenerator (uses .venv/bin/python3)
├── harness/
│   ├── manifest.json                      # {sha256, hf_repo, producer_cmd, llamacpp_commit, host_arch, …}
│   ├── kldref_format.py                   # HFKLDR + HFKSEQ-v2 reader/writer
│   ├── kld_reduce.py                      # reducer + bootstrap CI + PPL column
│   ├── tokenizer_parity.py                # step 1.5
│   ├── canary.md                          # 11-seq fixture (10 short + 1 long-ctx)
│   └── README.md                          # how to add a quant
├── refs/
│   └── (downloaded blobs, .gitignored — fetched via scripts/fetch-eval-refs.sh)
└── results/
    └── 2026-05-XX-quant-pareto.md          # output table + plot

# Producer / candidate-side binaries live in
# crates/hipfire-runtime/examples/:
#   build_kld_ref.rs   — BF16 ref dump (llama-perplexity FIFO + top-K reduce)
#   eval_hipfire.rs    — hipfire candidate scorer (forward + KLD + NLL)
#   eval_gguf.rs       — GGUF candidate scorer (llama-perplexity FIFO + KLD + NLL)
#   tokenize_slice.rs  — slice tokenizer for parity check
```

(m9 — moved manifest.json under harness/ for dependency clarity.)

## Binary format

### llama.cpp's native `--kl-divergence-base` format (verified Step 0)

For reference; we **do not use this format directly** for the cached reference (see "Hipfire derived format" below). Verified against `tools/perplexity/perplexity.cpp` on commit `9dcf83552`:

```
Header (16 bytes):
  bytes  0-7   magic "_logits_"     (8 ASCII chars, no null) [perplexity.cpp:1709]
  bytes  8-11  n_ctx                (uint32)                  [line 1718]
  bytes 12-15  n_vocab              (int32)                   [line 1726]
  bytes 16-19  n_chunk              (int32)                   [line 1727]

Tokens:
  n_ctx × n_chunk × int32 token IDs                           [line 1737]

Per-chunk × per-scored-token (only n_ctx − 1 − n_ctx/2 tokens scored per chunk):
  nv = 2 × ((n_vocab + 1) / 2) + 4   uint16 values             [line 1752]
    [0..1]    scale          (fp32, packed as 2 × uint16)
    [2..3]    min_log_prob   (fp32, packed as 2 × uint16)
    [4..nv]   stored[i]      (uint16, FULL vocab)              [line 100-101 producer]
    Reconstruction: log_prob[i] = scale * stored[i] + min_log_prob   [line 222-225 consumer]
    Quantization: stored = nearest_int((logit[i] − min_logit) / scale), 0 if logit ≤ min_logit
    min_logit clipped to max_logit − 16
```

**No magic-version field. No top-K. Full vocab.** Storage at Qwen 248,320 vocab × 1175 chunks × 1023 scored tokens/chunk = **~597 GB per reference** (corrected after Step 1.6 measured the actual chunking — llama-perplexity emitted 1175 chunks not the 1024 we'd planned). For 2 models = ~1.2 TB. **Unhostable on HF.**

### Numerics (verified Step 0)

llama.cpp's reference producer uses:
- `double sum_exp = 0.0` accumulator [line 87] — fp64 ✓
- `log_sum_exp = log(sum_exp)` formulation [line 91] — log-sum-exp ✓

`build_kld_ref` (below) and `eval_hipfire.rs` MUST match these conventions. Verify by computing top-K + residual on 10 random tokens of the 9B canary fixture and comparing fp32-tolerance against the values llama-perplexity produces in-memory (uses fp64 accumulator → fp32 output).

### Hipfire-derived top-K format (option B — what we actually use)

Step 0 surfaced that llama.cpp's native format is unhostable. We adopt **option B**: produce a hipfire-controlled top-K-reduced format from llama.cpp's full-vocab inference output. This trades "shared metric impl with llama.cpp" for "fits on HF" — `kld_reduce.py` becomes the canonical KLD reducer; `llama-perplexity --kl-divergence-base` is no longer used as our consumer.

**Format spec:**

```
Header (32 bytes):
  bytes  0-7   magic "HFKLDR\0\0"   (8 ASCII chars, null-padded)
  bytes  8-11  version              (uint32, currently 1)
  bytes 12-15  n_ctx                (uint32)
  bytes 16-19  n_vocab              (uint32)            [for sanity vs candidate]
  bytes 20-23  n_chunk              (uint32)
  bytes 24-25  top_k                (uint16, e.g. 256)
  bytes 26-27  flags                (uint16, currently 0)
  bytes 28-31  reserved             (uint32, zero)

Tokens:
  n_ctx × n_chunk × uint32 token IDs

Per-chunk × per-scored-token (n_ctx − 1 − n_ctx/2 tokens per chunk):
  uint32 top_indices[top_k]         [vocab IDs, descending log-prob]
  fp32   top_log_probs[top_k]       [log P(i), descending; fp32]
  fp32   sum_p_residual             [Σ P(i) for i NOT in top-K, in [0, 1]; fp32]
  fp32   reserved_pad               [zero, for 8-byte alignment]
```

Per-token storage: `top_k*4 + top_k*4 + 4 + 4 = 8 + 8*top_k` bytes. At top_k=256: **2,056 B per token** (~150× smaller than llama.cpp's native 304 KB). At 1175 × 1023 = 1,202,025 scored tokens × 2,056 B = **~2.47 GB per reference**, or ~4.94 GB for 2 references (9B + qwen3.6-27B) — fits HF. (n_chunk=1175 corrected from earlier 1024 estimate after Step 1.6 measured the actual llama-perplexity chunking on the slice; n_chunk depends on tokenized slice length / n_ctx.)

Reconstruction at consumer: `log P(i) = top_log_probs[j]` if `i == top_indices[j]` for some `j`. For `i` not in top-K, the bulk mass is bounded by `sum_p_residual`. KLD math proceeds with top-K from candidate × top-K from reference, plus a residual cross-term.

**Why log-probs (β) rather than raw logits + max_logit + log_sum_exp (α)?** llama.cpp's `--kl-divergence-base` format encodes log-probs already (the per-token `min_log_prob` + `scale` reconstruct to log-probs, not raw logits — see `tools/perplexity/perplexity.cpp:222-225`). Storing log-probs directly avoids backing out `max_logit` / `log_sum_exp` from llama.cpp's encoding (an underdetermined problem when the logit range is exactly clipped to 16). KLD operates on log-probs natively, so no information loss for our use case.

**Why this format vs llama.cpp's:**
- 150× smaller per token (top-K + residual vs full uint16)
- Stored values are fp32 not affine-quantized uint16 → no quantization noise in the reference
- top_k explicit + stored in header → no dependency on n_vocab divisibility
- Full vocab IDs preserved (top_indices) so consumer can match against candidate's top-K
- Token ID stride preserved so `eval_hipfire.rs` can align without re-tokenizing the slice

**Producer:** `crates/hipfire-runtime/examples/build_kld_ref.rs`. Architecture:

```
build_kld_ref --bf16-gguf qwen3.5-9b-bf16.gguf \
              --slice slice.txt \
              --top-k 256 \
              --output refs/qwen3.5-9b-bf16.kldref.bin

[1] mkfifo /tmp/kldref-<pid>.fifo
[2] spawn: llama-perplexity -m <gguf> -f <slice> --kl-divergence-base /tmp/kldref-<pid>.fifo -c <n_ctx> --kl-divergence
[3] read llama.cpp header from FIFO (16 bytes) → parse n_ctx, n_vocab, n_chunk
[4] read tokens from FIFO (n_ctx * n_chunk * 4 bytes) → write hipfire header + tokens to output
[5] for each per-token block (nv*2 bytes from FIFO):
       parse scale (fp32), min_log_prob (fp32) from first 8 bytes
       reconstruct log-probs: log_p[i] = scale * stored[i] + min_log_prob   [fp32 → fp64]
       top-K-reduce (nlargest by log_p)
       compute sum_p_residual = 1 - Σ_{i ∈ top_k} exp(log_p[i])             [fp64]
       (clamp ≥ 0 to absorb fp roundoff)
       write hipfire β per-token block (8 + 8*top_k bytes)
[6] join llama-perplexity, unlink fifo
```

The FIFO sidesteps the 318 GB transient: llama.cpp streams full-vocab uint16, hipfire's reducer reads + reduces in-flight, ~2.15 GB lands on disk.

**Top-K choice (M1):** at top_k=256, residual-mass sanity (Step 1.6) checks median `sum_p_residual` value on 10 canary tokens. <0.5% → 256 OK; >2% → raise to 512. Record actual K in manifest. Earlier rev-3 estimate stands.

**Numerics:** all sums (log-prob reconstruction, top-K accumulators, residual computation) use fp64 internally, written as fp32. Matches llama.cpp's accumulator precision.

## Reference dump → HF Hub

Per-model BF16 reference logit dumps live at **`hipfire-models/hipfire-eval-refs`** (m1).

**On the org choice:** `hipfire-models` is a new HF org created 2026-05-08 (see project memory; we already use it for the dev-stage Lloyd quants at `hipfire-models/qwen3.5-{4b-dev,9b-dev,9b}` and `hipfire-models/qwen3.6-27b-dev`). The legacy registry (`cli/registry.json`) still references `schuttdev/hipfire-*` for non-Lloyd quants where the canonical owner has write access. **Eval refs go to `hipfire-models` because Kevin (org admin) has write access; `schuttdev` is read-only.** This will look like an inconsistency in committed state at PR review time — flag it in the PR description.

Files (corrected after Step 0; format = hipfire top-K-reduced, not llama.cpp full-vocab):

```
qwen3.5-9b-bf16.kldref.bin     (~2.47 GB, top-K=256 + residual + meta, see Binary format §)
qwen3.6-27b-bf16.kldref.bin    (~2.47 GB — same size at top_k=256, vocab-independent)
README.md                       (producer commands, slice md5, llama.cpp commit pin, host_arch=gfx1151, model SHA256)
```

(Both refs are the same size at top_k=256: per-token block size is `8 + 8*top_k = 2,056 B` independent of n_vocab. n_vocab is only stored in the header for sanity-check.)

In-tree `manifest.json` carries `{sha256, hf_url, producer_cmd, model_sha256, llamacpp_commit, slice_md5, host_arch, top_k}` per reference. `scripts/fetch-eval-refs.sh` reads manifest, pulls via `hf` CLI into `~/.cache/hipfire-eval-refs/`, verifies SHA256.

## Reference dump methodology (one-time per model, on gfx1151)

```bash
cargo run --release --example build_kld_ref -p hipfire-runtime -- \
  --bf16-gguf  models/qwen3.5-9b-bf16.gguf \
  --slice      benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
  --top-k      256 \
  --output     refs/qwen3.5-9b-bf16.kldref.bin
```

**BF16 only**, produced via llama.cpp's inference (B3). The hipfire runtime does not support BF16 inference — see `crates/hipfire-runtime/src/llama.rs:481`, which panics on any `GgmlType` outside `{F32, F16, Q4_0, Q8_0, Q4K, Q6K}`. `build_kld_ref` orchestrates llama-perplexity as a subprocess, reads its full-vocab uint16 stream via FIFO, computes top-K + residual in flight, and writes the hipfire-internal format. `eval_hipfire.rs` runs hipfire's quantized variants only (Q8, MQ3-uniform, MQ4-uniform, MQ3-Lloyd, MQ4-Lloyd) and scores them against the externally-produced BF16 reference.

gfx1151 has 137 GB UMA — 27B BF16 fits with huge headroom. Disk needs are minimal (~2 GB output) since the FIFO sidesteps the 318 GB transient.

**Producer command + SHA recorded** alongside the reference SHA in `manifest.json` so divergence-debugging 6 months from now is trivial. The pinned llama.cpp commit (`9dcf83552`) is also recorded — if a future contributor rebuilds llama.cpp from a different commit and the producer format changes, the SHA mismatch surfaces it.

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

**Bench-host commitment is much larger than rev-2 estimated.** Pre-reserve before starting. (Costs updated post-rev-3.1 to reflect dropping qwen3.5-27B; matrix is now 2 models × 5 variants × 2 archs = 20 hipfire runs.)

- **gfx1100 subset:** 5 hipfire variants × 2 models × ~67 min/run = **~11 GPU-hours**.
- **gfx1151 subset:** 5 hipfire variants × 2 models × ~3 hours/run = **~30 GPU-hours**.
- **GGUF anchors:** 7 runs on gfx1151 × ~1.5 hr = **~10 GPU-hours**.
- **BF16 reference dumps:** 2 models × ~1.5 hr (gfx1151) = **~3 GPU-hours**.
- **9B CPU BF16 sanity (m3):** ~30 min.
- **Total: ~55 GPU-hours wall-clock.** Add ~20% padding for retries and harness shakedown → **~65 GPU-hours.**

Storage (corrected after Step 0; format β: 8 + 8*top_k bytes/token):
- Slice text: ~1 MB in git.
- Slice tokens.bin (post-bridge): ~8 MB if 1024 × 2048 × 4 B, in git.
- Per-model BF16 reference (hipfire top-K β format): **~2.47 GB** on HF Hub (top_k=256: 8+8*256 = 2,056 B/token × 1175 chunks × 1023 tokens/chunk). Two models = **~4.94 GB external**. Comfortable on hipfire-models org.
- Reference-build transient: ~597 GB of full-vocab uint16 streams from llama-perplexity per model dump, but never lands on disk — FIFO-piped through `build_kld_ref`'s top-K reducer. Disk requirement during a ref dump = ~3 GB scratch + the ~2.47 GB output.
- Per-quant per-sequence KLDs (CI reproducibility): 1024 × 4 B per row = 4 KB; total across 20+7 rows = ~110 KB; in git or attached to the result file.
- Optional `--keep-logs` for per-token logits (debugging only): ~2-5 GB per variant per arch, not persisted by default.

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
| 11 (new) | Top-K choice for 248K vocab | Step 1.6 measured (2026-05-08, on partial 9B BF16 ref): median 0.41% < 0.5% gate, p99 17.9%, max 72.0%. **Top-K=256 confirmed**, with long-tail caveat for tokens with very flat distributions (~1% of all tokens). |
| 12 (new) | sum_exp_residual numerics | log-sum-exp + fp64 accumulation, verified against llama.cpp on canary. |
| 13 (new) | Historical baselines comparable? | No. Document in result-file preamble. |
| 14 (new) | KV mode for eval | `asym3`, recorded per-row. |
| 15 (new) | DFlash τ included? | Yes, as a column on rows where a draft exists. Canonical merge_sort prompt + md5. |
| 16 (rev-3.1) | Reference format | **Hipfire-internal top-K-reduced** (option B), derived from llama.cpp's full-vocab native format via FIFO streaming in `build_kld_ref`. Native format is too large (~318 GB/ref). |
| 17 (rev-3.1) | llama-perplexity tokens-as-input flag | Confirmed absent on commit `9dcf83552`. Tokenizer-parity (Step 1.5) is the primary mitigation; if it fails, drop GGUF anchor track (rev-3 plan stands). |
| 18 (rev-3.2) | Reference format detail (α vs β) | β: log-probs + sum_p_residual + reserved_pad. 8 + 8*top_k bytes/token. KLD-equivalent to α; avoids machinery to back out llama.cpp's encoded `min_log_prob` to raw logits. Used as final spec. |
| 19 (rev-3.2) | qwen3.5-27B in eval matrix | **Dropped.** Superseded by qwen3.6-27B (newer training corpus, same parameter budget). Matrix is now 9B (qwen3.5) + 27B (qwen3.6). Drops ~20 GPU-hours from the bench commitment. |

## GGUF anchor architecture (rev-3.3, locked 2026-05-08)

After Step 1.5's verdict ("tokenizer parity fails but doesn't block"), the
GGUF anchor track architecture became clearer:

**The cached BF16 reference (hipfire-β format, top-K + residual) is
sufficient input for BOTH hipfire candidates AND GGUF candidates.**
No FIFO tee, no re-dump, no llama.cpp-native cached ref needed.

Why: KLD is computed candidate-side, looking up cand's log-prob at the
*ref's* top_indices. `eval_hipfire.rs` already does this; we add
`eval_gguf.rs` mirroring the same KLD math but using a different
source for cand's logits.

| binary | cand-logit source | per-token KLD math |
|---|---|---|
| `build_kld_ref` (DONE) | (writes ref, no cand) | — |
| `eval_hipfire` (DONE) | hipfire `forward_scratch` returns full vocab logits | for each tok: read ref's top_indices + top_log_probs from ref file; compute log_p_cand[ref_top_indices] from hipfire's full vocab; KLD = Σ P_ref * (log_p_ref − log_p_cand) over ref's top-K |
| `eval_gguf` (NEW, ~half day) | spawn `llama-perplexity --kl-divergence-base <fifo>` on the GGUF candidate; read full-vocab uint16 from FIFO; reconstruct candidate's log-probs for any token via `scale * stored[i] + min_log_prob` | identical math to eval_hipfire |

Both candidate-side binaries write HFKSEQ (per-sequence mean+p99 KLD)
that `kld_reduce.py` aggregates. `eval_gguf` is a near-clone of
`build_kld_ref`'s FIFO-orchestration code but processes blocks
differently (KLD against ref's top_indices instead of own top-K
reduction).

**Residual-term enhancement (rev-3.3):** both candidate binaries should
include a residual cross-term in the KLD computation, not drop it
entirely. Approximation:
- `kld_residual ≈ sum_p_residual_ref * (log(sum_p_residual_ref) − log(sum_p_residual_cand))`
  - where `sum_p_residual_cand = 1 − Σ_{i in ref_top_K} P_cand(i)`
- This assumes both distributions miss similarly in the tail; reduces
  bias on flat-distribution tokens (~1% of all scored tokens have
  residual mass > 17%, per Step 1.6).
- Add to both `eval_hipfire.rs` and `eval_gguf.rs` in their respective
  per-token KLD inner loops. ~5 lines each.

**No artifact rework needed:** the 9B BF16 ref dump in flight (and the
27B dump still to come) produce the correct ref artifact for both
tracks. eval_gguf is the only missing piece.

## Sequencing (rev-3.3; status snapshot 2026-05-09 ~20:30)

Legend: ✓ done · ⏳ in-progress · ⏸ blocked / queued · — pending

**✓ Step 0 (DONE 2026-05-08):** read `tools/perplexity/perplexity.cpp` on the pinned llama.cpp commit `9dcf83552`. Verified the `--kl-divergence-base` binary layout (full-vocab uint16, **not** top-K), `sum_exp_residual` numerics (log-sum-exp + fp64 ✓), and that `llama-perplexity` does **not** accept pre-tokenized input. Storage estimate corrected from "3-8 GB/ref" to "~318 GB/ref native, ~2.5 GB/ref hipfire-derived". **Decision:** adopt option B (hipfire-internal top-K-reduced format derived from llama.cpp's full-vocab via FIFO streaming) — ~150× smaller, fits HF, avoids 1-2 days of llama.cpp C++ patching.

**✓ Step 1 (DONE):** harness skeleton landed. `benchmarks/quality-baselines/{slice,harness,refs,results}/` with READMEs, kldref_format.py (β format reader/writer + HFKSEQ sidecar), kld_reduce.py (bootstrap CI + result-table emitter; smoke-tested with 8 synthetic per-seq files), manifest.json schema, make_slice.sh generator (uses .venv/bin/python3), canary.md skeleton. Commits b68864a + later.

**✓ Step 1.5 (DONE 2026-05-08, verdict: parity FAILS but doesn't block):** ran `tokenizer_parity.py` on the 9B BF16 GGUF (`/data/models/unsloth/Qwen3.5-9B/Qwen3.5-9B-BF16.gguf`) + slice md5 `83b0205a`. hipfire produced 2,407,713 tokens, llama.cpp 2,407,712 (Δ = 1). 45.9% of positions differ; structural pattern ([2071, 110] ↔ [220, 28495]) — known llama.cpp-vs-HF Qwen BPE divergence. **Doesn't block the eval pipeline:** eval_hipfire reads token IDs from the reference file (which llama-perplexity wrote during ref dump); never re-tokenizes. See §"Step 1.5 verdict" for the full reasoning. GGUF anchor track stays viable.

**✓ Step 1.6 (DONE 2026-05-08, verdict: top-K=256 confirmed):** sampled 50,000 blocks from the partial 9B BF16 ref. Median residual 0.41% (under the 0.5% gate), mean 1.81%, p99 17.94%, max 72.0%. Top-K=256 confirmed. Long-tail caveat (~1% of tokens have residual >17%) flagged for the eval write-up. Surfaced two corrections to estimates: n_vocab is 248,320 (not 151,936), n_chunk is 1175 (not 1024) — folded into the plan.

**✓ Step 2 (DONE):** wrote `crates/hipfire-runtime/examples/build_kld_ref.rs` — Rust orchestrator that mkfifo's, spawns llama-perplexity, reads the full-vocab uint16 stream, top-K-reduces in-flight (fp64 accumulators), writes the hipfire β format. End-to-end-validated by running the 9B BF16 dump (Step 4). Throughput: 375 reduced tokens/sec on gfx1151. Commit a31d4f6.

**✓ Step 3 (DONE):** wrote `crates/hipfire-runtime/examples/eval_hipfire.rs` — reads hipfire β reference, runs hipfire variants chunk-by-chunk via `forward_scratch`, computes per-token KLD via top-K-of-ref + residual cross-term (rev-3.3 enhancement), bins per-sequence, writes HFKSEQ. Builds clean (`cargo build --release --features deltanet`). Commits d4adac8 (initial) + f9dd19e (residual cross-term). Not yet run end-to-end; queued for Step 5.

**✓ Step 4-9b (DONE 2026-05-08):** dumped 9B BF16 reference on gfx1151. Output: `benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin`, **2.48 GB**, 53 min wall-time, 375 reduced tokens/sec. Header: n_ctx=2048, n_vocab=248320, n_chunk=1175, top_k=256, total scored = 1,202,025 blocks. sha256 `06948cd36bab71fce2df5d9af1be03c9cfb4090637d881056a6937a29caa65a7`; manifest.json populated (commit `0214e8c`). **Pending sub-steps:** upload to `hipfire-models/hipfire-eval-refs`, verify canary on both NORMALIZE_PROMPT settings.

**⏳ Step 5 (in progress on gfx1100):** running hipfire track on 9B × {gfx1100, gfx1151} (5 variants × 2 archs = 10 runs). First canary candidate (qwen3.5-9b.mq4 vs 9B BF16 ref) launched 2026-05-08 on gfx1100; throughput ~47.5 tok/s, total run ~7 hours (slower than the plan's 75 min estimate — gfx1100 is power-capped at the kernel-throughput ceiling per the rocm-smi snapshot). The first run doubles as the canary-expected-KLD source — its per-sequence KLDs become the committed `canary.md` expected-values when complete. Pre-Step-5 review-hardening fix bundle landed (commits `e407128` + `0214e8c` + `e7898f4` + `fd06646` + `3dee758` + `41b5977` + `dec913e`): C1 scoring window off-by-one, H1 commit-pin enforcement, H2 eval-mode env vars, H3 stale dead code, M1 sha256 ref validation, M3 PPL accumulator + HFKSEQ v2, M4 slice md5 enforcement, M5 fetch-eval-refs.sh, H7 cand-vs-ref token compare.

**✓ Step 6 part-A (DONE 2026-05-09):** dumped 27B BF16 reference on gfx1151. Output: `benchmarks/quality-baselines/refs/qwen3.6-27b-bf16.kldref.bin`, **2.48 GB**, 2h 3m wall-time, 162 reduced tokens/sec. Same shape as 9B (per-token block size depends on top_k, not n_vocab). sha256 `8af83b38710fbc8e5ee46ce2b84b3545381c834f17bf6dfaa15fd817e4734446`; manifest.json populated (commit `52879fe`). **Required `--no-mmap`** (now passed automatically by build_kld_ref): demand-paging on 50 GB BF16 weights caused eviction-cycle stall on the 124 GB UMA host until enabled. GGUF copied locally to `~/models-local/Qwen3.6-27B/` from NFS to reduce per-page-fault read latency. VM `enertree` shut down to free working-set RAM. **Pending Step 6 part-B:** run 5 hipfire variants × 2 archs = 10 runs on the 27B ref.

**✓ Step 7.A (DONE 2026-05-08):** wrote `crates/hipfire-runtime/examples/eval_gguf.rs` — mirrors eval_hipfire.rs's KLD math but uses FIFO-streamed llama-perplexity as the candidate-logit source. Same residual cross-term as eval_hipfire. Builds clean. Not yet run end-to-end. Commit f9dd19e.

**⏸ Step 7.B:** run all 7 GGUF candidates (Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q3_K_M, Q3_K_S) on the qwen3.6-27B BF16 ref (cached from Step 6). Each ~1.5 hr on gfx1151.

**— Step 8:** measure DFlash τ (canonical merge_sort prompt, md5-pinned) for each variant where a draft model exists. Adds the `DFlash τ` column to the result table.

**— Step 9:** write up `results/2026-05-XX-quant-pareto.md` with the full table + Pareto plot + caveats preamble. Post comments on #113 (positioning), #116 (positioning + recalibration input), #197 (MQ4-Lloyd promotion-or-not editorial recommendation).

**— Step 10 (optional):** Unsloth corroboration plot if same-model dense data exists.

**— Step 11 (follow-up, not this PR):** p99-gate calibration once mean-KLD CI variance is understood empirically. GEMV single-acc port (universal multi-acc drift fix) if the gfx1100-vs-gfx1151 KLD delta is >10%.

### Code-state snapshot

| File / artifact | Status |
|---|---|
| `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt` | ✓ committed (10.5 MB, md5 `83b0205a304bf4e52172ecdb05f2e895`) |
| `benchmarks/quality-baselines/slice/slice.md5` | ✓ committed |
| `benchmarks/quality-baselines/slice/make_slice.sh` | ✓ committed (uses .venv/bin/python3) |
| `benchmarks/quality-baselines/harness/manifest.json` | ✓ committed; both 9B and 27B refs registered with sha256 + producer_cmd; `hf_url` null pending HF upload |
| `benchmarks/quality-baselines/harness/kldref_format.py` | ✓ committed (β format reader/writer) |
| `benchmarks/quality-baselines/harness/kld_reduce.py` | ✓ committed (bootstrap CI + table emitter) |
| `benchmarks/quality-baselines/harness/tokenizer_parity.py` | ✓ committed (full impl, tested 2026-05-08) |
| `benchmarks/quality-baselines/harness/canary.md` | ✓ committed (11 sequences populated; expected KLDs TBD) |
| `benchmarks/quality-baselines/harness/eval_gguf.sh` | deleted (was stub); superseded by `crates/hipfire-runtime/examples/eval_gguf.rs` |
| `crates/hipfire-runtime/examples/build_kld_ref.rs` | ✓ committed; end-to-end validated on both 9B + 27B; auto-passes `--no-mmap` to llama-perplexity (commit `52879fe`) |
| `crates/hipfire-runtime/examples/eval_hipfire.rs` | ✓ committed (with residual cross-term, NLL accumulator, env-var forcing, sha256 verify); first end-to-end run on gfx1100 in progress |
| `crates/hipfire-runtime/examples/eval_gguf.rs` | ✓ committed (with residual cross-term, NLL accumulator, cand-vs-ref token compare, commit-pin verify); not yet run end-to-end |
| `crates/hipfire-runtime/examples/tokenize_slice.rs` | ✓ committed (used by tokenizer_parity.py) |
| `scripts/fetch-eval-refs.sh` | ✓ committed (M5; uses `.venv/bin/python3` + huggingface_hub) |
| `benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin` | ✓ produced (2.48 GB, gfx1151, 53 min wall, 2026-05-08); local only (gitignored) |
| `benchmarks/quality-baselines/refs/qwen3.6-27b-bf16.kldref.bin` | ✓ produced (2.48 GB, gfx1151, 2h 3m wall, 2026-05-09); local only (gitignored) |

### Sibling work executed

- **PR #201 cherry-picked** into this branch (commits 1d8c140 + 4f99c19): O(N²) → O(N log N) BPE encoder fix. Reduced full-slice tokenization from a projected ~30 min to **19 sec actual**. Made `tokenize_slice.rs` + `tokenizer_parity.py` tractable.
- **llama.cpp `build-strix` rebuilt** against ROCm 7.12 + gfx1151 with all targets (50+ binaries: llama-perplexity, llama-tokenize, llama-cli, etc.). Local path: `/home/kread/git/llm/llama.cpp/build-strix/bin/`. Required because the existing build was linked against `libamdhip64.so.6` (rocm-7.1) which doesn't exist on the system; rocm-7.12 ships `.so.7`.
- **Project venv at `.venv/`**: holds `numpy`, `datasets`, `huggingface_hub`, etc. for the harness's Python scripts. Per the repo-preference memory: Python deps go in venv, not pip --user.

### What's NOT yet done (blocking summary)

1. **9B BF16 ref upload to HF** — file ready, sha256 in manifest; needs `hf upload hipfire-models/hipfire-eval-refs ...` + setting the manifest's `hf_repo` field.
2. **27B BF16 ref upload to HF** — same as 9B; both refs were produced on the same gfx1151 host.
3. **Canary candidate run completion** (qwen3.5-9b.mq4 vs 9B BF16 ref) — running on gfx1100 since 2026-05-08; ETA was ~7 hours from launch. Output populates canary.md's expected-values.
4. **Bulk hipfire-track runs** (5 variants × 2 archs × 2 models = 20 runs minus the canary). At 47.5 tok/s on gfx1100 ≈ 7h/run × 19 ≈ ~5–6 days of wall-clock if serialized; gfx1151 is the parallel slot. The plan's original "~25 GPU-hours" estimate was at 9B-on-gfx1151 throughput; revised upward by the gfx1100 power-cap finding.
5. **GGUF anchor runs** (7 variants × 1 model on gfx1151) — currently ~1.5 hr × 7 = ~10 GPU-hours. Not yet end-to-end-validated.
6. **DFlash τ + write-up** — Step 8 + 9.

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
