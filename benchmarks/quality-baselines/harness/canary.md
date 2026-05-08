# Canary fixture — harness output reproducibility guard

11 wikitext-2 sequences (10 short, 1 long-ctx) with expected per-sequence
KLDs. Run before each eval; KLD divergence beyond per-sequence tolerance
aborts the eval with "harness regressed" — distinct from "reference
replaced" (caught by SHA256 in `manifest.json`).

**Purpose:** detect regressions in `eval_hipfire.rs` (kernel changes, fp
accumulator changes, prompt-handling drift) — not reference identity
changes. See plan §"Reference-drift canary (clarified)".

**Candidate model for canary:** `qwen3.5-9b.mq4` (canonical hipfire MQ4-uniform
9B; available locally at `~/.hipfire/models/qwen3.5-9b.mq4`). Candidate is
fixed across all canary runs; ref is the same as the bulk-eval ref.

## Sequences

Numbered s1–s11. Source: wikitext-2 train, distinct paragraphs picked for
character/topic diversity. Token counts measured with Qwen3.5/3.6 tokenizer.

| ID  | Length | Source description |
|-----|-------:|--------------------|
| s1  | ~120   | Wikipedia article opening — biography (early prose) |
| s2  | ~180   | Article body — historical events (multi-clause sentences) |
| s3  | ~250   | List-heavy section with named entities (high vocab variation) |
| s4  | ~220   | Geographic article — coordinates / numerical content |
| s5  | ~310   | Scientific article — technical terminology, formulas as text |
| s6  | ~280   | Cultural article — uncommon proper nouns, transliterations |
| s7  | ~350   | Plot summary — narrative, dialogue snippets |
| s8  | ~410   | Sports article — statistics, dates |
| s9  | ~470   | Long opinion-style passage — argumentative prose |
| s10 | ~490   | Mixed prose + tabular data |
| s11 | ~1900  | **Long-ctx sequence** (M8). Catches RoPE / KV-cache drift that only emerges late in the context window. Multi-paragraph article concatenation. |

(Lengths are targets; actual lengths land in the +/-10% range after token
count via the pinned tokenizer.)

## Sequence text

The actual text bytes are committed below (within markdown code blocks
delimited by `<!-- s1 -->` ... `<!-- /s1 -->` markers so the harness can
extract them programmatically). **Status:** placeholder — populated by a
follow-up commit that pulls passages from `../slice/wikitext2-1024s-2048ctx.txt`
once the slice is generated.

<!-- s1 -->
TBD
<!-- /s1 -->

<!-- s2 --> TBD <!-- /s2 -->
<!-- s3 --> TBD <!-- /s3 -->
<!-- s4 --> TBD <!-- /s4 -->
<!-- s5 --> TBD <!-- /s5 -->
<!-- s6 --> TBD <!-- /s6 -->
<!-- s7 --> TBD <!-- /s7 -->
<!-- s8 --> TBD <!-- /s8 -->
<!-- s9 --> TBD <!-- /s9 -->
<!-- s10 --> TBD <!-- /s10 -->
<!-- s11 --> TBD <!-- /s11 -->

## Expected per-sequence KLDs (TBD — fills in after Step 4)

| Seq | NORMALIZE_PROMPT=0 | NORMALIZE_PROMPT=1 | Tolerance |
|-----|-------------------:|-------------------:|----------:|
| s1  |               TBD  |               TBD  |   ±15%    |
| s2  |               TBD  |               TBD  |   ±15%    |
| s3  |               TBD  |               TBD  |   ±15%    |
| s4  |               TBD  |               TBD  |   ±15%    |
| s5  |               TBD  |               TBD  |   ±15%    |
| s6  |               TBD  |               TBD  |   ±15%    |
| s7  |               TBD  |               TBD  |   ±15%    |
| s8  |               TBD  |               TBD  |   ±15%    |
| s9  |               TBD  |               TBD  |   ±15%    |
| s10 |               TBD  |               TBD  |   ±15%    |
| s11 |               TBD  |               TBD  |   ±15%    |

The "both NORMALIZE settings" hedge addresses Gemini m2 — if KLDs diverge
meaningfully between the two, that's a finding (and the eval-mode default
of OFF is revisited).

## How the harness uses this

1. Eval harness extracts s1–s11 from the markers above.
2. Tokenizes each via the candidate's tokenizer.
3. Runs candidate forward (already done — same as bulk eval).
4. Computes per-sequence KLD against the corresponding bytes of the
   BF16 reference.
5. Compares to the table above. Per-sequence delta within tolerance →
   pass. Any sequence outside tolerance → fail with which sequence(s)
   regressed.
