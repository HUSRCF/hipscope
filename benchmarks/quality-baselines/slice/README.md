# Eval slice — wikitext-2 train, 1024 sequences × 2048 context

The frozen prompt bytes used by every quant-quality eval. Committed
directly so all comparable cohorts share byte-identical inputs (per
CLAUDE.md's "Prompt-structure τ sensitivity" rule, generalized:
shared bytes = comparable numbers).

## Files

- `make_slice.sh` — deterministic generator. Fetches wikitext-2 train,
  picks 1024 non-overlapping 2048-token windows (counted with a
  pinned tokenizer), concatenates into `wikitext2-1024s-2048ctx.txt`.
  Records md5 in `slice.md5`. Run once when first standing up the
  eval; output committed.
- `wikitext2-1024s-2048ctx.txt` — the slice. ~10 MB. **Generated; do
  not edit by hand.**
- `slice.md5` — md5 tripwire. The harness asserts this matches before
  any eval run; mismatch = abort.
- `tokens.bin` — u32 token IDs, post-tokenizer-parity check. Present
  if Step 1.5 fell back to the token-input bridge (per the plan); in
  the happy case where llama.cpp's tokenizer matches hipfire's on
  this slice, this file is absent and both stacks tokenize the text
  natively.

## Why wikitext-2 train, not test?

Wikitext-2 test is ~245K tokens — far short of the 2.1M tokens the
matrix needs (1024 × 2048). Train is ~2.5M tokens, enough with no
overlap. Using train for eval is unconventional but acceptable here:
- We're evaluating quants of pretrained models — the models have
  presumably seen wikitext-2 train during pretraining (most large
  open models scrape Wikipedia), so PPLs / KLDs measure
  *quantization-induced perturbation*, not generalization.
- We need the same slice across every (model, variant, arch) to make
  numbers comparable; the eval is a cross-quant comparison, not a
  generalization claim.

## Why not wikitext-103?

Wikitext-103 has more text but uses a different vocabulary
construction; cross-comparison with published WT2 PPLs (which exist
for many models) is harder. WT2-train + the matrix's slice size is
the sweet spot.

## Reproducibility caveat

Once `make_slice.sh` runs and `wikitext2-1024s-2048ctx.txt` lands in
git, the slice is byte-stable and the md5 tripwire catches drift. If
the upstream HF wikitext-2 corpus changes, `make_slice.sh` re-run
would produce different bytes — the md5 mismatch surfaces this.
**The slice text in git is the source of truth**; the script is the
recipe, not the canonical artifact.

PR #115's predecessor corpus (`dev/bench/data/wikitext2-test.txt`) is
not in git and produces unreproducible historical PPLs. This new
slice resets the comparable cohort.
