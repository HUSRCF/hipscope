#!/usr/bin/env bash
# eval_gguf.sh — GGUF anchor-track wrapper for the quant-quality eval.
#
# Runs llama-perplexity on a GGUF candidate against the pre-uploaded BF16
# reference (see ../refs/), captures per-sequence KLDs in the hipfire
# HFKSEQ format that kld_reduce.py consumes.
#
# Status: STUB — full impl lands after the BF16 reference is dumped (Step 4
# of the plan). Implementation outline:
#
#   1. Spawn `llama-perplexity --kl-divergence \
#               --kl-divergence-base <bf16-ref-fifo> \
#               -m <candidate-gguf> -f ../slice/wikitext2-1024s-2048ctx.txt`
#      where bf16-ref-fifo is a FIFO that we *pre-fill* by piping the
#      hipfire-format BF16 ref back through a small "expand-to-llama-format"
#      Rust helper (since llama-perplexity expects llama.cpp's full-vocab
#      format, not our top-K).
#
#   2. Parse llama-perplexity's per-token KLD output (kld_values, p_diff).
#
#   3. Bin per-sequence (1024 chunks of (n_ctx − 1 − n_ctx/2) tokens each).
#      Compute mean + p99 per sequence.
#
#   4. Write to <variant>__gfx1151.kldseq via the HFKSEQ format.
#
# Usage (when implemented):
#   ./eval_gguf.sh <candidate.gguf> <variant-name>
#
# Plan: docs/plans/issue-113-quant-quality-eval.md §"Tokenizer alignment +
# bridge investigation" — note that this path depends on Step 1.5 passing
# (else the comparison is invalid).

set -euo pipefail
cd "$(dirname "$0")"

echo "eval_gguf.sh: STUB — not yet implemented." >&2
echo "  Lands in Step 7 of the plan, after BF16 refs exist." >&2
exit 2
