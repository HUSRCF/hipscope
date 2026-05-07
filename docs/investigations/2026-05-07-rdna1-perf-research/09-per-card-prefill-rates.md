# Exp #9: Per-card prefill rates — gfx1010 vs gfx1030 vs gfx1151

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment)

## Hypothesis under test

WMMA on gfx1151 (Strix Halo iGPU, RDNA 3.5) yields substantially higher prefill_tok_s than gfx1010 (RDNA 1, no WMMA, no vdot) and gfx1030 (RDNA 2, no WMMA, has vdot) on identical 9B mq4 prefill workload. This validates the asymmetric "WMMA-prefill-tier + RDNA1-decode-tier" architecture hypothesis empirically with zero code change.

## Lever

Same daemon binary, same model, same prompt. ROCR_VISIBLE_DEVICES selects which card runs the load. HIPFIRE_PREFILL_BATCHED=1 routes through GEMM-shape (compute-bound) kernels that exercise WMMA where available.

## Scenario

- Hardware: hipx, 3 cards visible
  - HVD=0: gfx1151 iGPU (Strix Halo, WMMA)
  - HVD=1: gfx1010 (5700 XT, no WMMA, no vdot)
  - HVD=3: gfx1030 (6950 XT, no WMMA, has vdot)
- Model: qwen3.5-9b.mq4 (fits all 3 cards solo)
- KV mode: asym3
- Prefill batched (PB=1) — exercises GEMM compute-bound path
- Prompt: NIAH 8k truncated to ~1024 tokens (warm-up amortizes JIT cold-start)
- max_tokens: 40 (just enough to capture decode_tok_s alongside prefill)
- 3 warm fresh-process runs per card, median + σ reported

## Win criterion (pre-registered)

WMMA primary lever validated (≥2× gfx1010):
- `gfx1151 prefill_tok_s / gfx1010 prefill_tok_s ≥ 2.0`

Moderate validation (≥1.5× gfx1010):
- ratio between 1.5 and 2.0

## Loss criterion

`gfx1151 prefill_tok_s ≤ gfx1010 prefill_tok_s`. Would indicate iGPU memory architecture overhead outweighs WMMA gain.

## No-change band

ratio between 1.0 and 1.5 — hypothesis weakly supported. WMMA helps but other factors (memory BW, compute throughput) dominate.

## Why this works without infrastructure changes

We are not testing the full "tier-aware PP" architecture today (which would require splitting prefill stage and decode stage onto different device sets). We're testing the per-card WMMA contribution in isolation, which tells us whether the architecture would pay off when assembled.

If gfx1151 prefill ≫ gfx1010 prefill: WMMA-prefill-tier architecture is empirically grounded.
If gfx1151 prefill ≈ gfx1010 prefill: WMMA isn't the lever; per-card memory BW or compute throughput is the differentiator. Architectural hypothesis weakens.

## Quality gate

Output coherence verified by visual check on the canonical needle prompt. NIAH retrieval not required (max_tokens=40 won't always reach the answer, this experiment isolates prefill rate, not generation correctness).

## Action on win

Document numbers. Update memory entry framing the WMMA-prefill-tier architecture as empirically validated for hipx topology. Propose: a v1.2 PRD for tier-aware Gpus assignment.

## Action on loss / no-change

Document numbers. The architecture hypothesis is weakened — WMMA is not the primary differentiator on iGPU silicon for our prefill kernels. Don't pursue tier-aware infra without re-validating on gfx1100/1101/1102 discrete (where WMMA + dGPU memory BW combine).

## Implementation note

Bench harness: standard hipfire daemon JSON protocol, fresh process per run. Prompt constructed from `benchmarks/longctx/niah/niah_8k.jsonl` filler text truncated to ~5500 chars (≈1024 tokens after BPE). Same prompt bytes across all 3 cards.
