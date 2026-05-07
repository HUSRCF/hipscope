# Exp #10: gfx1151 solo DFlash 27B baseline

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment)

## Hypothesis under test

DFlash spec-decode on gfx1151 (Strix Halo iGPU, RDNA 3.5, has WMMA) is **NET POSITIVE** vs AR baseline on 27B mq4. Per memory, DFlash is net-positive on RDNA3+ silicon (gfx1100 et al) and net-negative on RDNA1 (gfx1010, -41% per `project_gfx1010_5700xt_validated_2026_05_06`). gfx1151 is unmeasured for DFlash on this rig.

This anchors expected gains for the hetero DFlash architecture (Exp #11+, draft on gfx1010, target+verify on gfx1151). If gfx1151 SOLO DFlash is already net-positive, then offloading the draft duty to gfx1010 is gravy. If gfx1151 SOLO DFlash is net-negative, the hetero variant needs different justification.

## Lever

Standard DFlash via daemon JSON protocol: `params: {"max_seq": 4096, "draft": "/path/to/27b-dflash-mq4.hfq"}`. AR baseline: same load without `draft`.

## Scenario

- Hardware: hipx, gfx1151 iGPU (HVD=0).
- Target: `qwen3.5-27b.mq4` (~14 GB).
- Draft: `qwen35-27b-dflash-mq4.hfq` (DFlash-trained draft for the 27B target).
- KV mode: asym3.
- Prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt` (canonical 27B-3.5 LRU DFlash test prompt per CLAUDE.md, ~230 tokens).
- max_tokens: 120 (canonical for DFlash bench rule).
- Temperature: 0.0 (DFlash is greedy-only).
- 3 fresh-process warm runs per condition (AR + DFlash).

## Win criterion (pre-registered)

- **STRONG WIN**: DFlash_tok_s / AR_tok_s ≥ 2.0. Indicates spec-decode delivers expected RDNA3+ multiplier; hetero variant is highly attractive.
- **WIN**: ratio ≥ 1.3. Spec-decode is net-positive but modest; hetero variant adds the BW separation as additional gain.
- **NO_CHANGE**: ratio between 0.95 and 1.3. DFlash isn't a clear lever on gfx1151 for 27B; hetero variant questionable.
- **LOSS**: ratio ≤ 0.95. DFlash is net-negative on gfx1151 27B; hetero variant unjustified.

## Quality gate

Output coherence verified by visual check. DFlash on 27B-3.5 LRU prompt should produce fluent code-shape output (per CLAUDE.md, expected τ=10.36 on 7900 XTX with this exact prompt). Token loops or attractor patterns disqualify.

## Action on win

Document gfx1151 solo DFlash baseline in memory as the anchor for hetero DFlash architecture. Justifies pursuing Exp #11+ which would build the hetero plumbing.

## Action on loss / no-change

DFlash on gfx1151 alone isn't a lever; hetero variant needs a different justification (e.g., the BW separation argument alone). Lower priority for v1.2 PRD investment.

## Implementation note

No code changes. Standard daemon JSON load with optional `draft` field. Existing canonical bench prompt.
