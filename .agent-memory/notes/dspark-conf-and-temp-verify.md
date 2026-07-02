---
title: DSpark conf_threshold sweep (qwen3=0.1) + CLI-shadow fix + temp>0 sampled verify (fused GPU sampler, still loses to AR)
date: 2026-07-02
tags: [dspark,spec-decode,conf-threshold,temp,sampling,qwen3,wiring]
---

Branch feature/dspark-qwen3. Follows [[dspark-gpu-resident-p4-p5]].

## DSpark runtime params
- **k / draft length = `block_size`** — baked in the sidecar JSON
  (`dspark_block_size`: qwen3=7, ds4=5), clamped [1,8]. It's the drafter's
  TRAINED horizon — NOT a runtime knob (no env/CLI; `MtpSpeculator` uses
  `arch.k()`==block_size). Nothing to sweep.
- **"budget" = `conf_threshold`** — the only real tunable. Confidence cutoff that
  truncates the drafted block before verify. Ladder: env
  `HIPFIRE_{QWEN3,DEEPSEEK4}_DSPARK_CONF_THRESHOLD` > CLI/config > per-arch carrier
  default (qwen3 0.1, ds4 0.5).

## conf sweep (gfx1151, qwen3-8b, max=256, fresh-proc, warmed)
Flat-to-cliff: **0.1 is optimal** on both prompt types; higher over-truncates.
code: 0.05→26.7 0.1→26.7 0.2→26.8 0.3→26.3 0.5→22.6 0.7→18.8.
prose: 0.05→26.5 0.1→26.5 0.2→23.8 0.3→22.8 0.5→17.7 0.7→16.8.
AR baseline 23.8 → DSpark greedy@0.1 is +12% vs AR.

## CLI-shadow bug FIXED (066d084a)
CLI unconditionally forwarded `dspark_conf_threshold=0.5`, and the qwen3 carrier
ranks that above its 0.1 default → qwen3 DSpark via the CLI silently ran at 0.5
(−15% code / −33% prose on the DEFAULT greedy path). Also the `run` flag only set
the ds4 env var (no-op on qwen3). Fix: CLI default `null`, forward only when the
user sets it (per-arch carrier default applies); `run` flag sets both arch env
vars; docs corrected ("deepseek4-only" was wrong — drives both).

## temp>0 sampled verify (bc4df7c2 + adb90438)
DSpark was greedy-only (`requires_greedy`); temp>0 bypassed it to AR. Added
distribution-preserving sampled verify: drafter stays a point-mass guess, TARGET
samples t_i~p_T(temp,top_p,top_k), accept draft iff ==sample. Wiring:
`SpecTarget::verify_block_sampled_capture_gpu` (default Err; llama impl, ds4
pending), `MtpDrafter::set_sampling`/`supports_temp_verify`, DsparkDrafter branches
greedy/sampled on temp, `build_dspark_speculator(supports_temp)`. bench gains
`HIPFIRE_QWEN3_TEMP/TOP_P/TOP_K`.
**Fused the softmax into the existing `sample_top_p_pf` GPU kernel** (softmax +
nucleus + top_k + categorical in ONE launch, 4-byte D2H) — same sampler AR uses,
so committed tokens are distribution-IDENTICAL to AR (not an approximation).
First cut host-softmaxed (want_logits D2H + host exp) = 17.2; fused GPU = 20.3.

**KEY RESULT: temp>0 still LOSES to AR** (code, top_p0.95/topk40): temp=0 26.7
(greedy, +12% vs AR); 0.7 20.3; 1.0 20.2; **AR 23.8**. τ holds (~1.87). Root
cause is FUNDAMENTAL: spec samples ~b(=7) positions/window but commits ~τ(1.87),
so ~3.7× the (expensive, ~250µs) sampler calls per committed token vs AR's 1.
Fusion closed part of the gap, not all.
→ **Serving gated greedy-only** (`supports_temp=false` in carrier; temp>0→AR
fallback); capability exercised via the bench (`supports_temp=true`).
→ Next lever (NOT done): **lazy/prefix sampling** — sample only until the first
rejection (~τ calls/window not b). Projected ~24.6 → marginally beats AR 23.8.
Marginal + a verify-loop refactor (interleave accept into verify), so deferred.

## traps
- AR decode speed is temp-invariant (per-token sampler cost ≈ forward-bound), so
  AR greedy tok/s is a fair temp>0 baseline.
- `sample_top_p_pf` is heavy (150k-vocab top-k scan); multiplying it by block
  size is what kills spec at temp>0. Greedy argmax is cheap → greedy spec wins.
- ds4 conf sweep never produced result lines (dspark_bench prints header only for
  ds4 — separate harness bug, unfixed); ds4 conf optimum still unknown.
