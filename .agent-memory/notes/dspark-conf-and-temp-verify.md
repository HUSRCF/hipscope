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

Progression (qwen3 code, top_p0.95/topk40, warm): host-softmax 17.2 → fused-GPU
20.3 → **LAZY 29.6** (temp0.7) / 28.8 (temp1.0). AR 23.8, greedy 26.7.

**LAZY PREFIX SAMPLING is the win (0f007882).** Acceptance is a prefix: once a
verify position's sample != its drafted token, all later positions reject — so
STOP the per-row head+sample loop at the first mismatch (pad rejected picks;
accept_greedy_prefix only reads up to the mismatch; the batched forward already
captured all b hidden, so P5 unaffected). The expensive 152k-vocab lm_head then
runs ~τ times/window instead of b. Per-window committed output identical to eager
(only the RNG stream diverges). **qwen3 temp>0 now BEATS AR (+24%) AND greedy** —
it does far fewer lm_head GEMVs than either. temp=0 unchanged (greedy path
untouched). Applies ONLY to the sample branch (argmax has multiple consumers that
read all picks — do NOT blanket-lazy the argmax path).

**Earlier "temp>0 fundamentally loses" was WRONG** — it assumed you must sample
all b positions. You don't. Lazy fixes it.

## deepseek4 (Stage 2, 3fe37f27) — verify_block_sampled_capture_gpu +
`final_norm_and_sample_all_batched_lazy` (per-position fused sampler + lazy stop).
Measured (warm): CODE AR 10.19 / greedy 6.12 / temp0.7-lazy **11.40** (temp>0
beats both — ds4 greedy on code is head-bound, all 5 lm_heads at τ1.36; lazy runs
~τ, τ↑1.84). PROSE (natural-EOS, noisy) AR 12.48 / greedy ~14 / temp>0 8.9-11.7
(competitive-to-losing; ds4's prose win is greedy's high accept 0.32, eroded by
temp>0 sampling). Serving still gated (supports_temp=false); bench-enabled.

## serving-enable DECISION (open)
qwen3 temp>0 now clearly wins → worth flipping carrier supports_temp=true + the
llama daemon gate (route temp>0 to the spec loop like the arch-7 gate at
daemon.rs:~6646; call speculator.set_sampling with request top_p/top_k). NOT done
yet — serving-behavior change. ds4 stays gated (prose-loses/code-wins is murky).

## BONUS opportunity (not done): apply the lazy prefix-stop to the GREEDY argmax
path too → ds4 greedy code 6.12 is head-bound; lazy would speed it up
(output-identical). BUT verify_block_argmax has MULTIPLE consumers (n-gram,
DFlash chain/tree) — must audit each does prefix-accept before blanket-lazy;
safest as a DSpark-only greedy variant.

## traps
- AR decode speed is temp-invariant (per-token sampler cost ≈ forward-bound), so
  AR greedy tok/s is a fair temp>0 baseline.
- ds4 dspark_bench needs unique EOS-free comparison; natural-EOS prose gens vary
  token count → noisy tok/s. Use raw/fixed-max for clean A/B.
- grep trap: `grep "temp="` matches the bench's header line → `head -1` hides the
  result; filter on `tokens=.*tok/s`.
- ds4 conf sweep optimum still unknown (never cleanly measured).
