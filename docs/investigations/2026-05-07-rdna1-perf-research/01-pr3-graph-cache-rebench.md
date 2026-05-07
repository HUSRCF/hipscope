# Exp #1: PR3 graph-cache re-bench on ROCm 7.2.2

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment bench)

## Hypothesis under test

The previous LOSS verdict for PR3 (per-shape hipGraph cache extending PR2 to all 4 fused families + plain GEMV) was recorded as `project_gemv_graph_cache_pr3_2026_05_07.md` with measurements of 0.8B -18.3% / 9B -5.4% on single-card 5700 XT decode. That measurement may have been taken before ROCm 7.2.2's runtime improvements were active. Three landed-since changes are the candidate explanation:

- **ROCm 7.1+ doorbell-ring batching** for graph launches.
- **ROCm 7.2.0 AQL-batch memset graph node optimization** (variable AQL packets per memset).
- **ROCm 7.2.0 async-handler lock-contention removal**.

Hypothesis: under current ROCm 7.2.2, PR3 may now yield neutral or positive results on the same scenario.

## Lever

`HIPFIRE_GEMV_GRAPH=1` (env-gated) on the `feat/gemv-graph-cache-pr3` branch. The branch contains commits PR1 (skeleton), PR2 (plain GEMV cache), and PR3 (fused families: fused_qkv, fused_qkvza, fused_gate_up, gemv_hfq4g256_residual + plain GEMV).

## Scenario

- Hardware: hipx, single RX 5700 XT (gfx1010, HIP_VISIBLE_DEVICES filtering down to one card via `ROCR_VISIBLE_DEVICES=1` = 0000:05:00.0 healthy fans).
- Power state: amdgpu default (auto DPM), no manual clock/power overrides.
- Models: `qwen3.5-0.8b.mq4` and `qwen3.5-9b.mq4`. The 0.8B is included because that scenario showed the largest prior regression (-18.3%).
- KV mode: asym3.
- Prompt: literal `"Why is the sky blue? Answer in two sentences."` (19 tokens).
- max_seq: 4096; max_tokens: 120; temperature: 0.0; deterministic.
- Bench harness: hipfire daemon JSON protocol, fresh process per run.
- 3 fresh-process runs per condition. Median + σ reported.

## Win criterion

PR3 (`HIPFIRE_GEMV_GRAPH=1`) shows ≥5% decode tok/s improvement vs baseline on at least one of the two models, with the median outside 2σ of the baseline distribution. PP=1 byte-equivalence held (existing PR3 correctness test passes).

## Loss criterion

PR3 shows ≥2% decode tok/s regression vs baseline on either model.

## No-change band

Between -2% and +5%, or within 2σ noise. Any result in this band is a NO_CHANGE verdict.

## Action on win

Document numbers and runtime evidence. Open a PR cherry-picking the PR3 cache code onto master, gated behind the existing `HIPFIRE_GEMV_GRAPH=1` env (default off). Update the prior `project_gemv_graph_cache_pr3_2026_05_07.md` memory entry with the new runtime context.

## Action on loss / no-change

Update the prior memory entry with the formal re-bench numbers and the methodology used (so future sessions don't re-ask this question). Do not commit code changes to master. Delete any treatment branches created for this experiment.

## Hardware state recording

Both baseline and treatment runs preceded by:
```
rocm-smi --showid --showclocks --showpower
sensors 2>/dev/null
cat /sys/class/drm/card*/device/pp_dpm_sclk
```

Captured to `/tmp/perf-research/hw-state/01-pr3-graph-cache-rebench-{baseline,treatment}.txt`.

## Quality gate

The PR3 branch ships `examples/test_gemv_graph_cache_correctness_pr3` which validates bit-equivalence across all 4 fused families × 100 calls (50 steady + 50 rotating eviction). This is run before the treatment bench. Any divergence aborts the experiment; perf measurement is not meaningful if correctness regresses.

## Pre-registration commit

This document is committed BEFORE any treatment bench runs. The criterion is locked.
