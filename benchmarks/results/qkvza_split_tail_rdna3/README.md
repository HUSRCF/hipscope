# RDNA3 QKVZA Split-Tail A/B

This directory contains the consolidated local W7900 evidence for the opt-in
`HIPFIRE_QKVZA_SPLIT_TAIL=1` route.

The benchmark toggles only `HIPFIRE_QKVZA_SPLIT_TAIL` and runs the production
`bench_qwen35_mq4` prefill path through
`scripts/bench_qwen36_qkvza_split_tail_ab.sh`.

Common setup:

- GPU: ROCm device 0, `gfx1100` / AMD Radeon Pro W7900, ROCm 7.2
- Prompt prefill tokens: `4096`
- Prefill runs per mode: `3`
- Generation tokens: `1` (smoke only; results below are prefill results)
- KV mode: `q8`
- DPM warmup: `2s`

Hardware evidence is recorded in `system_info.txt` from `rocm-smi` and
`rocminfo`. The test host has two `gfx1100` W7900-class GPUs; these benchmark
runs used `GPU_ID=0` / `HIP_VISIBLE_DEVICES=0`, mapped to the AMD Radeon Pro
W7900 device.

## Median Summary

| Model | off median tok/s | on median tok/s | Delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B MQ4 | 7974.7 | 8403.4 | +5.38% |
| Qwen3.5-4B MQ4 | 2685.0 | 2826.9 | +5.28% |
| Qwen3.5-27B MQ4 | 598.3 | 620.6 | +3.73% |
| Qwen3.6-27B MQ4 | 595.0 | 613.8 | +3.16% |

Interpretation:

- The opt-in split-tail route is consistently positive across 0.8B, 4B, and
  27B Qwen-family MQ4 checkpoints on gfx1100.
- The largest relative gains appear on smaller models where the QKVZA route is
  a larger share of total prefill time.
- This is a prefill-path result, not a decode-throughput claim.

Files:

- `summary.tsv`: median off/on throughput and delta by model.
- `raw_prefill.tsv`: per-run raw prefill timings used to compute the medians.
- `system_info.txt`: ROCm device inventory for the benchmark host.

## Beta Follow-up: Long Uncached Prefill Gate

The follow-up was rebased on upstream `beta`, where the default-off intent lane
is carried at marker `3020dcff`. The candidate is now limited to a resident,
uncached request beginning at position zero whose full request length is at
least 4096 tokens. Cache-hit continuation, eviction, speculative/captured,
pipeline-parallel, expert-parallel, and other nonstandard paths retain the
existing route.

Fresh-process synthetic prefill uses five counterbalanced process pairs per
length, q8 KV, MTP off, and a five-second DPM warmup per process:

| Prefill tokens | Off median tok/s | Active median tok/s | Delta | Paired median | Positive pairs |
|---:|---:|---:|---:|---:|---:|
| 4096 | 595.5 | 614.1 | +3.12% | +3.04% | 5/5 |
| 8192 | 469.8 | 491.1 | +4.53% | +4.64% | 5/5 |
| 16384 | 231.4 | 236.2 | +2.07% | +2.33% | 5/5 |

The official `scripts/serve_harness.py` user-facing path was then run with a
fresh daemon for every mode, q8 KV, MTP off, and the canonical long-prose NIAH
prompt. Tokenization produced 8147 uncached input tokens. All ten samples
reported `cached=0`; every active process reported one eligible request and one
split-tail route hit, while every off process reported zero:

| Pair | Order | Off tok/s | Active tok/s | Paired delta |
|---:|:---:|---:|---:|---:|
| 1 | off-on | 431.0 | 441.8 | +2.51% |
| 2 | on-off | 422.6 | 439.7 | +4.05% |
| 3 | off-on | 423.9 | 437.5 | +3.21% |
| 4 | on-off | 422.6 | 437.8 | +3.60% |
| 5 | off-on | 422.2 | 436.7 | +3.43% |

The user-facing medians are 422.6 tok/s off and 437.8 tok/s active
(`+3.60%` cross-sample, `+3.43%` paired median, 5/5 positive pairs). The
paired-delta range is `+2.51%` to `+4.05%` (IQR `+3.21%` to `+3.60%`).
`long_uncached_beta_summary.tsv` and `cold_serve_beta_raw.tsv` retain the
machine-readable values.

This remains workload-specific evidence from five process pairs for a
default-off long, cold prefill route on one W7900. It does not promote a broad
serving or decode-throughput claim.

## Production MMQ Screening And Warmup Follow-up

The original block benchmark did not run the production MMQ safety screen.
Follow-up runs now call the same screen as the daemon, perform a five-second
DPM warmup, discard the first block-benchmark prefill, and use counterbalanced
fresh processes. The prompt is `docs/testINPUT.md` (3361 tokens) and the
standalone path uses the daemon's outer 256-token request chunks.

| Model/path | Off tok/s | Active tok/s | Delta | Paired deltas |
|:---|---:|---:|---:|:---|
| Qwen3.5-0.8B screened block | 8812.3 | 9172.2 | +4.08% cross-sample | +3.48%, +4.24%, +2.68% |
| Qwen3.5-4B screened block | 2718.3 | 2879.7 | +5.94% cross-sample | +4.15%, +5.74%, +5.94% |
| Qwen3.5-9B screened block | 1755.3 | 1794.8 | +2.25% cross-sample | -0.75%, +2.25%, +1.11% |
| Qwen3.6-27B screened block | 530.6 | 551.5 | +3.94% | +3.18%, +4.11%, +3.98% |
| Qwen3.6-35B-A3B screened block | 1928.0 | 1960.1 | +1.66% cross-sample | -0.20%, +1.86%, +0.81% |
| Qwen3.5-0.8B primed uncached serving | 2836.3 | 2889.9 | +1.89% cross-sample | +0.07%, +1.89%, +2.55% |
| Qwen3.5-4B primed uncached serving | 1000.4 | 1023.0 | +2.26% cross-sample | +4.76%, +2.42%, +1.92% |
| Qwen3.5-9B primed uncached serving | 831.6 | 836.6 | +0.61% cross-sample | +0.69%, +0.70%, +0.06% |
| Qwen3.6-27B primed uncached serving | 268.9 | 273.5 | +1.69% | +1.11%, +2.27% |
| Qwen3.6-35B-A3B primed uncached serving | 783.9 | 784.7 | +0.09% | +0.05%, -0.29%, +0.96% |

The Qwen3.5 small-model rows use three counterbalanced fresh-process pairs.
Each process performs a five-second DPM warmup and excludes its first prefill;
the remaining three runs form the reported medians. All active processes
reported an eligible request and a split-tail route hit, while all off
processes reported neither. The 0.8B and 4B pairs are uniformly positive. The
9B signal is weaker and includes one negative pair, so these data do not show
a monotonic relationship between parameter count and benefit. The
machine-readable aggregate values are in `small_model_screened_followup.tsv`;
the measured samples are in `small_model_screened_raw.tsv`.

The corresponding small-model user-facing follow-up disables the prefix cache
and DeltaNet checkpoint resume, starts a fresh daemon for every arm, and uses
three counterbalanced process pairs. Each daemon receives five complete cold
contexts (`3386/6774/10162/13550/16938` tokens); turn one is retained in the
per-turn table but excluded from the aggregate as JIT/shape priming. These
sensitivity runs set the admission threshold to 2048 tokens; the code and
reproduction script default remain the more conservative 4096 tokens. Every
aggregated turn exceeds 4096 tokens. For every small model, all 15 active
requests were eligible, all three active daemons reported a route hit, and
every request reported `cached=0`.

| Model | Block paired median | Serving paired median | Serving positive pairs | Turn 1 -> turn 5 paired median |
|:---|---:|---:|---:|:---|
| Qwen3.5-0.8B | +3.48% | +1.89% | 3/3 | +3.57% -> +0.29% |
| Qwen3.5-4B | +5.74% | +2.42% | 3/3 | +6.11% -> +1.92% |
| Qwen3.5-9B | +1.11% | +0.69% | 3/3 | +1.27% -> +0.48% |

The production path preserves the block-benchmark ordering but attenuates the
gain. 4B is the strongest small-model serving result; 0.8B remains positive;
9B stays below one percent and should be treated as marginal rather than as a
default-enable result. The decline toward the longest context is consistent
with unaffected prefill work becoming a larger share of elapsed time, but the
experiment does not isolate that mechanism. Decode values cover only four
generated tokens per turn and are smoke data, not a decode-throughput claim.
Machine-readable aggregate and per-turn medians are in
`small_model_long_cold_serving_followup.tsv`. The measured request rows and
route counts are in `small_model_long_cold_serving_raw.tsv` and
`small_model_long_cold_serving_routes.tsv`.

The serving rows use the first complete uncached request as a discarded
long-prefill/JIT priming turn. Aggregate rates cover the remaining four cold
full-context requests (`6771/10158/13545/16932` tokens for 27B). All active
requests still enter the gate and every active daemon reports a route hit.
Per-turn 27B paired medians decrease from `+3.25%` at 3384 tokens (reported
separately, not aggregated) to `+1.41%` at 16932 tokens as unaffected
long-context work becomes a larger fraction of prefill.

The per-layer MMQ screen proxy is model-specific:

| Model | Linear-attention layers | Baseline MMQ | Split-tail-only | QKV/Z rejected |
|:---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 18 | 11 | 0 | 7 |
| Qwen3.5-4B | 24 | 21 | 0 | 3 |
| Qwen3.5-9B | 24 | 4 | 1 | 19 |
| Qwen3.6-27B | 48 | 33 | 3 | 12 |
| Qwen3.6-35B-A3B | 30 | 7 | 0 | 23 |

This proxy is not the complete runtime admission predicate; route diagnostics
remain authoritative. It helps explain the model split: for the tested
256-token RDNA3 path, the proxy classifies 36/48 27B layers as QKV/Z-safe, with
three in its split-tail-only class. It classifies only 7/30 35B-A3B layers as
QKV/Z-safe and none as split-tail-only. Admission should therefore account for
production MMQ screen compatibility as well as prompt length.

The small-model results reinforce that this static proxy is descriptive, not
predictive. In particular, 4B improves despite having no split-tail-only
layers in the proxy, while 9B has one such layer but only a weak paired-median
gain. Runtime route diagnostics and paired end-to-end prefill measurements
remain the admission evidence.
