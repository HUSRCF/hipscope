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
