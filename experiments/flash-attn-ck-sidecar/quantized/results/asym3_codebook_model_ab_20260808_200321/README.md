# Clean-card Asym3 codebook model A/B

This run compares the production CK sidecar's original inlined Asym3 centroid
switch with the 32-byte LDS codebook candidate on a clean Radeon Pro W7900.
Each point is a fresh Qwen3.6-27B MQ4 process using Asym3 KV, PP8192, three
prefill repeats, two warmup tokens, and eight generated tokens. Trial order
alternates to reduce drift.

| Mode | Prefill tok/s raw | Median | Decode range |
| --- | --- | ---: | ---: |
| switch | 528.9, 526.8, 526.6, 527.0, 526.3 | `526.8` | 32.7-32.9 |
| LDS codebook | 527.8, 526.2, 526.3, 526.3, 526.4 | `526.3` | 32.7-32.8 |

The LDS candidate is `0.9991x` (`-0.09%`) relative to the switch. All ten runs
reported the quantized CK route active. This fails the production improvement
gate, so the switch remains the default despite the LDS candidate's smaller
static instruction/resource footprint.

`meta.txt` records the executable, model, and sidecar SHA-256 values. The raw
per-trial data are in `results.tsv`; process logs and built artifacts are local
run products and are intentionally ignored by git.
