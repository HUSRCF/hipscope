# gfx11 Group256 Serial-Row Production A/B

This experiment routed the standalone group256 serial-row kernel through the
Qwen3.6-27B production prefill dispatch under an opt-in feature flag. The
default group128 path was unchanged. Tests ran on Radeon Pro W7900 (`gfx1100`),
GPU1, with PP8192, chunk size 2048, Asym3 KV, and the CK attention sidecar.

Five alternating fresh-process pairs were collected after prewarm. Each
process reported the median of three prefill runs; the aggregate below uses the
median across processes (trim-one gives the same result).

| Path | Prefill samples (tok/s) | Median | Decode median |
| --- | --- | ---: | ---: |
| Production group128 | 1059.8, 1061.8, 1059.4, 1057.1, 1057.6 | 1059.4 | 33.0 |
| Group256 serial-row | 1106.0, 1105.1, 1107.2, 1104.5, 1111.0 | 1106.0 | 33.0 |

The group256 candidate improves prefill by **4.40%** and all short benchmark
token IDs match. However, the separate real-prompt greedy test in
`../real_prompt_group256_serial_gpu1_20260810_013851/` first diverges at output
token 93 of 129. Group256 changes activation scaling and is therefore a
quality/performance tradeoff, not an exact production replacement. Keep the
feature disabled by default.
