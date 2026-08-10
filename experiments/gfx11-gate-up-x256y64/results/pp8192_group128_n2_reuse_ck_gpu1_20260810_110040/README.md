# PP8192 Group128 N2-Reuse Production A/B

Qwen3.6-27B MQ4 on Radeon Pro W7900 (`gfx1100`), GPU1, asym3 KV, optional quantized CK attention active, chunk size 2048. Five alternating pairs were run after both paths were prewarmed; the summary trims one sample from each side.

| Mode | Raw prefill tok/s | Median | Decode median |
|---|---|---:|---:|
| Production baseline | 1067.4, 1060.8, 1061.6, 1059.4, 1062.1 | 1061.6 | 33.0 |
| N2 fragment reuse | 1062.8, 1064.9, 1062.4, 1060.9, 1063.7 | 1062.8 | 33.0 |

Trimmed result: `1.0011x` (+0.11%). Token IDs match across every run. The standalone hot-shape gains do not translate into a meaningful full-model improvement, so the production route was removed.
