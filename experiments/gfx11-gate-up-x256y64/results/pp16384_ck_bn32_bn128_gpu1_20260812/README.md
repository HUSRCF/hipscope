# PP16384 CK BN32 / BN128 A/B

W7900/gfx1100, Qwen3.6-27B MQ4, Asym3 KV, strict group128/F32 semantics, 2048-token chunks, three prefill repetitions per process, three alternating pairs, and 20-second idle intervals.

| Mode | Process median prefill | Raw process medians |
| --- | ---: | --- |
| BM64 x BN32 | 1124.90 tok/s | 1148.9, 1124.9, 1122.4 |
| BM64 x BN128 | 1124.80 tok/s | 1123.3, 1129.8, 1124.8 |

Paired ratios: `0.977718x`, `1.004356x`, `1.002138x`; paired median `1.002138x`, `2/3` positive. Token IDs match within every pair and across both modes. This effectively neutral run shows no production-level BN128 win, so BN32 remains the default.
