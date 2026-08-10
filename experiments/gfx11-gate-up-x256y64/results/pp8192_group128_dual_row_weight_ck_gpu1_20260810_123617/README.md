# PP8192 dual-row packed-weight production A/B

W7900 (`gfx1100`), GPU1, Qwen3.6-27B MQ4, Asym3 KV, `PREFILL_MAX_BATCH=2048`, quantized CK attention sidecar active. Five alternating-order pairs were run after baseline and candidate prewarm. The benchmark binary was built with `--features flash-attn-ck`.

| Mode | Raw prefill tok/s | Median | Decode median |
|---|---|---:|---:|
| retained group128 row2 | 1065.0, 1060.8, 1059.8, 1063.0, 1057.8 | 1060.8 | 32.9 |
| dual-row packed-weight staging | 1090.0, 1087.8, 1085.6, 1088.3, 1084.9 | 1087.8 | 32.9 |

With one value trimmed from each side, the medians remain `1060.8` and `1087.8 tok/s`: `1.0255x` (`+2.55%`). Every pair favors the candidate and complete token IDs match across all runs (`1` unique sequence per mode).

The candidate remains opt-in through `HIPFIRE_RDNA3_Q8_GROUP128_DUAL_ROW_WEIGHT=1`, restricted to `gfx1100` and aligned X256/Y64 group128 dispatch. It is not enabled on gfx10/gfx12, and unaligned shapes retain the previous route.

Reproduction:

```bash
cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 --features flash-attn-ck
GPU_ID=1 TRIALS=5 TRIM_EACH_SIDE=1 SLEEP_SECS=5 \
  experiments/gfx11-gate-up-x256y64/run_pp8192_group128_dual_row_weight_ck_ab.sh
```
