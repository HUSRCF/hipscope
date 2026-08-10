# FFN width work-reduction admission probe

This standalone GPU1 probe measures the retained gfx11 group128 packed-MQ4
primitive while reducing the Qwen3.6-27B FFN active width. It does not modify a
checkpoint and is not evidence that arbitrary channel removal preserves model
quality. Its purpose is to quantify how much real Wave32-WMMA work must be
removed before a model-level approximation can meet the performance target.

The complete FFN width is 17,408. Gate/up reduce output rows (`M`) while down
reduces input columns (`K`). The nearest supported down widths are aligned to
256 columns.

| Active width | Retained | Gate ms | Gate speedup | Down ms | Down speedup |
|---:|---:|---:|---:|---:|---:|
| 17,408 | 100.0% | 4.4718 | 1.000x | 4.6203 | 1.000x |
| ~15.1K | ~87.0% | 3.9094 | 1.144x | 4.0340 | 1.145x |
| 13,056 | 75.0% | 3.3124 | 1.350x | 3.4520 | 1.338x |
| ~10.8K | ~62.0% | 2.8257 | 1.583x | 2.8686 | 1.611x |
| ~10.0K | ~58.0% | 2.6573 | 1.683x | 2.6678 | 1.732x |

Weighting gate and up twice and down once, the ~58% width point improves the
three large FFN projections by about 1.70x. With those projections accounting
for about 49% of measured PP8192 wall time, the optimistic Amdahl projection is
about 1.253x overall, or roughly 1.49k tok/s from the 1.189k best controlled
baseline. Reaching 1.5k therefore requires approximately 42-44% FFN work
removal, not a small structured-pruning tweak.

The next admission gate is model quality. A candidate must define a calibrated
channel/layer selection and preserve the MQ rotation contract; contiguous tail
truncation is not a production design.

Run:

```bash
GPU=1 PAIRS=11 \
  experiments/gfx11-mq4-v2/ffn-width-work-reduction/run.sh
```
