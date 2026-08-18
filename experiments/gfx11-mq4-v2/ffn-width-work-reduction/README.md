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

An exact target-boundary screen used the same retained primitive, `N=2048`,
and 11 internal pairs in one process per shape on GPU1:

| Groups | Active width | Retained | Gate ms | Down ms | Weighted FFN speedup | Projected PP8192 from 1189 |
|---:|---:|---:|---:|---:|---:|---:|
| 41/68 | 10,496 | 60.3% | 2.7452 | 2.7975 | 1.6609x | 1477.0 tok/s |
| 39/68 | 9,984 | 57.4% | 2.5785 | 2.6680 | 1.7592x | 1507.9 tok/s |
| 38/68 | 9,728 | 55.9% | 2.5665 | 2.6035 | 1.7793x | 1513.9 tok/s |

The weighted local speedup uses two gate/up projections and one down
projection. The end-to-end projection uses the previously measured 49% wall
share and the best controlled 1189 tok/s baseline. It is not a serving result:
from the conservative 1115.4 tok/s reference the same points project only
1385.6, 1414.5, and 1420.2 tok/s. The 1.5k target therefore requires both the
39/68 work reduction and preservation of the existing best-path gains.

Command outputs are archived in
`results/exact-target-widths-gpu1-20260811/`. The primary 39/68 boundary also
has fresh-process ABBA trials produced by `run_exact_target_abba.sh`; use that
paired result rather than mixing the older sweep baseline with a new process.

The five-pair fresh-process ABBA result for full width versus 39/68 was:

```text
paired weighted speedup: min 1.7246x, median 1.7378x, max 1.7426x
projected overall:       1.2627x
projected from 1189:     1501.33 tok/s
projected from 1115.4:   1408.40 tok/s
```

This is the primary performance admission number. Per-process logs and the raw
TSV are under `results/exact-9984-abba-gpu1-20260811_121500/`.

Weighting gate and up twice and down once, the exact 39/68 ABBA median improves
the three large FFN projections by 1.7378x. With those projections accounting
for about 49% of measured PP8192 wall time, the optimistic Amdahl projection is
1.2627x overall, or 1501.33 tok/s from the 1189 tok/s best controlled baseline.
Reaching 1.5k therefore requires approximately 42-44% FFN work removal, not a
small structured-pruning tweak.

The next admission gate is model quality. A candidate must define a calibrated
channel/layer selection and preserve the MQ rotation contract; contiguous tail
truncation is not a production design.

Run:

```bash
GPU=1 PAIRS=11 \
  experiments/gfx11-mq4-v2/ffn-width-work-reduction/run.sh
```

Run the exact full-vs-39/68 fresh-process gate with:

```bash
GPU=1 PAIRS=5 INTERNAL_PAIRS=11 \
  experiments/gfx11-mq4-v2/ffn-width-work-reduction/run_exact_target_abba.sh
```
