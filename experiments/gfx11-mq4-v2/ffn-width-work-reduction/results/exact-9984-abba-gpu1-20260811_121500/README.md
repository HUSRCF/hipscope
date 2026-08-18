# Full width versus 39/68 fresh-process ABBA

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
fresh-process pairs: 5
internal pairs per process: 11
order: alternating FT / TF
```

| Pair | Order | Full gate ms | 39/68 gate ms | Full down ms | 39/68 down ms | Weighted speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | FT | 4.5215 | 2.6097 | 4.6189 | 2.7026 | 1.72455188x |
| 1 | TF | 4.5439 | 2.5931 | 4.6320 | 2.7087 | 1.73780542x |
| 2 | FT | 4.5274 | 2.5976 | 4.6586 | 2.7064 | 1.73552192x |
| 3 | TF | 4.5409 | 2.5966 | 4.6758 | 2.7016 | 1.74261539x |
| 4 | FT | 4.5926 | 2.6055 | 4.6350 | 2.7383 | 1.73854302x |

```text
weighted median = 1.73780542x
overall Amdahl  = 1 / (0.51 + 0.49 / 1.73780542) = 1.26268234x
from 1189       = 1501.329 tok/s
from 1115.4     = 1408.396 tok/s
```

`results.tsv` and all 20 per-process logs are retained in this directory. The
projection is not an end-to-end result and depends on preserving the existing
1189 tok/s best controlled configuration.
