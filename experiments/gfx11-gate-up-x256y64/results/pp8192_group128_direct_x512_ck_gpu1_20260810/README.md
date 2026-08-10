# Group128 X512 Direct-Activation Production A/B

Qwen3.6-27B MQ4, PP8192 with 2048-token chunks, Asym3 KV, optional CK
attention sidecar, Radeon Pro W7900 (`gfx1100`), GPU1. The candidate affects
only the `17408 x 5120` gate/up projections and is controlled by the opt-in
`HIPFIRE_RDNA3_Q8_GROUP128_DIRECT_X512=1` flag.

Five alternating fresh-process pairs were collected after both paths were
prewarmed. Each process used the median of three prefill runs.

| Path | Prefill samples (tok/s) | Median | Decode median |
| --- | --- | ---: | ---: |
| Production LDS baseline | 1066.2, 1061.6, 1059.6, 1062.0, 1060.1 | 1061.6 | 33.2 |
| X512 direct activation | 1054.4, 1056.6, 1050.9, 1048.6, 1049.8 | 1050.9 | 33.2 |

The candidate is **0.9899x (-1.01%)** and all short-output token IDs match.
All five process pairs favor the baseline. The wider workgroup's tiny
standalone staging benefit does not transfer to the full model, where reduced
grid parallelism and higher VGPR allocation dominate. Keep this path disabled,
do not extend it to X1024, and retain X256/Y64 as the production shape.
