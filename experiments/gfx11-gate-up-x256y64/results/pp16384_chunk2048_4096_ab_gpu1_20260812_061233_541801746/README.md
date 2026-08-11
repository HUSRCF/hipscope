# PP16384 chunk 2048/4096 A/B

This run checks whether extending the retained gfx1100 X256/Y64 group128 MQ4
routes from `N=2048` to aligned `N=4096` makes a larger prefill chunk useful.
The model, Asym3 KV mode, staged CK attention sidecar, group128 activation
contract, FP32 FFN intermediate, and generated-token count are held constant.
The launch order alternates by pair.

| chunk | PP16384 prefill median | raw prefill tok/s | decode median |
| ---: | ---: | --- | ---: |
| 2048 | 1122.1 tok/s | 1165.9, 1120.1, 1122.1 | 31.9 tok/s |
| 4096 | 1127.8 tok/s | 1145.6, 1127.8, 1123.8 | 32.0 tok/s |

`chunk=4096` is only `1.0051x` (`+0.51%`) faster by median. More importantly,
each chunk setting is internally deterministic but the generated token IDs
differ between settings. The larger chunk changes the GDN state-update boundary,
so this result does not satisfy the retained semantic contract. The production
route guards remain restricted to `N=2048`.

The raw logs, manifest, checksums, and parser output are retained beside this
file. Reproduce with:

```bash
GPU_ID=1 TRIALS=3 \
  experiments/gfx11-gate-up-x256y64/run_pp16384_chunk2048_4096_ab.sh
```
