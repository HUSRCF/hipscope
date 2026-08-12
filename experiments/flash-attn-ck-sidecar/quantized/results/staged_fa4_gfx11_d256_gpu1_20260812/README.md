# FA4 gfx11 D256 staged attention validation

This result validates the exact FA4 CK source revision
`be194c0792e79ae26f71bf507e51b4d9136cf22c` on a Radeon Pro W7900 / gfx1100
with ROCm 7.14. The candidate changes the dense D256 stage behind the existing
strict-semantics Asym3-K/Q8-V bridge. It does not change model weights, KV
encoding, masking, GQA mapping, or decode.

## Component A/B

Ten alternating process pairs used Q=2048, 24 query heads, 4 KV heads, and
D=256. Medians below are staged-route GPU event times; raw rows are in the
sibling `staged_native_recipe_gpu1_20260812_1/raw.csv` result.

| K | pinned CK ms | FA4 CK ms | ratio |
| ---: | ---: | ---: | ---: |
| 2,048 | 1.404482 | 1.281179 | 1.0962x |
| 4,096 | 3.364502 | 2.930388 | 1.1481x |
| 6,144 | 5.455183 | 4.734181 | 1.1523x |
| 8,192 | 7.869337 | 7.138237 | 1.1024x |

The aggregate paired median was 1.1268x and all 10 pairs were positive. The
candidate reported `max_abs=0` against its packed-route reference at every
measured K. A clean rebuild from the pinned commit repeated the ABI smoke and
component result successfully.

## PP16384 production A/B

Five alternating fresh-process pairs used Qwen3.6-27B MQ4, three prefill runs
per process, 15 seconds between processes, and the strict FP32 FFN-intermediate
configuration.

The exact harness is committed as
`experiments/gfx11-gate-up-x256y64/run_pp16384_ck_tile_ab.sh`; its SHA-256 is
recorded in `manifest.txt`. The run used:

```bash
GPU_ID=1 PREFILL_TOKENS=16384 PREFILL_RUNS=3 TRIALS=5 SLEEP_SECS=15 \
BASELINE_LABEL=pinned_ck CANDIDATE_LABEL=native_recipe \
BASELINE_LIB=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
CANDIDATE_LIB=experiments/flash-attn-ck-sidecar/build-native-gfx11-d256/libhipfire_flash_attn_ck_quantized_staged.so \
OUT_DIR=experiments/gfx11-gate-up-x256y64/results/pp16384_ck_native_recipe_gpu1_20260812_5pair \
./experiments/gfx11-gate-up-x256y64/run_pp16384_ck_tile_ab.sh
```

The absolute paths above document the original invocation. Rebuilt libraries
may be supplied through the same `BASELINE_LIB` and `CANDIDATE_LIB` variables;
the original library hashes are pinned in `manifest.txt`.

| backend | process median prefill |
| --- | ---: |
| pinned CK | 1120.6 tok/s |
| FA4 CK | 1130.0 tok/s |

The paired-ratio median was **1.007950x**, all 5/5 pairs were positive, and all
greedy token IDs matched. See `results.tsv`, `summary.txt`, and `manifest.txt`.

## Scope

The selected long-query kernel is M64/N32 and uses FA4's Wave32 register-P
redistribution. Generated instances explicitly select the default O epilogue;
native-O is not active. Resource metadata for the selected dense kernel was
32 KiB LDS, 364 bytes private/scratch storage, 128 SGPR, and 256 VGPR. This is
a small, stable PP16384 gain rather than a claim that the full model improves
by the component-level 12.7%.
