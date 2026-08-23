# Latest-beta W7900 PP8192 A/B

This is a three-pair alternating-process comparison of the native Asym3/Q8 prefill path and the optional staged CK sidecar on `upstream/beta@80a572c8`. Each process executes three PP8192 prefill runs; the reported process value is the in-process median. A ten-second idle interval separates arms.

This branch contains the CK integration only. It does not contain the separate gfx11 packed-MQ4 production stack (`X256/Y64`, permuted group128/256 weights, fused SwiGLU packing, and FP16 FFN intermediate). Consequently, these numbers validate the incremental CK backend on official beta; they are not the final all-optimizations throughput result. The retained full-stack PP8192 measurements are approximately `1.19-1.21k tok/s`.

| Mode | Process medians (tok/s) | Median (tok/s) |
| --- | --- | ---: |
| Native | `592.4`, `593.9`, `593.7` | **593.7** |
| Staged CK | `869.2`, `862.1`, `865.8` | **865.8** |

The paired speedup median is **1.4583x** (`+45.83%`), with all three pairs positive. Native and CK produced the same eight greedy token IDs in every pair. Decode remained neutral at `35.1-35.3 tok/s`; the sidecar is a prefill-only route.

Reproduce with:

```bash
GPU_ID=1 \
PREFILL_TOKENS=8192 \
PREFILL_RUNS=3 \
TRIALS=3 \
MODEL="$HOME/.hipfire/models/qwen3.6-27b.mq4" \
CK_LIB="$PWD/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so" \
  ./scripts/bench-gfx11-ck-prefill-ab.sh
```

The exact rows and artifact hashes are in `results.tsv` and `manifest.txt`. The raw process logs are intentionally not committed.
