# gfx11 Group128 K128 CK-on A/B

This experiment compares the retained X256/Y64 group128 row2 MMQ path with a
K128 cooperative WMMA probe. Both arms use the quantized FlashAttention CK
sidecar, Asym3 KV, 2048-token prefill chunks, and identical production flags.
Only `HIPFIRE_RDNA3_Q8_GROUP128_K128` changes.

Hardware: Radeon Pro W7900 (`gfx1100`), selected with
`HIP_VISIBLE_DEVICES=1`. Each arm ran in five alternating fresh processes;
each process reported the last of three PP8192 passes. One sample from each
side was trimmed.

| Path | Raw prefill tok/s | Median | Decode median |
| --- | --- | ---: | ---: |
| X256/Y64 group128 row2 | 1061.3, 1055.4, 1056.1, 1051.4, 1055.4 | **1055.4** | 33.0 |
| K128 cooperative WMMA | 1037.9, 1032.8, 1034.0, 1032.0, 1035.2 | **1034.0** | 33.0 |

K128 is **0.9797x (-2.03%)**. All generated token IDs match exactly. The
larger K tile reduces loop/synchronization structure in its standalone family,
but it does not beat the current production MMQ row2 organization. It remains
an opt-in diagnostic only and must not be enabled by default.

The persistent sidecar artifact used for this matched A/B is recorded in
`artifacts.sha256`. Raw measurements are in `results.tsv`; `summary.txt`
contains the aggregate.

```bash
HIP_VISIBLE_DEVICES=1 \
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized.so \
GPU_ID=1 TRIALS=5 TRIM_EACH_SIDE=1 PREFILL_RUNS=3 SLEEP_SECS=5 \
OUT_DIR=experiments/gfx11-gate-up-x256y64/results/pp8192_group128_k128_ck_gpu1_5pair_20260809 \
bash experiments/gfx11-gate-up-x256y64/run_pp8192_group128_k128_ck_ab.sh
```
