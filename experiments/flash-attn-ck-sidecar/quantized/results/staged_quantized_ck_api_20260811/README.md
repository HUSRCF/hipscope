# Exported staged CK API benchmark

Same-process W7900/gfx1100 benchmark of the exported packed and staged C APIs.
Each point uses five warmups and seven alternating GPU-event trials. Both paths
consume the same FP32 query and packed Asym3-K/Q8-V inputs, use bottom-right
causal GQA (24 query heads, 4 KV heads, D256), and return FP32 output.

```bash
STAGED=1 OUT=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
  ./experiments/flash-attn-ck-sidecar/quantized/build_quantized_sidecar.sh

QUANTIZED_SIDECAR=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
  ./experiments/flash-attn-ck-sidecar/quantized/build_staged_ck_bench.sh

HIP_VISIBLE_DEVICES=1 \
  ./experiments/flash-attn-ck-sidecar/quantized/build/staged_quantized_ck_bench
```
