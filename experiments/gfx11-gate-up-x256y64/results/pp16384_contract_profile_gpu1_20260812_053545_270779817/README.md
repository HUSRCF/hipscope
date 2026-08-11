# PP16384 group128/F32 contract profile

This profile measures the Qwen3.6-27B MQ4 prefill path used as the semantic-contract reference: group128 activation quantization, F32 FFN intermediate, retained gfx1100 quad-row packed-MQ4 kernels, Asym3 KV, 2048-token chunks, and the staged quantized-KV CK sidecar. The benchmark performs an unprofiled JIT warm-up before one profiled PP16384 pass.

## Overall

| Metric | Value |
| --- | ---: |
| Prefill wall | 14520.5 ms |
| Prefill throughput | 1128.3 tok/s |
| Internally tracked kernels | 13020.8 ms |
| Internally untracked wall | 1499.7 ms (10.33%) |
| Decode throughput | 32.1 tok/s |

The benchmark currently prints the last row as `startup_overhead_ms`, but that label is not valid for this run: warm-up completed before profiling, while the external CK sidecar and its bridges do not participate in the internal timer registry. Treat `1499.7 ms` as untracked wall, not cold-start or CPU overhead.

## Main tracked costs

Percentages below use the complete 14520.5 ms prefill wall rather than the internally tracked subtotal.

| Component | Time | Wall share |
| --- | ---: | ---: |
| Gate/up MQ4, M17408 K5120 | 5024.6 ms | 34.60% |
| FFN down MQ4, M5120 K17408 | 2574.1 ms | 17.73% |
| QKV MQ4, M10240 K5120 | 1101.1 ms | 7.58% |
| Residual MQ4, M5120 K6144 | 933.0 ms | 6.43% |
| GDN projection MQ4, M6144 K5120 | 658.9 ms | 4.54% |
| QKVZA MQ4, M12288 K5120 | 433.7 ms | 2.99% |
| Six group128 packed-MQ4 projections | 10725.4 ms | **73.86%** |
| Gated DeltaNet core | 949.9 ms | 6.54% |
| Fused SwiGLU/rotate/Q8 | 317.6 ms | 2.19% |
| Conv1D/SiLU | 315.6 ms | 2.17% |

Adding the smaller MQ4 tails keeps the full packed-weight family near 75% of wall. This confirms that the contract-preserving PP16384 path remains dominated by one execution primitive. The CK/bridge share cannot be recovered exactly from the internal profile and is included in the untracked wall.

## Reproduction

```bash
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
GPU_ID=1 \
experiments/gfx11-gate-up-x256y64/run_pp16384_contract_profile.sh
```

`profile.log` is the raw output. `profile-summary.txt`, `manifest.txt`, and `artifacts.sha256` retain the concise report and exact artifacts.
