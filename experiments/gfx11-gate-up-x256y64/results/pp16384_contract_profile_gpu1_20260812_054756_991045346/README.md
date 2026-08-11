# PP16384 strict-contract profile with CK attribution

Radeon Pro W7900 (`gfx1100`), GPU1, Qwen3.6-27B MQ4, Asym3 KV, 2048-token prefill chunks, group128 activation quantization, FP32 FFN intermediate, quad-row packed-MQ4 kernels, and the staged quantized-KV CK sidecar.

The CK sidecar call is timed at the Rust ABI boundary only while `HIPFIRE_PROFILE=1`. The production path is unchanged when profiling is disabled. The benchmark runs two untimed warmup tokens before the profiled prefill, so the generic `first prefill run includes kernel JIT` footer is not evidence that this sample is cold.

| Metric | Result |
| --- | ---: |
| Prefill length | 16,384 tokens |
| Prefill wall | 14,373.0 ms |
| Throughput | 1,139.9 tok/s |
| Serialized tracked time | 14,101.0 ms |
| Remaining untracked wall | 272.1 ms (1.9%) |
| Staged CK attention | 1,218.7 ms (8.6%), 128 calls |
| Gated DeltaNet core | 936.1 ms (6.6%), 384 calls |
| Six group128 packed-MQ4 shapes | 10,610.2 ms (73.8%) |

The two FFN projections remain the largest individual families: gate/up is 4,971.0 ms (35.3%) and down/residual is 2,546.2 ms (18.1%). Adding the QKV, residual, GDN-projection, and QKVZA packed-MQ4 shapes brings the same primitive to approximately 74% of complete wall time.

## Staged-KV cache boundary

The staged sidecar decodes packed Asym3 K and Q8 V into FP16 scratch before dense CK attention. The existing standalone predecode benchmark was rebuilt from this commit and run on the same GPU:

| K rows | Predecode median |
| ---: | ---: |
| 2,048 | 0.0756 ms |
| 4,096 | 0.1574 ms |
| 6,144 | 0.2971 ms |
| 8,192 | 0.3951 ms |

All K outputs were exact; Q8-V conversion had the expected FP16 maximum absolute difference of `9.765625e-4`. At a conservative linear 0.05 ms per 1K rows, decoding every 2K/4K/.../16K prefix for all 16 attention layers costs about 58 ms, or 0.4% of this PP16384 wall. A per-layer persistent FP16 K/V cache would require roughly 1 GiB at 16K and cannot recover enough time to justify its lifecycle and capacity cost. The staged CK time is therefore dominated by dense attention rather than packed-KV predecode.

Reproduction:

```bash
GPU_ID=1 PREFILL_TOKENS=16384 \
  experiments/gfx11-gate-up-x256y64/run_pp16384_contract_profile.sh

GPU_ARCH=gfx1100 \
  experiments/flash-attn-ck-sidecar/quantized/build_predecode_bench.sh
HIP_VISIBLE_DEVICES=1 \
  experiments/flash-attn-ck-sidecar/quantized/build/quantized_kv_predecode_bench
```

