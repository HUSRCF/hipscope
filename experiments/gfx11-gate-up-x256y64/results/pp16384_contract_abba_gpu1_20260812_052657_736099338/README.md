# PP16384 activation-contract ABBA

This Qwen3.6-27B MQ4 run separates the group128/F32 activation contract from the optional throughput-tuned group256/F16 configuration. Both modes use the same Asym3 KV cache, 2048-token prefill chunks, retained gfx1100 quad-row packed-MQ4 kernels, and staged quantized-KV CK sidecar. Here, `contract` means that activation quantization remains group128 and the FFN intermediate remains F32; it does not claim bitwise equivalence between CK and the original attention implementation.

## Results

AMD Radeon Pro W7900 (`gfx1100`), GPU1, `contract/tuned/tuned/contract` order, 30-second idle intervals, and two prefill measurements per process:

| Sample | Mode | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| 01 | contract | 1150.1 | 32.1 |
| 02 | tuned | 1151.7 | 32.0 |
| 03 | tuned | 1142.2 | 31.9 |
| 04 | contract | 1117.3 | 32.0 |

The arithmetic means are `1133.70 tok/s` for contract and `1146.95 tok/s` for tuned, a `1.0117x` (`+1.17%`) difference. The two adjacent tuned/contract ratios are `1.0014x` and `1.0223x`; their mean is `1.0118x`. All four runs emit the same 11 logged token IDs, and decode remains within `31.9-32.1 tok/s`.

The final contract sample is slower than the first, so the absolute mean includes residual run-order or thermal drift. The ABBA pairing nevertheless places the group256/F16 contribution near 1%, not near the 10% scale needed for the next backend. Future exact-contract optimization uses group128/F32 as the reference and reports the group256/F16 route separately as a quality/performance option.

## Reproduction

```bash
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
GPU_ID=1 \
experiments/gfx11-gate-up-x256y64/run_pp16384_contract_abba.sh
```

`artifacts.sha256`, `manifest.txt`, `results.tsv`, `summary.txt`, and the four raw logs preserve the run inputs and measurements.
