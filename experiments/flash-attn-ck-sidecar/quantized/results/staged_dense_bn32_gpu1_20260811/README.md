# Staged dense CK D256 N32 probe

This is a rejected, source-backed complete-attention-block probe. It replaces
the gfx11 dense CK D256 `M64/N64` recipe with `M64/N32` for both FP16 and BF16
table lookups; the packed quantized kernel, predecode, bridges, workspace, and
production ABI remain unchanged. The measured staged path used FP16 dense K/V.

The W7900/gfx1100 test used ROCm 7.14, Q=2048, 24 query heads, 4 KV heads,
D=256, 15 alternating fresh-process pairs, five internal warmups, and seven
alternating GPU-event trials per process. `raw.tsv` is the retained output.

| K | N64 median | N32 median | paired median speedup | positive pairs |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1.407801 ms | 1.351320 ms | 1.0433x | 15/15 |
| 4,096 | 3.391130 ms | 3.149486 ms | 1.0748x | 15/15 |
| 6,144 | 5.461308 ms | 5.096494 ms | 1.0689x | 15/15 |
| 8,192 | 7.868808 ms | 7.604399 ms | 1.0320x | 14/15 |

The aggregate paired median is 1.0561x. The candidate is bit-identical to the
packed-view M64/N32 result for these fixtures; the N64 baseline differs by at
most `6.10e-5`, consistent with the already accepted FP16 accumulation-order
boundary.

A one-run `rocprofv3 --kernel-trace` resource audit reports the same 32 KiB
LDS, 400-byte scratch field, and 256 VGPR for both dense recipes. The local gain
therefore does not open a new occupancy tier. `resource_audit.tsv` retains the
reported fields and source trace paths. The independently archived PP8192
runtime timeline attributes 11.27% of wall time to CK quantized attention; at
that share, the measured local gain has only about 0.6% modeled end-to-end
value and misses the 1.10x attention-local admission threshold. No production
build option is kept.

`gfx11_ck_d256_bn32.patch` applies after the repository's gfx11 recipe to CK
revision `13f6d635653bd5ffbfcac8577f1ef09590c23d78`; both kernel traces are
archived beside this README. The patch is retained solely to reproduce the
rejected candidate.

Wall-share provenance:

```text
experiments/gfx11-gate-up-x256y64/results/
  pp8192_runtime_timeline_gpu1_20260811/README.md
```
