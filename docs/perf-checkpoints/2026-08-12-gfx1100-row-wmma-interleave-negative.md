# gfx1100 row-fragment WMMA interleave probe

## Scope

This probe preserves the exact packed-MQ4, group128-Q8, scale, zero-correction, and per-accumulator K-order contracts. It changes only the instruction order of two independent row-fragment Wave32 IU8-WMMA dependency chains. It is standalone-only and is not selected by serving dispatch.

## Direct hot-shape result

Five process-level trials used 31 alternating pairs per process on W7900/gfx1100. Both paths were compared against the same generic group128 reference and produced bit-exact output.

| Shape | Existing quad-row median | Interleaved median | Interleaved / quad-row | Exact |
|---|---:|---:|---:|---:|
| gate/up set, M17408 K5120 N2048 | 4.6327 ms | 4.6458 ms | 1.0028x | yes |
| down/residual add, M5120 K17408 N2048 | 4.7208 ms | 4.7253 ms | 1.0010x | yes |

Result: no hot-primitive improvement over the production quad-row path.

## PP16384 production check

The controlled production check fixed asym3 KV, quantized CK attention, chunk size 2048, three prefill runs per process, and three alternating process pairs. Only the temporary row-WMMA route was changed.

| Mode | PP16384 median | Decode median |
|---|---:|---:|
| existing quad-row | 1120.3 tok/s | 31.9 tok/s |
| interleaved | 1119.4 tok/s | 31.8 tok/s |

The median ratio was `0.9992x`; paired ratios were `0.9807x`, `0.9994x`, and `0.9966x`. All token IDs matched. The first pair includes a higher cold-order baseline, but the later pairs and direct hot-shape comparison independently show no positive result.

## ISA/resource audit

Both full-set kernels compile to wave32 with 256 VGPRs, four VGPR spills, a 20-byte private segment, and 128 IU8-WMMA instructions. The interleaved schedule reduces static `s_waitcnt` count from 117 to 115 but does not reduce VGPR or spill pressure. That instruction-level difference does not improve elapsed time.

## Decision

Reject production promotion. Keep the probe and reproducible direct comparison as negative evidence; do not add a FeatureFlag or serving route.

Artifacts:

- `experiments/gfx11-gate-up-x256y64/results/interleave_row_wmma_gpu1_20260812_074043/`
- `experiments/gfx11-gate-up-x256y64/results/pp16384_interleave_row_wmma_gpu1_20260812_073056_831307132/`
