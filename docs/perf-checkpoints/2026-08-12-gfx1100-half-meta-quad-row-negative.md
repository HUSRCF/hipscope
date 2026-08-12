# gfx1100 half2 metadata plus quad-row MQ4 probe

## Scope

This standalone-only execution-format probe keeps the 128-byte HFQ4 nibble payload, Q8 group128 activation, affine scale/zero correction, WMMA order, and output contract unchanged. It preconverts each weight group's two FP32 metadata values to one FP16 `half2` at load/repack time, then combines that representation with the selected quad-row packed-payload loader. The production reference already performs the same FP32-to-FP16 conversion before LDS staging, so the candidate is bit-exact relative to that arithmetic path. It is not selected by serving dispatch.

## Direct hot-shape result

Five fresh processes per shape used 31 alternating within-process baseline/candidate pairs on W7900/gfx1100. Output tensors used deterministic non-zero initialization, so the add path verifies residual preservation.

| Shape | Existing quad-row median | Half-meta quad-row median | Candidate / baseline time | Exact |
|---|---:|---:|---:|---:|
| gate/up set, M17408 K5120 N2048 | 4.5224 ms | 4.5536 ms | 1.0080x | yes |
| down/residual add, M5120 K17408 N2048 | 4.6103 ms | 4.6082 ms | 0.9998x | yes |

Preconverting metadata removes the per-group FP32-to-FP16 conversion and reduces metadata LDS replication, but neither dominant production shape improves. The payload decode, WMMA feed, and affine accumulation remain unchanged and dominate the saved metadata work.

## Decision

Reject production promotion. The candidate does not qualify for a PP16384 serving check, and a load-time full-model repack would add complexity without measured kernel benefit. Keep the standalone probe as exact-format boundary evidence; do not add a FeatureFlag or serving route.

Artifacts:

- `experiments/gfx11-gate-up-x256y64/results/half_meta_quad_row_gpu1_20260812_093000/`
