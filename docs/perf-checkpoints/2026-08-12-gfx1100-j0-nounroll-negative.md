# gfx1100 MQ4 N-subtile no-unroll probe

## Scope

This standalone-only probe preserves the packed-MQ4, group128-Q8, scale, zero-correction, WMMA K-order, and output contracts. It changes only the compiler unroll directive on the four local N subtiles from full unroll to `unroll 1`. The goal was to reduce peak VGPR pressure and the four compiler-generated spills in the production quad-row kernel. It is not selected by serving dispatch.

## Direct hot-shape result

Five process-level trials used 31 alternating pairs per process on W7900/gfx1100. The output tensors were initialized with deterministic non-zero values so the add path also verifies residual preservation. Both candidates produced bit-exact output.

| Shape | Existing quad-row median | N-subtile no-unroll median | Candidate / baseline time | Exact |
|---|---:|---:|---:|---:|
| gate/up set, M17408 K5120 N2048 | 5.9309 ms | 7.9982 ms | 1.3511x | yes |
| down/residual add, M5120 K17408 N2048 | 5.8358 ms | 7.8556 ms | 1.3508x | yes |

Each process alternated the production quad-row baseline and candidate directly; the ratio is the median of the five within-process ratios. The candidate is substantially slower on both dominant `N=2048` production chunk shapes, so it did not qualify for a PP16384 serving check.

## ISA/resource audit

The full-set/full-add candidate compiles to wave32 with 192 VGPRs and zero reported VGPR spills, down from 256 VGPRs and four spills in the existing kernel. This apparent resource improvement is misleading: `private_segment_fixed_size` grows from 20 bytes to 272 bytes, and the full-set disassembly grows from 8 to 81 static scratch load/store instructions. Without compile-time N-subtile indices, the compiler materializes the accumulator array in private memory.

Full N-subtile unrolling is therefore part of the register-resident accumulator contract, not an incidental tuning choice. The existing four scalar spills are cheaper than dynamic private-memory accumulator addressing.

## Decision

Reject production promotion and do not run PP16384. Keep the standalone probe and direct comparison as negative evidence; do not add a FeatureFlag or serving route.

Artifacts:

- `experiments/gfx11-gate-up-x256y64/results/j0_nounroll_gpu1_20260812_091000_sameprocess/`
