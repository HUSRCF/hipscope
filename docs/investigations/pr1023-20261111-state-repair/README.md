# PR #1023 state-repair evidence — 2026-11-11

This documentary checkpoint is based on `dd58ad80733516d3a29ed138225a18bf26db59c9` (tree `438524b27d8a91976d0db6121221c760ad9afd46`). It records completed **CPU checks and one standalone convolution GPU regression**. It makes no native model-pair, cache-on equivalence, service-quality or performance claim.

Production commit `acd7f994` adds seven lines to preserve BF16-rounded convolution history during accepted-prefix repair. The final test commit adds the actual-producer regression and a typed QSA selection contract: partial repair checks complete S0 selection buffers, all other target state against AR, then the next AR full-vocabulary logits and complete active state. These are explicit state checks; no state field is silently discarded.

## Current CPU result

All nine commands completed on the exact clean dd58 source: precommit, full railgun-cert, unfiltered workspace libraries, native oracle controls, actual sampled-routing controls, four lab builds, rebuilt-oracle CPU controls, posthoc corruption controls and plain default products. Eight commands passed. The workspace command remains **exit101 / FAIL_RETAINED**:

| Library coverage | Passed | Failed | Ignored | Filtered |
| --- | ---: | ---: | ---: | ---: |
| 61 actual library harnesses | 12145 | 1 | 120 | 0 |

HFQ `direct_ranges_mixed_residency` passed in this run. The sole failure is the pre-existing KT48 assembler-parity test: unchanged source and the exact normalized 379 bank diagnostics match the independently preserved base evidence. The separately issued scoped CPU capsule preserves this failure and `strict_all_workspace_pass=false`; it does not call the workspace green. The current normal CLI/daemon manifest binds five actual Cargo package feature vectors and excludes lab/timing leakage.

The expensive canonical ABI/no-GPU scan remains attributed to parent818. The exact final three-file Git diff establishes unchanged kernel, ABI/catalog, registry, build and workflow bytes; that historical scan is not relabelled as a dd58 rerun.

## Current standalone GPU result

The sealed current-head Qwen4 unit executable ran `state::tests::row_capture_64_prefix_abort_and_continuation_match_singletons` on **actual gfx1201**. Raw exit0: **1 passed, 0 failed, 0 ignored, 2412 filtered** (4.57s libtest wall time). The fixture covers all 64 retained-prefix lengths, F32/Q8, both capture halves, abort and continuation against the actual singleton producer. Source before and after stayed clean at dd58. This is a small no-model regression, not complete native/LRU model qualification or a speed measurement.

The original tiny RUN retains an inherited PLAN field `GPU_touched=false`. Actual executable invocation, `GPU dev 0: gfx1201`, the explicit test device marker and the successful raw result establish GPU execution. The contradictory PLAN field has not been rewritten.

## Scope and provenance

The matching-head hosted-CI observation at `2026-11-11T16:13:09.524729+00:00` found zero runs, check runs or check suites and a pending combined status: **CURRENT_CI_UNOBSERVED_PENDING**. No hosted result is inferred from the local checks.

[receipts.json](receipts.json) contains relative external artifact paths and SHA-256 identities for the root-issued CPU capsule, original command results, current lab/product manifests, unit executable provenance, tiny raw logs and CI observation. Raw logs and binaries remain outside Git. The earlier failure/diagnosis checkpoint is preserved unchanged. Native model and canonical serve results belong to separate receipts and remain outside this checkpoint's qualification scope.
