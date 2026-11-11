# PR #1023 state-repair evidence — 2026-11-11

This documentary checkpoint is based on `dd58ad80733516d3a29ed138225a18bf26db59c9` (tree `438524b27d8a91976d0db6121221c760ad9afd46`). It records completed **CPU checks, a standalone convolution GPU regression, and paired forced-K3 native MTP/cache correctness checks**. Canonical serving, default chooser behavior, asynchronous overlap and performance remain outside its qualification scope.

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

## Paired forced-native model result

The cache-off and cache-on oracle arms both passed on dd58, one model load per arm, with an independently revalidated strict posthoc comparison. Both code and fiction fixtures generated **512 actual token IDs** matching the independent AR reference and the paired arm. The profile was native K3, head reserve K7, R1/2/4/8 target transactions, 18000 MiB hot budget, max sequence16384, FP8 VMM, plain Q8 and PLEq4.

| Fixture | Prompt tokens | Native windows | Published native victim turnovers | Native fills completed |
| --- | ---: | ---: | ---: | ---: |
| Code | 84 | 135 | 2637 | 8556 |
| Fiction | 33 | 186 | 3214 | 11904 |

Each fixture observed zero, interior and full native acceptance, passed all30 target-prefix cells and both formal prefix-restore branches, and completed terminal repair with actual saved native frames. Active target/head/boundary state, pending-hidden bits and same-call-schedule physical ring evidence remain strict. Full-vocabulary evidence covers the consumed1..9 AR backbone and the R1/2/4/8 transaction/next-AR checks; no all512-step full-vocabulary claim is made.

Turnover requires real paired GU/DN slot addresses, newer published generations and pending-fill identities created during native work. Prior AR warmup or counter increases alone do not qualify it. Bulk prefill restores increased by zero **during each measured native interval**; other process stages are outside that delta.

### Why selection uses a composite state contract

Production `rollback_rows_retain` restores the **complete pre-verify S0 target selected_indices buffer**. Trunk FullAttention recomputes selection before its next read; the separate MTP head owns persistent ReuseSelection state. Thus a partial-prefix check validates all12 target selection owners against full S0 bytes and all208 other canonical rows literally against AR at the kept position. Full acceptance validates all220 rows against AR. An actual next AR step then validates full-vocabulary logits and all220 canonical state rows against AR at keep+1. Every field has an explicit checked source.

The original48-row discrepancy remains preserved:36GDN convolution rows and12QSA selection rows. The independently tested seven-line convolution repair is required alongside this selection contract. The original failure receipts and checkpoint were retained.

The original CPU/tiny [receipts.json](receipts.json) remains byte-identical to checkpoint `601a2ba`; its narrower scope describes that earlier snapshot. [native-pair.json](native-pair.json) extends this checkpoint and binds the strict native proof to its raw off/on reports and root-revalidated summary. State/head hashing synchronizes HIP and can drain DMA: this result supplies no asynchronous-overlap or speed claim. Forced K3 execution supplies no default/adaptive chooser, sampled HTTP, canonical service or native batched-PM4 qualification. Those remain separate gates.

## Scope and provenance

The matching-head hosted-CI observation at `2026-11-11T16:13:09.524729+00:00` found zero runs, check runs or check suites and a pending combined status: **CURRENT_CI_UNOBSERVED_PENDING**. No hosted result is inferred from the local checks.

[receipts.json](receipts.json) contains relative external artifact paths and SHA-256 identities for the root-issued CPU capsule, original command results, current lab/product manifests, unit executable provenance, tiny raw logs and CI observation. Raw logs and binaries remain outside Git. The earlier failure/diagnosis checkpoint is preserved unchanged. The forced-native paired result is recorded separately in native-pair.json. Canonical serve, sampled producer and quality results remain outside this checkpoint's scope.
