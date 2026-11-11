# PR #1023 state-repair evidence — 2026-11-11

This documentary checkpoint is based on `dd58ad80733516d3a29ed138225a18bf26db59c9` (tree `438524b27d8a91976d0db6121221c760ad9afd46`). It records completed **CPU checks, a standalone convolution GPU regression, paired forced-K3 native MTP/cache correctness, a canonical same-server service regression, and AR Redline parity**. Each result retains its own scope. Direct sampled producer telemetry, generated-code execution, native batched PM4, asynchronous overlap and performance remain unqualified.

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

Production `rollback_rows_retain` restores the **complete pre-verify S0 target selected_indices buffer**. Trunk FullAttention recomputes selection before its next read; the separate MTP head owns persistent ReuseSelection state. Thus a partial-prefix check validates all 12 target selection owners against full S0 bytes and all208 other canonical rows literally against AR at the kept position. Full acceptance validates all220 rows against AR. An actual next AR step then validates full-vocabulary logits and all220 canonical state rows against AR at keep+1. Every field has an explicit checked source.

The original48-row discrepancy remains preserved:36GDN convolution rows and12QSA selection rows. The independently tested seven-line convolution repair is required alongside this selection contract. The original failure receipts and checkpoint were retained.

The original CPU/tiny [receipts.json](receipts.json) remains byte-identical to checkpoint `601a2ba`; its narrower scope describes that earlier snapshot. [native-pair.json](native-pair.json) extends this checkpoint and binds the strict native proof to its raw off/on reports and root-revalidated summary. State/head hashing synchronizes HIP and can drain DMA: this result supplies no asynchronous-overlap or speed claim. Forced K3 execution supplies no default/adaptive chooser, sampled HTTP, canonical service or native batched-PM4 qualification. The later service and AR-only results below retain their own separate scopes.

## Canonical same-server service result

The normal current-source CLI/daemon completed one load and all 12 requests on one service instance: warmup, five greedy questions, one ordinary temperature 0.3 code question, then the same five greedy questions. All 12 returned HTTP 200, one terminal stop and one DONE with no late payload, stream error or length cap. Independent raw-SSE folding verified decoded responses and request/prompt hashes. All five greedy before/after pairs have identical full decoded UTF-8 and protocol fields; request IDs and timings vary, and served token-ID identity is outside this check.

Actual greedy native traces were positive before and after sampling: **191/183 native windows**, including **189/180 batched multirow windows**, with observed maximum R8. The product chooser was not forced to a fixed window. The five decoded outputs remain identical despite different native schedules; this supplies no speed or general chooser-policy claim.

The temperature 0.3 request completed naturally with 75 tokens and zero native events, with ordinary sampled policy left on and no sampled opt-out. **Direct runtime AR producer identity remains UNOBSERVED.** Same-head builder/router CPU proof plus successful exact-body execution without native/refusal events supports the intended sampled fallback path; event absence alone is not a direct producer label. Sampled native correctness/distribution is unqualified.

Independent review found no wrong response or truncation within these six short task prompts. Generated Python was syntax-parsed and manually read, never executed; functional code correctness and broad benchmark quality remain unqualified. The warmup is protocol-only.

## AR Redline result

The separate **MTP-off, single-input-row AR** run passed: three stable captures, an actual 31-kernel contract inventory matched to loader-probed code-object membership and explicit/padded ABI, and **16 distinct positions 513–528**. Ordinary HIP, recorded HIP and retained PM4 produced bit-exact raw logits and canonical state at every tested position. The observed 31 is this run's inventory, not a hardcoded historical acceptance threshold.

The loader explicitly reported AR cache verify capacity 1, FP8 VMM and plain Q8. This is AR PM4 evidence; native batched PM4 remains unqualified. Fault coverage is prepared-position pre-submit refusal followed by clean-HIP recovery. No live kernel fault or unknown-quiescence GPU injection is claimed.

[service-ar.json](service-ar.json) seals both completed runs and the independent service review. The earlier CPU/tiny and forced-native packets remain byte-identical; their narrower exclusions describe their own snapshots. QC35 and any active long-context run are outside this extension and have not been inspected here. Current CPU still retains its sole classified KT48 failure.

## Scope and provenance

The matching-head hosted-CI observation at `2026-11-11T16:13:09.524729+00:00` found zero runs, check runs or check suites and a pending combined status: **CURRENT_CI_UNOBSERVED_PENDING**. No hosted result is inferred from the local checks.

[receipts.json](receipts.json) contains relative external artifact paths and SHA-256 identities for the root-issued CPU capsule, original command results, current lab/product manifests, unit executable provenance, tiny raw logs and CI observation. Raw logs and binaries remain outside Git. The earlier failure/diagnosis checkpoint is preserved unchanged. The forced-native paired result is recorded separately in native-pair.json. The service and AR-only extension is recorded separately in service-ar.json; direct sampled producer, code execution, native batched PM4 and long-context/QC35 remain outside its qualification scope.
