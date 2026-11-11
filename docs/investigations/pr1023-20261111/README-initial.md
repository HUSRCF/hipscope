# PR #1023 native MTP diagnosis checkpoint — 2026-11-11

This is an investigation checkpoint based on `f2d25a0ba30a697d08ad196916fc8b902f2211a9`, not a product fix or a qualification result. The diagnostic model oracle stopped in its **cache-off** arm. No cache-on model arm or off/on comparison passed. All original raw logs, state receipts and binaries remain in the external QC artifact root labelled `pr1022-pr1023-review-20261111`; [receipts.json](receipts.json) uses paths relative to that root and records their SHA-256 identities.

## Approaches and observed results

1. **Initial oracle fixture failed at hidden-capture capacity.** At parent `818ea1104620fecc5f8c452b1ff48ce6dae8c5ee`, the oracle allocated and freed an R8 scratch before native prefill. The retained hidden capture was too small for native prefill's 4096-row request. Commit `f2d25a0` deletes only those two early sizing calls; the R8 admission assertion remains. Six scoped CPU steps passed, and the next model attempt got beyond this error. Production source was unchanged by this fixture correction.
2. **The next cache-off model attempt failed the accepted-prefix target-state check.** The original failure and cleanup receipts remain recorded. This failure predates any cache-on comparison; it cannot be attributed to LRU from these runs.
3. **A diagnostic-only overlay retained the assertions and printed the first mismatch.** The attached [diagnostic-only.patch](diagnostic-only.patch) adds milestone logging and mismatch serialization only. Its independently sealed executable was used with the same off-arm profile: hot budget 18000 MiB, max sequence 16384, native K3 with head reserve K7. The temporary overlay was restored to clean f2 after its CPU build.

The first mismatch is **R2, accept=0, keep=1**, through `generic_commit_prefix`, after an 84-token prompt. Both expected and actual positions are 85; block IDs are `[248068, 271]`. Of 220 compared semantic state rows, 48 differ: **36 GDN convolution histories and 12 QSA selected-index tensors**. No convolution or QSA field was removed from the check.

Before that failure, the first code fixture's internal native stage generated 512 tokens through 135 actual native windows. It observed zero, interior and full acceptance and completed its internal terminal-repair check. R1/R2 full-verify logits and semantic target-state checks also completed before the failing partial-prefix operation. These are completed internal stages of a failed diagnostic run, not a complete model case or paired native/LRU qualification.

## Current hypothesis and its limits

The CPU source-contract reproduction gives finite-value counterexamples to restoring convolution history by copying raw F32 projection captures: the kernel's convolution history uses a BF16 round trip, while the restored capture can retain unrounded F32 bits. Its four small cases cover start positions 84/85/86 and an R8/keep7 ring; the modeled BF16 round trip matches the modeled kernel-prefix history. Exact production source identities and the complete CPU receipt are sealed in `receipts.json`.

This establishes a source storage-contract hypothesis. The GPU diagnostic saved hashes, not convolution element values, so it does not prove which exact live elements differ or that a proposed hardware repair works. The QSA selection mismatch remains a separate unresolved contract question. No production fix or weakened assertion is included here.

## Evidence scope

The parent full CPU attempt remains `FAIL_RETAINED`, exit 101, with the separately classified local HFQ/KT48 failures. The f2 fixture follow-up has scoped CPU receipts; it does not relabel the parent full suite as passing. The oracle's numerical snapshots synchronize HIP and can drain DMA, so this investigation supplies no asynchronous-overlap or speed claim. A hardware repair, complete native model proof, cache-on turnover proof, paired comparison and canonical service checks remain unqualified.
