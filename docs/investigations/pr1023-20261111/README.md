# PR #1023 native MTP diagnosis checkpoint — 2026-11-11

This checkpoint retains the original failed model attempts and adds a narrowly
scoped GPU regression diagnosis. It is **not a final model, cache-off/on pair,
service or performance qualification**. The [initial checkpoint](README-initial.md)
and its [original receipts](receipts.json) remain byte-for-byte available; their
unresolved-hypothesis language describes that earlier evidence boundary.

## What the added GPU evidence establishes

The corrected existing test obtains convolution history from the actual scalar
convolution producer rather than manually copying raw F32 captures. Its initial
history is nonzero BF16-exact 0.25; the test retains its prefix, abort and
continuation assertions. The two temporary test sources are identical except for
the seven-line production fix.

On actual gfx1201, the unfixed binary failed at **F32, base=0, keep=1**: only the
convolution digest differed, while recurrence, position, token history and
feedback matched. Exit 101 and the complete small stderr/stdout are retained.
With the production fix, the same existing single-layer test passed for its F32
and plain-Q8 matrix: 64 capture rows, bases 0 and 64, keep 0 through 64, with abort
and continuation checks. The observed test report is one test passed, not a
full-model result. Both attempts report an actual GPU initialization and an owned
child exit; neither was replaced by a retry.

[The follow-up receipt](gpu-conv-prefix-20261111/FOLLOWUP.json) binds the raw
attempt/build records and externally retained executable hashes. The small raw
records and logs are copied here; large native binaries stay in the original
external QC root.

## Production correction and remaining scope

[Commit acd7f994](gpu-conv-prefix-20261111/production-acd7f994.patch) adds the
existing in-place BF16 round trip after partial captured-row history repair.
Raw F32 capture bytes now obey the convolution producer's BF16-expanded F32
storage boundary. Ordinary AR and full acceptance do not enter this repair.
Each partial repair adds one small existing kernel per GDN layer; no performance
claim follows. The diagnostic documentation worktree contains this patch as
evidence only, not as applied product source.

The permanent version of the corrected test restores the established try_gpu()
skip behavior and emits an actual-GPU marker after successful initialization.
The positive and negative proof binaries used required GPU initialization;
their source/build identities are separately retained.

The original full-model failure included **36 convolution histories plus
12 target QSA selected-index tensors**. This bounded GPU proof addresses the
convolution storage defect; it does not prove that every original live mismatch
is repaired. QSA partial-prefix repair restores complete pre-verify selection
owners while updating active marks. That separate oracle contract must compare
the complete twelve owners to S0, every other canonical row to AR, full acceptance
to AR, and the next AR full-vocabulary logits and canonical state to AR. No owner
may simply be ignored. The new model oracle and its posthoc verifier still need
their own source-pinned CPU and hardware qualification.

The parent full CPU attempt remains FAIL_RETAINED with its separately
classified HFQ/KT48 failures. No original failure, final source qualification,
cache-on pair, service gate, async-overlap gate or speed result is relabelled by
this update.
