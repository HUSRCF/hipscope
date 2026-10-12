# Rebased-candidate checks (2026-10-12)

Branch `perf/gfx1100-xts-combined`, base
`b44cc08bbb2e47d9a53e02b24cf0035258c905c5`. This document does not extend
the frozen-base performance claim to this candidate.

Production/test commit: `0a600a3c1` (after formatting-only `1cdac8a53`).
The following evidence commit changes documentation and attachments only.

Tested product build (`cargo build --release --locked`, exit0): daemon MD5
`3a4a12cd6fa5e66d08899e692a9f5b15`, CLI MD5
`7c42bef625b1474fb30be3316554174d`. Raw local records are under
`results/xts-combined-pr/`; `tested-source.patch` and `tested-base.txt`
identify the GPU-tested source. Subsequent formatting-only changes in the
two touched Rust modules and documentation do not change the kernel source
or embedded code objects. Raster and source-scope tests pass again after
formatting; the hardware run predates that formatting commit.

| Check | Result |
|---|---|
| Release product build | PASS |
| Raster interpreter | 1 test PASS (270 grid/epilogue combinations) |
| Existing gfx11 builder tests | 18 PASS |
| HIN/half-V source scope | 6 PASS; static checks, not GPU numeric proof |
| `no-gpu-ci.sh` final isolated rerun | PASS, including 579 pytest tests |
| Ratchets | PASS, 21 metrics |
| Crate maps / scratch growth / changelog | PASS |
| 32K runtime capture, AQL contracts, HIP/PM4/blob parity | PASS; logits/KV/recurrent/GDN exact |
| Standard serving battery, Q8/VMM, AR, max_tokens1024 | PASS, 5/5 finish=stop, no flags/empty; text inspected |
| Loaded HIN/attention ELF and raster bundle | Match frozen candidate objects |
| `cargo test --locked --lib --workspace` | FAIL, exit101: four ISA tests cannot create hardcoded `/home/kaden/...` directories; fail-fast, later crates not covered by this command |
| Release workspace/all-targets build | PASS (`--release --workspace --all-targets --locked`) |
| Changed-file rustfmt | PASS after separate, user-approved two-file formatting-only commit `1cdac8a53` |
| Fixed 4B speed gate | FAIL: PP32 1152.3 passes, decode148.2 below floor169.1 |
| Legacy agentic hook, protocol-adapted temporary copy | PASS, one cell, zero soft warnings; not the current acceptance route |

Failure history retained: first no-GPU run needed the new compile-time HIN
macro registered by the lifecycle doc generator; fixed, no schema/default
change. Second no-GPU run overlapped a legitimate serving test; its final
leak detector flagged that active process. Final rerun after serving exited
passes unchanged checks. First serve command used invalid backend `pm4`
and was refused before inference; corrected to `redline` + transport `pm4`,
rerun in `serve-corrected/`, exit0. First speed-gate wrapper selected a
different lock path; corrected to inherit the same per-card lock, then the
real speed test returned exit1. None of these original logs was rewritten.

The historical clean frozen-base 4B controls also miss the floor (~146 tok/s).
That supports an existing local/reference discrepancy, not a passing gate
or a precise diagnosis. The W7900 and reference device are different; this
absolute-rate gate is not a same-device A/B regression verdict. Request a
rerun on the reference/baseline machine before promotion.

The user explicitly authorized a one-time local speed-hook exemption to
create a Draft PR. A temporary copy exempts speed-gate execution and changes
its success message; other applicable checks remain active.
No repository hook, threshold, test assertion, or CI policy is changed.
This exemption does not relabel the speed test as passed. The two-file
formatting-only commit was verified against rustfmt applied to the base.

The stock legacy agentic hook also fails before generation: it omits the
required `attempt_id` field (`generate missing attempt_id`). A temporary
script copy adds `attempt_id: 1` to each uniquely identified generate request,
without changing prompts, settings or assertions. The hook runs that copy;
neither repository script is changed. This old harness explicitly selects
asym3 (fwht3) on Qwen3.6-27B and automatically loads legacy KV, reported as
`HIPFIRE_KV_BACKEND=legacy` with reason `device gfx1100 has no certified VMM
KV path for the selected mode`. This is not the flagship Q8/VMM battery and
is not claimed as its acceptance route. The original failure and the exact
temporary adapter/hook diffs are retained with the validation evidence.
The adapted run passes (one cell, zero soft warnings), on the all-targets
build daemon MD5 `6330ab7adb7441d38ac8311e1c1746fd`. Its raw token-stream
snapshot ends at `commit_ready`; the original script deletes its temporary
daemon log after evaluating it. The final report and full hook log are retained.

The four workspace failures are unchanged baseline tests:
`residual_xbatch_m7`, `xbatch_pm_m7`, `lane_rows_mq6_m7`, `q8_lane_rows_m7`.
Each fails at `create_dir_all` with permission denied for `/home/kaden/qcal/`.
The ISA crate reports 80 passed / 4 failed. No permission workaround or
unrelated source repair is included in this PR.
