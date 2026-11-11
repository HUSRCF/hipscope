# PR #1022: frozen-beta exact-cache qualification

Tested production source: fe6b29f60ab9d26b4cef875d384fbe5708fe0630
Tested tree: 4a0eae200e9b173a6e989caef01a1ac42d639ae8
Base: f453d10da0346d1935f7cc6488ef4aea958b7022

This checkpoint adds documentary receipts over the tested source. Its new
documentation commit was not compiled as a product. PR #1022 retains the
tested fe6 source. Raw logs, numerical receipts and request/SSE artifacts
remain on X570 under hipfireQC/tmpOutput/pr1022-pr1023-review-20261111;
compact files here bind their identities and disclose their limits.

## CPU and product identity

The full workspace/all-target release build, all 61 library test binaries
and canonical scripts/no-gpu-ci.sh executed on d06d93ed6. Only two example
fixtures change from d06 to fe6; production code is byte-identical.
All 61 binaries together report 12,136 passed, one failed, 120 ignored and
zero filtered tests. The failed KT48/LLVM parity test is reproduced on
clean beta with the same 379 normalized diagnostics and unchanged lift
crate. It remains a failure, not an all-workspace pass.

The fixture successor separately passes final precommit, full railgun-cert,
three lab-tool release builds and plain default-feature CLI/daemon builds.
The no-GPU CI pass belongs to the parent, not a claimed full fe6 rerun.
source-and-products.json binds actual Cargo compiler-artifact feature
records and copied binary SHA-256/MD5. Products exclude lab, reference-parity,
test-utils and kernel-timing; all-targets/lab builds are not product evidence.

## GPU correctness

Hardware: X570, gfx1201 R9700. Model: qwen3.8-flash-next.mq4-xts,
125,288,544,792 bytes. The complete model file was not freshly rehashed
for this campaign; actual source-index/hotset identities are in the
numerical receipts.

- Tiny indirection: 44 cases across 11 placements and four execution paths,
  including 11 PM4 replays, wrong-pointer controls and canaries.
- Paired publication passes the real mutable-residency mechanism.
- Production-loader cache-off/on numerical comparison: 582 full-vocabulary
  and active-state records per arm match. Original semantic partition and
  hotset match; physical mirrored placement intentionally differs.
- Code and fiction dirty-short prefills each avoid donor restoration.
  Each long-prefill case restores 10,486,579,200 bytes.
- Canonical Redline: three stable captures, 31 observed kernel contracts,
  ordinary HIP / recorded HIP / PM4 full logits and state match at
  positions 513 through 528. Kernel count is observed, not a fixed gate.
- Known no-independent-submit cancellation recovers initial state/logits.
  Unknown-fault injection and native MTP are not qualified here.

Numerical synchronization drains copy DMA. These checks do not prove
asynchronous overlap or performance.

## Served comparison and quality

The 16K served comparison has 26/26 equal decoded outputs, request metadata
and terminal fields. Battery and chain each naturally stop on all five
rows. Session includes six capped responses and a naturally stopped
unfinished Rust response; the 1,024-token longdecode is also unfinished.
These limitations match clean beta and are not semantic/code passes.
Generated code was not executed.

The 65K LongBench 30 plus LongDecode 5 comparison is documented separately
in qc35-quality-scope.json: all 35 decoded outputs and specified request/terminal fields match. Both arms score 11/30 strict LongBench correct, with six wrong and 13 truncated/unparsed answers; all five LongDecode responses naturally stop (38,268 tokens). The canonical attractor failure remains retained. HTTP SSE does not expose
complete output token IDs. The old driver field relative_byte_and_id_equal
only compares its listed JSON fields; it is not token-ID evidence.
Do not re-tokenize decoded text to manufacture an ID comparison.

Actual loader state is QSA FP8 K/V in VMM and plain GDN Q8; loaded ACK
reports automatic VMM, legacy=false, reason=null. Generic harness null
backend fields remain UNOBSERVED.

## Hosted CI and limits

hosted-ci.json binds the actual current merge-checkout tree to fe6.
Sixteen of nineteen jobs succeed. Windows clock_gettime linkage and
int-rest required-oracle/no-ROCm-policy failures match clean beta; the downstream failed job
only aggregates shard status. Required hosted CI is not green.
The prior mixed-residency CI failure did not recur; its cause is unproven.

No new-beta speedup, peak-VRAM, live Q8Ef, native MTP, sampled-native MTP,
adaptive policy, or broader-device claim is made. Maintainer review and
required CI remain merge gates.

CI discovery precision: the explicit no-ROCm policy returns no oracle while
the test helper requires one. This does not establish physical tool absence;
see ci-policy-clarification.json. Required CI remains red.
