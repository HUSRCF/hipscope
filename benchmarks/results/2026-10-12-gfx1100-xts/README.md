# gfx1100 XTS: raster4 + half-V + HIN one-pass

Frozen, combined A/B evidence; **not a performance measurement of the rebased
PR head**. Production changes comprise four files; no model format, dispatch
policy, FA-cache fusion, or extra global device buffers are introduced.

## Identities and protocol

- A: beta `6d71d955e711dc816aa87f066364a8b59545bb76`, daemon MD5
  `7d191dfa5e66552e798739be624f20b5`.
- B: production-equivalent `38d77049593e7c92f4d86995d98fe5aaa1c8b492`
  plus three archived patches; daemon MD5 `1940c6b40973e389067448b6e7177518`.
- Shared CLI MD5 `788307b52efb83de93ff16fbbca39040`.
- Qwen3.8-27B MQ4-XTS SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`.
- AR/spec-off, Q8/VMM, PM4 in both arms, HIP 7.15, automatic clocks.
- GPU0: W7900, PCI `0000:53:00.0`, ABBAAB. GPU1: W7900 Dual Slot,
  PCI `0000:9e:00.0`, BAABBA. Three fresh processes per arm, not pooled
  across cards. Work on the two cards overlapped.
- Serving: same 23,972-token prompt, MD5 `0934df59b617db97059b861e0ce4443b`,
  two requests/process, row0 warmup, row1 measured, no prefix-cache hits,
  greedy/thinking-off, max_seq65536, max_tokens1024.
- Synthetic: PP4096/8192, TG1024 at ctx8192, max_seq16384, one throwaway
  per PP size and three TG warmups, one measurement/process.

## Results

Median [min, max]; tok/s unless noted. Instrumented traces excluded.

| Instrument | Metric | A | B | Median change |
|---|---|---:|---:|---:|
| GPU1 synthetic | PP4096 | 2364.1 [2360.2,2372.9] | 2423.8 [2422.9,2424.8] | +2.53% |
| GPU1 synthetic | PP8192 | 2266.7 [2264.6,2272.9] | 2327.6 [2325.0,2328.3] | +2.69% |
| GPU1 synthetic | TG1024 | 40.682 [40.637,40.693] | 40.831 [40.543,40.855] | +0.37% |
| GPU0 serving | Prefill | 1802.1 [1798.8,1809.1] | 1840.2 [1833.5,1841.8] | +2.11% |
| GPU0 serving | Decode | 38.1 [38.1,38.2] | 38.4 [38.3,38.6] | +0.79% |
| GPU0 serving | TTFT (s) | 13.372 [13.320,13.397] | 13.097 [13.085,13.144] | -2.06% |

Synthetic decode ranges overlap: effectively flat, not a robust speed claim.
50ms board-peak VRAM medians differ by +8 KiB serving and +4 KiB synthetic;
no material increase observed. Sampling is a lower bound, not allocation
accounting. All 12 formal serving response records match; they are capped
stories, not evidence that each story completed. Six processes return exit0.

## LongBench and correctness limits

B-only LongBench-v2 hard30, original long inputs (rendered ctx20405–30248),
greedy/thinking-off, Q8/VMM, max_seq65536, output cap1024:
30/30 completed, zero runtime errors/empty outputs, 25 parseable choices,
11 correct. All 30 checked transcripts, prompt/request hashes, context/token
counts and finish reasons match saved A. A was not rerun; its per-run binary
and template provenance is incomplete, so it is not a new matched timing A/B.

Both have 3 attractor-detector flags and 5 capped outputs; stock exit1 is
preserved. This is **no observed regression on this fixture**, not a passing
quality gate or proof of universal quality. The 1-based flagged questions are
23/25/27; repeated round listings/arithmetic occur in their text, with 25/27
ending in answers. The detector is not waived or modified.

Separate runtime A check passes at ctx32704 (HIP/PM4/blob logits, KV,
recurrent state and GDN frame parity). Frozen B reuses exact-binary HIN-round
parity evidence. Combined loaded-object audit: only the three expected
modules change among 37; the other 34 images and metadata agree.

## Raw evidence

[combined-evidence.tar.gz](combined-evidence.tar.gz), SHA-256:
`2956eb4f22d34b560342fc9e68fc73ad5cfba71ce3894933e0e54daf53a8fa57`.
Contains raw outputs/logs/telemetry, loaded-object caches, all speed samples,
source hashes, build manifest/patches, LongBench fixture and saved A output.
No model weights or daemon binaries. Extract in a fresh directory; absolute
paths in records are original provenance, not portable replay commands.

New CPU checks: `cargo test -p hipfire-isa --features toolchain --test v2c_raster`
and `python3 scripts/test_gfx1100_xts_scope.py`. Existing bundle-emission checks:
`cargo test -p hipfire-isa --features toolchain --test gfx11_builder`.
Static/source checks do not replace GPU numerical validation.

## Benchmark revision disclosure

New-base checks and limitations: [rebased-validation.md](rebased-validation.md).
The full flagship serving JSON is [battery.json](battery.json); raw build/test
logs, runtime JSON, invocation scripts, source patch and temporary hook diffs
are in [rebased-validation.tar.gz](rebased-validation.tar.gz), SHA-256
`21a60c30ea4c4edfd36f3444fb9b40e95734f1ecb518a15354a8ec040becb29e`.
Kernel caches, private HOME directories and model binaries are excluded.

These numbers are based on frozen beta `6d71d955e` and its three-optimization
candidate, **not the rebased PR head based on beta `b44cc08bb`**. New-head
build, CPU and hardware validation results are recorded separately; no
new-head throughput measurement is implied.
