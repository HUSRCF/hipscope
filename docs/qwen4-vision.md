<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next vision: implementation and reproduction

Status (2026-10-23): branch implementation for [issue #938](https://github.com/warpfront/hipfire/issues/938).
The independent QSA and vision-tower checks below have receipts, and the local
single-GPU image HTTP route passed 17/17 cases on Strix Halo (gfx1151). The registry has **no published, paired Flash Next
vision sidecar**. Export one and pass its path explicitly. The current validation
route is [Flash-Next image pathway in VALIDATION.md](VALIDATION.md).

## Checkpoint and artifact

The pinned source is `Qwen/Qwen3.8-Flash-Next` at
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. Its
[official tensor index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json)
contains **333 `model.visual.*` tensors**, all in
`model-00001-of-00131.safetensors`. These include the projector/merger. The
[official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
specifies a 27-block, 1152-wide vision tower with 16 attention heads, patch size
16, temporal patch size 2, spatial merge size 2, and **2560-wide output**. The
Qwen3.8-27B tower's 5120-wide output is incompatible with this trunk.

The implementation reuses hipfire's Qwen35-VL tower equations and image
preprocessing, with a Flash Next-specific configuration and complete-tensor
preflight. The text trunk remains architecture 16; image capability requires a
resident, validated tower.

Build and export from the pinned remote checkpoint:

```bash
cargo build --release
cargo run --release -p hipfire-quantize --bin hipfire-quantize -- \
  --qwen4-flash-next --vision-only \
  --input hf://Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output "$HOME/.hipfire/models/qwen3.8-flash-next-vision.hfq"
```

Production component mode is the default. This route fetches the visual tensor
payload ranges and necessary metadata, without downloading the text weights.
It preserves the original configuration and preprocessor configuration; matrices
are stored as F16 and vectors/tables as F32. The generated local artifact was
checked against all 333 source tensors:

| Property | Local validation artifact |
| --- | --- |
| Filename | `qwen3.8-flash-next-vision.hfq` |
| Size | 904,008,640 bytes |
| Tensor layout | 111 F16 matrices, 222 F32 vectors/tables |
| SHA-256 | `acef5567d9ee0f253d9f8ebde994bd9c37e285754db6d227b3f1df2fea67cc08` |
| Visual BF16 source subset SHA-256 | `6eb9e554981f944118b1d8fd4d12ec5213d996a4510ecd21d3398f8f7d775804` |

The subset hash identifies a safetensors file with a rewritten visual-only
header and unchanged source payloads. These are local receipt identities,
not a registry publication or a claim that a future export has identical metadata.

## Serving and execution scope

With the separately downloaded text trunk already available:

```bash
HIPFIRE_VISION_MODE=on target/release/hipfire serve \
  --model qwen3.8:flash-next --spec off \
  --vision "$HOME/.hipfire/models/qwen3.8-flash-next-vision.hfq"
```

`--vision` overrides the environment/registry sidecar path; `vision_mode=off`
still prevents the tower from loading. The loader allocates the tower before
planning text placement and reserves the maximum admitted image scratch until
text and optional draft allocation finish. Device allocation backing is queried
separately from integrated-GPU identity: a BIOS VRAM carve-out is not charged
again against the host RAM left outside it. Image discovery
reflects the loaded tower, not merely a configured filename. See the existing
[vision configuration contract](CONFIG.md#vision-tower) for `on`, `auto`, and `off`.

The implemented image route is **one image, one GPU, cold autoregressive
generation**. Video, multiple images, and distributed image execution are
rejected. Speculative image decoding and image prefix/session reuse are outside
this route. Standard sampling, stop sequences, and the normalized Qwen reasoning
controls are forwarded to the common AR producer. Final resize dimensions are
checked before allocating the resized image; the attention hard limit is 16,128
patches. Temporary GPU owners are transactional across upload, each layer, and
the merger. A failed fence quarantines allocations instead of recycling memory
that may still have in-flight readers.

The carrier follows these upstream semantics:

- Replace the expanded image-pad embedding rows with the tower's 2560-wide
  outputs **before HC stream expansion**. Keep the original image-pad token IDs
  for PLE/hash inputs.
- Retain the complete prompt's T/H/W positions. QSA query, key, and indexer query
  use interleaved mRoPE sections `[11, 11, 10]`; decoding uses the sequence
  position plus the image's `rope_delta`.
- Pool raw indexer keys, normalize, then rotate using the **first token's**
  T/H/W position in each compression block. Chunking cannot replace this with a
  scalar position or the block's last position.
- During the image request, bypass the NPU trunk plan, one-dimensional QSA
  fusions, lane promotion, retained PM4/AQL replay, and token-only prefix/session
  caches. Graph capture is refused for this carrier. Request reset releases the
  image context after GPU completion and restores normal text request policy.

The implementation lives in `hipfire-arch-qwen4`'s `vision` and
`bundle/image_context` modules, `hipfire-generate/src/qwen4_vision.rs`, and the
optional QSA mRoPE bindings in `rdna-compute/src/qsa_mrope.rs`.

## Validation

CPU checks:

```bash
cargo test -p hipfire-arch-qwen4 --lib vision
cargo test -p hipfire-arch-qwen4 --lib image_context
cargo test -p hipfire-loader --lib vision
cargo test -p hipfire-generate --lib qwen4_vision
```

For the independent QSA kernel reference, build before acquiring the shared GPU
lock; run it separately from a resident serving process:

```bash
cargo build -p rdna-compute --features lab --example qsa_mrope_parity
mkdir -p .codeinsight+research/vision-938
source scripts/gpu-lock.sh
gpu_acquire qwen4-vision-mrope
trap gpu_release EXIT
HIPFIRE_REPLAY_BACKEND=hip timeout 120s target/debug/examples/qsa_mrope_parity \
  > .codeinsight+research/vision-938/qsa-mrope-parity.json \
  2> .codeinsight+research/vision-938/qsa-mrope-parity.log
gpu_release
trap - EXIT
```

The gfx1151/HIP 7.16 receipt covers rectangular image positions, gated Q,
contiguous K, indexer Q, chunk boundaries, decode delta, and incremental F32/BF16
key pooling. Maximum norm/rotation absolute error was `8.94e-7`; chunked versus
unchunked outputs and both pooling references matched exactly. Full local
receipt: `.codeinsight+research/vision-938/QSA-MROPE-RECEIPT.md`.

For tower validation, [dump_qwen4_reference.py](../benchmarks/vision/dump_qwen4_reference.py)
loads only the 333 original BF16 visual tensors into the official Transformers
vision model on CPU. Feed its `pixel_values.bin` and grid dimensions to
`cargo run --release -p hipfire-arch-qwen4 --features lab --example qwen4_vision_dump --`
with `--model`, `--patches`, `--grid-h`, `--grid-w`, and `--out`. Rerun the CPU
script with `--compare` pointing to that dump to compare every stage. Receipts
record source, input, configuration, implementation, and binary identities.
The existing gfx1151 comparison used a `[1,12,18]` grid and `[54,2560]` projected
output: final maximum absolute error `6.69e-4`, relative L2 `0.001404`, cosine
`0.999999017`. Its local receipt is
`.codeinsight+research/vision-938/tower-stage-comparison-complete.json`.
These are component numerical checks; neither supplies E2E quality or timing
acceptance.

Against the running service, save the HTTP report and inspect every answer:

```bash
python3 scripts/check_qwen4_vision.py --base-url http://127.0.0.1:11435 \
  --model qwen3.8:flash-next --expect-vision on \
  --output .codeinsight+research/vision-938/http-on.json
```

This covers streaming/non-streaming image requests, image replacement, OCR and
description fixtures, disconnect/follow-up behavior, and text after an image.
Repeat after restarting the service with `HIPFIRE_VISION_MODE=off`, using
`--expect-vision off` and a separate report; discovery must omit image support,
image requests must be refused, and text must still work. Record exact trunk,
sidecar, CLI, and daemon identities alongside both reports.

Local result (2026-10-23, gfx1151/HIP 7.16, single GPU, q8 KV, eager HIP,
`--spec off`, `--max-seq 8192`, NPU trunk off): **17/17 image HTTP cases passed**.
Red/green/blue answers matched across streaming modes; all five French direction
signs were transcribed correctly; the dog was correctly described as lying on
its back with closed eyes and raised paws. Stop sequences and low-effort capped
reasoning were honored. An extreme aspect-ratio image was rejected before tower
execution, and cancellation followed by text and another image remained usable.
The separate vision-off run passed all ten image refusals plus a text request.
Decoded OCR and description answers were inspected against the committed images.
These are functional results, not a performance comparison. Mixed speculative
text/image serving and NPU-enabled text loads were not exercised in this run.

Final image-report identities:

- Trunk SHA-256: `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`.
- CLI MD5: `9418bfeb06e3829e41253a61f51465c8`.
- Daemon MD5: `e69bc229887fc6a0ba4d4f5625c5a4c7`.
- Local full report: `.codeinsight+research/vision-938/http-final.json`, including
  exact image and prompt digests, complete answers, and stream terminals.
- Final text battery/chain reports and capture diagnostic live beside it;
  the first text attempt with only `--thinking off` retained template reasoning
  and exhausted its small output budget. Successful text runs explicitly pass
  `--thinking-effort none --max-tokens 512`; the unsuccessful receipt is retained.

### Text capture diagnostic on Halo

The additional text-only `redline_daemon_harness.py --qwen4 --pm4` check uses
BF16 QSA state, context 128, maximum context 2048, and five shadow positions
129–133. On gfx1151, `HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` is required to compare
the same attention reduction route: eager split-K is enabled by default on
Halo while recording uses the per-head kernel. With split-K disabled, all
positions matched ordinary HIP, recorded HIP, and PM4 bit for bit, with stable
1,177-launch capture and 33 passing kernel contracts. The same binary with
split-K enabled failed that exact-state comparison. Both reports are retained;
the default configuration is not claimed to pass. This diagnostic does not
admit a Flash Next PM4 product route on Halo, and image requests remain eager.
