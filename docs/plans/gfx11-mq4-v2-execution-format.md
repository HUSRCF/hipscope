# gfx11 MQ4-v2 execution-format plan

Status: research plan. This is not a serving route or a new checkpoint format.

## Objective

The retained gfx1100 Qwen3.6-27B prefill path is mature at about 1.12k tok/s.
The best controlled combination measured so far, including staged CK,
group256, and the F16 FFN intermediate, is about 1.189k tok/s.
The next backend must improve the common packed-MQ4 primitive used by gate,
up, down, and the other large projections. It must not be another launch-tile
sweep over the existing dataflow.

The current production implementation uses Wave32 WMMA, not MFMA. Its source
weight contract is 136 bytes per 256 weights:

```text
8 B   FP32 affine scale/zero
128 B packed 4-bit payload
```

The kernel expands the 4-bit payload to the IU8 WMMA feed representation in
LDS. The measured full-set/full-add code objects use 256 VGPRs, four VGPR
spills, and 57,344 bytes of dynamic LDS per workgroup.

## Performance admission line

Use 1115.4 tok/s as the conservative stable reference and 1189 tok/s as the
best controlled configuration. Reaching 1500 tok/s requires:

```text
stable reference: 1500 / 1115.4 = 1.3448x, 25.64% wall reduction
best reference:   1500 / 1189.0 = 1.2616x, 20.73% wall reduction
```

The runtime timeline attributes about 71.7% of prefill wall time to the
packed-MQ4 family. If no other component changes, that whole family must reach
about 1.56x local speedup from the conservative reference or 1.41x from the
best controlled configuration to reach 1500 tok/s.

That family label does not mean weight bytes dominate the kernel. For the
retained X256/Y64 geometry at N=2048, the explicit movement model is:

```text
gate/set M17408 K5120:
  Q8 activation staging  ~= (M/64) * N * K * (144/128) = 3.21 GB
  MQ4 weight staging     ~= (N/256) * M * K * (136/256) = 0.379 GB
  FP32 output write      ~= N * M * 4 = 0.143 GB
  total                  ~= 3.73 GB / 4.275 ms = 872 GB/s effective

down/add M5120 K17408:
  Q8 activation staging  ~= 3.21 GB
  MQ4 weight staging     ~= 0.379 GB
  FP32 output access     ~= 0.042 GB per read or write
```

The effective rate includes cache/LDS behavior and is not a claim that every
byte reaches HBM. It does establish that repeated activation-tile movement,
not only packed-weight decode, is a first-order part of the primitive. A new
weight representation alone cannot be assumed to deliver the 1.56x family
speedup; it must either materially change the whole feed path or be paired
with a dataflow that reduces activation replication.

A new experiment is admitted only if its plausible affected wall share is at
least 10%, or it can be implemented and rejected within one short probe. A
new execution backend is promoted beyond standalone only after the large
gate/up and down shapes each reach at least 1.30x. A 1-3% local result does not
justify production integration.

## Measured same-math ceiling

A mature CK A16 x I4 screen and a dense-FP16 rocBLAS roofline now bound the
remaining same-math execution-format space on this host. The best CK FP16 x I4
instances reached only 1.061x on gate/up and 1.015x on down. Dense FP16, which
optimistically removes packed decode, affine scale/zero handling, and Q8 feed
conversion, reached 1.138x and 1.092x respectively. Applying the better 1.138x
ceiling to the entire measured packed-MQ4 wall share yields only about 1.095x
overall, or roughly 1.30k tok/s from the 1.189k controlled baseline.

Consequently, MQ4-v2 is no longer admitted as a layout-only rewrite. A future
same-math backend must first demonstrate a mechanism that materially exceeds
the measured dense rocBLAS/CK roofline. Otherwise the only candidates with a
credible path to 1.5k must reduce effective model work and pass an explicit
quality gate.

## Model-work reduction boundary

An active-width sweep of the retained packed-MQ4 primitive establishes the
performance requirement before any checkpoint surgery:

| FFN width retained | Combined gate/up/down local speedup | Projected overall |
|---:|---:|---:|
| 75% | ~1.35x | ~1.36k tok/s |
| 62% | ~1.60x | ~1.46k tok/s |
| 58% | ~1.70x | ~1.49k tok/s |

The projection uses the measured 49% wall share of the three large FFN
projections. It shows that reaching 1.5k requires approximately 42-44% less FFN
work. Contiguous tail truncation is not a quality proposal: MQ checkpoints use
a rotation contract around the down projection, so any channel-pruned artifact
needs calibration, a rotation-aware repack or retraining, and task-level
quality evaluation. The reproducible timing record is in
`experiments/gfx11-mq4-v2/ffn-width-work-reduction/`.

A per-token post-SwiGLU energy oracle tested the most favorable dynamic version
of this idea. Keeping 41/68 rotated 256-channel groups, close to the required
work reduction, increased PPL by 9.25% on the original document and 52.79% on
an out-of-domain WikiText2 slice. Keeping 60/68 groups increased PPL by 1.02%
and 6.92% respectively; even its ideal affected wall share is below 10%. The
route is closed as a transparent optimization. The oracle is diagnostic only:
it zeroes the packed Q8 input after gate/up and does not reduce work. Results
and the reproducer are in `experiments/gfx11-mq4-v2/dynamic-ffn-oracle/`.

## Closed directions

These results apply to the current HFQ4/MQ4 affine contract and Wave32 WMMA
dataflow. Unless noted otherwise, they were measured on the large gate/up set
shape and therefore close that candidate architecture for the first admission
gate; they do not establish a down/add result. They also do not disprove future
WMMA kernels with a different numerical contract.

| Direction | Full-shape result | Decision |
|---|---:|---|
| Packed weights in LDS, X256/Y64 | 0.482x | closed |
| Packed weights direct from global memory | 0.421x | closed |
| Packed weights in LDS, X128/Y64 | 0.560x | closed |
| Exact Q8 as two IU4 planes | 0.462x | closed |
| FP16 metadata-only repack | 0.934-1.006x | closed |
| Expanded-I8 execution copy, aligned quad-row | 0.9997x gate/up, 1.0045x down | rejected |
| Remove affine zero correction | 0.9995x gate/up, 0.9910x down | rejected |
| Remove scale accumulation and zero correction | 1.0166x gate/up, 1.0431x down | rejected upper bound |
| K128 phased X256/Y64 | 0.416x | closed |
| K128 X256/Y128, 512 threads | 0.622x | closed |
| Signed-A4 gate/up | +4.89% PP8192, failed long-prompt quality | rejected |
| Symmetric signed-int4, no affine zero correction | 1.027x gate/up, 1.039x down | rejected |
| Mature Q8_0 WMMA execution copy, 2x weight bytes | 0.279x gate/up, 0.310x down | rejected |
| Current HFP4G32 kernel implementation | 0.361x gate/up, 0.330x down | rejected implementation, not format |
| Current HFQ3-G256 Wave32 WMMA implementation | 0.397x gate/up, 0.293x down | rejected implementation, not all 3-bit formats |
| Current MQ3-Lloyd-G256 Wave32 WMMA core | 0.348x gate/up, 0.300x down; pre-rotation excluded | rejected implementation, not all codebook formats |
| Current MQ2-Lloyd FP16-WMMA four-wave core | 0.1786-0.1834x in same-process paired dense FFN runs; quality already rejected | rejected implementation and current MQ2 quality contract |
| rocBLAS rowwise-W8A8 full hot path | 1.06x gate/up, 1.01x down, about 1.88x bytes | rejected |
| X128/Y128 with retained quad-row loader | 0.9955x gate/up, 0.9849x down | rejected |
| Lane-major exact MQ4, packed LDS + register decode | 0.448x gate, 0.408x down; zero spills | closed |
| Row-I8, row-scale, full-K int32 accumulation | 0.404x gate, 0.414x down; 1.88x bytes | closed |
| Row-Q4, row-scale, full-K int32 accumulation | 0.480x gate, 0.434x down; 0.94x bytes | closed |
| Independent K16 IU8-WMMA halves plus exact i32 merge | 0.8886x gate, 0.8940x down; bit-exact | closed |
| Lane-owned affine metadata plus `ds_bpermute` | 0.7672x gate, 0.7607x down; bit-exact | closed |
| Module-exact gfx11 CU-mode scheduling | PP8192 1144.6 -> 1123.7 tok/s (0.9817x); decode 33.3 -> 33.2 tok/s | closed |
| Late-lane epilogue index reconstruction | spills 4 -> 2, private 20 -> 12 B/thread; weighted set/add +0.04% | rejected |
| Group256 activation staged in LDS | candidate throughput is 0.755x serial-row gate/set and 0.704x serial-row down/add | closed |

The lossless packed-layout experiments show that eliminating or relocating
nibble expansion alone did not help the measured gate/up shape. The exact IU4
experiment shows that two native IU4 WMMA passes cost more than the expansion
they remove on that shape. A globally expanded INT8 execution copy is also not
an acceptable default: it nearly doubles resident weight bytes and reduces the
context capacity that makes the 27B single-GPU configuration useful.

The late-lane epilogue probe confirms that the reported four-spill count is not
itself a useful optimization target. Reconstructing the Wave32 lane ID at the
epilogue removed two spills, but the code object remained in the 256-VGPR tier
and the production-weighted timing was neutral. The logical wave index cannot
be reconstructed from `HW_ID1.WAVE_ID`, because that register reports a
physical SIMD wave slot rather than the wave's logical index within its
workgroup. The exact probe and resource record are in
`experiments/gfx11-mq4-v2/results/late-lane-epilogue-gpu1-20260812/`.

The existing 140-byte compact group128 activation record is also not admitted
as a new single-output backend. It saves only four bytes from the 144-byte
wire record. More importantly, its 35-dword row stride rotates 16-byte
alignment across LDS rows while the Wave32 WMMA feed uses 16-byte fragment
loads. A safe exact path must expand or repack it to the canonical 36-dword
LDS stride before the WMMA loop, leaving only a 2.78% global metadata saving
and adding a repack. This remains useful inside bounded standalone probes, but
it has no credible path to the 1.30x primitive admission line.

The lane-major execution copy additionally removes the row-major access
pattern as a confounder. It keeps the exact 136-byte affine MQ4 contract but
reorders each 16-row, 256-K tile as
`payload[subK32][packed_word][row]`, allowing contiguous global staging and
conflict-light LDS fragment reads. Despite 217 VGPRs, zero spills, and zero
private bytes, it reached only 0.448x on gate and 0.408x on down. Packed-nibble
register decode inside the WMMA loop, rather than the old physical row order,
therefore remains the dominant failure in this architecture. The full record
is in `experiments/gfx11-mq4-v2/results/lane-major-packed-lds-gpu1-20260811/`.

The measured mature Q8_0 backend made this tradeoff worse rather than better:
it doubled resident weight bytes and was 3.2-3.6x slower than retained MQ4 on
the full gate/up and down shapes. It is therefore not an execution-format upper
bound for this workload.

The row-scale probes additionally changed the accumulation contract: integer
Wave32-WMMA accumulators remained live across the complete K range and scales
were applied only in the epilogue. Row-I8 nearly doubled resident weight
bytes; row-Q4 was slightly smaller than retained MQ4. Both were less than half
as fast despite at most 93 VGPRs and zero spills. This closes the specific
full-K-accumulator/K128-staging topology; the negative result is not a claim
against Wave32 WMMA itself.

## Open design space

### 1. Symmetric int4 weight contract

The first probe replaced affine unsigned int4 weights with signed int4
weights and one scale per group. This is not the existing HFQ4G128 path, which
also stores an affine scale and zero. The intended arithmetic is:

```text
w ~= scale_w * signed_i4
x ~= scale_x * signed_i8
dot ~= scale_w * scale_x * WMMA(signed_i4_or_i8, signed_i8)
```

Removing the weight zero point also removes the activation-sum correction from
the GEMM contract. A useful implementation should therefore reduce both
metadata and the scale/correction live range, rather than merely saving four
header bytes. Start with group128 and group256 quality/performance probes made
from original BF16/FP16 weights. Requantizing the existing MQ4 artifact is
acceptable for plumbing but not for the final quality judgment because it
adds a second quantization error.

The standalone implementation continued to unpack signed nibbles to IU8 WMMA
operands. It reached only 1.027x on gate/up and 1.039x on down, so removing the
affine correction alone is closed and no checkpoint-level quantizer was added.
Native IU4 remains a separate follow-up only if one pass can preserve the
activation contract; the exact two-pass IU4 result is already closed.

### Coarse-scale int32 accumulation (closed)

A matched Wave32-WMMA probe shared weight and activation metadata across four
adjacent group128 blocks and retained the i32 accumulator for all 512 K values
before applying scale and zero correction. This reduced dequant/correction
flushes by 4x and reduced standalone resident weight bytes by about 8%, while
remaining exact on a synthetic input whose four sub-groups intentionally used
the same metadata. Two independent GPU1 runs measured only 1.016x-1.034x on
gate/up and 1.066x-1.078x on down. The kernels used 73 VGPRs, Wave32, no LDS,
and no spills, so resource failure does not explain the small gain.

This closes scale coarsening and delayed dequantization as a >=1.30x execution
backend candidate. It is not a checkpoint-quality result: changing a trained
model from group128 to group512 remains an approximate quantization change.
The full standalone record is in
`experiments/gfx11-mq4-v2/coarse-scale-int32-accum/`.

### 2. Bounded-correction IU4 contract (closed)

The only native 4-bit path with a positive throughput signal used signed-A4
activations, but its uncorrected full-shape gate/up kernel reached only 1.087x
over retained Q8 and failed the long-prompt quality gate after 32 matching
output tokens. Because 1.087x is already below the 1.30x admission line, any
non-free correction cannot produce an admitted backend. Group32 A4 was slower
and less accurate. Do not implement a correction kernel unless a new IU4 main
primitive first demonstrates at least 1.30x on both gate/up and down.

### 3. Other quantized weight contracts

A new weight format may change the affine scale/decode contract rather than
only rearranging the same 136-byte groups. It must be designed backwards from
the gfx11 WMMA operand and vector-load layout. This requires checkpoint-level
requantization and a quality study; it is not a transparent runtime repack.

The first format probe must report:

- resident bytes per weight, including metadata;
- load-to-WMMA conversion instructions;
- dynamic LDS, VGPRs, spills, and workgroup geometry;
- gate/up and down full-shape medians;
- relative L2, cosine, and long-prompt token-quality results.

### 4. Mature alternate primitive

A vendor or external INT8/INT4 GEMM primitive is relevant only if it consumes
the model's affine quantization semantics without a full expanded-weight copy,
or if its measured speedup is large enough to justify that copy as an explicit
high-memory profile. A dense FP16 peak comparison is not evidence for this
path.

The current HFP4G32 implementation and a complete rocBLAS rowwise-W8A8 hot
path have both been screened and rejected. These results do not disprove all
HFP4 numerical formats or future vendor primitives, but neither existing
implementation is a viable production fallback for this workload.

The existing HFQ3-G256 and MQ3-Lloyd-G256 Wave32 WMMA cores were
also screened at the same full FFN shapes. Their smaller resident weight
footprints did not offset their current cross-byte/codebook decode and operand
feed costs; both were substantially slower than retained MQ4. They are closed
as implementation reuse candidates, not as format-level impossibility results.
The Lloyd timing excludes its required activation pre-rotation and is therefore
an optimistic core upper bound.

The mature MQ2-Lloyd grouped FP16-WMMA kernel was mapped to a one-expert,
top-k-1 dense projection as a deliberately optimistic lower-bit upper bound.
Same-process paired medians across two GPU1 processes were 26.39-26.75 ms for
gate/up and 26.29-26.72 ms for down. The retained MQ4 set-mode control measured
4.71-4.78 ms and 4.82-4.90 ms respectively, placing this MQ2 implementation at
only 0.1786-0.1834x. The result rejects the current implementation without
assigning the slowdown to one unisolated mechanism. MQ2-Lloyd had already
failed the model-quality gate, so no dense serving adapter or checkpoint
conversion is justified by this result.

## Required coverage

The first backend slice must expose one single-projection primitive usable by
all of these shapes:

```text
gate/up:       M=17408, K=5120
down:          M=5120,  K=17408
attention QKV: M=12288, K=5120
GDN QKVZA:     M=10240, K=5120
```

Gate/up multi-output fusion is deferred until the single-projection primitive
has resource headroom. Fusing outputs on top of the current 256-VGPR,
57-KiB-LDS kernel is likely to increase pressure rather than remove it.

The balanced X128/Y128 activation-reuse probe was repeated with the retained
quad-row loader to remove the loader confound from the earlier comparison. It
remained neutral-to-negative on both full FFN shapes despite using 228 VGPRs
and zero spills. This closes nearby tile-geometry work under the current
packed-MQ4 contract when combined with the earlier topology screens; this
probe by itself rejects the balanced X128/Y128 candidate. Future candidates
must change effective model work or demonstrate a primitive that exceeds the
measured dense-FP16 backend roofline.

## Promotion gates

1. Standalone gate/up and down are both at least 1.30x faster than retained
   group128 quad-row Q8.
2. No unexpected scratch; resource use is recorded from the code object.
3. Resident execution-format overhead is at most 10% by default. A larger
   footprint must be a separately named high-memory profile.
4. Exact formats match the retained output tolerance. Approximate formats pass
   long-prompt generation and a task-level quality suite before model routing.
5. PP8192 ABBA confirms an end-to-end gain with identical routing except for
   the candidate backend.
6. The old backend remains available as the stable A/B reference and fallback.

## Frozen retained improvements

The optional F16 FFN intermediate path is retained as an opt-in candidate. Its five-pair PP8192 test
measured a 1.31% paired median gain, four of five pairs positive, unchanged
decode throughput, and identical recorded token IDs. It is a validated
incremental optimization, but not the MQ4-v2 architecture or a broad quality
approval.
