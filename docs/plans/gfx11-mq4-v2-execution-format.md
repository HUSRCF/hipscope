# gfx11 MQ4-v2 execution-format plan

Status: research plan. This is not a serving route or a new checkpoint format.

## Objective

The retained gfx1100 Qwen3.6-27B prefill path is mature at about 1.12k tok/s.
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

Use the production-style F16-intermediate median, 1115.4 tok/s, as the stable
reference. Reaching 1500 tok/s requires:

```text
overall speedup = 1500 / 1115.4 = 1.3448x
wall-time reduction = 25.64%
```

The runtime timeline attributes about 71.7% of prefill wall time to the
packed-MQ4 family. If no other component changes, that whole family must reach
about 1.56x local speedup to reach 1500 tok/s.

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
| rocBLAS rowwise-W8A8 full hot path | 1.06x gate/up, 1.01x down, about 1.88x bytes | rejected |
| Lane-major exact MQ4, packed LDS + register decode | 0.448x gate, 0.408x down; zero spills | closed |

The lossless packed-layout experiments show that eliminating or relocating
nibble expansion alone did not help the measured gate/up shape. The exact IU4
experiment shows that two native IU4 WMMA passes cost more than the expansion
they remove on that shape. A globally expanded INT8 execution copy is also not
an acceptable default: it nearly doubles resident weight bytes and reduces the
context capacity that makes the 27B single-GPU configuration useful.

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
