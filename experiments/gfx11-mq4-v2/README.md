# gfx11 MQ4-v2 experiments

This directory is reserved for execution-format experiments that satisfy
`docs/plans/gfx11-mq4-v2-execution-format.md`.

Do not add another tile-only variant of the retained group128 quad-row kernel.
Every candidate must state which numerical or execution-format contract it
changes, its resident-byte overhead, and the packed-MQ4 wall share it can
plausibly affect.

P0, a symmetric signed-int4 weight contract with no affine zero-point
correction, was rejected at 1.027x gate/up and 1.039x down. Existing HFQ4G128
is not such a contract. P1 is a bounded-correction IU4 activation path only if
the correction remains sparse and the combined kernel clears the performance
and quality gates.

The existing Q8_0 Wave32 WMMA backend was also rejected as a high-memory
execution copy: it used 2x the weight bytes and reached only 0.279x gate/up and
0.310x down relative to retained MQ4.

An operand-ready expanded-I8 upper bound was then tested with the current
quad-row workgroup topology. It preserves the exact affine MQ4 and Q8
activation math, but expands each 136-byte weight group to 264 bytes at load
time so the kernel can stage IU8 WMMA operands without runtime nibble unpack.
The quad-row probe pads this to 272 bytes per group so every 256-byte payload
starts at a 16-byte-aligned address.
The standalone benchmark is `bench_mq4v2_expanded_quad`.

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
pairs: 7

shape                         retained MQ4    expanded I8    speedup    max_abs
gate/up M=2x17408 K=5120        9.7776 ms       9.7808 ms    0.9997x          0
down add M=5120 K=17408         4.8630 ms       4.8410 ms    1.0045x          0
```

Both full-set/full-add code objects remained at 256 VGPRs, four VGPR spills,
20 bytes of private state per work-item, and wave32. Eliminating nibble unpack
therefore did not reduce the limiting resource footprint, while the execution
copy doubles resident weight bytes. This closes expanded-I8 as both the
default and high-memory MQ4-v2 path.

The full-set ISA comparison is consistent with the timing result. Expanded-I8
removed all 16 `v_perm_b32` nibble-expansion instructions, but the static
instruction count only changed from 1422 to 1404. Both variants still emitted
128 Wave32 WMMA instructions and 128 integer-to-FP32 conversions, while
expanded-I8 used 77 global loads versus 75 for retained MQ4. Loader-only
execution formats are therefore below the MQ4-v2 admission bar: a viable
format must also reduce the accumulator/live-range or scale/decode work that
survives into the compute loop.

An existing timing-only ablation also removed the affine zero-point
correction while setting all synthetic weight zero metadata to zero. This
kept the group128 execution path otherwise unchanged:

```text
shape                          reference    skip zero    speedup
gate/up M=17408 K=5120 N=2048   4.8360 ms    4.8382 ms    0.9995x
down add M=5120 K=17408 N=2048  4.9322 ms    4.9770 ms    0.9910x
```

Removing zero correction alone has no performance budget on these shapes.
This agrees with the rejected symmetric-int4 probe and rules out a new weight
contract whose only execution advantage is deleting the affine zero term.

A stronger timing-only upper bound kept the retained quad-row loads, Wave32
IU8 WMMA count, and FP32 output accumulation, but deleted both per-group
`scale_w * scale_x` application and affine zero correction. Its outputs are
intentionally not numerically meaningful:

```text
shape                          reference    scale-free    upper bound
gate/up M=2x17408 K=5120         9.6717 ms     9.5142 ms       1.0166x
down add M=5120 K=17408          4.7937 ms     4.5958 ms       1.0431x
```

The scale-free code object eliminated the four VGPR spills and 20 bytes of
private state, but still allocated 256 VGPRs and remained Wave32. Even this
unrealistic upper bound is far below the 1.30x admission line. A new scale or
zero-point contract is therefore insufficient unless it also changes the
WMMA feeding/accumulation architecture.

The shipped HFP4G32 wave32-WMMA backend was screened as a genuinely different
scale/decode contract before requantizing a full checkpoint. The standalone
benchmark is `bench_hfp4g32_mq4v2`; it compares the production FFN shapes in
one process with alternating launch order after kernel and DPM warmup. The
retained reference used the group128 quad-row MQ4 path.

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
pairs: 7

shape                         retained MQ4    HFP4G32    HFP4 speedup
gate/up M=2x17408 K=5120       12.5819 ms     34.8922 ms     0.3606x
down add M=5120 K=17408         6.4021 ms     19.3803 ms     0.3303x
```

The existing `test_gemm_hfp4g32` correctness anchor passed its QKV, gate/up,
and residual checks before timing. The benchmark creates independent synthetic
MQ4 and HFP4 weights and does not compare cross-format outputs. It therefore
rejects the current HFP4G32 kernel implementation as an MQ4-v2 performance
candidate, not the HFP4 numerical format in general. The current MFP4G32 path
uses the same HFP4 payload/decode and adds an activation FWHT, so it is not
prioritized for this experiment; this is not a format-level impossibility
claim.

A high-memory rowwise-W8A8 sidecar was also screened through rocBLAS INT8 GEMM.
The optimistic raw GEMM initially reached 3.01 ms on gate/up and 3.12 ms on
down, so `rocblas_w8a8_pipeline.cpp` added the missing rowwise activation
quantization and INT32 scale epilogue. Weight conversion remained load-time
work and was excluded. After 1000 pipeline warmup iterations to reach stable
W7900 clocks, the apparent raw advantage disappeared:

```text
GPU: AMD Radeon Pro W7900 / gfx1100, ROCm 7.14
N: 2048
warmup: 1000
trials: 15

shape                    quant       rocBLAS      epilogue     total
gate/up M17408 K5120     0.1434 ms   4.0968 ms    0.4106 ms   4.6515 ms
down M5120 K17408        0.3964 ms   4.2116 ms    0.1260 ms   4.7339 ms
```

The matching retained-MQ4 medians are about 4.91 ms per gate/up projection and
4.78 ms for down. Even a hypothetical free quantizer and epilogue leave only
about 1.20x and 1.13x raw-GEMM ceilings, below the 1.30x admission threshold.
Including row scales, the rowwise-I8 sidecar uses 89,198,592 resident bytes for
gate/up and 89,149,440 for down, versus 47,349,760 bytes for MQ4, an
approximately 88.3-88.4% increase. A 32-output synthetic sample did not expose
an immediate numerical blocker (`relative_l2=0.00644/0.00697`,
`cosine=0.999979/0.999977`), but it is not a full-output or task-level quality
result. Steady-state throughput and capacity already reject the candidate, so
do not add rocBLAS/hipBLASLt W8 serving integration for this path.

The existing gfx11 3-bit Wave32 WMMA implementations were then screened with
`bench_hfq3g256_mq4v2`. The harness uses the retained group128 quad-row MQ4
primitive as its reference and tests both HFQ3-G256 affine weights and
MQ3-Lloyd-G256 codebook weights. It runs the production FFN shapes in one
process with alternating order after kernel warmup and a five-second DPM
warmup. MQ3-Lloyd inputs are FWHT-rotated before timing; excluding that required
rotation makes its reported time an optimistic GEMM-core upper bound. The
independently generated synthetic formats are not compared for quality; this
is an execution-format admission screen only.

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
pairs: 7

shape                         retained MQ4    HFQ3-G256    speedup
gate/up M=2x17408 K=5120       10.1220 ms      25.5000 ms   0.3969x
down add M=5120 K=17408         5.2344 ms      17.8467 ms   0.2933x

shape                         retained MQ4    MQ3-Lloyd    speedup
gate/up M=2x17408 K=5120       10.0392 ms      28.8407 ms   0.3481x
down add M=5120 K=17408         5.0735 ms      16.9011 ms   0.3002x
```

For gate, up, and down combined, resident weights are `142,049,280` bytes for
MQ4, `108,625,920` bytes for HFQ3 (-23.5%), and `116,981,760` bytes for
MQ3-Lloyd (-17.6%). Code-object inspection found Wave32 throughout. HFQ3 gate
and residual use 102 VGPRs, no private segment, and no LDS; MQ3-Lloyd uses 105
VGPRs, no private segment, and 256 bytes of LDS. The respective gate/residual
objects contain 1433/1013 and 1466/1050 static instructions, with eight static
WMMA instructions each. The candidates therefore fail despite avoiding the
retained kernel's spill and large-LDS footprint. Current 3-bit decode/operand
feeding is not a viable MQ4-v2 shortcut, so neither path proceeds to checkpoint
requantization or serving integration. This rejects these implementations, not
all possible 3-bit or codebook execution formats.

The first accepted probe must cover both production FFN shapes:

```text
gate/up: M=17408 K=5120 N=2048 set
down:    M=5120  K=17408 N=2048 add
```

Required output:

```text
reference_ms
candidate_ms
speedup
max_abs
relative_l2
cosine
resident_bytes_per_weight
dynamic_lds_bytes
vgpr_count
vgpr_spill_count
```

Candidates below 1.30x on either large FFN shape stop at standalone.

Passing the local speed threshold is necessary but not sufficient for routing.
Exact candidates must also match the retained output tolerance. Approximate
candidates must pass long-prompt generation and a task-level quality suite.
Every promoted candidate then needs a PP8192 ABBA test with identical routing
apart from the backend under test, plus an explicit execution-format memory
accounting. The retained backend remains the fallback and A/B reference.
