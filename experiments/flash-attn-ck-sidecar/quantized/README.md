# Quantized CK D256 experiment

This directory implements and tests a fixed gfx11 attention contract behind an
explicit serving opt-in:

- causal GQA with 24 query heads and 4 KV heads;
- head dimension 256;
- FP32 hipfire query/output with internal FP16 CK staging;
- hipfire Asym3 packed K and Q8 packed V;
- the existing CK `BlockFmhaPipelineQRKSVS` and default epilogue.

The selected D256 regular-attention policy follows the fast local FA4/CK
implementation under
`flash_attn_ck/flash-attention-fa4-v4.0.0.beta4_20260319c18_release2`:
gfx11 uses an M64/N32/r4x1 regular tile for this route. Hipfire adds only the
packed Asym3/Q8 logical views, Q rotation/output bridges, stable C ABI, and
fail-closed runtime gate; it does not copy the PyTorch extension or its broad
dispatcher.

The CK query is in the same Givens-rotated coordinate system as the stored
Asym3 K. The native A/B path includes an FP32-to-FP16 Q rotation kernel with
the same block-local pair contract as hipfire's Asym3 writer. The pairs are
register-local; no lane shuffle is required for this layout.

`PackedKvBufferView` exposes the packed caches as logical FP16 CK tensor views.
The CK tile loader invokes the view's `get<X>` method, which decodes only the
requested K or V elements. There is no dense-KV materialization pass.

## Tests

Prepare the pinned CK source with the parent sidecar build, then run on gfx1100:

```bash
HIP_VISIBLE_DEVICES=1 GPU_ARCH=gfx1100 \
  ./experiments/flash-attn-ck-sidecar/quantized/run_quantized_tile_loader_smoke.sh

HIP_VISIBLE_DEVICES=1 GPU_ARCH=gfx1100 \
  ./experiments/flash-attn-ck-sidecar/quantized/run_quantized_ck_pipeline_smoke.sh

HIP_VISIBLE_DEVICES=1 GPU_ARCH=gfx1100 TRIALS=5 \
  ./experiments/flash-attn-ck-sidecar/quantized/run_quantized_ck_native_ab.sh

HIP_VISIBLE_DEVICES=0 GPU_ARCH=gfx1100 TRIALS=3 \
  ./experiments/flash-attn-ck-sidecar/quantized/run_quantized_ck_prefill_ab.sh

GPU_ARCH=gfx1100 \
  ./experiments/flash-attn-ck-sidecar/quantized/build_quantized_sidecar.sh

GPU_ARCH=gfx1100 STAGED=1 \
  OUT=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
  ./experiments/flash-attn-ck-sidecar/quantized/build_quantized_sidecar.sh

HIP_VISIBLE_DEVICES=0 GPU_ARCH=gfx1100 \
  LIB=/tmp/libhipfire_flash_attn_ck_quantized_asym4_loader.so \
  ./experiments/flash-attn-ck-sidecar/quantized/run_asym4_loader_smoke.sh

./experiments/flash-attn-ck-sidecar/quantized/audit_quantized_ck_resources.sh
```

The staged build installs a copy of `libhipfire_flash_attn_ck.so` beside
`OUT` and records `RUNPATH=$ORIGIN`. Keep that pair together when moving or
packaging it; no build-tree path is required at runtime. The top-level
`scripts/package-gfx11-ck-bundle.sh` validates this layout before producing a
versioned archive.

The loader smoke covers hand-packed fixtures and hipfire's real Asym3/Q8 KV
writer kernels, including non-monotonic absolute positions. The pipeline smoke
checks bottom-right causal masking, GQA head mapping, non-tile-aligned lengths,
and a non-default HIP stream against an FP32 CPU reference.

The sidecar now has a controlled hipfire runtime replacement. It is compiled
only with the `flash-attn-ck` Cargo feature and activates only when
`HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB` names a compatible library. Unsupported
shapes, layouts, architectures, and builds retain the native path.

The optional C ABI is allocation-free and stream-ordered. The caller owns one
workspace containing rotated FP16 Q followed by FP16 CK output. The sidecar
launches Q rotation, the CK attention kernel, and a vectorized half2-to-FP32
bridge without a host synchronization. Its current gate intentionally accepts
only bottom-right causal D256 24Q/4KV prefill with Q>=128, contiguous Asym3 K,
and contiguous Q8 V. The built gfx1100 legacy/staged libraries are about
120/124 KiB, excluding the separately built dense CK sidecar.

An additive staged ABI provides the selected production route for the same
contract. It decodes every physical Asym3-K/Q8-V KV head once into dense FP16
scratch, then invokes the mature dense CK pipeline. This removes repeated
quantized decode from each query-head/tile load while retaining GQA reuse. Old
ABI-v1 sidecars remain loadable: the Rust loader resolves the staged symbols
optionally and falls back to the packed-view route when they are absent,
unsupported, or do not fit the caller-owned scratch buffer. `STAGED=1` is an
explicit build variant and links the dense CK sidecar; the default build remains
standalone and does not acquire that dependency. The staged route is still
excluded from graph capture and performs no allocation or host sync.

An independent Asym4 loader ABI decodes each physical `(position, kv_head)`
exactly once into dense FP16 K/V staging. Givens-Asym4 and FWHT4 use the same
loader because their packed K caches already contain transformed coordinates;
the corresponding Q transform remains a separate attention-front-end concern.
The Givens-Asym4 production route activates automatically when a compatible
sidecar is explicitly loaded and the validated D256 24Q/4KV prefill contract
matches. It requires a 64-row flash-partials allocation at PP2048 and above;
the reusable PP8192 A/B harness selects that capacity for Asym4.

On W7900, three alternating PP8192 process pairs measured scalar Asym4 versus
the staged CK route at `632.9 -> 1243.1`, `629.7 -> 1240.1`, and
`629.6 -> 1237.2 tok/s`. The paired-median speedup was **1.9651x** with 3/3
positive pairs and identical greedy token sequences. Decode remained neutral
at about `33.3-33.5 tok/s`. Raw evidence is under
`results/asym4_ck_pp8192_abba_w7900_20260827/`.

At Q=2048 with 24 query heads, 4 KV heads, and D256, the complete reusable
workspace (rotated Q, FP16 output, dense K, and dense V) is about 59/67/75/84
MiB for K=2K/4K/6K/8K. A same-process, alternating seven-trial benchmark of
the exported production C API measured:

| K rows | Packed CK | Staged CK | Attention-local speedup | Max abs |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | `3.447 ms` | `1.417 ms` | `2.433x` | `6.10e-5` |
| 4,096 | `9.474 ms` | `3.324 ms` | `2.850x` | `1.91e-6` |
| 6,144 | `14.698 ms` | `5.489 ms` | `2.678x` | `9.54e-7` |
| 8,192 | `20.725 ms` | `7.867 ms` | `2.634x` | `4.77e-7` |

The corresponding Qwen3.6-27B PP8192 production A/B measured a **+6.05%**
median paired throughput gain across three alternating process pairs:
`1127.3 -> 1188.1`, `1106.9 -> 1180.3`, and `1106.5 -> 1173.5 tok/s`.
This is a model-level backend improvement, not a route to 1.5k tok/s by itself;
packed-MQ4 projections remain the dominant wall-time target. Raw logs and the
summary are under `results/staged_model_ab_warm_20260811/`.

The sidecar was subsequently rebased onto official `beta@80a572c8` and tested
with the branch's native defaults. This CK-only beta branch does not include
the separate packed-MQ4 production stack whose retained PP8192 result is about
`1.19-1.21k tok/s`. Three alternating W7900 process pairs,
each using three PP8192 runs, measured `593.7 -> 865.8 tok/s`: a **1.4583x**
paired-median prefill speedup with 3/3 positive pairs. All pairs produced the
same eight greedy token IDs and decode remained neutral at approximately
`35.2 tok/s`. Compact evidence and the exact reproduction command are under
`results/beta_w7900_pp8192_ab_20260823/`.

After migrating the complete packed-MQ4 production stack into the same beta
branch, a new three-pair PP8192 run measured `748.4 -> 1221.5 tok/s`, a
**1.6321x** paired-median improvement with 3/3 positive pairs and exact greedy
token-ID agreement. This is the retained full-stack beta result; compact
evidence is under `results/beta_w7900_pp8192_fullopt_ab_20260823/`.

A follow-up replaced only the staged dense D256 `M64/N64` CK recipe with
`M64/N32`. Fifteen alternating process pairs measured a stable 1.0561x
aggregate attention-local gain, with per-K results from 1.0320x to 1.0748x.
The resource trace reports the same 32 KiB LDS, 400-byte scratch field, and 256
VGPR for both recipes, so the candidate does not open a new occupancy tier. At
the measured 11.27% PP8192 attention wall share, this is only about 0.6%
modeled end-to-end value, below the 1.10x local admission threshold. The
production N64 recipe is retained;
the rejected patch and raw data are archived under
`results/staged_dense_bn32_gpu1_20260811/`.

The full mature FA4 gfx11 D256 source path was then tested separately from the
local N32-only patch. It combines the D256 M64/N32 long-query dispatch with
FA4's Wave32 register-P redistribution; generated sources keep the default O
epilogue. Ten alternating component pairs on an idle W7900 measured a 1.1268x
aggregate attention-local paired speedup, with all ten pairs positive and
elementwise `max_abs=0` against the candidate's packed reference. A five-pair
strict-semantics Qwen3.6-27B PP16384 validation measured `1130.0` versus
`1120.6 tok/s`, a 1.00795x paired median with 5/5 positive pairs and exact
greedy token IDs. This is a small production prefill improvement consistent
with attention's limited wall-time share, not a general 12.7% model speedup.
The source is pinned to FA4 commit
`be194c0792e79ae26f71bf507e51b4d9136cf22c`; compact evidence is under
`results/staged_fa4_gfx11_d256_gpu1_20260812/`.

The loader smoke also compares scalar element decoding with a 32-dimension
batch decoder. The latter shares one Asym3 cnorm, four packed words, and one Q8
scale per work item. It is a deliberate ablation, not the selected path: on
the W7900 short loader cases it is 1.95x-3.19x slower than the scalar grid.
The selected CK view instead recognizes its naturally aligned eight-element
vector loads. It shares one Asym3 cnorm and packed word, or one Q8 scale, across
the eight returned FP16 elements without changing CK's K/V LDS staging.

An additional build-time experiment replaces the inlined eight-way Asym3
centroid switch with an indexed device constant codebook. The switch expands
the same FP32 literals at every unrolled decode site. On the generated M64/N32
kernel, the codebook reduces FMHA disassembly from 13,560 to 7,548 lines and
private storage from 400 to 396 bytes per work-item, while preserving output.
A five-pair directional run measured `4.5180 ms` versus `2.6655 ms` at
Q=128/K=8192 (`1.695x`), but both GPUs had resident daemons. Therefore
`ASYM3_CODEBOOK=1` remains an experimental build option. The raw data and
contamination note are under
`results/asym3_constant_codebook_directional_20260808/`.

An LDS-backed follow-up keeps the compact indexed lookup while moving the
eight FP32 centroids into 32 bytes of per-workgroup LDS. Relative to device
constant storage, it reduces FMHA global loads from 419 to 301, wait
instructions from 491 to 468, and VGPR spills from 102 to 101; private storage
remains 396 bytes. A three-pair run under the same fully occupied GPU condition
measured `3.9048 ms` for constant storage and `3.5862 ms` for LDS at
Q=128/K=8192 (`1.089x`), with identical `2.1435e-5` maximum error. A subsequent
clean-card Qwen3.6-27B PP8192 run did not carry that micro-level result into the
model: five alternating fresh-process pairs measured `526.8 tok/s` for the
switch and `526.3 tok/s` for LDS (`-0.09%`). The switch therefore remains the
sidecar default. The model-level data are under
`results/asym3_codebook_model_ab_20260808_200321/`.

A follow-up unaligned packed-load prototype was rejected before timing. Using
`memcpy`-lowered 16/64-bit loads for the three-byte Asym3 group and eight Q8
values increased FMHA disassembly from 7,548 to about 7,880 lines, private
storage from 396 to 404 bytes, and VGPR spills from 102 to 103. Both GPUs were
fully occupied, so no contaminated timing was retained and no build option was
left behind.

The FA4 gfx11 D256 `M64/N64` short-Q recipe was also checked as a single
source-backed alternative to the current `M64/N32` quantized tile. It is a
poor fit once K/V decode is fused into the CK view: private storage rises from
396 to 460 bytes, VGPR spills from 102 to 117, and SGPR use from 48 to 101.
This was rejected statically without opening a tile sweep or retaining another
runtime/build knob.

A guarded wave-scale broadcast was rejected for the same reason. Runtime
offset validation plus ballot/branch/shuffle increased private storage from
396 to 408 bytes and VGPR spills from 101 to 104 versus the LDS-codebook
candidate. Scale sharing therefore needs to live in a layout-specialized CK
staging policy, not inside the generic buffer-view `get()` path.

A direct rotated-Q view was also rejected. It removed the standalone FP32-Q
Givens-to-FP16 kernel and reduced FMHA private storage from 396 to 368 bytes
and VGPR spills from 101 to 91, but moved the rotation arithmetic onto the
critical Q staging path. Three alternating Q=128/K=8192 pairs measured a
`2.0173 ms` median for the separate bridge and `2.0478 ms` for direct Q
(`-1.51%`). Both GPUs were occupied, so this is directional rejection evidence,
not a clean performance result. The raw timings are retained under
`results/direct_rotated_q_directional_20260808/`; no build knob remains.

## Native A/B boundary

The decode-shaped comparison uses the same packed K/V, causal positions, GQA
mapping, and Givens coefficients for the CK candidate and hipfire's native
Asym3 tile+reduce kernels. Q rotation is included in CK total time. Initial
W7900 measurements show that the generic CK D256 recipe is not competitive:

| Q rows | K | CK total | Native | CK / native |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | `1.272 ms` | `0.0739 ms` | `17.23x slower` |
| 1 | 8,192 | `5.143 ms` | `0.1447 ms` | `35.53x slower` |
| 1 | 16,384 | `10.275 ms` | `0.2494 ms` | `41.19x slower` |
| 16 | 2,048 | `1.291 ms` | `0.2390 ms` | `5.40x slower` |
| 16 | 8,192 | `4.959 ms` | `0.9959 ms` | `4.98x slower` |
| 16 | 16,384 | `9.923 ms` | `1.9728 ms` | `5.03x slower` |

These are five-process medians from the original scalar-view experiment; each
process uses five warmups and twenty GPU-event repeats. The standalone Q
rotation costs only about 4-6 microseconds. Maximum CK/native output error
across the matrix is `3.01e-5`. Regular CK remains unsuitable for decode, where
padding a Q=1 or Q=16 input to a matrix tile wastes most of the work.

## Prefill crossover

The eight-element packed-view decode and the gfx11 D256 tiles from the local
FA4/CK implementation change the result for prefill. The production-window
measurement includes FP32 Q rotation, CK attention, and the FP16-to-FP32 output
bridge. The reproducible run is:

```text
quantized/results/fa4_vector8_f16_bridge_prefill_ab_20260808/results.tsv
```

W7900 / gfx1100, ROCm 7.14, three fresh-process trials per point:

| Tile | Q | K | CK production median | Native median | Native / CK |
| --- | ---: | ---: | ---: | ---: | ---: |
| M64/N32 | 128 | 2,048 | `1.037 ms` | `1.639 ms` | `1.58x` |
| M64/N32 | 128 | 8,192 | `3.664 ms` | `7.393 ms` | `2.02x` |
| M64/N32 | 512 | 2,048 | `1.502 ms` | `6.061 ms` | `4.04x` |
| M64/N32 | 512 | 8,192 | `5.645 ms` | `31.507 ms` | `5.58x` |

The earlier tile matrix shows M64/N32 is 3.7%-9.8% faster than M64/N64. Its
resource metadata is 32 KiB LDS, 47 SGPR, 256 VGPR, 102 VGPR spills, and 400
bytes of fixed private storage per work-item. The output bridge uses 5 VGPR and
has no spill. The small CPU-reference cases remain within `3.08e-4`; the
prefill native A/B remains within `4.13e-5`.

Writing FP32 directly from the CK epilogue was rejected. Its medians were
`1.082/3.959 ms` at Q=128 and `2.735/10.070 ms` at Q=512 for K=2K/8K,
respectively. Retaining CK's FP16 epilogue and converting with the separate
half2 bridge is materially faster, especially at Q=512.

This establishes a narrow prefill candidate. The runtime gate retains
hipfire's native kernel for decode and admits CK only for contiguous
Asym3-K/Q8-V causal GQA prefill with at least 128 query rows.

## Controlled runtime A/B

The optional runtime replacement was tested on a Radeon Pro W7900 / gfx1100
with ROCm 7.14 and Qwen3.6-27B MQ4 using Asym3 KV.

| Workload | Native | CK sidecar | Result |
| --- | ---: | ---: | ---: |
| synthetic PP8192, 3-run median | `539.1 tok/s` | `691.5 tok/s` | `1.283x` |
| real 3369-token text, greedy AR (single controlled run) | `328.78 tok/s` | `631.15 tok/s` | `1.920x` |
| real-text decode | `35.53 tok/s` | `35.48 tok/s` | `0.999x` |

The real-text A/B emitted the same 32 greedy token IDs. The PP8192 result uses
three prefill runs after model initialization; the real-text result is a
single controlled correctness run. These are prefill/backend results, not a
claim of full serving throughput across arbitrary workloads.

The runtime gate excludes graph capture. Hipfire inflates `max_ctx_len` to the
physical capacity while capturing its native replay-safe kernels, whereas this
sidecar ABI currently carries only one scalar live K length and no per-row
positions. Non-capture prefill therefore remains the supported contract.

## Rejected direct projection bridge

An optional ABI experiment fused CK's FP16 output conversion, Qwen's sigmoid
gate, MQ FWHT, and Q8_1 quantization for the following output projection. The
bridge is byte-exact and 2.45x-4.91x faster than the four-pass component path,
with wave32, 42 VGPR, 18 SGPR, and no spill. However, five alternating
Qwen3.6-27B PP8192 process pairs measured `682.9 tok/s` for the normal CK route
and `681.1 tok/s` for the direct bridge (`-0.26%`). The bridge can save only
about 31 ms across the model's 16 full-attention layers, below 0.3% of total
prefill time. Its automatic production route was rejected; the ABI, standalone
smoke, scripts, and raw results remain as boundary evidence.

## W7900 result

Radeon Pro W7900 / gfx1100, ROCm 7.14:

| Case | GPU median | max abs | mean abs |
| --- | ---: | ---: | ---: |
| Q=64, K=96, 24Q/4KV | `0.1605 ms` | `3.263e-4` | `2.840e-5` |
| Q=73, K=137, 24Q/4KV | `0.1344 ms` | `1.695e-4` | `2.248e-5` |

Each process measurement uses five warmups and twenty GPU-event repeats; the
table is the median of five fresh processes. These small cases validate the
pipeline and expose occupancy sensitivity, but are not a native-backend A/B.

Resource metadata for the final A/B binary is 32 KiB LDS, 101 SGPR, 256 VGPR,
117 VGPR spills, and 460 bytes of fixed private storage per work-item.
The matching generated dense CK D256 instance uses the same 32 KiB LDS and 256
VGPR but reports 241 VGPR spills and 884 bytes of private storage. The packed
loader therefore did not introduce the dominant gfx11 D256 spill condition,
although the absolute resource use remains too high for a production claim.
