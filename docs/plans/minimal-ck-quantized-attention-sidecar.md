# Minimal CK quantized-attention sidecar plan

Status: controlled opt-in runtime prototype; not part of the default hipfire build or dispatch.

## Repository ownership

The production candidate should vendor the audited minimal CK source subset in
the hipfire repository. Users must not need a FlashAttention checkout, a full
Composable Kernel installation, PyTorch, or a network fetch during the hipfire
build. Hipfire owns the quantized-KV loaders, fixed-shape instance definitions,
ABI, tests, and any narrowly scoped compatibility changes in this subset.

The vendored code remains derived from upstream CK rather than being presented
as original hipfire code. Every imported file must retain its upstream license
header and be recorded with its upstream commit, original path, and SHA256. An
extraction/update tool must regenerate the subset from the pinned upstream tree
so that future CK updates are reviewable as source diffs instead of opaque
binary replacements.

Do not vendor the complete CK or FlashAttention repositories. The checked-in
surface is limited to the transitive dependency closure needed by the selected
gfx11/gfx12 FMHA instances and hipfire's quantized global-to-LDS loaders.

## Objective

Turn the current optional dense-FP16 FlashAttention CK sidecar into a small,
reproducible hipfire component without vendoring the complete FlashAttention or
Composable Kernel repositories. The eventual backend should read hipfire's
quantized KV cache directly, beginning with Q8_0 on gfx11, while retaining a
separate gfx12 policy and validation target.

This is not a plan to fork all of CK. It is a plan to preserve only the
generated FMHA translation units, templates, and headers that participate in
the validated build, plus hipfire-owned KV loaders and a stable C ABI.

## Current boundary

The existing optional sidecar supports dense FP16 Q/K/V and output for head
dimensions 64, 128, and 256. A standalone prototype now exposes hipfire's
packed Asym3 K and Q8 V as logical FP16 CK tensor views and runs them through
the existing D256 causal GQA pipeline without materializing dense K/V. It has
passed small CPU-reference cases and real-writer loader tests on gfx1100.

The prototype is fixed to 24 query heads, 4 KV heads, head dimension 256, and
the contiguous Asym3/Q8 cache layout. It now exposes a versioned sidecar ABI
and a controlled serving route, but does not handle paged or VMM addressing.
Regular CK is not a decode replacement: the original scalar-view recipe is
17.23x-41.19x slower for Q=1 and 4.98x-5.40x slower for Q=16 across
K=2K/8K/16K.

The prefill result is different. An aligned eight-element packed-view decode,
combined with the local FA4/CK gfx11 D256 M64/N32 tile, beats hipfire's native
Asym3 tile+reduce path once Q reaches 128 rows. Three-process medians range
from 1.58x at Q128/K2K to 5.58x at Q512/K8K; Q64 is approximately the
crossover and remains on the native side of the conservative gate. The
M64/N32 tile is 3.7%-9.8% faster than M64/N64 across the measured prefill
matrix. A standalone FP32-to-FP16 Givens Q rotation costs 4-19 microseconds in
these cases, and CK/native outputs agree within `4.21e-5`.

The sidecar is intentionally optional:

- the default Cargo build does not compile or link CK;
- no generated shared object is committed;
- the Rust loader requires an explicit trusted path;
- absence or rejection of the sidecar leaves the native backend unchanged.

## Proposed source layout

```text
optional/ck-attention/
  README.md
  LICENSES/
    composable-kernel.txt
  manifest/
    ck-revision.txt
    source-sha256.txt
  include/ck_tile_min/
  src/common/
    flash_attention_api.cpp
    q8_block.hpp
    q8_global_to_lds.hpp
  src/gfx11/
    policy.hpp
    q8_fmha_instances.cpp
  src/gfx12/
    policy.hpp
    q8_fmha_instances.cpp
  tools/
    extract_ck_subset.sh
    verify_ck_subset.sh
  build_sidecar.sh
```

The exact path may remain under `experiments/flash-attn-ck-sidecar` until the
quantized path passes correctness and performance gates. Promotion into
`optional/` means checking the audited source subset into hipfire; it does not
mean adding an external CK runtime or build-time download. Dense-FP16 component
benchmarks alone are insufficient for promotion.

## Reproducible CK extraction

Do not select CK headers manually. Template dependencies are deep and manual
selection is difficult to review or update safely.

The extraction tool should:

1. pin a known CK commit;
2. generate only the selected gfx11/gfx12 FP16 FMHA instances;
3. compile those translation units with dependency-file emission;
4. copy the transitive source/header closure named by the depfiles;
5. preserve relative paths and the upstream license;
6. write a SHA256 manifest for every copied file;
7. rebuild from the extracted tree and compare the resulting kernel inventory.

CI should be able to regenerate the subset from the pinned CK checkout and
detect drift. The extracted subset must not contain unrelated CK examples,
tests, backends, or generated instances.

## Backend design

Keep algorithm, KV format, and architecture policy separate:

```cpp
template<class ArchPolicy, class KvLoader>
struct HipfireFmha;

struct DenseFp16KvLoader;
struct Q8KvLoader;

struct Gfx11Policy;
struct Gfx12Policy;
```

The loader layer consumes hipfire's packed KV blocks directly. The common Q8
loader consumes Q8_0 blocks in this form:

```text
[FP16 scale][32 signed INT8 values]
```

The preferred implementation fuses decode into the global-to-LDS stage:

```text
Q8 HBM load -> scale and INT8 decode -> FP16 LDS tile -> CK FP16 FMHA
```

It must not materialize a full dense FP16 KV cache. A tile-local FP16 register
or LDS representation is allowed and expected; it is not a persistent
dequantized cache. Softmax and accumulation remain floating point, and the
established CK FP16 matrix pipeline is reused. An INT8 QK matrix path is a
separate experiment and is not required for the first production candidate.

For the current Qwen3.6 serving configuration, the first production-shaped
instance should be gfx11, causal GQA, head dimension 256, with an Asym3 K
loader and Q8 V loader. A Q8 K/V loader remains useful as a simpler contract
test and for other cache modes, but it should not delay validation of the
actual Asym3/Q8 path.

## Architecture scope

### gfx11 first target

- wave32 RDNA3 dGPU policy;
- Asym3 K and Q8 V for the first serving-shaped instance;
- head dimension 256 first, with other dimensions requiring separate instances;
- causal dense prefill first;
- MHA/GQA mapping used by hipfire;
- explicit feature and runtime gate;
- no tree/speculative-verify route until separately validated.

### gfx12 reserved target

Keep a separate policy and generated-instance file from the beginning. Do not
assume gfx11 LDS, occupancy, or WMMA choices are optimal on gfx12. The existing
dense-FP16 sidecar smoke on gfx1201 is the compatibility floor; Q8 remains off
until R9700 correctness and benchmark evidence exists.

### Deferred formats

Q8 K/V remains a useful simpler format but is not the serving-shaped first
target. FWHT and paged/VMM layouts are deferred. These formats should reuse the
stable C ABI and architecture policies only after the Asym3/Q8 implementation
establishes a measured benefit.

## Current prototype boundary

The first production-style C ABI now exists under
`experiments/flash-attn-ck-sidecar/quantized/`. It is deliberately narrow:

- gfx11 D256, 24 query heads, 4 KV heads;
- bottom-right causal prefill with at least 128 query rows;
- contiguous Asym3 K and Q8 V rows;
- caller-owned workspace and caller-provided HIP stream;
- no allocation, event creation, or synchronization in the forward call;
- FP16 CK epilogue followed by a vectorized FP32 output bridge.
- non-capture execution only; graph capture retains the native backend until
  the sidecar ABI carries replay-safe per-row live-length metadata.

On W7900, the complete ABI-shaped GPU path is 1.58x-5.58x faster than the
current native tile+reduce path over Q=128/512 and K=2K/8K. This is operator
evidence. The production route now carries an explicit caller contract for a
dense KV prefix plus contiguous query suffix, allowing every bottom-right
causal prefill chunk to use the sidecar rather than only the first `Q == K`
chunk.

For Qwen3.6-27B MQ4 with Asym3 KV, PP8192, and 2048-token prefill chunks, three
alternating fresh-process trials measured native at 596.8/591.2/593.9 tok/s
and CK at 806.4/808.1/817.2 tok/s. The medians are 593.9 and 808.1 tok/s,
respectively: `1.3607x` or `+36.07%`. Decode remained neutral at 33.0-33.1
tok/s because the route is prefill-only. A separate 3369-token real-prompt
greedy-AR check used legal 256-token chunks and emitted identical 16-token ID
sequences on native and CK; that correctness path measured 580.62 versus
635.93 prefill tok/s. The earlier 539.1-to-691.5 and 328.78-to-631.15
single-run measurements remain historical diagnostics, not the primary claim.

## Runtime integration

Build the optional quantized sidecar:

```bash
GPU_ARCH=gfx1100 \
  ./experiments/flash-attn-ck-sidecar/quantized/build_quantized_sidecar.sh
```

Runtime remains explicit and fail-closed:

```bash
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=/path/to/libhipfire_flash_attn_ck_quantized.so \
  hipfire ...
```

The runtime gate must check architecture, ABI version, KV format, dtype, head
dimension, attention semantics, strides, and graph/stream safety. A rejected
case falls back to the native hipfire kernel without changing output buffers.

## Delivery phases

### Phase 0: minimal vendoring

- implement extraction and manifest verification;
- reproduce the current D64/D128/D256 dense-FP16 smoke from the subset;
- confirm the default hipfire binary and dependency graph are unchanged.

### Phase 1: gfx11 quantized-KV correctness

- add Asym3 K and Q8 V tile-local loaders;
- support the D256 causal GQA serving shape;
- compare against CPU and native quantized attention outputs across short and long cases;
- audit generated ISA, LDS, VGPR, scratch, and kernel inventory.

### Phase 2: production-style gfx11 route

- add preallocated Q/output conversion scratch if the caller remains FP32;
- add feature and length/workload gates;
- validate graph capture, non-default streams, repeated invocation, and
  fallback behavior;
- report synchronized end-to-end operator time, not only CK kernel time.

### Phase 3: gfx12 validation

- generate gfx12-specific instances and policy;
- run the same correctness matrix on R9700;
- establish an independent gate from measurements rather than copying gfx11.

### Phase 4: additional KV formats

- evaluate Q8 K/V and the deferred FWHT layouts after the Asym3 K/Q8 V prefill
  route is integrated and measured end to end;
- keep format-specific gates rather than assuming the Asym3 result transfers.

## Acceptance criteria

- no full-KV dequantization buffer;
- no change to default builds or dispatch when the feature is disabled;
- CPU/reference and native-backend correctness tests pass for every admitted
  shape;
- loader resource deltas relative to the matching dense CK instance are
  documented and bounded by a repeated end-to-end operator win;
- repeated median improvement of at least 5% after all bridge costs;
- gfx11 and gfx12 results are reported independently;
- all vendored CK files have revision, hash, origin path, and license records.

## Explicit non-claims

- Dense-FP16 CK speedups do not prove a Q8 or asym3 serving speedup.
- RDNA3 results do not establish RDNA4 performance.
- This sidecar is not a replacement for the complete CK project.
- Quantized storage does not imply native quantized softmax; decode and
  accumulation remain floating point in the first design.
