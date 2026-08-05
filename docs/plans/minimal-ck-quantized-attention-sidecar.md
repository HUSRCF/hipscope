# Minimal CK quantized-attention sidecar plan

Status: future work; not part of the default hipfire build or dispatch.

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

The existing sidecar supports dense FP16 Q/K/V and output for head dimensions
64, 128, and 256. It does not understand hipfire's Q8_0, asym3, FWHT, paged, or
VMM KV layouts. In particular, it has no Q8_0 scale decoding and cannot be
routed to the current asym3 serving configuration.

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
Q8 path passes correctness and performance gates. Promotion into `optional/`
must not happen based on dense-FP16 component benchmarks alone.

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

The initial Q8 loader consumes hipfire Q8_0 blocks directly:

```text
[FP16 scale][32 signed INT8 values]
```

The preferred first implementation fuses decode into the global-to-LDS stage:

```text
Q8 HBM load -> scale and INT8 decode -> FP16 LDS tile -> CK FP16 FMHA
```

It must not materialize a full dense FP16 KV cache. Softmax and accumulation
remain floating point, and the established CK FP16 matrix pipeline is reused.
An INT8 QK matrix path is a separate experiment and is not required for the
first production candidate.

## Architecture scope

### gfx11 first target

- wave32 RDNA3 dGPU policy;
- Q8_0 K and V;
- head dimensions 128 and 256;
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

asym3 is not a trivial Q8 extension. It uses rotated packed 3-bit K and Q8 V,
so it requires a distinct K loader and rotation-aware query contract. FWHT and
paged/VMM layouts are also deferred. These formats should reuse the stable C
ABI and architecture policies only after the Q8 implementation establishes a
measured benefit.

## Runtime integration

Build example:

```bash
./build_sidecar.sh \
  --arch gfx1100 \
  --kv q8 \
  --head-dims 128,256
```

Runtime remains explicit and fail-closed:

```bash
HIPFIRE_CK_ATTENTION=1 \
HIPFIRE_CK_ATTENTION_LIB=/path/to/libhipfire_ck_attention_gfx1100.so \
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

### Phase 1: gfx11 Q8 correctness

- add Q8 global-to-LDS loader;
- support D128/D256 causal prefill and GQA;
- compare against the native Q8 attention output across short and long cases;
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

- evaluate a dedicated asym3 K loader with Q8 V;
- only proceed when it beats the existing native asym3 implementation after
  conversion, dispatch, and workspace costs are included.

## Acceptance criteria

- no full-KV dequantization buffer;
- no change to default builds or dispatch when the feature is disabled;
- CPU/reference and native-backend correctness tests pass for every admitted
  shape;
- zero unexpected scratch and documented LDS/VGPR usage;
- repeated median improvement of at least 5% after all bridge costs;
- gfx11 and gfx12 results are reported independently;
- all vendored CK files have revision, hash, origin path, and license records.

## Explicit non-claims

- Dense-FP16 CK speedups do not prove a Q8 or asym3 serving speedup.
- RDNA3 results do not establish RDNA4 performance.
- This sidecar is not a replacement for the complete CK project.
- Quantized storage does not imply native quantized softmax; decode and
  accumulation remain floating point in the first design.
