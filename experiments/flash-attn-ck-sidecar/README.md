# Optional FlashAttention CK sidecar

This experiment gives hipfire a stable raw-pointer boundary to the official
FlashAttention ROCm Composable Kernel backend. It deliberately does not link
FlashAttention into hipfire's default build.

The official extension is a PyTorch/pybind module and is hundreds of megabytes.
Its public Python ABI is not usable from hipfire's Rust/raw-HIP runtime. This
experiment instead compiles a selected CK instance set into a small library and
exports a versioned C ABI. The library remains an optional runtime artifact.
ABI v4 enumerates exact-architecture layout capabilities, distinguishes dense
element strides from packed row-byte strides, and exposes a caller-owned
workspace query. The first quantized cell stages through persistent caller
scratch without allocating inside a stream-ordered launch.

Current scope:

- dense FP16 forward attention;
- causal or non-causal masks;
- MHA, MQA, and GQA;
- dense FP16 head dimension 64;
- gfx1100 F32-Q/Q8-K/Q8-V causal GQA at head dimension 256;
- gfx1100 F32-Q/Asym3-Givens-K/Q8-V causal GQA at head dimension 256;
- raw HIP stream and element-stride inputs.

The Q8 cell vector-decodes both packed caches into F16, invokes the CK D256
pipeline, and converts output back to F32. It is a correctness-first staged
adapter; direct quantized CK and asym/FWHT/Lloyd layouts remain future cells.

ABI v4 gives Givens and FWHT K caches distinct format IDs; Q8 V remains a
separate format, and callers provide explicit K/V row and head byte strides
plus both transform tables. The D256 Givens cell rotates Q and decodes packed K
and Q8 V into caller-owned F16 staging before invoking CK. D512 Givens and both
FWHT shapes remain `recognized-no-cell`: their layouts are validated but no
capability is published. This keeps each unimplemented packed loader fail-closed.
Packed staging currently requires contiguous `[row, head, dim]` Q and output;
the ABI validator rejects non-contiguous element strides rather than ignoring them.
The Rust loader accepts ABI v3 sidecars for their original dense/Q8 cells by
passing the unchanged v3 struct prefix; v3 quantized format IDs 4+ are rejected
because their old generic meaning is not the explicit v4 Asym3 contract.

## Build

The current upstream CK generator still rejects `gfx1100`, despite gfx1100 being
accepted by top-level build metadata. To keep this experiment reproducible,
`build_sidecar.sh` archives CK revision
`13f6d635653bd5ffbfcac8577f1ef09590c23d78` into the build directory and applies
the bundled `gfx11_ck_recipe.patch` before generating any kernels. Changes in
the caller's CK worktree are not consumed. The minimal recipe changes four CK
files: it adds the gfx11 tile and GEMM epilogue contract, remaps the
softmax-P and output-accumulator distributions through LDS, and uses an
explicit tile redistribution where the gfx11 register layouts differ. It does
not include the research tree's vLLM, split-K, or debug changes. The
patch SHA256 is:

```text
b43ea8d12e14cef04518225acaa69b63e62991ba4a83efcd596fc108105ac765
```

`FLASH_ATTN_ROOT` must point to a FlashAttention checkout whose
`csrc/composable_kernel` Git repository contains that revision. The build also
fails unless the patched generator emits the exact D64/D128/D256 instance set.

```bash
FLASH_ATTN_ROOT=/path/to/gfx11-enabled-flash-attention \
ROCM_PATH=/opt/rocm \
GPU_ARCH=gfx1100 \
./experiments/flash-attn-ck-sidecar/build_sidecar.sh
```

The build uses the selected CK sources directly and has no PyTorch dependency.
No `.so` is copied into or committed to hipfire.

The validated local build used ROCm 7.14:

```text
HIP version: 7.14.60850-0000000
RUNPATH: /opt/rocm/core-7.14/lib
NEEDED: libamdhip64.so.7
```

The resulting selected-instance sidecar is under 1 MiB. The full PyTorch extension is
not required and remains outside hipfire.

## Smoke

```bash
HIP_VISIBLE_DEVICES=0 \
  experiments/flash-attn-ck-sidecar/build/smoke_raw_abi
```

The pure-HIP smoke runs dense FP16 MHA/MQA/GQA, packed Q8 D256 GQA,
and Asym3-Givens D256 GQA cases against CPU references. Quantized references
use reconstructed values, so they check packed loading and attention
independently of quantization error.

Validated on Radeon Pro W7900 / gfx1100 with ROCm 7.14:

| Case | max abs | mean abs |
| --- | ---: | ---: |
| FP16 GQA D64, non-causal, default stream | `4.172325e-05` | `5.800672e-06` |
| FP16 GQA D64, causal, non-default stream | `5.501509e-05` | `7.329229e-06` |
| FP16 MHA D64, non-causal, default stream | `4.062802e-05` | `6.646507e-06` |
| FP16 MQA D64, non-causal, default stream | `4.367530e-05` | `6.348691e-06` |
| F32/Q8/Q8 GQA D256, causal | `4.766881e-05` | `7.248458e-06` |
| F32/Asym3-Givens/Q8 GQA D256, causal | `6.110966e-05` | `1.009769e-05` |

## Optional Rust loader

`rdna-compute` exposes the versioned C ABI only when built with the
`flash-attn-ck` feature. Enabling the feature does not search for or load a
library and does not alter attention dispatch. A caller must pass an explicit,
trusted sidecar path to the unsafe
`rdna_compute::flash_attn_ck::FlashAttnCk::load` boundary. A successfully loaded
library is intentionally pinned for the process lifetime because its HIP
launches are asynchronous.

The loader can be checked without changing the default backend:

```bash
HIPFIRE_FLASH_ATTN_CK_TEST_LIB="$PWD/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so" \
  cargo test -p rdna-compute --features flash-attn-ck \
  flash_attn_ck::tests::explicit_test_sidecar_loads
```

Serving selection remains fail-closed and occurs only after native KV-tier
resolution. Loading a sidecar alone cannot route an unsupported layout to CK.

Build a serving binary with `flash-attn-ck`, then opt in with both an exact-arch
artifact and an explicitly sized startup allocation:

```bash
cargo build --release -p hipfire-daemon --features deltanet,flash-attn-ck
HIPFIRE_FLASH_ATTN_CK_LIB=/opt/hipfire/ck/gfx1100/libhipfire_flash_attn_ck.so \
HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES=536870912 \
  target/release/daemon
```

The workspace is allocated once when `Gpu` is created. Missing or insufficient
workspace produces a one-time route reason and retains native attention. Use
`scripts/bench_ck_q8_prefill_ab.sh` for a reproducible production-path A/B.
