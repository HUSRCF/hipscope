# Optional FlashAttention CK sidecar

This experiment gives hipfire a stable raw-pointer boundary to the official
FlashAttention ROCm Composable Kernel backend. It deliberately does not link
FlashAttention into hipfire's default build.

The official extension is a PyTorch/pybind module and is hundreds of megabytes.
Its public Python ABI is not usable from hipfire's Rust/raw-HIP runtime. This
experiment instead compiles a selected CK instance set into a small library and
exports a versioned C ABI. The library remains an optional runtime artifact.

Current scope:

- dense FP16 forward attention;
- causal or non-causal masks;
- MHA, MQA, and GQA;
- head dimension 64;
- raw HIP stream and element-stride inputs.

It does not support hipfire's Q8/asym/FWHT KV layouts. Those layouts must stay
on hipfire's native attention kernels. The first possible runtime consumer is
`AttnFullF16`, after adding reusable FP32-to-FP16 query/output scratch because
that hipfire boundary currently carries Q and output in FP32.

## Build

The current upstream CK generator still rejects `gfx1100`, despite gfx1100 being
accepted by top-level build metadata. To keep this experiment reproducible,
`build_sidecar.sh` archives CK revision
`13f6d635653bd5ffbfcac8577f1ef09590c23d78` into the build directory and applies
the bundled `gfx11_ck_recipe.patch` before generating any kernels. Changes in
the caller's CK worktree are not consumed. The minimal recipe changes four CK
files: it adds the gfx11 D64 tile and GEMM epilogue contract, remaps the
softmax-P and output-accumulator distributions through LDS, and uses an
explicit tile redistribution where the gfx11 register layouts differ. It does
not include the research tree's vLLM, split-K, D128/D256, or debug changes. The
patch SHA256 is:

```text
b43ea8d12e14cef04518225acaa69b63e62991ba4a83efcd596fc108105ac765
```

`FLASH_ATTN_ROOT` must point to a FlashAttention checkout whose
`csrc/composable_kernel` Git repository contains that revision. The build also
fails unless the patched generator emits exactly eight FP16/D64 kernel
instances plus one dispatcher.

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

The resulting D64-only sidecar is about 324 KiB. The full PyTorch extension is
not required and remains outside hipfire.

## Smoke

```bash
HIP_VISIBLE_DEVICES=0 \
  experiments/flash-attn-ck-sidecar/build/smoke_raw_abi
```

The pure-HIP smoke runs FP16 MHA, MQA, and GQA cases against a CPU reference.
The causal GQA case uses a non-default HIP stream. A runtime route is out of
scope until this passes under the same ROCm runtime used to build hipfire.

Validated on Radeon Pro W7900 / gfx1100 with ROCm 7.14:

| Case | max abs | mean abs |
| --- | ---: | ---: |
| FP16 GQA D64, non-causal, default stream | `4.172325e-05` | `5.800672e-06` |
| FP16 GQA D64, causal, non-default stream | `5.501509e-05` | `7.329229e-06` |
| FP16 MHA D64, non-causal, default stream | `4.062802e-05` | `6.646507e-06` |
| FP16 MQA D64, non-causal, default stream | `4.367530e-05` | `6.348691e-06` |

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

The current hipfire `AttnFullF16` contract still uses FP32 Q and output tensors,
so it is not routed to this all-FP16 sidecar. A production route requires
preallocated conversion scratch owned outside the attention hot path.

## Performance probe

`bench_vs_native.sh` compares four paths at the same D64 shape and attention
semantics:

- hipfire's current scalar `attention_dflash_f32` kernel;
- direct all-FP16 CK attention;
- an `AttnFullF16`-style bridge that casts FP32 Q to FP16 and output back to
  FP32 while consuming already-FP16 K/V;
- a conservative full-FP32 bridge that also casts K/V on every invocation.

GPU-event columns isolate submitted stream work. Synchronized wall-clock
columns also include the sidecar's host dispatch and kernel enqueue cost and
are the primary integration metric. The direct number compares all-FP16 CK
against the native FP32 scalar kernel; it is a mixed-precision reference, not a
strict upper bound because producer/conversion kernels can change cache state.
The Q/O bridge estimates the conversion tax at hipfire's `AttnFullF16`
boundary; the full-FP32 bridge measures the cost of adapting the scalar
fallback ABI. None is a production `AttnFullF16` replacement claim: that path
is D128, whereas this validated CK instance set is D64.

Check that the selected GPU is idle before running:

```bash
rocm-smi --showuse --showmemuse --showpids

GPU_ID=1 \
GPU_ARCH=gfx1100 \
ROCM_PATH=/opt/rocm \
WARMUP=3 \
TRIALS=9 \
ITERATIONS=20 \
  experiments/flash-attn-ck-sidecar/bench_vs_native.sh
```

The script writes a CSV to
`experiments/flash-attn-ck-sidecar/build/bench_vs_native.csv`.

### W7900 result

Radeon Pro W7900 / gfx1100, ROCm 7.14, three warmups, nine trials, twenty
iterations per trial. The table reports synchronized wall-clock medians:

| Q | K | Hq/Hkv | native F32 ms | CK F16 ms | Q/O bridge ms | full-F32 bridge ms | direct FP16 | Q/O bridge | full-F32 bridge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 8/8 | 0.0329 | 0.0276 | 0.0390 | 0.0545 | 1.19x | 0.85x | 0.60x |
| 64 | 2048 | 8/8 | 0.1152 | 0.0898 | 0.0983 | 0.1220 | 1.28x | 1.17x | 0.95x |
| 256 | 2048 | 8/8 | 0.4425 | 0.1007 | 0.1121 | 0.1345 | 4.39x | 3.95x | 3.29x |
| 512 | 4096 | 8/8 | 2.2907 | 0.2333 | 0.2190 | 0.2567 | 9.82x | 10.46x | 8.93x |
| 256 | 2048 | 8/2 | 0.3223 | 0.1028 | 0.1058 | 0.1161 | 3.13x | 3.05x | 2.78x |

The maximum output difference against the native FP32 kernel was
`1.32e-6`. GPU-event and synchronized wall-clock medians were close, showing
that host dispatch is not the dominant cost in these batched measurements.
Direct, Q/O-bridge, and full-FP32 paths use separate FP16 working sets. The Q/O
bridge appearing faster than direct CK in one large case reflects the
conversion kernel warming the FP16 query; it is not evidence that conversion
is free.

The useful boundary is the result, not a single peak number: conversion
overhead loses on the smallest Q tile, while Q=256--512 dense prefill shapes
show a clear net win over the current D64 scalar kernel. This benchmark does
not compare against hipfire's D128 WMMA path and therefore does not establish
a production-backend speedup for that route.
