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

The build selects the CK recipe from `GPU_ARCH`. The current upstream CK
generator still rejects `gfx1100`, despite gfx1100 being accepted by top-level
build metadata, so gfx11 applies the bundled compatibility recipe. gfx12 uses
the upstream recipe unchanged. To keep both paths reproducible,
`build_sidecar.sh` archives CK revision
`13f6d635653bd5ffbfcac8577f1ef09590c23d78` into the build directory and applies
the bundled `gfx11_ck_recipe.patch` only for gfx11 before generating any
kernels. Changes in the caller's CK worktree are not consumed. The minimal
gfx11 recipe changes four CK files: it adds the D64 tile and GEMM epilogue
contract, remaps the softmax-P and output-accumulator distributions through
LDS, and uses an explicit tile redistribution where the gfx11 register layouts
differ. It does not include the research tree's vLLM, split-K, D128/D256, or
debug changes. The patch SHA256 is:

```text
b43ea8d12e14cef04518225acaa69b63e62991ba4a83efcd596fc108105ac765
```

`FLASH_ATTN_ROOT` must point to a FlashAttention checkout whose
`csrc/composable_kernel` Git repository contains that revision. The build also
fails unless the pinned generator emits the validated architecture-specific
set: eight FP16/D64 kernels plus one dispatcher for patched gfx11, or four
kernels plus one dispatcher for upstream gfx12.

```bash
FLASH_ATTN_ROOT=/path/to/flash-attention \
ROCM_PATH=/opt/rocm \
GPU_ARCH=gfx1100 \
./experiments/flash-attn-ck-sidecar/build_sidecar.sh
```

Use `GPU_ARCH=gfx1201` for the native gfx12 recipe. Unsupported architecture
families fail before generation.

The build uses the selected CK sources directly and has no PyTorch dependency.
No `.so` is copied into or committed to hipfire.

The validated local build used ROCm 7.14:

```text
HIP version: 7.14.60850-0000000
RUNPATH: /opt/rocm/core-7.14/lib
NEEDED: libamdhip64.so.7
```

The validated D64-only sidecars were about 324 KiB for gfx1100 and 168 KiB for
gfx1201. The full PyTorch extension is not required and remains outside
hipfire.

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

The same smoke passed on Radeon AI PRO R9700 / gfx1201 with ROCm 7.14.
Maximum absolute error across its four cases was `5.501509e-05`.

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
| 64 | 512 | 8/8 | 0.0329 | 0.0280 | 0.0389 | 0.0546 | 1.17x | 0.84x | 0.60x |
| 64 | 1024 | 8/8 | 0.0573 | 0.0485 | 0.0596 | 0.0816 | 1.18x | 0.96x | 0.70x |
| 64 | 2048 | 8/8 | 0.1140 | 0.0888 | 0.0974 | 0.1210 | 1.28x | 1.17x | 0.94x |
| 64 | 4096 | 8/8 | 0.2245 | 0.1927 | 0.1850 | 0.2100 | 1.17x | 1.21x | 1.07x |
| 96 | 1024 | 8/8 | 0.0768 | 0.0514 | 0.0597 | 0.0744 | 1.49x | 1.29x | 1.03x |
| 96 | 2048 | 8/8 | 0.1497 | 0.0974 | 0.1047 | 0.1277 | 1.54x | 1.43x | 1.17x |
| 128 | 512 | 8/8 | 0.0534 | 0.0268 | 0.0346 | 0.0459 | 1.99x | 1.54x | 1.16x |
| 128 | 1024 | 8/8 | 0.1006 | 0.0470 | 0.0558 | 0.0706 | 2.14x | 1.80x | 1.43x |
| 128 | 2048 | 8/8 | 0.2015 | 0.0970 | 0.1000 | 0.1215 | 2.08x | 2.01x | 1.66x |
| 192 | 1024 | 8/8 | 0.1519 | 0.0482 | 0.0808 | 0.0939 | 3.15x | 1.88x | 1.62x |
| 192 | 2048 | 8/8 | 0.3264 | 0.0998 | 0.1216 | 0.1439 | 3.27x | 2.68x | 2.27x |
| 256 | 512 | 8/8 | 0.1062 | 0.0280 | 0.0385 | 0.0495 | 3.80x | 2.76x | 2.15x |
| 256 | 1024 | 8/8 | 0.2043 | 0.0492 | 0.0607 | 0.0744 | 4.15x | 3.37x | 2.74x |
| 256 | 2048 | 8/8 | 0.4477 | 0.1052 | 0.1045 | 0.1241 | 4.25x | 4.28x | 3.61x |
| 512 | 4096 | 8/8 | 2.2773 | 0.2427 | 0.2199 | 0.2589 | 9.38x | 10.36x | 8.80x |
| 256 | 2048 | 8/2 | 0.3183 | 0.1021 | 0.1025 | 0.1125 | 3.12x | 3.11x | 2.83x |

Raw data:
[`results/w7900_gfx1100_rocm7.14_20260730.csv`](results/w7900_gfx1100_rocm7.14_20260730.csv).

The maximum output difference against the native FP32 kernel was
`1.49e-6`. GPU-event and synchronized wall-clock medians were close, showing
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

For this exact D64 dense layout, a conservative experimental gate is
`seqlen_q >= 128 && seqlen_k >= 512`. Every measured Q/O-bridge point in that
region was at least 1.54x faster. `seqlen_q >= 96 && seqlen_k >= 1024` is an
aggressive profile; Q=64 remains marginal even at long K and should stay on
the native path by default. This is a prefill/cross-attention gate on query
chunk length, not a decode-context gate: Q=1 decode is outside the measured
profitable region.

### R9700 sidecar result

Radeon AI PRO R9700 / gfx1201, ROCm 7.14, used the clean upstream gfx12 CK
recipe from the same pinned revision. The sidecar was 168 KiB, and the Rust
optional-loader integration test passed against it. The benchmark used the
same three warmups, nine trials, and twenty iterations per trial as the W7900
sweep. The table reports synchronized wall-clock speedups:

| Q | K | Hq/Hkv | native F32 ms | Q/O bridge ms | direct FP16 | Q/O bridge | full-F32 bridge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 8/8 | 0.0372 | 0.0173 | 3.11x | 2.14x | 0.82x |
| 64 | 1024 | 8/8 | 0.0579 | 0.0254 | 2.94x | 2.28x | 1.51x |
| 64 | 2048 | 8/8 | 0.1201 | 0.0408 | 3.41x | 2.95x | 2.14x |
| 64 | 4096 | 8/8 | 0.2686 | 0.0720 | 4.05x | 3.73x | 2.77x |
| 96 | 1024 | 8/8 | 0.0793 | 0.0248 | 4.14x | 3.20x | 2.14x |
| 96 | 2048 | 8/8 | 0.1622 | 0.0387 | 4.90x | 4.19x | 3.00x |
| 128 | 512 | 8/8 | 0.0597 | 0.0451 | 4.97x | 1.32x | 0.82x |
| 128 | 1024 | 8/8 | 0.1008 | 0.0529 | 5.07x | 1.91x | 1.57x |
| 128 | 2048 | 8/8 | 0.2032 | 0.0685 | 5.71x | 2.97x | 2.42x |
| 192 | 1024 | 8/8 | 0.1395 | 0.0261 | 7.05x | 5.35x | 3.72x |
| 192 | 2048 | 8/8 | 0.2812 | 0.0417 | 7.95x | 6.75x | 4.92x |
| 256 | 512 | 8/8 | 0.0991 | 0.0470 | 8.12x | 2.11x | 1.30x |
| 256 | 1024 | 8/8 | 0.1822 | 0.0548 | 9.01x | 3.32x | 2.75x |
| 256 | 2048 | 8/8 | 0.3575 | 0.0706 | 9.93x | 5.07x | 4.13x |
| 512 | 4096 | 8/8 | 1.8731 | 0.0950 | 24.27x | 19.71x | 15.48x |
| 256 | 2048 | 8/2 | 0.4396 | 0.0706 | 12.34x | 6.23x | 4.45x |

Raw data:
[`results/r9700_gfx1201_rocm7.14_20260730.csv`](results/r9700_gfx1201_rocm7.14_20260730.csv).
The benchmark maximum absolute difference was `1.49e-6`.

For the exact D64 dense Q/O-bridge contract, every measured R9700 point with
`seqlen_q >= 64 && seqlen_k >= 512` was faster, with a minimum measured
speedup of 1.32x. This measured boundary is not evidence for Q below 64, Q=1
decode, D128 WMMA attention, or quantized/paged KV. The large peak speedup is
primarily a comparison against hipfire's scalar D64 fallback.

## gfx1201 default-path regression check

The optional sidecar and its Cargo feature are default-off. To verify that the
change does not affect the existing gfx1201 path, commit `a2fcd552` was compared
with its base `b69b28c2` on an AMD Radeon AI PRO R9700 (gfx1201, ROCm 7.14).
Both isolated trees passed `cargo check -p rdna-compute`; the feature tree also
passed with `--features flash-attn-ck`. The existing
`test_dots_ocr_wmma_gfx12` correctness example passed with maximum absolute
errors `1.948e-5` for GEMM and `6.038e-6` for attention.

For performance, the already-built base and feature binaries alternated over
five rounds of `bench_attention_vision --iters 20`. The production gfx12
DotsOCR v5 path was unchanged:

| build | per-run ms/iter | median ms/iter |
| --- | --- | ---: |
| base `b69b28c2` | 112.7, 113.2, 113.2, 113.4, 113.5 | 113.2 |
| feature `a2fcd552` | 112.8, 113.1, 113.4, 113.5, 113.6 | 113.4 |

The median difference was `+0.18%`, within the observed run-to-run variation.
No measurable regression was observed in the default gfx1201 DotsOCR attention
path; this is not a FlashAttention CK performance result on gfx1201.
