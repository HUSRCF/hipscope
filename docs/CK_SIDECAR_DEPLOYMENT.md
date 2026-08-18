# CK sidecar preview deployment

This guide applies to the temporary `publish/qwen36-ck-prefill-preview`
branch. It packages the optional gfx11 CK prefill route together with the
Qwen3.6 packed-MQ4 production paths without adding a CK binary to the
repository or to hipfire's default build.

The validated route is deliberately narrow:

- Radeon Pro W7900 / `gfx1100` with ROCm 7.14;
- Qwen3.6-27B MQ4;
- causal dense-prefix prefill with Asym3 K and Q8 V;
- 24 query heads, 4 KV heads, and head dimension 256;
- no graph capture, tree mask, or multi-slot attention;
- prefill only; decode and DFlash verification retain their native paths.

Unsupported calls fail closed to hipfire's native attention backend. The
sidecar is loaded only when the binary contains the `flash-attn-ck` Cargo
feature and `HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB` names a compatible library.

## Check out the preview

For a new clone:

```bash
git clone https://github.com/HUSRCF/hipscope.git
cd hipscope
git switch --track origin/publish/qwen36-ck-prefill-preview
```

For an existing checkout, preserve local work before switching:

```bash
git fetch origin
git switch --track origin/publish/qwen36-ck-prefill-preview
```

To migrate another development branch instead of switching to the preview,
merge the published snapshot explicitly and resolve any local kernel or
FeatureFlag changes:

```bash
git switch my-development-branch
git merge --no-ff origin/publish/qwen36-ck-prefill-preview
```

## Prerequisites

The build requires a working ROCm/HIP toolchain, a C++20 compiler, Python 3,
Git, Rust, and the pinned FA4 source tree. PyTorch is not required.

```bash
git clone https://github.com/HUSRCF/flash-attn-rocm-rdna3.git ../flash-attn-rocm-rdna3
git -C ../flash-attn-rocm-rdna3 checkout be194c0792e79ae26f71bf507e51b4d9136cf22c
```

Confirm the target and toolchain before compiling:

```bash
rocminfo | rg 'Name:.*gfx1100'
/opt/rocm/bin/hipconfig --version
git -C ../flash-attn-rocm-rdna3 rev-parse HEAD
```

The final command must print
`be194c0792e79ae26f71bf507e51b4d9136cf22c`.

## Build the sidecars

First build the selected dense D256 FA4/CK instance. The script archives the
pinned source revision into its build directory, so uncommitted changes in
the FlashAttention checkout are not consumed.

```bash
FLASH_ATTN_ROOT="$PWD/../flash-attn-rocm-rdna3" \
ROCM_PATH=/opt/rocm \
GPU_ARCH=gfx1100 \
HEAD_DIMS=256 \
CK_USE_FA4_GFX11_D256_RECIPE=1 \
MAX_JOBS=16 \
  ./experiments/flash-attn-ck-sidecar/build_sidecar.sh
```

Then link the staged Asym3-K/Q8-V adapter against that dense sidecar:

```bash
ROCM_PATH=/opt/rocm \
GPU_ARCH=gfx1100 \
STAGED=1 \
DENSE_SIDECAR="$PWD/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so" \
OUT="$PWD/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so" \
  ./experiments/flash-attn-ck-sidecar/quantized/build_quantized_sidecar.sh
```

The quantized build runs its ABI smoke automatically. Keep both `.so` files
in their build directories: the staged library records an rpath to the dense
library used during the link. Verify the resulting dependency before launch:

```bash
ldd experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so \
  | rg 'hipfire_flash_attn_ck|amdhip64'
```

## Build hipfire

Build the daemon with the optional loader enabled:

```bash
cargo build --release --locked \
  -p hipfire-runtime \
  --example daemon \
  --features deltanet,flash-attn-ck
```

The default build without `flash-attn-ck` does not load or link either
sidecar.

## Enable the validated production configuration

The sidecar path enables CK attention. The remaining variables select the
packed-MQ4 production configuration used for the W7900 PP8192 validation.

```bash
export HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="$PWD/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so"
export HIPFIRE_KV_MODE=asym3
export HIPFIRE_PREFILL_MAX_BATCH=2048
export HIPFIRE_FLASH_PARTIALS_BATCH=32
export HIPFIRE_QKVZA_SPLIT_TAIL=1
export HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64=1
export HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64=1
export HIPFIRE_RDNA3_HFQ4_AUX_X256Y64=1
export HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE=1
export HIPFIRE_RDNA3_Q8_GROUP128=1
export HIPFIRE_RDNA3_Q8_GROUP128_ROW2=1
export HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT=1
export HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128=1
export HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1
export HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW=1
export HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP=0

HIP_VISIBLE_DEVICES=0 ./target/release/examples/daemon
```

On the first admitted prefill call, stderr must include both messages:

```text
loaded optional quantized FlashAttention CK sidecar: ...
staged quantized FlashAttention CK prefill active: ...
```

The first message alone proves only that the ABI loaded. The second proves
that the runtime gate admitted a real prefill call. Absence of the second
message means the call stayed on the native backend.

## Reproduce the PP8192 check

Build the benchmark with the same feature and run the retained alternating
A/B script:

```bash
cargo build --release --locked \
  -p hipfire-runtime \
  --example bench_qwen35_mq4 \
  --features deltanet,flash-attn-ck

GPU_ID=0 \
PAIRS=5 \
TRIM_EACH_SIDE=1 \
COOL_SECS=10 \
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="$PWD/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so" \
OUT_DIR=/tmp/hipfire-q36-ck-pp8192 \
  ./experiments/gfx11-gate-up-x256y64/run_pp8192_group256_staged_ck_ab.sh
```

The W7900 integration run on 2026-08-18 measured `1201.4 tok/s` for the
group128 route and `1205.1 tok/s` for group256 after trimming one sample from
each side. Output token IDs matched. These are PP8192 prefill results on the
validated model and GPU, not general gfx11 or end-to-end serving guarantees.

## Disable or remove

To disable CK without rebuilding, unset the sidecar path and restart the
process:

```bash
unset HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB
```

To return fully to the default build, rebuild the daemon without the optional
feature. Generated sidecars and CK sources are ignored build artifacts and
can be removed independently of the repository history.
