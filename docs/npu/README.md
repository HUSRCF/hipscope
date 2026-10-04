# NPU (XDNA2 / AIE2P) research stack — experimental, dark

hipfire's own path to the Strix / Strix Halo NPU: an AIE2P program builder, a
pure-CPU AIE2P simulator, and a direct `amdxdna` runtime (DRM ioctls only, no
libxrt, no vendor compiler). Folded in from the standalone NPU workspace at
`c1bf57a`. **Nothing in the daemon, engine or loader calls any of it**; the
default hipfire build and runtime path are unchanged.

## Layout

| Piece | Where | What |
|---|---|---|
| `pm-npu` | `crates/pm-npu` | AIE2P emitter: VLIW encoder/scheduler (`isa`, tables generated from llvm-aie TableGen), CDO/PDI writer, DPU transaction writer, DMA/route/registers, kernel designs (`kernels`: int8 GEMM, whole-array V8–V10/G80, IU4/IEF15, ring, experts, bw probe), and the offline simulator (`sim`). Bins `pm-npu-objdump`, `pm-npu-sim`. |
| `railgun::npu` | `crates/railgun/src/npu` (feature `npu`, default off) | Runtime: device query, DEV_HEAP, hardware context, CONFIG_CU, BOs, eager/prepared/chained submit, persistent ring, `tape`, FCLK guard (`fclk`), dlopen'd HIP interop (`hip_runtime`). |
| `npu-tools` | `crates/npu-tools` | Every binary: `npu-tools` (`aie-status`, `txn-dump`, `soc-metrics`, `m4`, `tblgen2rs`, `gpu-bw`, `fclk`), `npu-gemm`, `npu-hello`, `npu-experts`, `npu-ring`, `npu-railgun`, `npu-bw`, `npu-coop`, and `coop27` (`--features lab`). |
| Scripts | `tools/npu/` | Silicon-window harnesses (`npu-window.sh` pins FCLK and takes the timing lock), batch/bisect drivers, the `a4-ec` and `fold-model` offline analyses (standalone cargo projects). |
| Design notes | `docs/npu/*.md` | `railgun-npu.md` (runtime design), `gemm_i8.md`, `fold-contract.md`, `ief15-n0.md`, `g80-feasibility.md`, `dense27b.md`, `coop-*.md`, `levers-*.md`, `a4-ec.md`. |
| Evidence | `docs/npu/report.md` | Silicon results and measurements; the raw window logs stay in the NPU workspace's `logs/` (not copied). `docs/npu/CHANGELOG.md` is the workspace's own history. |

## Relationship to `crates/hipfire-xdna`

`hipfire-xdna` is the older, minimal `amdxdna` ioctl control plane (one
session, one ERT packet, opt-in gfx1151 spillover experiments driven by
`HIPFIRE_XDNA_*`). It stays as is. `railgun::npu` is a separate, broader
runtime (prepared/chained submit, ring, tape, HIP dma-buf interop) built for the
PM-npu designs; the two share no code and neither is on the product path.
Converging them is future work, not part of this fold.

## Building and testing

```sh
cargo build -p railgun --features npu      # runtime (CI builds this)
cargo build -p npu-tools --features lab    # all binaries incl. coop27 (CI builds this)
cargo test  -p pm-npu --release            # emitter + simulator suites, pure CPU
cargo test  -p railgun --features npu --lib
```

All `pm-npu` tests are pure CPU (`golden_bytes`, `lean`, `ring`, `experts`,
`g80`, `iu4_ief15`, …); `array` and `g80` take minutes in debug. Vendor
byte-oracle cross-checks run only with `AIE2P_VENDOR_CORPUS` set (they skip
otherwise); `isa_decode::vendor_decode_oracle_roundtrip` is `#[ignore]`d
because it also needs an llvm-aie `llvm-objdump` (`AIE2P_OBJDUMP`). Running the
binaries needs XDNA2 silicon, and the co-op paths a gfx1151 iGPU with a pinned
fabric clock (`NPU_FCLK_GUARD`). Environment variables are listed in
[`../env-vars.md`](../env-vars.md#manual--npu-research-stack-dark-not-hipfire_).

## Licensing

Material derived from Xilinx/llvm-aie, Xilinx/mlir-aie, Xilinx/aie-rt and the
Linux amdxdna UAPI is listed, with the affected files, in the root `NOTICE`.
