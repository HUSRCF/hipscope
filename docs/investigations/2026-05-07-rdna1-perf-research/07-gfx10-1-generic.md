# Exp #7: gfx10-1-generic compile target

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment)

## Lever

LLVM groups gfx1010 (Navi 10), gfx1011 (Navi 12), gfx1012 (Navi 14) into the `gfx10-1-generic` target family. A single binary compiled with `--offload-arch=gfx10-1-generic` should run on any RDNA1 silicon. If our kernels compile cleanly to this target and produce bit-identical output, the per-arch JIT cache directory layout (currently `.hipfire_kernels/gfx1010/`, `.hipfire_kernels/gfx1011/`, etc.) could collapse to a single `.hipfire_kernels/gfx10-1-generic/` directory.

## Why

This is a compile-and-load test, not a perf-tuning experiment. If it works:

1. The BC-160 (gfx1011) plan no longer needs a separate per-arch JIT step — the existing gfx10-1-generic kernels would just work.
2. Build cache simplification — one directory covers all RDNA1 family.
3. Forward compatibility — any future Navi-1.x silicon revision that lands in the gfx10-1 family inherits cached kernels for free.

## Implementation

Single env override `HIPFIRE_TARGET_ARCH` added to `crates/rdna-compute/src/dispatch.rs::init_with_device`:

```rust
// BEFORE
let arch = hip.get_arch(id).unwrap_or_else(|_| "gfx1010".to_string());

// AFTER
let detected_arch = hip.get_arch(id).unwrap_or_else(|_| "gfx1010".to_string());
let arch = std::env::var("HIPFIRE_TARGET_ARCH")
    .ok()
    .filter(|s| !s.is_empty())
    .unwrap_or(detected_arch);
```

The override flows through `KernelCompiler::new(&arch)` → `--offload-arch={arch}` to hipcc. Cache key derives from `arch` so a different value triggers fresh JIT.

## Scenario

- Hardware: hipx, single RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1).
- Test invocation: `HIPFIRE_TARGET_ARCH=gfx10-1-generic` daemon load + 9B inference.
- Baseline: same daemon WITHOUT env (defaults to detected `gfx1010`).
- Prompt: `"Why is the sky blue? Answer in two sentences."` (deterministic, temperature=0.0).
- Comparison metric 1: byte-identical output token sequence.
- Comparison metric 2: decode tok/s within 1% of baseline.

## Win criterion (pre-registered)

ALL of:
1. hipcc accepts `--offload-arch=gfx10-1-generic` and compiles all hipfire kernels without error on first JIT.
2. Daemon loads and emits the canonical "loaded" event.
3. First-token output matches the gfx1010-specific build byte-for-byte (deterministic decode).
4. Decode tok/s within ±1% of gfx1010-specific baseline.

## Loss criteria (pre-registered)

ANY of:
- Compile failure on any kernel.
- Daemon load failure.
- Output divergence beyond first 16 tokens.
- Decode tok/s outside ±1% (regression OR improvement — both indicate the binaries differ in a way that affects perf).

## Action on win

Document. Propose a small follow-up PR: make `gfx10-1-generic` the default for any gfx101x device. Cache key collapse to one family-level directory. Update `CLAUDE.md` user_hardware notes accordingly.

## Action on loss

Revert. Document failure mode. Master unchanged.
