# hipfire v0.1.3-alpha — Plan

## Theme: Multi-arch stability + community tools

v0.1.2 shipped tester fixes and CLI dev tools. v0.1.3 focuses on making hipfire
reliably work across all supported GPUs and giving the community tools to
self-diagnose, benchmark, and contribute.

---

## P0 — Must ship

### 1. `hipfire bench` CLI command
**Source:** AMD Discord user request
**Scope:** Run standardized benchmarks for models that fit the user's VRAM.
Compare against llama.cpp if installed.

```
hipfire bench                        # Auto-detect GPU, run all fitting models
hipfire bench --model qwen3.5:9b     # Bench specific model
hipfire bench --compare-llamacpp     # Side-by-side with llama.cpp
```

Output: structured table with tok/s, prefill speed, VRAM usage.
Testers paste this into GitHub issues for community benchmark collection.

### 2. `hipfire update --stable` vs `hipfire update`
**Source:** Kaden (this session)
**Scope:** Two update channels:
- `hipfire update` — current behavior, pulls master HEAD, builds from source (nightly)
- `hipfire update --stable` — downloads latest GitHub Release binary (no Rust toolchain needed)

This is critical for Windows users who can't easily build from source.

### 3. gfx1151 (Strix Halo) support
**Source:** AMD Discord user
**Scope:**
- Add gfx1151 to arch detection in install.sh, install.ps1, cli/index.ts, dispatch.rs
- Investigate reported memory leak on Qwen3.5
- Compile + ship pre-compiled kernels for gfx1151
- May need a gfx1151-specific kernel variant if VGPR budget differs

### 4. 27B HFQ4 quality investigation
**Source:** DomKo (GitHub #2)
**Scope:** HFQ4 produces gibberish on coding tasks at 27B scale. HFQ6 is fine.
- Compare dequantized weights against reference (FP16) at key layers
- Check if group_size=256 is too coarse for 27B
- Consider HFQ4-G128 (smaller groups, more scales) as alternative
- If unfixable: document clearly and default `qwen3.5:27b` to HFQ6

---

## P1 — Should ship

### 5. Daemon model preloading + warm cache
Currently each `hipfire run` spawns a fresh daemon, loads model, runs, exits.
For repeated use, keep the daemon alive and the model warm:
- `hipfire serve` already does this for HTTP
- Add `hipfire daemon start` / `hipfire daemon stop` for CLI reuse
- Or: `hipfire run` detects running daemon and reuses it

### 6. Runtime arch-variant kernel selection
Currently the runtime compiler always uses generic kernel source. Arch-specific
variants (gfx1100 Q8, gfx1200 Q8) are only used via pre-compiled blobs.
- Embed arch-specific source via `include_str!()`
- Select at runtime based on `gpu.arch`
- Eliminates the hash validation gap for arch-variant blobs

### 7. Improved serve mode
- Fix 27B daemon crash on load through serve (likely OOM or timeout)
- Add `/health` endpoint with GPU/model/VRAM info
- Add `--model` flag to preload on startup
- Document SSE streaming format for frontend integration

---

## P2 — Nice to have

### 8. `hipfire validate <model.hfq>`
Check HFQ file integrity: header, tensor count, expected sizes, checksums.
Catch corrupted downloads before they cause mysterious inference failures.

### 9. GPU temperature monitoring in diag
Read `/sys/class/drm/card*/device/hwmon/` for temperature + clock speed.
Detect thermal throttling during benchmarks.

### 10. Research: minimal HIP runtime in Rust
Explore replacing libamdhip64.so dependency with direct amdgpu KMD ioctls.
~15 HIP functions → ~10 amdgpu ioctls. Would eliminate ROCm install requirement.
- Start with a PoC: hipMalloc + hipMemcpy + hipModuleLaunchKernel via ioctl
- If PoC works on gfx1010, test portability to gfx1100+
- Document findings in docs/RESEARCH-HIP-PORT.md

---

## Release criteria

- [ ] `hipfire bench` produces correct benchmarks on gfx1010
- [ ] `hipfire update --stable` downloads and installs release binary
- [ ] gfx1151 detected and kernels compile (if hardware available for testing)
- [ ] 27B HFQ4 either fixed or clearly documented as not recommended
- [ ] All P0 items tested on gfx1010, P0 items 1-2 tested on Windows
- [ ] Regression checklist passes (CONTRIBUTING.md protocol)
- [ ] DomKo, effectiveass, Vamned, AMD Discord user pinged with release
