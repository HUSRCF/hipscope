# Redline vs HIP Dispatch Benchmarks

## Hardware & Environment

| | |
|---|---|
| GPU | AMD RX 5700 XT (gfx1010, RDNA1, 40 CUs, 8.6 GB GDDR6) |
| Kernel | Linux 6.17.0-14-generic |
| DRM | Version 3.64 |
| ROCm | 6.3.60304 |
| HIP | libamdhip64.so.6.3.60304 |
| Date | 2026-04-04 |

## What is Redline?

Redline bypasses the entire HIP/HSA/ROCm runtime stack. It talks directly to `libdrm_amdgpu.so` (55 KB), the thin userspace wrapper around the amdgpu kernel driver. It builds PM4 command buffers in Rust, submits them via `amdgpu_cs_submit`, and waits via `amdgpu_cs_query_fence_status`.

```
HIP path:     app → libamdhip64 → libhsa-runtime64 → libdrm_amdgpu → kernel
Redline path: app → libdrm_amdgpu → kernel
```

## Per-Dispatch Latency

Identical kernel (`vector_add`, 256 elements, `__launch_bounds__(256)`), identical buffers.
Each dispatch includes: kernarg upload, PM4 build (redline) or runtime call (HIP), submit, and fence wait.

| Backend | Median | Mean | P99 | Min | Max | Iterations |
|---------|--------|------|-----|-----|-----|------------|
| HIP | **18.0 µs** | **17.6 µs** | **21.0 µs** | **13.9 µs** | 51.7 µs | 1,000 |
| Redline | 68.0 µs | 69.5 µs | 93.3 µs | 64.7 µs | 1,008 µs | 10,000 |

**HIP is 3.8x faster per-dispatch.** This is expected: HIP uses user-mode queuing (doorbell writes to shared memory), while Redline goes through `amdgpu_cs_submit` (a kernel ioctl) on every dispatch. The ioctl round-trip adds ~50µs of syscall overhead.

### Why Redline is slower (and how to fix it)

Redline currently uses `amdgpu_cs_submit` for every dispatch. This is the "safe" path — the kernel validates the command buffer before submitting. The fast path would be **user-mode submission** (AQL queues with doorbell writes), which is what HIP/HSA uses internally. This is a future optimization.

## Sequential Multi-Dispatch (200 kernels)

| Method | Total Time | Per-Kernel |
|--------|-----------|------------|
| HIP sequential | **3.39 ms** | **16.9 µs** |
| Redline sequential | 13.62 ms | 68.1 µs |

Same ratio as single dispatch — each redline dispatch pays the ioctl overhead.

## Startup Time

Time from process start to first kernel dispatch completing.

| Backend | Device Init | Notes |
|---------|-------------|-------|
| Redline | **0.52 ms** | `open(/dev/dri/renderD128)` + `amdgpu_device_initialize` |
| HIP | 13.33 ms | `hipInit` + `hipSetDevice` (loads HSA, firmware, creates queues) |

**Redline is 25x faster to start.** HIP must initialize the full HSA runtime, load firmware, create internal queues, and set up the user-mode submission infrastructure.

*Note: Both exclude kernel compilation time (hipcc). Total wall time for redline including compile was 675ms.*

## Memory Overhead

RSS after init + warmup dispatches, before any large allocations.

| Backend | RSS |
|---------|-----|
| Redline | **2.7 MB** |
| HIP | 134.8 MB |

**Redline uses 50x less memory.** HIP loads the entire ROCm stack into the process: libamdhip64 (23 MB), libhsa-runtime64 (3.3 MB), libamd_comgr (139 MB — the compiler/code manager), plus internal heap allocations for queues, signal pools, and memory managers.

## Binary Size

| Binary | Unstripped | Stripped |
|--------|-----------|----------|
| Redline poc_vector_add | 582 KB | 471 KB |
| Redline bench_dispatch | 612 KB | 496 KB |
| HIP bench_hip | 26 KB | 23 KB |

The HIP binary is smaller because it's a thin C++ wrapper — all the work is in the shared libraries. Redline statically links its dispatch logic.

### Runtime Dependencies

| Backend | Libraries Required | Total Disk |
|---------|--------------------|-----------|
| Redline | `libdrm_amdgpu.so` (55 KB) | **~55 KB** |
| HIP | `libamdhip64.so` (23 MB) + `libhsa-runtime64.so` (3.3 MB) + `libamd_comgr.so` (139 MB) + 12 more | **~165 MB** |

**Redline needs 3000x less library footprint.** `libdrm_amdgpu.so` ships with every system that has the amdgpu kernel driver — no ROCm installation required.

## Summary

| Metric | Redline | HIP | Winner |
|--------|---------|-----|--------|
| Per-dispatch latency | 68 µs | 18 µs | HIP (3.8x) |
| Startup time | 0.52 ms | 13.33 ms | **Redline (25x)** |
| Memory footprint | 2.7 MB | 134.8 MB | **Redline (50x)** |
| Library deps | 55 KB | ~165 MB | **Redline (3000x)** |
| Requires ROCm? | No | Yes | **Redline** |
| Works on gfx1010 natively? | Yes | Unofficial | **Redline** |

### When to use Redline
- Embedded/edge deployment (minimal footprint)
- Systems without ROCm installed (just need amdgpu driver)
- Latency-sensitive startup (serverless, cold-start)
- Consumer RDNA GPUs that AMD refuses to support in ROCm

### When to use HIP
- Maximum dispatch throughput (user-mode queuing)
- Need ROCm ecosystem (rocBLAS, MIOpen, etc.)
- Officially supported hardware

### Future: closing the dispatch gap
Redline's per-dispatch overhead is dominated by the `amdgpu_cs_submit` ioctl (~50µs). Two paths to fix:
1. **IB chaining** — submit 100+ dispatches per ioctl (infrastructure built, barrier WIP)
2. **User-mode queues** — use AQL doorbell submission like HIP does internally (eliminates ioctl entirely)

## How to Reproduce

```bash
# Redline benchmark
cargo run -p redline --example bench_dispatch --release -- 10000

# HIP benchmark
hipcc -O3 --offload-arch=gfx1010 -o /tmp/bench_hip benchmarks/bench_hip_dispatch.hip
/tmp/bench_hip 1000

# Binary analysis
cargo build -p redline --examples --release
strip target/release/examples/poc_vector_add
ldd target/release/examples/poc_vector_add
```
