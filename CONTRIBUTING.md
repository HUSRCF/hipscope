# Contributing to hipfire

hipfire is an alpha project. Testers and contributors are essential — we need benchmark
numbers across RDNA2/3/4 cards, bug reports, kernel optimizations, and new model support.

## Focus areas

- Benchmarks on new GPUs (6800 XT, 7900 XTX, 9070 — we need your numbers)
- Model testing (correctness, coherence, edge cases)
- Kernel optimization (RDNA3/4 have different VGPR budgets)
- Bug reports with full reproduction steps
- New model architecture support (beyond Qwen3/3.5)

---

## For Testers (non-developers)

You don't need to know Rust. If you have a supported AMD GPU, you can help.

### 1. Install hipfire

Follow the quickstart in [README.md](README.md):

```bash
# Linux
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex
```

### 2. Run benchmarks

```bash
# Automated — runs all models and KV modes
bash scripts/megabench-q35.sh 2>&1 | tee my-bench-results.txt

# Quick single-model check
timeout 60 target/release/examples/infer models/qwen3.5-4b.q4.hfq --no-think \
  "Explain the three laws of thermodynamics" 2>&1 | grep "Done"
```

### 3. Submit results

Open a GitHub issue titled **"Benchmarks: [your GPU name]"** and paste:

```
GPU: [card name] ([gfx ID], [VRAM]GB)
OS: [Linux distro + kernel / Windows version]
ROCm version: [dpkg -l | grep rocm-core | awk '{print $3}']
Model: [model name and quant, e.g. Qwen3.5-9B HFQ4]
tok/s: [number from === Done line]
Coherence: [OK / LOOP / REPET / SHORT]
KV mode: [Q8 / turbo4 / turbo2]
Context: [short (~10 tok input) / long (~400 tok input)]
Notes: [anything unusual]
```

Results get added to [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Coherence codes:**
- `OK` — output is coherent and on-topic
- `LOOP` — model repeats the same phrase in a loop
- `REPET` — output degrades into repetition after N tokens
- `SHORT` — output stops abnormally short (< 20 tokens)

### 4. Report bugs

Open a GitHub issue with:
- GPU model and OS
- The exact command you ran
- Full error output (not just the last line)
- Output of `.skills/hipfire-diag/run-diagnostics.sh` if available

---

## For Developers

### Setup

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/hipfire
cd hipfire

# Build (requires ROCm SDK for kernel compilation; pre-compiled kernels ship for 4 arches)
cargo build --release --features deltanet -p engine
```

### Running tests

```bash
# Unit tests (no GPU required)
cargo test

# GPU kernel tests
scripts/test-kernels.sh
```

### Branch naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/xyz` | `feature/rope-interleave` |
| Bug fix | `fix/xyz` | `fix/oom-on-27b` |
| Benchmark | `bench/gpu-name` | `bench/7900-xtx` |

### PR process

1. Describe what changed and why
2. Include benchmark numbers if your change affects performance (before/after tok/s)
3. Keep PRs focused — one logical change per PR
4. Run `cargo fmt` and `cargo clippy` before submitting

### Pre-push regression checklist

Before pushing any change that touches kernels, dispatch, sampling, or the forward pass:

```bash
# 1. Build clean
cargo build --release --features deltanet --example daemon --example infer -p engine

# 2. Kernel tests (no model needed, ~30s)
target/release/examples/test_kernels

# 3. Speed regression — compare against documented baselines
target/release/examples/infer models/qwen3.5-9b.q4.hfq --max-tokens 64 "Hello"
# Expected: ~45 tok/s on gfx1010. Flag if >10% slower.

# 4. Correctness — model should answer this correctly
target/release/examples/infer models/qwen3.5-9b.q4.hfq --max-tokens 32 \
  "What is the capital of France? One sentence."
# Expected: output contains "Paris"

# 5. Qwen3 path (if touching sampling or daemon)
# Tests the non-DeltaNet code path + repeat_window clamping
echo '{"type":"load","model":"models/qwen3-8b.q4.hfq"}' | timeout 15 target/release/examples/daemon
```

**Speed baselines (gfx1010 / RX 5700 XT):**

| Model | Binary | Expected tok/s | Alarm if below |
|-------|--------|----------------|----------------|
| Qwen3.5-0.8B HFQ4 | infer | 220 | 180 |
| Qwen3.5-9B HFQ4 | infer | 45 | 38 |
| Qwen3.5-9B HFQ4 | daemon (turbo4) | 40 | 33 |
| Qwen3-8B HFQ4 | daemon | 38 | 30 |

**Multi-arch notes:**
- We can only test gfx1010 locally. Changes to kernels, dispatch, or compilation
  paths MUST be validated by a tester on gfx1100+ before release.
- Kernel hash files must be regenerated after any `.hip` source change:
  `cargo run --release -p rdna-compute --example gen_kernel_hashes`
- Pre-compiled blobs without matching hashes are rejected at runtime (the engine
  falls back to hipcc recompilation).

### Code style

- `cargo fmt` — required, enforced in CI
- `cargo clippy` — no warnings
- **No Python in the hot path.** Python is allowed for tooling and benchmarks, never
  in the inference engine itself.
- Comment HIP kernel parameters: explain VGPR counts, wave occupancy, LDS usage.

---

## Quantizing New Models

To add support for a model that isn't in the supported list:

### 1. Check prerequisites

- Model must be on HuggingFace in safetensors format
- Model architecture must be supported (currently: Qwen3, Qwen3.5, Qwen3.5-VL)

### 2. Build the quantizer

```bash
cargo build --release -p hipfire-quantize
```

### 3. Run the quantizer

```bash
# HFQ4 — recommended starting point
target/release/hipfire-quantize \
  --input Owner/ModelName \
  --output models/modelname.q4.hfq \
  --format hfq4

# HFQ6 — better quality
target/release/hipfire-quantize \
  --input Owner/ModelName \
  --output models/modelname.hfq6.hfq \
  --format hfq6
```

### 4. Test it

```bash
target/release/examples/infer models/modelname.q4.hfq "Hello, who are you?"
```

### 5. Submit a PR

If it works, open a PR that adds the model to the supported list in README.md.
Include your benchmark numbers.

---

## Benchmark Submission Template

For the benchmark database ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)), include:

```
GPU: [card name] ([gfx ID], [VRAM]GB)
OS: [Linux distro + kernel / Windows version]
ROCm version: [e.g. 6.3.1]
hipfire version: [git rev-parse --short HEAD]
Model: [model name and quant]
tok/s: [number]
Coherence: [OK / LOOP / REPET / SHORT]
KV mode: [Q8 / turbo4 / turbo2]
Context: [short / long (~400 tok input)]
Notes: [anything unusual]
```

Run at least two prompts per configuration for stable numbers:
1. Short: `"Hello, who are you?"`
2. Long: a multi-paragraph prompt or use `scripts/megabench-q35.sh`

---

## Architecture Overview

```
crates/hip-bridge/       → FFI to HIP runtime (dlopen, no link dep)
crates/rdna-compute/     → GPU dispatch, kernel management
crates/engine/           → Inference orchestrator, model loading
crates/hipfire-quantize/ → Model quantizer (safetensors → .hfq)
cli/                     → Bun TypeScript CLI (hipfire serve/run/list)
kernels/src/             → HIP kernel sources (.hip)
kernels/compiled/        → Pre-compiled .hsaco blobs per GPU arch
```

The key architectural constraint: hip-bridge uses `dlopen` to load the HIP runtime at
startup, so no ROCm SDK link dependency is needed at build time. Pre-compiled `.hsaco`
kernel blobs ship in the repo for each supported arch, so end users don't need `hipcc`.

---

## What We Need Help With

### High priority

- **Benchmarks on 6800 XT (gfx1030)** — we have no external gfx1030 numbers yet
- **Benchmarks on 7900 XTX (gfx1100)** — same
- **Benchmarks on 9070 (gfx1200)** — brand new arch, very interested in these
- **Qwen3.5-27B benchmarks** — needs 16GB+ VRAM; submit numbers from any card that fits

### Medium priority

- **Windows testing** — the PowerShell installer is new and needs real-world testing
- **Kernel optimization for RDNA3/4** — gfx1100/gfx1200 have larger VGPR budgets and
  different wavefront scheduling; the current kernels are tuned for gfx1010

### Lower priority (but welcome)

- **New model architectures** — Llama 3, Mistral, Phi-4, etc.
- **OpenAI API compatibility testing** — does `hipfire serve` work with your tool chain?
- **Long-context testing** — turbo4/turbo2 KV compression at 4K+ context lengths
