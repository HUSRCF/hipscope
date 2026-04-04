# DeltaNet on AMD RDNA: Technical Documentation

Hipfire is currently the **only known implementation** of DeltaNet (Gated Delta Net /
Qwen3.5) inference running on AMD GPUs. This document describes exactly how it works,
what bugs were found and fixed along the way, and how to diagnose the known failure
modes on different RDNA architectures.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The GDN Recurrence Kernel](#the-gdn-recurrence-kernel)
3. [Kernel Variants and Dispatch](#kernel-variants-and-dispatch)
4. [Qwen3.5 Forward Pass](#qwen35-forward-pass)
5. [State Quantization](#state-quantization)
6. [Bug History: The Road to Coherent Output](#bug-history)
7. [Issue #2: gfx1100 Stale Kernel Bug](#issue-2-gfx1100-stale-kernel-bug)
8. [Porting to New RDNA Architectures](#porting-to-new-rdna-architectures)

---

## Architecture Overview

Qwen3.5 is a **hybrid architecture**: some layers use DeltaNet (linear attention with
a learned delta rule), others use standard grouped-query attention (GQA) with KV cache.
The layer type sequence is defined in the model config's `layer_types` array.

```
Per-layer dispatch (Qwen3.5-0.8B example: 24 layers, 18 DeltaNet + 6 FullAttention):
  L00: DeltaNet    L06: DeltaNet    L12: DeltaNet    L18: DeltaNet
  L01: DeltaNet    L07: DeltaNet    L13: DeltaNet    L19: DeltaNet
  L02: DeltaNet    L08: DeltaNet    L14: DeltaNet    L20: FullAttn
  L03: DeltaNet    L09: DeltaNet    L15: DeltaNet    L21: FullAttn
  L04: DeltaNet    L10: FullAttn    L16: DeltaNet    L22: FullAttn
  L05: DeltaNet    L11: FullAttn    L17: DeltaNet    L23: FullAttn
```

Key dimensions (Qwen3.5):
- **DeltaNet heads**: 16 key heads, 16 value heads
- **Head dimension**: 128 (both key and value)
- **S matrix**: 128x128 per head, 16 heads = 262,144 elements per DeltaNet layer
- **Conv1d kernel**: size 4 (causal, applied to QKV before DeltaNet recurrence)

The full-attention layers use standard GQA with 8 query heads, 2 KV heads, head_dim=256,
and partial RoPE (only the first 25% of dims get rotary embeddings).

### Code Locations

| Component | File | Purpose |
|-----------|------|---------|
| Config & forward pass | `crates/engine/src/qwen35.rs` | Model loading, per-layer dispatch |
| GDN kernels | `kernels/src/gated_delta_net*.hip` | HIP compute kernels |
| Kernel dispatch | `crates/rdna-compute/src/dispatch.rs` | Grid/block config, kernel launch |
| Kernel compilation | `crates/rdna-compute/src/compiler.rs` | Pre-compiled blob loading + hipcc fallback |
| Kernel source embedding | `crates/rdna-compute/src/kernels.rs` | `include_str!()` for runtime compilation |
| Pre-compilation script | `scripts/compile-kernels.sh` | Build `.hsaco` blobs for target archs |

---

## The GDN Recurrence Kernel

### Mathematical Definition

The Gated Delta Net recurrence processes one token at a time, updating a persistent
state matrix S per head:

```
Given: q, k, v ∈ R^d, α (decay gate, scalar), β (update gate, scalar)
       S ∈ R^(d×d) (persistent state matrix, d=128)

1. kv = S @ k                          # Current state's projection onto key
2. delta = (v - α * kv) * β            # Gated delta update (scalar)
3. S' = α * S + k ⊗ delta              # State update (rank-1, outer product semantics)
4. output = S' @ q                      # Project updated state onto query
```

Where `k ⊗ delta` means each row `i` of S gets `k[j] * delta` added (since delta is
derived from `v[row]`, this is actually computed per-row in the kernel).

### Critical Detail: S @ k, NOT S^T @ k

The `kv` computation must be a **row dot product** of S with k. For row `i`:

```c
// CORRECT: S @ k — read row i of S, dot with k
kv = sum_j( S[i][j] * k[j] )    // = S_row[j] * k[j]

// WRONG: S^T @ k — read column i of S, dot with k  
kv = sum_j( S[j][i] * k[j] )    // = state[j*HD + row] * k[j]
```

With row-major storage `S[i*HD + j]`, reading `S_row[j]` gives `S[i][j]` (row access),
while reading `state[j*HD + row]` gives `S[j][i]` (column access = transposed).

This distinction was the source of the primary correctness bug. See [Bug History](#bug-history).

### Gate Computation

The alpha gate is computed on-GPU via a fused kernel (`alpha_gate_f32`):

```
alpha_raw = linear(hidden_state)    # Per-head projection
alpha = -exp(A_log) * softplus(alpha_raw + dt_bias)
alpha_final = exp(alpha)            # Applied as multiplicative decay
```

The `A_log` tensor stores log-space decay rates (always negative after `-exp()`).
The `dt_bias` is a per-head learnable bias. The `softplus` ensures the gate is
always positive before the sign flip from `A_log`.

Beta is computed as `sigmoid(linear(hidden_state))`, giving a [0,1] update gate.

---

## Kernel Variants and Dispatch

### Variant Overview

| Kernel File | Threads | Grid | LDS | State Storage | Target |
|-------------|---------|------|-----|---------------|--------|
| `gated_delta_net.hip` | 32 | [n_heads, 32] | 4KB (S tile) | FP32 global | All archs |
| `gated_delta_net_q8.hip` | 32 | [n_heads, 32] | 4KB (S tile) | Int8 + per-row F32 scale | All archs (generic) |
| `gated_delta_net_q8.gfx1100.hip` | 128 | [n_heads, 1] | 256B (k+q) | Int8 → registers | RDNA3 (gfx1100) |
| `gated_delta_net_q8.gfx1200.hip` | 128 | [n_heads, 1] | 256B (k+q) | Int8 → registers | RDNA4 (gfx1200) |
| `gated_delta_net_q4.hip` | 128 | [n_heads, 1] | 256B (k+q) | 4-bit nibble-packed + per-row F32 scale | All archs |

### FP32 Variant (Tiled LDS)

**File:** `kernels/src/gated_delta_net.hip`

```
Grid:    [n_heads, HD/TILE_ROWS] = [16, 32]
Block:   [32, 1, 1]
LDS:     TILE_ROWS(4) × HD(128) × 4 bytes = 2KB per block
Strategy: Each block handles one head + 4 rows of S.
          32 threads cover 128 columns (4 per thread).
          Warp shuffle reduces kv and output dot products.
```

The kernel tiles the 128x128 S matrix into 4-row strips. Each threadblock loads one
tile into LDS, processes all tokens through that tile, then writes back. The tiling
keeps LDS usage at 2KB, allowing 8+ blocks per CU on RDNA1.

Launch bounds: `__launch_bounds__(32, 8)` — 32 threads/block, hint for 8 blocks/CU.

### Generic Q8 Variant (Tiled LDS with Dequant/Requant)

**File:** `kernels/src/gated_delta_net_q8.hip`

Same tiling strategy as FP32, but:
1. **Dequant at tile load**: `S_tile[i] = scale * (float)sq_base[i]`
2. **Compute in FP32**: Identical inner loop to FP32 variant
3. **Requant at tile writeback**: Per-row absmax → `inv_s = 127/max` → round to int8

The per-row scale is computed via warp-wide `__shfl_xor` max reduction, giving all 32
threads the same max value. Thread 0 writes the scale to global memory.

### RDNA3-Optimized Q8 (Register-Array)

**File:** `kernels/src/gated_delta_net_q8.gfx1100.hip`

```
Grid:    [n_heads, 1]
Block:   [128, 1, 1]
LDS:     k_lds[128] + q_lds[128] = 1KB
Strategy: One thread per S row. Entire row (128 floats) in registers.
          No S memory traffic during recurrence — pure ALU.
```

RDNA3 has 1536 VGPRs per SIMD unit. This kernel uses ~144 VGPRs per thread
(128 for `S_local[128]` + ~16 for compute), allowing 10 waves per SIMD.

Launch bounds: `__launch_bounds__(128, 10)` — 128 threads, 10 waves.

Key performance advantage: S state never touches memory during the inner loop.
Dequant happens once at kernel start, requant once at kernel end. The inner loop
is entirely register-to-register with `k` and `q` broadcast via LDS.

**This variant is only used by `compile-kernels.sh` for pre-compiled blobs.**
The runtime compiler always uses the generic Q8 source via `include_str!()`.
See [Issue #2](#issue-2-gfx1100-stale-kernel-bug) for why this matters.

### Q4 Variant (Nibble-Packed, Register-Array)

**File:** `kernels/src/gated_delta_net_q4.hip`

```
Grid:    [n_heads, 1]
Block:   [128, 1, 1]
LDS:     k_lds[128] + q_lds[128] = 1KB
Strategy: Same as RDNA3 Q8 — one thread per row, full row in registers.
          4-bit nibble unpacking: byte & 0xF (low) and byte >> 4 (high), minus 8.
          Symmetric quantization: -8..+7 range, scale = absmax/7.
```

VRAM savings vs Q8: 50% less state storage (8KB vs 16KB per head per layer).
Quality tradeoff: Faster quantization error accumulation across tokens.

### Dispatch Code

In `crates/rdna-compute/src/dispatch.rs`:

```rust
// FP32: 32-thread warp, tiled S
unsafe { self.hip.launch_kernel(func, [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, ...) }

// Q8 generic: same tiled layout
unsafe { self.hip.launch_kernel(func, [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, ...) }

// Q4: 128-thread register-array
unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [128, 1, 1], 0, ...) }
```

Note: `n_tiles = HD / TILE_ROWS = 128 / 4 = 32` for FP32/Q8 generic.

---

## Qwen3.5 Forward Pass

### DeltaNet Layer Forward (per token)

```
Input: x (hidden state), pos (sequence position)
State: S matrices (persistent), conv ring buffers (persistent), KV cache (full-attn only)

1. RMSNorm (Qwen3.5 variant: output = x * rsqrt(var+eps) * (1 + weight))
2. QKV projection: [q_raw, k_raw, v_raw] = Linear(normed_x)
   - q_raw, k_raw: [n_key_heads × key_head_dim] each
   - v_raw: [n_value_heads × value_head_dim]
3. Z (gate) projection: z = Linear(normed_x)  → [n_value_heads × value_head_dim]
4. Beta projection:  beta = sigmoid(Linear(normed_x))  → [n_value_heads]
5. Alpha projection: alpha = exp(-exp(A_log) * softplus(Linear(normed_x) + dt_bias))  → [n_value_heads]
6. Conv1d + SiLU (fused kernel): applies causal conv over QKV with ring buffer state
7. Split conv output → Q, K, V
8. L2 normalize Q and K (per-head)
9. Scale Q by 1/sqrt(key_head_dim)
10. Repeat Q/K heads if n_key_heads < n_value_heads (GQA-style)
11. GDN Recurrence: S' = α*S + k⊗delta; output = S'@q
12. Gated norm: rmsnorm(attn_out) * silu(z)  (using separate norm weight)
13. Output projection: o = Linear(gated_normed)
14. Residual: x = x + o
15. FFN: SwiGLU (gate*up → silu → down)
16. Residual: x = x + ffn_out
```

### Full Attention Layer Forward (per token)

```
1. RMSNorm (1 + weight variant)
2. Q projection (2x wide): [q_raw, gate_raw] = Linear(normed_x), interleaved per-head
3. K projection, V projection
4. Q/K norm (per-head RMSNorm)
5. Partial RoPE on first 25% of head dims
6. KV cache write at position
7. Attention (flash or standard depending on sequence length)
8. Sigmoid gate: output = attn_out * sigmoid(gate_raw)
9. Output projection → residual → FFN → residual
```

### Weight Loading Quirks

1. **Qwen3.5 RMSNorm stores offsets**: Weights represent `(1 + w)`, not `w` directly.
   `load_norm_weight()` adds 1.0 to all values before GPU upload. This applies to
   `input_layernorm`, `post_attention_layernorm`, `final_layernorm`, `q_norm`, `k_norm`.
   It does **NOT** apply to the DeltaNet gated norm (`Qwen3_5RMSNormGated`), which
   uses weights directly (initialized to ones in the model).

2. **Full-attention Q is 2x wide**: The Q projection is `[dim, 2 * n_heads * head_dim]`.
   The output is interleaved per-head: `[Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]`.
   Split after projection, not before.

3. **Conv1d weight layout**: PyTorch stores conv1d weights as `[out_channels, 1, kernel_size]`
   in row-major. The kernel expects reversed kernel ordering for causal convolution:
   `weight[c*4+3], weight[c*4+2], weight[c*4+1], weight[c*4+0]`.

4. **A_log conversion**: The raw `A_log` parameter stores `log(-A)`. The gate computation
   uses `-exp(A_log)` to recover the negative decay rate.

---

## State Quantization

### Overview

DeltaNet S matrices are persistent across tokens and dominate VRAM for long sequences.
Three quantization modes trade off accuracy vs. memory:

| Mode | VRAM per DeltaNet Layer | Per-Head | Quantization Error |
|------|------------------------|----------|-------------------|
| FP32 | 1 MB (16 heads × 128 × 128 × 4B) | 64 KB | None |
| Q8 | ~264 KB (+ 8KB scales) | ~16.5 KB | Low, compounds slowly |
| Q4 | ~136 KB (+ 8KB scales) | ~8.5 KB | Higher, compounds faster |

### Q8 Quantization Scheme

Per-row symmetric int8. For each row of 128 elements:
1. Find absmax across the row
2. `scale = absmax / 127`
3. `quantized[j] = round(value[j] / scale)`, clamped to [-128, 127]
4. Store scale as one FP32 per row → `s_scales[head * 128 + row]`

Dequant: `value = scale * (float)quantized[j]`

In the tiled kernel, the warp-wide max is computed via `__shfl_xor` butterfly reduction
(all 32 threads get the same result). Thread 0 writes the scale.

In the register-array kernel, the max is computed per-thread over the 128-element
register array (no cross-thread communication needed since each thread owns a full row).

### Q4 Quantization Scheme

Per-row symmetric 4-bit, nibble-packed. For each row of 128 elements:
1. Find absmax across the row
2. `scale = absmax / 7` (4-bit signed: -8..+7)
3. `quantized[j] = round(value[j] / scale) + 8`, clamped to [0, 15]
4. Pack two values per byte: `byte = low_nibble | (high_nibble << 4)`
5. Store: 64 bytes per row (128/2), plus one FP32 scale per row

Dequant: `value = scale * ((byte & 0xF) - 8)` for even indices,
         `value = scale * ((byte >> 4) - 8)` for odd indices.

### Quantization Error Accumulation

Because S is updated every token and requantized after each update, quantization noise
compounds over the sequence. This is an inherent property of recurrent state quantization.

Empirically:
- **Q8**: Coherent output for 2000+ tokens on Qwen3.5-0.8B, 500+ on larger models
- **Q4**: Coherent for ~200-500 tokens before degradation becomes noticeable
- **FP32**: No degradation (but uses 4x the VRAM)

Default is Q8, selectable at runtime via `--state-quant {fp32|q8|q4}`.

---

## Bug History

The DeltaNet implementation went through five critical bugs before producing coherent
output. Each is documented here because **the same bugs will appear on any new
DeltaNet implementation**. This is the reference for debugging.

### Bug 1: Conv1d Weight Layout (commit `44e8558`)

**Symptom:** NaN logits
**Root cause:** Conv1d weights loaded with column stride instead of row stride, and
kernel ordering was forward instead of reversed (for causal convolution).
**Fix:** Changed from `weight[c], weight[n_channels+c]...` to
`weight[c*4+3], weight[c*4+2], weight[c*4+1], weight[c*4+0]`.

### Bug 2: A_log Sign (commit `44e8558`)

**Symptom:** NaN or exploding values in gate computation
**Root cause:** Used raw `A_log` value instead of `-exp(A_log)`.
**Fix:** Apply `-exp()` to A_log during weight loading (matching llama.cpp GGUF conversion).

### Bug 3: Q/Gate Interleaving in Full Attention (commit `44e8558`)

**Symptom:** Attention output garbage in full-attention layers
**Root cause:** Q projection output assumed two contiguous halves `[Q_all | Gate_all]`
but is actually interleaved per-head: `[Q_h0, Gate_h0, Q_h1, Gate_h1, ...]`.
**Fix:** Split per-head with stride, not as two contiguous blocks.

### Bug 4: Missing Q Scaling (commit `0789980`)

**Symptom:** Hidden state norm 15x too small, model stuck generating `\n\n`
**Root cause:** Missing `1/sqrt(key_head_dim)` scaling on Q before GDN recurrence.
**Fix:** `gpu.scale_f32(&q_part, 1.0 / (config.linear_key_head_dim as f32).sqrt())`

### Bug 5: Qwen3.5 RMSNorm (1+weight) (commit `2fd1d9f`)

**Symptom:** All norm outputs 2-5x too small, cascading through every layer
**Root cause:** Qwen3.5's `Qwen3_5RMSNorm` stores weights as offsets from 1.0:
`output = x * rsqrt(var+eps) * (1 + weight)`, not the standard `* weight`.
**Fix:** Add `load_norm_weight()` that adds 1.0 to stored weights before GPU upload.
Applied to all `Qwen3_5RMSNorm` instances but NOT to `Qwen3_5RMSNormGated`.

### Bug 6: S^T @ k Transposition — THE PRIMARY BUG (commit `2dc96ed`)

**Symptom:** Garbage token output: `"oom SPRetur overlaprey Courtesyroat DNCENN..."`
**Root cause:** All three GDN kernel variants computed `S^T @ k` (column access into S)
instead of `S @ k` (row access).

```c
// WRONG — reads column `row` across all rows j:
kv_val += state[h*HD*HD + j*HD + row] * k_lds[j];

// CORRECT — reads row `row` across all columns j:
kv_val += S_row[j] * k_lds[j];
```

**Fix:** One-line change per variant: switch from column access `state[j*HD + row]`
to row access `S_row[j]` (or `S_local[j]` in register-array variants).

**After fix:** `"I'll start by breaking down the request. The user is asking me to
explain photosynthesis in simple terms."` — coherent reasoning output.

### Bug 7: Sampling (commit `87728d4`)

**Symptom:** Model gets stuck in reasoning loops ("I need to check... I need to check...")
**Root cause:** Bare argmax sampling. Not a kernel bug.
**Fix:** Added `sample_top_p(0.6, 0.8)` + `repeat_penalty=1.1`.

### Timeline

```
Mar 22  54f5f98  First end-to-end run. Garbage output.
Mar 22  44e8558  Fix conv1d + A_log + Q/gate split. NaN→real logits, stuck on \n\n.
Mar 22  0789980  Add Q scaling. Layer 0 matches reference, error accumulates.
Mar 23  2fd1d9f  Fix (1+weight) RMSNorm. Structured English, still incoherent.
Mar 23  836e1b6  Kernel rewrite for performance (introduces transposition bug).
Mar 29  2dc96ed  FIX S@k transposition. COHERENT OUTPUT ACHIEVED.
Mar 29  87728d4  Add repeat penalty + top-p. Breaks reasoning loops.
```

---

## Issue #2: gfx1100 Stale Kernel Bug

**GitHub:** [hipfire#2](https://github.com/Kaden-Schutt/hipfire/issues/2)
**Reporter:** Sludge2158 (RX 7900 XTX, gfx1100, openSUSE Tumbleweed, ROCm 6.4)

### Symptoms

- Model loads correctly (layers detected, correct vocab, correct dimensions)
- Every token decodes to `!` (exclamation mark)
- Speed is real (60.9 tok/s for 9B) — not a mock/fallback
- Affects both Qwen3.5-9B (HFQ4) and Qwen3.5-27B (HFQ6)

### Root Cause

The pre-compiled `.hsaco` blobs shipped in `kernels/compiled/gfx1100/` were built from
**stale kernel source files** that predate the S@k transposition fix (`2dc96ed`).

The compilation pipeline has a split-source problem:

```
compile-kernels.sh  →  reads kernels/src/*.hip  →  .hsaco blobs (on-disk)
                        (may be stale)

Runtime compiler    →  reads include_str!() in kernels.rs  →  .hsaco (temp dir)
                        (always matches binary)
```

`compiler.rs` loads pre-compiled blobs **unconditionally** with no hash validation
(lines 57-63). When pre-compiled blobs exist, the runtime compiler never runs, so stale
kernels get used even though the Rust binary embeds the correct source.

### Workaround

Delete the pre-compiled kernel directory to force runtime compilation:

```bash
rm -rf ~/.hipfire/bin/kernels/compiled/gfx1100
```

After this, the runtime compiler uses the embedded (correct) source and compiles fresh
`.hsaco` blobs. The user confirmed coherent output at 62.3 tok/s after this fix.

### Fix (Implemented)

**Hash validation in `compiler.rs`:** Pre-compiled blobs now require a `.hash` sidecar
file alongside each `.hsaco`. The hash is computed from the embedded source
(`include_str!()`) + the target arch, using Rust's `DefaultHasher`. If the hash file
is missing or doesn't match, the blob is skipped with a warning and the runtime
compiler takes over.

This means:
- Existing blobs without hash files are automatically bypassed (safe fallback)
- `compile-kernels.sh` builds blobs, then `write-kernel-hashes.sh` generates the
  sidecar hashes from the Rust binary (ensuring they match what the runtime expects)
- Stale blobs are detected and rejected even if someone forgets to rebuild

### Additional gfx1100 Note: Arch-Specific Variant Gap

The runtime compiler always uses the **generic** Q8 kernel source
(`gated_delta_net_q8.hip`), even on gfx1100 where an arch-specific variant exists
(`gated_delta_net_q8.gfx1100.hip`). The arch-specific variant uses a register-array
approach (128 threads, S in VGPRs, 10 waves/SIMD) that is better suited to RDNA3's
large VGPR file.

Currently, only `compile-kernels.sh` knows about arch-specific variants. The runtime
compiler would need to be taught to select `gated_delta_net_q8.gfx1100.hip` on gfx1100
to get the optimal kernel path without pre-compiled blobs.

---

## Porting to New RDNA Architectures

### What You Need

1. **Pre-compiled kernels OR hipcc on the target system.** The runtime compiler handles
   everything if hipcc is available. For deployment without ROCm SDK, run
   `compile-kernels.sh <arch>` on a build machine.

2. **Verify wave size.** All kernels assume wave64 (standard on RDNA1-4). If a future
   arch changes this, the warp shuffle reductions in the tiled kernels will break.

3. **Check VGPR budget.** The register-array variants (Q4, gfx1100 Q8, gfx1200 Q8) use
   128+ VGPRs per thread. Verify `maxVGPRs / VGPRs_per_thread >= target_waves`.

### Checklist for a New gfxNNNN

- [ ] Add arch string to `compile-kernels.sh` default list
- [ ] Add arch to `install.sh` GPU detection mapping
- [ ] Add min ROCm version to `dispatch.rs` init check
- [ ] Test generic Q8 kernel — likely works out of the box
- [ ] Profile and decide if an arch-specific variant is worthwhile
- [ ] If creating a variant: `kernels/src/gated_delta_net_q8.gfxNNNN.hip`
- [ ] Run `compile-kernels.sh` to build pre-compiled blobs
- [ ] Run inference on a known model, compare output against reference

### Known Architecture Differences

| Feature | gfx1010 (RDNA1) | gfx1030 (RDNA2) | gfx1100 (RDNA3) | gfx1200 (RDNA4) |
|---------|-----------------|-----------------|-----------------|-----------------|
| VGPRs per SIMD | 1024 | 1024 | 1536 | 1536 |
| Max waves/SIMD | 10 | 10 | 16 | 16 |
| Wave size | 64 | 64 | 64 | 64 |
| LDS per CU | 64 KB | 64 KB | 64 KB | 64 KB |
| Arch-specific Q8 | No (uses generic) | No | Yes (register-array) | Yes (register-array) |
| Min ROCm | 5.0 | 5.0 | 5.5 | 6.4 |

### Debugging "All `!` Output" on a New Arch

This is the stale-kernel pattern. Checklist:

1. Delete `~/.hipfire/bin/kernels/compiled/<arch>/` and retry
2. If that fixes it: the pre-compiled blobs are stale. Rebuild with `compile-kernels.sh`
3. If it doesn't fix it: check `hipcc --version` matches the expected ROCm version
4. Check kernel compilation warnings: `hipcc --genco --offload-arch=<arch> -O3 -o /dev/null kernels/src/gated_delta_net_q8.hip 2>&1`
5. If compilation fails: the generic kernel may use features unsupported on the new arch
