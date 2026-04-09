# Fix Proposal: TurboQuant KV Cache

## There is no turbo-specific code bug to fix.

The corruption is caused by Qwen3-0.6B being too small for 4-bit KV quantization.
ALL 4-bit KV modes (HFQ4, HFQ4s, Turbo4) produce garbage on this model. Only Q8 and
FP32 produce coherent output.

## Recommended actions:

### 1. Keep the ds_swizzle optimization (already applied)
- 3 fewer VGPRs in turbo attention kernels (31→28)
- 40 shuffle instructions upgraded from ds_bpermute to native ds_swizzle
- Compiles clean on all architectures
- Produces identical output to the __shfl_xor version
- This is a pure ISA improvement with no behavioral change

### 2. Validate turbo4 on Qwen3.5 models (requires deltanet feature)
- Build with `--features deltanet` to enable Qwen3.5 inference
- Test turbo4 on Qwen3.5-4B or 9B (known to be turbo-tolerant on other hardware)
- If it works, turbo is validated. If it doesn't, the bug is DeltaNet-specific.

### 3. Add a minimum model size guard for turbo KV
- In `KvCache::new_gpu_turbo_adaptive`, warn or refuse when dim < 2048 or n_layers < 32
- Print: "WARNING: turbo4 KV may degrade quality on small models. Use --q8kv for models < 2B."
- This prevents users from enabling turbo on models that can't tolerate it.

### 4. Test HFQ4 KV independently
- HFQ4 KV also fails on the 0.6B model (blank output)
- This is a separate issue from turbo and should be investigated independently
- It confirms that the 0.6B model cannot tolerate ANY 4-bit KV quantization

## What NOT to do:
- Do not revert the ds_swizzle change — it's a correct optimization
- Do not add precision-improvement hacks to turbo for small models — the math is correct,
  the model is too small
- Do not assume turbo is broken on larger models based on the 0.6B results
