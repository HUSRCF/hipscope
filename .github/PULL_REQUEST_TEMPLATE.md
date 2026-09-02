## Summary

<one or two sentences>

## Which crate(s) does this touch?

- [ ] `kernels/` (HIP source)
- [ ] `crates/rdna-compute` (kernel dispatch / RDNA arch routing)
- [ ] `crates/hip-bridge` (HIP/ROCm FFI)
- [ ] `crates/hipfire-runtime` (LM runtime: KV, sampler, guards, framing, paging, spec decode)
- [ ] `crates/hipfire-arch-qwen35`
- [ ] `crates/hipfire-arch-qwen35-vl`
- [ ] `crates/hipfire-arch-llama`
- [ ] `crates/hipfire-arch-toy` (template — touch only when refining the new-arch reference)
- [ ] `crates/hipfire-quantize`
- [ ] examples / daemon
- [ ] docs / CI / scripts

## Test plan

- [ ] `./scripts/no-gpu-ci.sh` passes, or equivalent CI job is green
- [ ] `cargo build --release --workspace --features deltanet` clean
- [ ] `cargo test --lib --workspace --features deltanet` passes
- [ ] **hw-gate** runs automatically once a maintainer applies the `hw-run` label (required CI check; see [`docs/VALIDATION.md`](../docs/VALIDATION.md) § hw-gate)
- [ ] I read the decoded text in the hw-gate evidence comment
- [ ] **Loader / daemon / runtime-load changes:** I ran `hipfire run <flagship tag>` locally on hardware and pasted the decoded output below
- [ ] If perf-relevant: `./scripts/speed-gate.sh` within ±2% of locked baselines

<details><summary>local hipfire run decoded output (loader/daemon/runtime-load)</summary>

```
paste decoded assistant text from a local `hipfire run <flagship tag>` here
```

</details>

Route policy lives in [`docs/VALIDATION.md`](../docs/VALIDATION.md). The required
CI evidence is **hw-gate** (`.github/workflows/hw-gate.yml` + `scripts/hw-gate/`).
`python3 -m tools.change_gate` is optional local planning only and is **not** CI
evidence. The retired `scripts/coherence-gate*.sh` batteries are **not**
acceptance evidence and no longer exist in-tree.

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
