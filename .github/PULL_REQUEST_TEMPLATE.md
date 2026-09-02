## Summary

<one or two sentences: what changes, in behavioral terms>

## Which surface(s) does this touch?

hw-gate selects hardware routes from the diff (`scripts/hw-gate/select.py`); tick what applies so a reviewer can check the selection.

- [ ] **kernel** — `kernels/`, `crates/rdna-compute`, `crates/hipfire-dispatch`, `crates/hip-bridge`, `crates/saddle-core`
- [ ] **load** — `crates/hipfire-loader`, `crates/hipfire-daemon`, runtime load path (`model_load`, `hfq`, `loader_api`, `config`, `safetensors_source`, `weight_backend`, `multi_gpu`), arch `load*`/`weights*`/`carrier.rs`, `hipfire-config`, `hipfire-registry`, `registry/`, Cargo manifests
- [ ] **serve** — `crates/hipfire-engine`, `crates/hipfire-generate`, daemon slots/serve, runtime emit/eos/dflash/dspark/spec/reset/triattn
- [ ] **arch crate(s)**: <list, e.g. `hipfire-arch-qwen35`, `hipfire-arch-gemma4`, `hipfire-arch-lfm2moe`>
- [ ] `crates/hipfire-quantize` / quant formats (update `docs/quant-formats/qt-register.txt`)
- [ ] control plane — `hipfire-cli`, `hipfire-client`, `hipfire-tui`
- [ ] docs / CI / scripts only (no hardware route)
- [ ] **policy files** — `.github/workflows/`, `CODEOWNERS`, `scripts/hw-gate/`, `leanup-thresholds.txt`, `layering.txt`, `registry/` (always needs a human; the bot can never greenlight these)

## Test plan

- [ ] `./scripts/no-gpu-ci.sh` passes, or the CI jobs are green
- [ ] `cargo build --release` clean
- [ ] `cargo test --lib --workspace` passes
- [ ] **load / serve / kernel changes:** I ran `python3 scripts/serve_harness.py --model <flagship artifact> --mode battery --out battery.json` on hardware myself and attached `battery.json` below — before asking for `hw-run`. A `hipfire run` transcript is not evidence.
- [ ] If perf-relevant: `./scripts/speed-gate.sh` within ±2% of locked baselines
- [ ] If this raises a ceiling in `scripts/leanup-thresholds.txt`: the commit message carries `RATCHET-RAISE: <metric> <old> -> <new>, traded for <reason>` **and** the PR carries the `ratchet-raise` label (CI fails without both)

<details><summary>local serve_harness battery.json (load / serve / kernel changes)</summary>

```json
paste the harness --out JSON here (per-turn rows with assistant_content, attractor, empty, finish, expected_substrings), plus the artifact sha256 and daemon md5
```

</details>

## How this merges (hw-gate)

`hw-gate` is the required CI check. Docs-only diffs pass immediately. Anything touching a hardware surface follows this flow:

1. **A maintainer applies `hw-run`.** Nothing executes on hardware before that — it is the maintainer's "I read this diff". The label is removed after every run and cleared on every push; a new commit needs a fresh `hw-run`.
2. **The runner builds this PR and drives every pinned fixture** (`scripts/hw-gate/fixtures.json`) through `serve_harness.py` — battery for load changes, battery + chain for serve changes, plus Redline parity for kernel changes. Every turn's decoded text is posted verbatim in the **evidence** comment. A missing or mismatched fixture, an attractor, an empty or runaway turn, or a missed expect-substring fails the gate.
3. **The reviewer model posts a prelim and a verdict** (`greenlight` / `needs-human` / `block`) inside a script-enforced floor (`scripts/hw-gate/review.py`). Its review is informational; the `hw-gate` status carries the decision:
   - `greenlight` → check green.
   - `needs-human` → check red until a maintainer who has **read the evidence and verdict** applies `human-reviewed`. Cleared on every push.
   - `block` → check red; only a new commit clears it.
4. **Everyone reads the decoded text.** A green check alone is not review. Numbers never prove coherence.

Route policy: [`docs/VALIDATION.md`](../docs/VALIDATION.md) § hw-gate. `python3 -m tools.change_gate` is optional local planning and is **not** CI evidence; the retired `scripts/coherence-gate*.sh` batteries no longer exist.

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
