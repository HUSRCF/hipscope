# Kernel Atlas — three-layer architecture

The Kernel Atlas is hipfire's **measurement corpus for kernel performance
data**. It's separate from Astrea, which is the **measurement corpus for
quantization quality**. Astrea's PR explicitly documents an "Atlas
handoff" — once a quant passes Astrea's KLD/MSE quality gates, Atlas
takes over to measure how fast its kernels actually run.

This doc captures the three-layer split that survived discovery, why
each layer ends up in the language it does, and the migration path
from "stdout-scrape" collection to typed in-process emission.

## The three layers

### Layer 1 — Collection (Rust, in-process)

**Job:** Capture per-row measurements from running bench/inference
binaries. One row per (workload × phase × variant). Each row is a typed
`AtlasRow` struct serialized to JSONL.

**Why Rust:** The data comes from the bench binaries themselves
(`bench_qwen35_mq4`, `dflash_spec_demo`, …). Having those binaries emit
typed `AtlasRow` values directly — via the `hipfire-atlas` crate — is
strictly safer than the legacy approach of:

1. Bench binary prints `SUMMARY  prefill_tok_s=1432 …` to stderr
2. Python wrapper regexes the line into a dict
3. Python wrapper writes JSONL

The regex contract is silently fragile: a new metric requires changes
in both places, and a format drift breaks Python collection without
the bench noticing. Today's `PREFILL_SUMMARY` addition is a recent
example — adding a new field worked by *luck* (the regex
`[A-Za-z0-9_]+=[0-9.]+` happened to match it). Typed in-process
emission eliminates that whole class of breakage.

**API surface:**

```rust
use hipfire_atlas::AtlasRow;

let mut row = AtlasRow::new("prefill", "bench_qwen35_mq4");
row.set_metric_f64("prefill_tok_s", 1432.5)
   .set_metric_f64("prefill_kernel_ms", 88.4)
   .set_metric_str("arch", "gfx1100")
   .set_metric_u64("prefill_tokens", 128);
row.append_to_jsonl(atlas_path)?;
```

**Wired today:**
- `bench_qwen35_mq4 --emit-atlas <path.jsonl>` writes a `prefill` row
  after the prefill phase and a `decode_ar` row after the gen phase.

**Wired tomorrow (deferred):**
- `dflash_spec_demo --emit-atlas <path.jsonl>` writes a `decode_dflash`
  row including τ, accepted tokens, cycle count.
- Per-kernel profile capture inside the binary — instead of grepping
  the `=== PROFILE ===` stdout block, the binary writes
  `artifacts.profile_kernels` directly from `rdna_compute::profile::stop()`.
- ISA manifest capture via in-process `clang-offload-bundler` +
  `llvm-readelf` invocations — replaces the current Python subprocess
  chain in `scripts/kernel_atlas.py`.

### Layer 2 — Analysis (Python, out-of-process)

**Job:** Rank rows, render ASCII fit tables, suggest tuning targets,
generate optimization-task JSON bundles for an agent or human.

**Why Python:** Ad-hoc analysis iterates faster in Python. The pandas
ecosystem, matplotlib for plotting, jupyter for poking — these are
*much* nicer than equivalent Rust setups for the
"load 1000 rows, group by quant × arch × shape, show the median"
workflow that dominates Atlas analysis.

**Tool:** `scripts/kernel_atlas.py` on the HIPa branch. Keep it.

**The migration:** Once Layer 1 is wired everywhere, the Python tool
stops being a *collector* (no more subprocess `bench && grep`) and
becomes purely an *analyzer* (`load_rows(path)` + ranking + render).
That removes ~200 lines of subprocess plumbing from the Python script
and makes it a much smaller surface.

### Layer 3 — Advisor (future, language TBD)

**Job:** Consume the corpus and *suggest* tuning changes — register-
budget retunes, K-unroll factors, dispatch heuristics — either via a
hand-coded ranker or an autotuner / LLM advisor pipeline.

**Why not committed yet:** We don't have enough corpus volume to make
this worth building. When we do, it'll likely live in Rust for the
embed-in-engine case (advisor runs alongside inference) and Python for
the offline-research case (advisor runs in a notebook).

## Concrete migration steps

| Step | Change | Status |
|---|---|---|
| 1 | `crates/hipfire-atlas/` crate with typed `AtlasRow` + JSONL writer | ✓ this PR |
| 2 | `bench_qwen35_mq4 --emit-atlas <path>` writes prefill + decode_ar rows | ✓ this PR |
| 3 | `dflash_spec_demo --emit-atlas <path>` writes decode_dflash rows | TODO |
| 4 | Profile capture in-process (no `=== PROFILE ===` stdout grepping) | TODO |
| 5 | `scripts/kernel_atlas.py` switches from subprocess+grep to `subprocess hipfire-atlas` for collection orchestration; analysis stays Python | TODO |
| 6 | ISA manifest capture in-process via Rust wrappers around `clang-offload-bundler` | TODO |

## On-disk schema invariants

- One row per line (JSONL)
- `schema` field always reads `"hipfire.kernel_atlas.v0"` (semver later)
- `phase` is one of: `"prefill"`, `"decode_ar"`, `"decode_dflash"`
- `workload_kind` identifies the binary or workload class
  (`"bench_qwen35_mq4"`, `"dflash_spec_demo"`, etc.)
- `metrics` holds numeric measurements (canonical types: `f64` for
  throughput/latency/bandwidth, `u64` for counts, `String` for
  categorical labels like `arch` or `kv_mode`)
- `artifacts` holds nested-structure data (per-kernel profile tables,
  ISA manifests, lineage refs)
- Extra fields flatten into the row object — useful for
  `captured_at_unix_s`, `git_sha`, `hostname` etc. without polluting
  `metrics`

Compatible with the JSONL emitted by `scripts/kernel_atlas.py` so a
mixed corpus from both layers is parseable by either tool.

## Why not pick one language?

Because the layers have genuinely different workloads:

- **Collection needs trust.** A regex-fragile Python parser that
  silently drops a new field is worse than no collection. The bench
  binary that knows the metric should write it directly.
- **Analysis needs iteration speed.** Re-running a Rust analyzer to
  change one ranking heuristic is friction; Python is a notebook away.
- **Advisor needs both.** When we get there, the in-engine variant
  goes in Rust (no Python in the inference hot path, per the
  project rule), and the offline variant stays Python.

## Relationship to Astrea

Astrea (`scripts/astrea.py`, agent skill at `.agents/skills/astrea/`)
covers the **quality** axis: KLD vs BF16 reference, per-layer error
attribution, PyTorch oracle replay, calibration recipes.

Atlas covers the **performance** axis: tok/s, GiB/s, per-kernel timing,
ISA manifests.

The two are designed to compose:

1. Astrea picks a candidate quant variant, verifies its KLD floor.
2. Astrea hands off to Atlas via a workflow contract documented in the
   skill (`Atlas handoff` section).
3. Atlas measures the kernels for that variant, writes rows.
4. Both corpora indexed by `git_sha` + workload, so a single experiment
   joins to one Astrea row + N Atlas rows.

Neither tool duplicates the other; merging them would force a single
language choice that hurts both axes.

## Open follow-ups

- **AOT-shipped HSACOs per gfx ID.** The Atlas
  `startup_overhead_ms` and `cold_overhead_pct` fields will collapse
  to ~0 when bench tools no longer pay JIT compile cost on first call.
  See PR #253 description for the perf gap this would close (~50% on
  both gfx1100 and gfx1201 today).
- **`bench_qwen35_mq4` argmax-NaN panic at `llama.rs:3549`** during
  graph-captured gen warmup blocks the decode_ar atlas row from
  emitting. Prefill row still writes (emitted before the panic).
  Filing as its own bug.
- **`dflash_spec_demo --emit-atlas`** and **profile-capture-in-process**
  are the next concrete deliverables for Layer 1.
