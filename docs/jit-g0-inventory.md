# G0 inventory and CPU kill experiment

**G0 PASS, with the explicitly excluded FN-gfx1100/hipx fixture.** Baseline `83cb363e8a`; branch `perf/jit-during-load`; worktree `/home/kaden/ClaudeCode/warpfront/wt-jit`. No A1–A5 product change is included. Temporary observation logging was removed; its reproducer patch is archived outside the product worktree.

## Six pinned inventories

| Manifest | Modules | Default required symbols | Explicit helper superset symbols | Evidence |
|---|---:|---:|---:|---|
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/27b-xts-gfx1201.routes.json` | 45 | 51 | 8 | observed-point-routes |
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/27b-xts-gfx1100.routes.json` | 25 | 30 | 6 | observed-point-routes |
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/27b-xts-gfx1151.routes.json` | 27 | 31 | 4 | observed-point-routes |
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/flash-next-gfx1201.routes.json` | 42 | 106 | 8 | observed-point-routes |
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/flash-next-gfx1100.routes.json` | 29 | 67 | 6 | source-predicted; excluded fixture |
| `/home/kaden/qcal/release-0.4.2/jit-peacemaker/flash-next-gfx1151.routes.json` | 17 | 28 | 4 | failed-default-request-observation |

Each HIP entry pins the exact source path/SHA256, full flags, architecture, scheduler profile, cache ABI, toolchain and required symbols; embedded entries pin image bytes/targets/exports and bypass HIP compilation. Default AR/native-MTP membership is separate from historical explicit-spec helper supersets and serial-compiler kill arms. The observed FN-gfx1201 AR/PM-off arm is separately recorded. `HIPFIRE_PACK_JOBS=1` is the existing experiment control; `HIPFIRE_JIT_JOBS=1` remains A1/A2-owned, not implemented by G0.

All 24 current ColdJit042 R9700 source hashes from `/home/kaden/qcal/release-0.4.2/cold-jit/gfx1201-cold-r1/` are covered by selected-symbol entries and `/home/kaden/ClaudeCode/warpfront/wt-jit/crates/rdna-compute/tests/fixtures/jit-g0/cold-24.json`. Mixed fixtures never establish default membership. All 508 TSV source MD5s matched their pinned files.

These are G0 evidence inventories, **not closed A3 ready-plan oracles**. All-row/tail/max-context/graph selector coverage and separately admitted drafter headers are explicitly A3-owned. gfx11 dense inventories include completed prefill observations, not the reset ordinary-decode attempts. The 28 FN-gfx1151 entries are requests before a failed native prefill, not a successful full route or numerical proof. A3 must not extrapolate from that failure to downstream readiness.

The exact canonical Flash-Next model SHA256 is `8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972` (verified local and remote); its header SHA256 is `8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013`. The 27B header SHA256 is `2a4f495e9077a1c68ec4b923b00b7fba5a2206acdad17c7c73ede9ea934a5de0`, verified equal local/remote. Full tensor metadata is retained in committed header fixtures; only tokenizer metadata is omitted.

## Excluded fixture, not a JIT-resource defect

Canonical FN default AR on gfx1100 is source-predicted only. On hipx's 24 GiB XTX + approximately30 GiB host, it is inadmissible: 96 trunk expert tensors alone total64,172,851,200 bytes (59.765625 GiB). Even allocating **all**24 GiB VRAM to experts leaves at least35.765625 GiB host expert payloads. Restored hipx `/proc/meminfo` reports MemTotal31,960,744 KiB (30.480141 GiB), so the lower-bound shortfall is5.285484 GiB before MTP, fixed device weights, staging, scratch, safety or compiler jobs. This fixture cannot load there at all; **no load-time JIT guarantee is claimed**. Admission and production loader reserves remain A4-owned. Detail: `/home/kaden/qcal/release-0.4.2/jit-peacemaker/flash-next-gfx1100-fixture-exclusion.json`.

## CPU-only kill experiment

Receipts: `/home/kaden/qcal/release-0.4.2/jit-peacemaker/kill-27b-gfx1201-r1/`, including frozen `input-manifest.json`, `registry.tsv`, `provenance.json`, `summary.json`, `required-exports.json` and nine per-process command/strace/RSS/time/result receipts.

Used the existing bounded pack compiler, implementation `1767881e81e1` from the read-only prerequisite worktree `/home/kaden/ClaudeCode/warpfront/wt-parallel-packs` (tip `74a8abf5e8`); its compiler source matches baseline. Builds used `-j 28`, `CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.2/target-jit`, and `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`. No new execution runtime or compiler executor was added.

Planned set:44 HIP module aliases, **43 full source/flags/profile/ABI recipe groups**,47 required HIP exports, plus one embedded image supplying4 required exports (51 total). The two `qwen35_fa_prep_*` aliases retain their names and union both required symbols while compiling once. All47 required HIP exports were found in the actual measured ELF objects. Installed-hit arms used a private executable-relative installed pack seeded from serial1, with fresh private HOME/cache/lock/output directories. Every arm has three fresh pack processes; CPU-only env is pinned in each command receipt.

Budget:64 affinity CPUs; default CPU request56; MemAvailable68,166,197,248 bytes; reserve63,215,166,464 bytes (three times27B file bytes +16 GiB staging/scratch +1 GiB safety); remaining4.611007 GiB;1536 MiB/job clamp => **3 workers**. All inspected cgroup ancestors had `memory.max=max`, with no tighter finite cap. This is a conservative experiment admission bound, not a production loader-memory estimator. Keep1536 MiB/job despite lower observed RSS.

| Arm | Fresh process | Wall s | User+sys s | Sampled peak tree MiB | HIP compile invocations | Peak active compilers |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1 | 25.711841 | 25.87 | 240.10 | 43 | 1 |
| bounded | 1 | 9.333515 | 27.04 | 580.79 | 43 | 3 |
| hit | 1 | 0.103486 | 0.22 | 52.90 | 0 | 0 |
| bounded | 2 | 9.249586 | 26.87 | 577.93 | 43 | 3 |
| serial | 2 | 25.927736 | 26.06 | 239.52 | 43 | 1 |
| hit | 2 | 0.103468 | 0.20 | 58.64 | 0 | 0 |
| serial | 3 | 25.765523 | 25.91 | 239.68 | 43 | 1 |
| bounded | 3 | 9.428430 | 27.23 | 587.24 | 43 | 3 |
| hit | 3 | 0.124684 | 0.26 | 148.05 | 0 | 0 |

All nine return codes were0. Tree RSS was sampled every20 ms; the highest bounded sample was587.24 MiB, below the reserved compiler budget. `strace` counted compile invocations and peak active compiler processes; installed hits compiled **zero** recipes.

Every run's129 object/hash/index files were byte-hash-identical. Full per-file SHA256 maps are in each `result.json`; the common sorted filename-to-SHA256 artifact-set digest is:

`61000acde94372919a5bcec7e175fae0376c469436fa28a8cf0c4a5a7116d59a`

Frozen input manifest SHA256: `c5cb33aa4983fc7de0856e5124b27a00cfa97afec76f273e3761e66d24e186ff`.
Pack binary SHA256: `b20aa944c62144848cb86464a9bc258142e2104c370e0049a678da3673fc28c3`.

Median serial25.765523 s, bounded9.333515 s, installed hit0.103486 s: **63.7752% compile-wall reduction**. Using the already-observed2.834 s weight interval, `max(weight, compile)` predicts25.765523 =>9.333515 s (63.7752% reduction), with installed-hit critical path2.834 s. Thus the CPU kill gate exceeds the20% compile-wall and10% predicted critical-path thresholds with identical artifacts and a safe bounded resource model: **PASS**.

This does not measure real load overlap, driver loading, alias publication, ready/first-response latency, numerical equivalence or throughput. The prediction is not an observed end-to-end load improvement. No Part A arithmetic/source reorder is included; the intended steady-state throughput ceiling remains0% change.

## Source findings handed to A3

- Four live tensor_ops q8 exports are absent from legacy registry declarations/TSV export lists but present in actual ELF objects: `indexed_attention_attention_q8_batched`, `indexed_attention_attention_q8_batched_hg4`, `indexed_attention_cache_append_q8_batched`, `indexed_attention_decode_prologue_q8`. Inventories record the declaration gaps rather than hiding them; no product registry correction is included.
- Current canonical Halo HC-down selected `gemm_wmma_lds_160_64_32_64_k64_p`; the historical four-entry LDS census is not a closed default route. HC row-fold and all selected shapes must be audited before port ranking.
- Native MQ6 shape coverage is not blanket coverage: gfx1151's current public thresholds include M>=2560 and N>=2048; gfx1201 BF16/regions paths remain HIP. gfx1100 has no corresponding native image.
- `HIPFIRE_QWEN4_MOE_SYM_PM=0` restores symmetric MoE HIP only on gfx1151, not gfx1201; the HIP TU is already required for its symmetric-check export. No whole-TU saving is inferred.
- FN HC defaults differ (gfx1151 level3, gfx1201 level1, gfx1100 level0); gfx1100 host-mapped default is AR, not Halo's native-MTP default. QSA/HC producer and consumer coverage must share resolved policy, not mixed-fixture names.

## Observation validity and helper safety

Recovered remote result mtimes:27B-gfx1100 `2026-10-08T19:55:27.343Z`;27B-gfx1151 `19:58:40.891Z`;original FN-gfx1151-native `20:02:13.228Z`. They precede Main's invalid OOM window20:25–20:36Z. The original FN run returned1 after an HSA fault; its request receipt is archived, with **no causal kernel-fault attribution** and no memory/process-pressure telemetry. All queued reset/timed-out followups and runs in the invalid window are excluded from acceptance evidence.

Owned helper directories `/home/kaden/qcal/release-0.4.2/jit-peacemaker/` and `/home/kaden/pm-wave/jit-g0/` were checked for Python stdlib-name collisions: none. G0 helper shebangs and remote replicas now use `python3 -I`; future invocations must do so. No retry wrapper or recursive relaunch was added.

## Pure route-data tests and main-agent handoff

Seven pure tests are added at `/home/kaden/ClaudeCode/warpfront/wt-jit/crates/rdna-compute/tests/jit_g0_route_data.rs`, with six manifests, two header fixtures and the24-source fixture under `/home/kaden/ClaudeCode/warpfront/wt-jit/crates/rdna-compute/tests/fixtures/jit-g0/`. They cover baseline factories/recipe identities/embedded exports, all24 cold sources, header quant/shapes and separated defaults, full-recipe alias union, conflicting public-entry image refusal, existing architecture/shape-selector constants, and G0-pass versus A3-readiness/excluded-fixture distinctions. They do not initialize HIP or add a model-route execution runtime.

**Verified on Main's explicit follow-up assignment: 7 passed, 0 failed (0.03 s)**, CPU-hidden, release/locked, `-j 28`, target-jit. Main supplied two type corrections; the remaining failure was a mis-keyed test lookup: serial/spec arms use `entries_ref`, not inline `entries`. The test now resolves those named manifest fields and validates every referenced recipe without skipping or weakening assertions. Passing output: `artifact://119879`. Exact command:

```sh
env ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.2/target-jit cargo test --release --locked --manifest-path /home/kaden/ClaudeCode/warpfront/wt-jit/Cargo.toml -j 28 -p rdna-compute --features deltanet --test jit_g0_route_data
```

G0 concludes **PASS with the FN-gfx1100 inadmissible-host exclusion**, per Main's scope clarification. A1 has not started; main will dispatch A1–A3 separately.
