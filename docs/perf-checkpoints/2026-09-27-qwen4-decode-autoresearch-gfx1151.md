# Qwen4 (Qwen3.8-Flash-Next mq6q8-pleq8) autoregressive decode — autoresearch campaign (gfx1151) — 2026-09-27

**Lifecycle:** `historical`

**Disposition:** measured local deltas on `gfx1151`, landed on branch
`autoresearch/improve-autoregressive-decode-performance-for-qw-20260926`
(`922308c38` → `939cadb71`). Not a G5 admission, not a retained-replay
certification, not a cross-architecture result, and not a product speed-floor
update.

## Fixture

- Host `halo`, gfx1151, 128 GB UMA, HIP 7.2.
- Model `~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq`, size
  `125467331096`, sha256
  `58fb4f586403000b3394413c38f58b0ec0d8845675f81c3d3c0b5de2cdaa4aed`
  (the Flash-Next pin in `AGENTS.md`).
- Prompt `benchmarks/prompts/glimmer_prefill_1024.txt`, md5
  `0ee8f86ada3683eda452bc294ec824a9`, 1131 prompt tokens, 128 decoded tokens.
- Harness `autoresearch.sh` on the campaign branch.

## Method

- Every run: build `hipfire-daemon`, start a fresh daemon under the GPU lock,
  load with `max_seq` 2048, KV q8, MTP off, `HIPFIRE_GRAPH=0
  HIPFIRE_AR_GRAPH=0 HIPFIRE_CASK_OFF=1 HIPFIRE_DPM_WARMUP_SECS=10`, one warmup
  and three measured greedy generates. The metric is the median client-side
  decode rate (token arrival times, first token excluded).
- Quality gate: `qwen4_kld eval --decode` (8 wikitext-2 chunks × 255
  single-token forwards against the BF16-source teacher) must not exceed
  `KLD_MAX`, and at least the first 4 greedy token ids must match `REF_IDS`.
  The harness also prints the sha256 of every decode logit; a keep is called
  bit-exact when that hash is unchanged.
- One KLD-gated keep (run 200, `c1c032392`: the decode HC long-K BF16 GEMVs
  split each row's K across four waves) changed numerics: 8-chunk KLD
  0.072049 → 0.069977; full 32-chunk decode KLD 0.075160 → 0.074299, NLL
  1.715235 → 1.713378, top-1 0.8892 → 0.8923. `KLD_MAX`/`REF_IDS` were rebased
  in the same commit. Every other keep leaves the logit hash
  (`71f5ed6b70a4…`) unchanged.
- Kernel attribution: rocprofv3 `--kernel-trace` of a 64-token generate after a
  JIT-warming run, split into per-token windows at `embedding_q8_batched`.
  Profiles are local only (`.codeinsight+research/qwen4/ar-dec-20260926/`);
  run logs in `~/.omp/autoresearch/--home-bjoern-hipfire--/runs/NNNN/`.
- The GPU was shared with external `llama-server` processes.
- Serve route: `scripts/serve_harness.py --mode battery --sampling greedy
  --thinking high --max-think-tokens 512 --max-tokens 700 --mtp off
  --speculation off --max-seq 2048` against the run-247 daemon (md5
  `33edbdb4b107f593cb0913a68816e6f3`): 5/5 turns `finish=stop`, no
  runaway/empty/attractor, answers read and correct.

## Result

| | start (run 181, `922308c38`) | end (run 269, `939cadb71`) |
|---|---:|---:|
| decode tok/s | 19.20 | **29.63** (+54.3%) |
| 8-chunk decode KLD | 0.072049 | 0.069977 |
| prefill tok/s | 1238.2 | 1264.8 |

Per-token profile from mid-campaign (run 200 state, 25.7 tok/s) to run 247:
1685 → 1039 dispatches, span 40.4 → 35.5 ms (profiler-inflated), idle gaps
between dispatches 4.8 → 3.0 ms; runs 250-254 removed about 60 more. The large projections
(`gemv_mq6g256v2_x4`, the MoE gate/up and down GEMVs, the BF16 HC GEMVs, the
Q8 LM head) run at 190-230 GB/s, close to what this box streams; what remains
above them is per-dispatch overhead (~2.7 us gap plus ~1.5 us minimum kernel)
and a few latency-bound kernels (QSA attention ~79 us/layer, ~60 after
`78aa7664a`; GDN step ~35
us/layer, the fused HC write kernel ~7 us/sublayer).

## What worked — reusable levers

### 1. Guarded loads serialize (bit-exact)

`x = i < n ? p[i] : 0` inside an unrolled loop compiles to one branch plus
`s_waitcnt vmcnt(0)` per element: N memory round trips instead of one. Give
the full-batch case an unguarded path and issue every load before the first
use. `hyper_norm` (`879ad6c4f`, `247ea5d8a`), MoE down GEMV (`0957e4fa1`,
+2.3% e2e), MoE gate/up (`ab52d62f0`), QSA key loads (`0b4e63fa1`,
`2325a5dfb`), the fused HC write kernel (`14e1fad2e`). Check the ISA for
`global_load` immediately followed by `vmcnt(0)`.

### 2. Launch count is the lever once GEMVs stream at the roof (bit-exact)

- Fuse the tiny kernels that sit between GEMVs, not work into the GEMVs:
  GDN step + gated norm (`bb41aca27`), HC write + next HC read norm
  (`4b0eb314e`), QSA decode prologue 6 → 1 (`b99160cf4`), shared-expert
  activation into the routed activation launch (`9a013301c`), Clear folded
  into the fused HC kernel (`920cbe9f6`).
- Compute a consumer's input where its producer already holds the data: the
  HC read writes `mq_rotate_x(mixed)` for the GDN/QSA projection
  (`f2dd9fd20`) and for the sealed MoE (`f4dc7f9b2`), the GDN step writes the
  out-projection's rotation (head pairs, arrival counter; `c2e6c3570`). A
  one-shot `ScratchState::prerotated` memo lets the consumer's `rotate_x_mq`
  skip; the step executor clears it after every step except the MoE's
  granular stages.
- The decode QSA select writes the persistent selected indices itself
  (`bbcdce045`), and decode pools only the QSA block its row completes
  (`4b9faa30b`; earlier blocks hold the same kernel's output for unchanged raw
  keys).
- Identical inputs, different weights: the HC write's norm uses the same
  streams (same RMS) as the preceding HC read, so the fused write+read kernel
  also produces the next write's normalized row and its four k4 gate quarter
  dots (same per-quarter lane order), and that write skips its norm and gate
  GEMV (`2d266d0d2`, +2.2% e2e).
- Redundant BF16 round trips and elementwise launches folded into
  neighbours (`2301c85a1`, `6bc8d61a9`, `090ff4520`, `337dac92f`,
  `a443fc2fd`, `69ef35586`), shared FWHT per basis (`cf887246c`), multi-matrix
  GEMVs on one input (`6147c1404`, `99991cbea`).

### 3. Latency-bound SIMT kernels (bit-exact)

- GDN decode step: two passes over the state column instead of four reads and
  two writes; write the update as an explicit `fmaf(kv, delta, s * decay)` —
  the plain expression let the compiler contract `fma(s, decay, kv * delta)`
  and changed the output (`f6d18f4be`).
- Router top-10: wave max with `row_xmask` DPP + `permlanex16`, then the minimum
  index among the holders — exact because max/min are order-independent
  (`baee3fc4a`).
- QSA select rank loop: float4 LDS reads (`3e57d0777`, `68d406672`): 29 → 13
  us/layer.
- QSA attention PV: the selected rows are kept as cache offsets (32-bit
  scalar-base loads instead of 64-bit per-lane address math) and each value
  batch loads one batch ahead of the in-order accumulation (`78aa7664a`):
  72 → 47 us warm in isolation.
- MoE gate/up (K = 2560): eight rows per 256-thread block read one LDS copy of
  x; the per-wave x re-reads (8× the 4-bit weight bytes) were L2-bound
  (`939cadb71`, 89 → 85 us in isolation).
- GDN q/k norm: BF16 squares computed in parallel, thread 0 only sums them in
  order (`bbcdce045`); MoE combine and the GDN K=4 conv load every operand up
  front (`6d09e50bb`).

### 4. Host

- Sealed MoE per-call validation O(1) and cached weight lookups: host program
  build 3.3 → 0.43 ms/token (`7a3245ce7`, `b5bf09c2d`).
- Pure-greedy sampling takes the argmax on the GPU and reads back 4 bytes
  (`2dfa3c920`); `tensor_ops::argmax_f32` now has `llama::argmax` semantics
  (first finite maximum, 0 when nothing is finite), covered by a parity test.

## What did not work

- Work added to big memory-bound GEMV epilogues or prologues: HC up GEMV +
  branch mix via a last-arrival counter, routed combine in the shared-down
  scaled add, shared activation recomputed in the shared-down GEMV, HC norm in
  the k4 gate GEMV (all bitwise, all slower).
- Fewer waves with more work each: MoE gate/up two rows per wave, QSA attention
  512/1024 threads, a wide split QSA score kernel (uncoalesced 1 KB key rows
  thrash L2 at higher concurrency), LDS-staged coalesced key chunks.
- DPP instead of `ds_bpermute` in the BF16 GEMV reduction, `hyper_norm` shuffle
  tail, wider MoE down loads, PV unroll 64, float4 router denominator reads,
  32-deep GDN state load batches: no change or slower.
- Non-bit-exact MoE down with 8 lanes per row: +2% but KLD 0.071630 > gate.
- Router top-10 in the last-arriving wave of the gate-side x4 GEMV, the shared
  expert's BF16 down rows as an 11th rank of the routed down launch (with the
  scaled add in the combine), and a 1024-thread QSA select with split rank
  counts: all bitwise, none faster (the work moves into a serialized tail).
- Pipelined QSA key loads: -6 us warm, +13 us with a cold cache; decode is
  the cold case.
- Deferring decode PLE staging past layers 0-1 and stream-ordered H2D copies:
  no change — the host prefix is ~0.1 ms/token (step build 87 us), enqueue
  1.8 ms/token, far below the 34 ms of GPU work.
- Prefetching the next dense GEMV's weights into the 32 MiB MALL (warm reads
  ~850 GB/s vs ~220 cold): a second stream paced by HIP events costs ~4 us
  per event; extra prefetch workgroups inside QSA attention made the GEMV 3×
  faster but slowed attention by the same time. Latency-bound kernels are not
  idle DRAM windows.
- LDS-shared x for the dense MQ6 and BF16 GEMVs and the MoE down: no gain.
- Retained PM4 replay (measured with a local ROCm root that finds ROCr): same
  decode rate as ordinary HIP — the inter-dispatch gap is GPU-side.
  `HIP_FORCE_DEV_KERNARG=1`, `ROC_ACTIVE_WAIT_TIMEOUT=0`: no change.

## Not claimed / not done

- **Other architectures.** The new fast paths reuse the existing gfx11+ SIMT
  gates or are generic HIP; nothing was measured off gfx1151.
- **Validation.** Serve battery only (above); `scripts/redline_daemon_harness.py`
  was not run and no retained route was certified.
- **Other contexts and sampling.** One 1131-token prompt, greedy. The GPU
  argmax path applies only to pure-greedy sampling; sampled decode downloads
  logits as before.
- **MTP / speculative decode** were not measured.
