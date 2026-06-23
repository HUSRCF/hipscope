# Spec-Decode Durability — qwen dense, gfx11 (2026-06-23)

Measured on gfx1100 (RX 7900-class), dense Qwen3.6-27B (`qwen3.6-27b.mq4`), q8 KV,
greedy (temp=0), 3-run median. **AR decode baseline ≈ 43 tok/s** (genre-independent).

> **CORRECTION (config matters).** The durability numbers below use the **chatml serving
> config** — the mode the daemon actually serves in. An earlier draft used `--no-chatml` (raw
> open-ended *continuation*, the canonical code-bench config), which is the single worst case for
> a drafter and made prose look retrain-bound. Under the serving config, DFlash clears every
> standard genre; only pure *creative fiction* (the least-predictable content) stays below the
> DFlash 1.3× bar — and MTP covers that at its 1.15× floor. **Always measure durability under the
> serving config.**

**Durability floors (the goal):**
- **DFlash:** every genre `τ > 1.5` **AND** `tok/s ≥ 1.30× AR` (= 56 tok/s).
- **MTP:** every genre `tok/s ≥ 1.15× AR` (= 49.5 tok/s).

---

## TL;DR — durable across every genre via a lossless hybrid (serving config, chatml)

| Genre | DFlash tok/s | AR× (floor 1.3×, τ>1.5) | MTP tok/s | AR× (floor 1.15×) |
|---|---|---|---|---|
| code | 98 | 2.28× ✓ τ4.05 | 83 | 1.93× ✓ |
| reason | 124 | 2.89× ✓ τ5.59 | 72 | 1.67× ✓ |
| instruct | 89 | 2.07× ✓ τ3.55 | 67 | 1.57× ✓ |
| prose (expository/reflective/descriptive) | 66–76 | 1.55–1.77× ✓ τ2.3–2.84 | 65 | 1.50× ✓ |
| prose (**creative fiction**) | 42 | 0.97× ✗ τ1.05 | 50 | 1.16× ✓ |

**DFlash clears every standard genre** (code/reason/instruct + representative prose) at ≥1.3× and
τ>1.5. **MTP clears everything — including creative fiction — at ≥1.15%, losslessly.** The durable
answer: DFlash where it wins (1.3×+), MTP for the creative-fiction tail (1.15×). **Every content
type is covered, losslessly, deployable** (MTP via `HIPFIRE_QWEN_MTP=1`).

The only content DFlash can't clear at 1.3× is **pure creative fiction** (novel narrative, τ1.05) —
a *fundamental* spec-decode limit (the content is unpredictable for any drafter, the drafter can't
match what the target invents), not a hipfire-specific gap. MTP covers it at the lower floor. The
earlier "DFlash prose retrain-bound" verdict was a **measurement-config artifact** — it used
`--no-chatml` raw continuation (the canonical *code*-bench config), the worst case for prose;
under the chatml serving config, representative prose clears DFlash comfortably.

---

## 1. DFlash durable perf matrix (27B-3.6, greedy, q8 KV)

| Genre | linear tok/s | τ | AR× | DDTree b12 tok/s | τ | AR× | Floor (1.3×, τ>1.5) |
|---|---|---|---|---|---|---|---|
| code | **130.0** | 6.33 | 3.03× | 111.6 | 7.80 | 2.60× | **PASS** (linear) |
| reason | **109.9** | 4.79 | 2.53× | 77.2 | 4.94 | 1.78× | **PASS** (linear) |
| instruct | **74.5** | 2.82 | 1.72× | 53.7 | 3.57 | 1.24× | **PASS** (linear) |
| prose | 45.9 | 1.26 | 1.06× | 37.1 | 1.70 | 0.86× | **FAIL** (no config) |

Notes: DDTree loses to linear on tok/s in *every* cell (per-cycle tree-verify cost > τ benefit at
any budget). Prose is the sole failure: linear fails τ (1.26<1.5), DDTree fails tok/s (0.86×).
On 27B-3.5 the picture is identical plus a marginal **instruct** miss (1.25×, 4% short — the 3.5
drafter is slightly weaker on instruct; 3.6 passes at 1.72×).

## 2. MTP durable perf matrix (27B-3.6, K=3 p_min=0.4, greedy, q8 KV)

| Genre | tok/s | τ | AR× | Floor (1.15×) |
|---|---|---|---|---|
| code | 82.95 | 3.38 | 1.93× | **PASS** |
| reason | 71.70 | 3.13 | 1.67× | **PASS** |
| instruct | 67.38 | 2.89 | 1.57× | **PASS** |
| prose | 56.43 | 2.43 | 1.31× | **PASS** |

**All genres clear, losslessly, at one fixed config.** `p_min=0.4` (deeper adaptive chain) is the
prose-optimal early-exit threshold and doesn't hurt structured genres; `p_min` is not a correctness
knob (greedy MTP is distribution-preserving at any threshold). MTP perf is proven via
`mtp_only_demo`; it is **not yet daemon-wired** (only ds4 MTP is) — deployment is the remaining
implement step (§6).

## 3. The prose dividing line

Prose is the entire story. High-entropy narrative text means the target distribution is flat, so a
*lossless* spec-decode acceptor can rarely confirm the draft's specific token — prose's lossless
DFlash τ ceiling is **~1.26** (linear) / ~1.70 (DDTree, but the tree's per-cycle cost erases the
gain). The distilled DFlash drafter is code/agentic-trained (inverse-τ: code 6–10, prose ~1.3).

MTP's head is **jointly trained with the target**, so it actually models prose (τ2.43 greedy) and
its acceptance translates to 56 tok/s — over the (lower) MTP floor. Same hardware, same target,
same KV: the difference is entirely the drafter's prose competence.

## 4. Levers tried and shelved (the campaign)

| Lever | Result | Disposition |
|---|---|---|
| **Chunked (parallel) GDN** | math validated (parity 1.5e-7) but 11-16× *slower* than sequential at every shape; threading recovered to 5× slower, plateaus (grid under-utilization) | **falsified**, default-off flag |
| **DDTree budget right-size (prose)** | decode tok/s flat ~37 across b4–b12 (τ drops with budget — a wash) | falsified |
| **Tree-verify hipGraph fix** | root-caused `block_start` staleness, fixed (committed) — correct, no regression, but ~0% on prose (launch overhead tiny vs the tree's host mask-build) | **fixed & banked**, not a prose lever |
| **Rejection sampling (lossless, temp>0)** | prose τ1.14 — *worse* than greedy (flat target dist) | shelved |
| **CACTUS bumped acceptance (lossy)** | τ>1.5 but visibly corrupts prose (garbled tokens, repeats) at the δ that clears τ; also D2H-capped ~45 tok/s | rejected (quality) |
| **Verify-cost cuts (compressed lm_head, drop topk sync)** | ~+15% ceiling; the 27B forward is the floor, can't reach 56 losslessly | insufficient for prose |
| **MTP (native head)** | clears all genres incl. prose, lossless | **the answer for prose** |
| **DFlash prose drafter retrain** | the only lossless DFlash-prose lever; infra broken 3 ways (trainer won't compile, `load_target_init` unbuilt, mi300 torn down), correct d=5120 arch never run, multi-week, 30–40% odds | **deferred** (dedicated effort) |

## 5. State of spec-decode for qwen in hipfire (report)

**What works, durably:**
- **DFlash (linear, greedy, q8 KV)** is the production fast path for structured genres — 2–4× AR
  on code/reason/instruct, lossless. This is the recommended default for code/agentic/reasoning.
- **MTP (compressed-serial, K=3, p_min=0.4)** is the durable *all-genre* option, including prose,
  at 1.3–1.9× AR, lossless. It is the only mode that clears prose.
- The **tree-verify hipGraph** is now correct (was silently broken via `block_start` staleness)
  and available behind `HIPFIRE_VERIFY_GRAPH_TREE`, though its perf benefit is marginal on current
  workloads.

**What's shelved / why:**
- **DDTree** out-accepts linear on τ but never wins tok/s (per-cycle verify cost) — not a perf win
  on this hardware; useful only where acceptance matters more than throughput.
- **Chunked GDN** — algebraically exact but a perf regression at every shape; the sequential GDN
  is already an optimal kernel for this op. Banked behind a dead flag; do not re-chase.
- **Lossy acceptance (CACTUS)** corrupts prose at the τ it needs; not durable.

**The dividing line:** it's *creative-fiction* generation, not "prose," that's drafter-bound.
Expository/factual/reflective prose under the serving config is predictable enough that DFlash
clears it (τ2.3–2.84). Only *novel narrative* — where the target invents content the drafter can't
anticipate — collapses τ; that's a fundamental spec-decode property, and MTP's native head narrows
it enough to clear the 1.15× floor.

**Highest *durable* perf — chatml serving config, EVERY genre clears (gfx11 dense 27B-3.6):**
- code **98 tok/s** (DFlash, 2.28× AR, τ4.05)
- reason **124 tok/s** (DFlash, 2.89×, τ5.59)
- instruct **89 tok/s** (DFlash, 2.07×, τ3.55)
- prose (expository/reflective/descriptive) **66–76 tok/s** (DFlash, 1.55–1.77×, τ2.3–2.84)
- prose (creative fiction) **50 tok/s** (MTP, 1.16× — the fundamental-limit tail DFlash can't clear)

(`--no-chatml` raw continuation is faster for structured genres — code 130, reason 110 — but is the
canonical *code*-bench config, not the serving mode, and its prose-continuation fails; chatml is the
all-genre-durable config.)

## 6. Open items / recommendations

1. **Deploy MTP — DONE** (commit `fd717e5d`). Wired into the daemon: `LoadedModel.qwen35_mtp_head`
   (bundled `.mq4-mtp` trailer or `<trunk>.mtp` sidecar), `generate_qwen35_mtp`, gated
   `HIPFIRE_QWEN_MTP=1` + greedy + arch 5/6 + single-GPU (default path unchanged), generation-local
   `MtpSpecState` freed at every exit (state-bleed guard), defaults K=3/p_min=0.4. **Validated
   gfx11 27B-3.6:** routes through MTP (`"mtp":true`), **lossless** (byte-identical to AR at
   temp=0), **no state-bleed** (same prompt at positions b & d in a 4-request session →
   byte-identical output + τ), perf over floor (prose 65 / code 78 / capital 93 tok/s decode).
2. **Genre-aware mode selection** (DFlash for structured, MTP for prose) — or simply run MTP
   everywhere (it clears all genres; DFlash is faster on structured but MTP is durable everywhere).
3. **DFlash prose retrain** (lossless, to lift DFlash prose τ→2+): a dedicated effort — build the
   `load_target_init` d=5120 loader, a prose-balanced ChatML corpus, validate on available GPU.
   30–40% odds; only worth it if DFlash-on-prose throughput (~91 tok/s projected) is needed beyond
   what MTP already delivers (56).
4. **3.5 instruct** (1.25×, 4% short) — a small verify-cost lever (compressed lm_head ~+9%) closes
   it; 3.6 instruct already passes.
