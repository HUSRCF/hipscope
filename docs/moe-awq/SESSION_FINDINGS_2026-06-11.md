# MoE Mixed-Quant Experts — Findings (2026-06-11)

Target model: **Qwen3.6-35B-A3B** (qwen3_5_moe, 256 experts/top-8, 40 layers, hidden
2048, moe_inter 512). All KLD vs an f32-native oracle (`q36a3b-f32-oracle.hfq`); refs
`q36a3b-{wt2,agentic}-f32.kldref.bin`. Branch `feat/moe-awq-experts` on mi300.
Decode = per-token scoring; "q8/fwht4" = KV mode.

---

## 1. Per-expert AWQ — DEAD END for quality (shipped + measured)
Down-proj AWQ wired end-to-end (commit 3e5f2e9c; indexed silu-rotate `x/s` before FWHT,
gated on per-expert `down.awq_scale` sidecars). Validated correct: down-AWQ file
forwards coherent (PPL 4.78) vs kill-switch garbage (PPL 164.9). **But the quality A/B
(dense-AWQ baseline vs +down-expert-AWQ, isolated):**

| corpus / KV | baseline KLD | +down-AWQ KLD | Δ |
|---|---|---|---|
| wt2 / q8 | 0.03385 | 0.03498 | +3.3% (hurts) |
| wt2 / fwht4 | 0.03572 | 0.03659 | +2.4% (hurts) |
| agentic / q8 | 0.16299 | 0.15961 | **−2.1% (helps)** |
| agentic / fwht4 | 0.16647 | 0.16814 | +1.0% (hurts) |

**Verdict:** expert-AWQ is a wash — helps only agentic/q8, ~free but redundant once you
spend real bits. Dense-AWQ MQ4 is the best-of-MQ4 floor. Quality comes from bits, not
activation-aware 4-bit scaling.

---

## 2. Expert precision ladder (uniform, all 256 experts same format)
Commits: MQ6 wiring 7b71833a (mixed gate_up/down dispatch by individual dtype, gfx942 —
no kernel port needed), MQ5 f7efb940 (full decode parity, ultracode workflow).

**KLD (q8 KV, max-chunks 32):**

| variant (gate_up/down) | bpw | size | wt2 KLD | agentic KLD |
|---|---|---|---|---|
| MQ4 / MQ4 | 4.25 | 19.7 GB | 0.03385 | 0.16299 |
| d6 = MQ4 / **MQ6** | — | 22.3 GB | 0.02739 | 0.13519 |
| **MQ5 / MQ5** | 5.25 | 23.7 GB | 0.01910 | 0.10603 |
| **+P = MQ6 / MQ6** | 6.25 | 27.7 GB | 0.01593 | 0.08677 |

**PPL** (f32 oracle: wt2 5.350, agentic 5.902): MQ4 wt2 5.433/ag 6.099 → MQ5 5.413/6.051
→ +P 5.396/5.967.

Key results:
- **gate_up is the DOMINANT lever, not down.** +P vs d6 (= adding gate_up MQ6 on top of
  down MQ6) buys *another* −32..−42% KLD AND is what moves PPL toward f32. **d6 (down-only
  MQ6) is a half-measure: −14..−19% KLD but PPL-FLAT** (down improves distribution
  fidelity, not next-token likelihood). The PPL win lives in gate_up.
- **KLD ≫ PPL gap at short ctx (512):** MQ6-down cuts KLD a lot but PPL barely moves;
  matches the old kmap bench (the MQ6 PPL win needs >3K ctx; Q8 KV masks it). Use asym4/
  fwht4 KV + longer ctx to surface the PPL win.
- **MQ5** captures ~80% of the MQ4→+P win; only 4 GB under +P → a *measured reference*,
  not a SKU (MQ4-Lloyd at 5.0 bpw would likely dominate it).
- kmap (full promotion) bench (gfx1151, 2026-05-08): MoE +1.7% PPL @ctx2048 Q8 but
  **−19.8% @ctx8192 asym4** (unmasked). kmap'd files need gfx12 for the dense MQ6 GEMV
  (`gemv_mq6g256_prerotated`) — won't forward on gfx942; experts-only promotion does.

---

## 3. REAP importance gate (commit 6381592d) — PASSES
Instrumented per-(layer,expert) `count / Σgate / Σ‖out‖ / Σ(gate×‖out‖)` capture
(`HIPFIRE_MOE_EXPERT_STATS`); TSVs `expert_stats_{agentic,wt2}.tsv`.

- **freq ≈ contribution: Spearman 0.92–0.93** → routing frequency (free from imatrix
  `.counts`) is a fine grader. The "freq≠contribution" worry that parked REAP is resolved.
- **Concentration:** Gini(contribution) 0.76–0.77; top-20% units = ~80% of contribution,
  top-50% = 96%. Deepens with depth (early Gini 0.68 → late 0.75).
- **NEW: hot set is DOMAIN-SPECIFIC** — agentic vs wt2 top-10% overlap only **24%**
  (Jaccard 0.14); top-20% 37%; top-30% 47%. A single graded quant needs the UNION hot-set
  (top-10% union ≈ 17.5% of units) or per-domain builds.

---

## 4. Tier size/perf model (A3B, per-expert graded by importance percentile)
- **Size (exact):** `2.55 GB fixed + 4.026 GB/bpw × avg_bpw`. Reproduces MQ4 19.66 /
  MQ5 23.69 / +P 27.71.
- **Perf (BW-bound est, anchor MQ4 = 150 tok/s gfx11):** per-token read ≈ 0.96 GB fixed
  (lm_head + shared + attn, constant) + ~0.54 GB routed-experts (MQ4) → routed = ~36% of
  read, so bpw swings give modest tok/s swings. Routed weighted by route-freq quintiles
  `[0.52,0.22,0.13,0.08,0.05]`. **Size is cold-driven; perf is hot-driven (the experts
  that fire are the high-bit ones).**
- **Quality est:** hot-dominated; **cold-MQ2-Lloyd tail is the unmeasured risk**
  (coherence-gate, not KLD, is the gate).

| Tier | blend (hot 20% → cold 20%) | avg bpw | size | est tok/s | est ag KLD | character |
|---|---|---|---|---|---|---|
| 1 | MQ4×5 | 4.25 | 19.7 GB | 150 | 0.163 (meas) | uniform baseline |
| 2 | MQ4/MQ4/MQ2L/MQ2L/MQ2L | 3.05 | 14.8 GB | ~157 | ~0.17 | compress (smaller+faster) |
| 3 | MQ6/MQ4/MQ4/MQ2L/MQ2L | 3.85 | 18.0 GB | ~141 | ~0.12 | balance |
| 4 | MQ6/MQ5/MQ4/MQ2L/MQ2L | 4.05 | 18.9 GB | ~138 | ~0.11 | quality-lean |
| 5 | MQ6/MQ5/MQ4/MQ3L/MQ2L | 4.25 | 19.7 GB | ~137 | ~0.10–0.11 | max-grade, **iso-MQ4-size** |
| ref | +P (all MQ6) | 6.25 | 27.7 GB | ~128 | 0.087 (meas) | uniform high |

Headlines: graded tiers 3–5 **dominate +P** (smaller + faster + near-same quality);
**Tier 5 = MQ4 footprint, near-+P quality, ~9% slower**; Tier 2 is the *fastest*
(compress). gfx11 all-resident only — the 5700XT-with-GTT perf model differs (cold = PCIe).

---

## 5. Mixed-precision decode kernel — scoping → **dtype-tag**
Need: routed gate_up + down GEMVs handle per-expert dtype (silu-rotate + combine are
weight-agnostic, unchanged; per-expert dtype already in `ExpertWeights[i].gpu_dtype`,
just collapsed to `[0]` today). Grid is **block-per-expert** (`blockIdx.y = krank`).

- **dtype-tag (one merged kernel, per-block branch): RECOMMENDED.** Block-uniform dtype →
  **no divergence**; 1 launch (matters for the launch-sensitive 5700XT); silu-rotate +
  combine untouched; just a per-expert u8 tag table + the merged kernel. Cost: union of
  two dequant paths (BW-bound → modest occupancy hit).
- two-pass (bucket by dtype, reuse existing MQ6 + MQ2-Lloyd indexed kernels): more
  launches + needs topk **permutation** threaded through silu/combine (bug-prone). The
  "kernel reuse" win is undercut by the permutation glue.

MQ2-Lloyd MoE indexed kernels already exist (ds4: `gemv_mq2g256_lloyd_moe_down_indexed*`).

---

## 6. 5700XT / gfx1010 (RDNA1) — origin loop CLOSED
qwen3.5-0.8B mq4 runs **coherent, native gfx1010** (no GFX override) at **256.8 tok/s
decode** on hipx (HIP_VISIBLE_DEVICES=0, 8.6 GB). RDNA1 forward-correctness gate passes.
A3B-on-5700XT plan: **GTT cold-expert offload** — host-map cold experts
(`hipHostMalloc...Mapped`); the indexed MoE GEMV ptr-table is location-agnostic, so no
pager — heavy-tail masks GTT slowness. Moved to a dedicated worktree
(hipx `~/hipfire-gfx1010`, branch `gfx1010-opt`).

---

## 7. Open — the matrix to run next
Gating question: **does a graded quant hold quality once the cold MQ2-Lloyd tail is real?**
Recommended order:
1. **Cold-tier floor (cheapest, NO new kernel):** uniform all-MQ2-Lloyd-GPTQ + all-MQ3-Lloyd
   experts (existing quant fns) → eval via the CPU-top-K fallback (how the f32 oracle
   already forwards) → KLD + **coherence-gate**. Completes the Lloyd ladder + de-risks the
   cold tier before any kernel.
2. If cold holds → build the **dtype-tag mixed kernel** (Tier 5 or 3) + confirm the graded
   quant matches the model. If it derails → cap cold at MQ3-Lloyd, re-table.
