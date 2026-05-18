# KLD Measurements — Master Catalog

Single canonical place for every hipfire KLD/PPL measurement against the
BF16 references. New cohort rows land here; per-cohort tables in plan docs
should link back rather than re-publish numbers.

**Methodology, format spec, and caveats:** see
[`issue-113-quant-quality-eval.md`](issue-113-quant-quality-eval.md) §5 (procedure),
§7 (gates), §8 (table format). Do NOT cross-compare rows of different `Mode`
(prefill / per-token / gguf — see §5.3).

**Producer convention:** `eval_hipfire --kv-mode <mode> --scoring-mode prefill
--max-chunks N` against `benchmarks/quality-baselines/refs/<model>-bf16.kldref.bin`,
followed by `harness/kld_reduce.py` for the row stats. Filename:
`<variant>__<arch>__prefill.kldseq` (3-segment form; reducer rejects others).
Embed kv-mode and chunk count in the `<variant>` slug
(e.g. `…-kvq8-c256`) so the row is self-describing.

**Mean KLD ± 95% CI** is bootstrap (10 000 resamples) over per-sequence
means; written by `kld_reduce.py`.

---

## 27B (qwen3.6-27b)

BF16 ref: `qwen3.6-27b-bf16.kldref.bin` (2.48 GB, sha256 `8af83b38…`,
produced on gfx1151 2026-05-09, uploaded to `hipfire-models/qwen-kldref`).

| Variant | Arch | Mode | n_chunks | Mean KLD ± 95% CI | p99 KLD | PPL | Notes |
|---|---|---|---:|---|---:|---:|---|
| mq4-plain-q8head (kv-q8, c256) | gfx1100 | prefill | 256 | 0.2034 (CI 0.1841–0.2237) | 19.009 | 8.584 | 2026-05-18; **no AWQ, no GPTQ**; `--kmap-dense` only (Q8 lm_head + default Promote6 on alt down_proj). Body kmap matches the AWQ+GPTQ row → clean A/B isolating AWQ+GPTQ Δ. ~29 min @ 153 tok/s |
| mq4-awq-gptq-f2-q8head-v100 (kv-q8, c256) | gfx1100 | prefill | 256 | 0.1257 (CI 0.1126–0.1398) | 16.666 | 8.697 | 2026-05-18; AWQ stage-A F2 + GPTQ body + Q8 lm_head; ~27 min @ 162 tok/s |

**AWQ+GPTQ Δ (this cohort, identical kv-q8 + Q8 lm_head + body kmap):**
mean KLD **-38%** (0.2034 → 0.1257), CIs non-overlapping. p99 KLD -12%. PPL +1.3%
(8.584 → 8.697) — distinct from KLD because the plain quant happens to assign
slightly higher probability to the ground-truth next token; AWQ+GPTQ is more
faithful to the BF16 *distribution* (the KLD axis). KLD is the canonical
faithfulness metric here.

**Open rows for context:**
- mq4-awq-gptq-f2 (MQ4 lm_head, same body) — pre-Q8-head v100 quant; would
  isolate Q8 lm_head Δ. Coherence-broken on this stack (premature-EOS
  mid-reasoning, documented 2026-05-18) so the KLD number would still be
  numerically valid but the failure mode masks part of the lm_head signal.
- Same variant on gfx1151 — second-arch row, required by §8 "per arch".
- Same variant at full slice (1175 chunks) — current row is the 256-chunk
  smoke (CI half-width 0.014 = 11% of mean, acceptable for relative
  comparison; full slice expected to tighten by ≈2×).

---

## 9B (qwen3.5-9b)

BF16 ref: `qwen3.5-9b-bf16.kldref.bin` (sha256 `06948cd3…`, produced on
gfx1151 2026-05-08, uploaded to `hipfire-models/qwen-kldref`).

### MQ3 cohort (gfx1151)

| Variant | Arch | Mode | n_chunks | Mean KLD ± 95% CI | p99 KLD | PPL | Notes |
|---|---|---|---:|---|---:|---:|---|
| mq3-rtn-kvq8-c256 | gfx1151 | prefill | 256 | 0.5449 (CI 0.532–0.559) | 16.927 | 13.45 | 2026-05-18; **no AWQ, no GPTQ** — naked MQ3 RTN baseline |
| mq3-awq-gptq-kvq8-c256 | gfx1151 | prefill | 256 | 0.1967 (CI 0.189–0.205) | 9.705 | 11.65 | 2026-05-18; AWQ + GPTQ at 3-bit |

**AWQ+GPTQ Δ at 3-bit (identical kv-q8, body otherwise matched):**
mean KLD **−64% / 2.77×** (0.5449 → 0.1967). CIs nowhere near overlap
(rtn lower bound 0.532 vs awq-gptq upper 0.205). p99 KLD −43%
(16.927 → 9.705) — AWQ's outlier-preserving design pays off in the tail
at low bit-widths. PPL −13% (13.45 → 11.65); unlike the 27B MQ4 cohort,
AWQ+GPTQ wins on both KLD and PPL here. 3-bit is far enough into the
lossy regime that AWQ+GPTQ helps even the next-token-loss metric.

**Tightening vs prior n=20 asym3 smoke** (from `2026-05-15-mq3-awq-uplift/`):
CIs shrank ≈3.6× as expected from √(256/20). mq3-awq-gptq centre moved
from 0.189 → 0.197 (+4%, within original CI); rtn from 0.569 → 0.545
(−4%, within original CI). The qualitative gap held; n=20 wasn't lying
about the direction, only about the precision.

### Other 9B variants

See [`qwen35-mq4-quality-gap.md`](qwen35-mq4-quality-gap.md) for the
pre-existing 9B catalog (MQ4 kv-mode ablations, Q8 lm_head variants,
MQ3-Lloyd, MFP4G32, etc.). That doc pre-dates this master and is still
the canonical reference for non-MQ3 9B rows; fold here when convenient.

---

## 0.8B (qwen3.5-0.8b)

See `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-stage-a-awq-0.8b/`
and the Phase A cohort write-ups. No central 0.8B table in docs yet.
