# REAP keep-map test harness (generic MoE)

Evaluate a **REAP-pruned** MoE variant (e.g. 0xSero `DeepSeek-V4-Flash-162B`,
256→144 routed experts) **without re-quantizing** — by partial-loading the kept
experts out of an existing full quant (`deepseek-v4-flash.mq2lloyd`).

The keep-map loader is now **arch-generic** (crate `hipfire-reap`), wired into
`deepseek4`, `qwen35`, `lfm2moe`, `minimax`. Activate on any of them with
`HIPFIRE_REAP_PLAN=<dir>` (a `reap_plan.json`). `cohere2moe` gets the same wiring
once it merges to master. See `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md`.

A REAP prune is a *pure expert selection*: the kept experts (and the router rows
for them) are byte-identical to the full model; only the hash-router `tid2eid`
tables (layers 0–2) are remapped. So the loader can keep only `keep[l]` experts
per layer, packed into compact slots `0..kept`, and reproduce the pruned model
exactly from the full quant.

## Loader hook

`HIPFIRE_REAP_PLAN=<dir>` activates the generic keep-map (any wired MoE arch); for
ds4, the legacy `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP=<dir>` still works as an alias
(loads a keep-only plan from `keep_by_layer.json`). Default-off ⇒ the load path is
byte-identical. When active: routed-expert count → kept count; each kept expert is
loaded/packed from `experts.{keep[l][slot]}`; router/gate (+ per-expert bias, ds4
hash `tid2eid`) rows are gathered to `keep[l]` via the shared `gather_rows`. All
exact byte ops — no dequant. REAP and EP-sharding are mutually exclusive.

> ⚠️ The cross-arch **keep-all identity gate** and the ds4 **K144 PPL/KLD smoke**
> below are **GPU-deferred** (the box's GPU is in use). The loader code compiles and
> all `hipfire-reap` CPU unit tests pass; the 10-decimal NLL gate must be run once
> the GPU frees. See the SP1 plan's GPU-embargo note.

## Workflow

```bash
# 1. Build the keep-map sidecar from the pruned repo's reap_plan.json + safetensors
python3 scripts/reap/build_reap_keepmap.py
#    -> /data/hipfire-models/reap_keepmap_162B_k144/{keep_by_layer.json, tid2eid_l{0,1,2}.i32}

# 2. (optional) Keep-all identity plan to validate the machinery is an exact no-op.
#    Generic (any arch): emits reap_plan.json for HIPFIRE_REAP_PLAN.
python3 scripts/reap/build_keepall_sidecar.py --num-layers <L> --num-experts <E> \
        --arch <name> --out /data/hipfire-models/reap_keepall_<E>
#    ds4 convenience (also emits tid2eid + legacy keep_by_layer.json):
python3 scripts/reap/build_keepall_sidecar.py --ds4
#    Then HIPFIRE_REAP_PLAN=<out> must reproduce that arch's no-plan baseline NLL.

# 3. Build the PPL harness
cargo build --release -p hipfire-arch-deepseek4 --example deepseek4_perplexity

# 4. Run full-vs-pruned PPL + KLD
scripts/reap/run_ppl_kld.sh 1024 8
```

`deepseek4_perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--dump-logits PATH]`
computes NLL/PPL via `decode_step`; `--dump-logits` writes per-position full-vocab
logits (`DS4PPL01` format) for `kld_compare.py` (stdlib-only — this box's numpy is
broken). Set the keep-map env var to score the pruned variant; unset for the full
baseline.

## Result (0xSero 162B, K144, mq2-lloyd, wikitext2, ctx=1024)

| | full-256 | pruned-144 |
|---|---|---|
| PPL | 7.56 | 17.73 |
| NLL/tok | 2.023 | 2.875 |

KL(full‖pruned) = 1.14 nats, KL(pruned‖full) = 1.64, top-1 agreement = 57.6%.
The K144 checkpoint is heavily degraded (experimental, partial calibration).
