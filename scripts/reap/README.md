# REAP keep-map test harness (DeepSeek V4)

Evaluate a **REAP-pruned** DeepSeek-V4 variant (e.g. 0xSero `DeepSeek-V4-Flash-162B`,
256→144 routed experts) **without re-quantizing** — by partial-loading the kept
experts out of an existing full quant (`deepseek-v4-flash.mq2lloyd`).

A REAP prune is a *pure expert selection*: the kept experts (and the router rows
for them) are byte-identical to the full model; only the hash-router `tid2eid`
tables (layers 0–2) are remapped. So the loader can keep only `keep[l]` experts
per layer, packed into compact slots `0..kept`, and reproduce the pruned model
exactly from the full quant.

## Loader hook

`HIPFIRE_DEEPSEEK4_REAP_KEEPMAP=<dir>` activates the keep-map in the ds4 loader
(`crates/hipfire-arch-deepseek4/src/{deepseek4.rs,arch.rs}`). Default-off ⇒ the
load path is byte-identical (validated: a keep-all-256 identity sidecar
reproduces the full baseline NLL to 10 decimals). When active: `n_routed_experts`
→ kept count; expert blob loads `experts.{keep[l][slot]}`; Q8 gate + F16 bias rows
are gathered to `keep[l]`; hash `tid2eid` is read from the sidecar. All exact byte
ops — no dequant.

## Workflow

```bash
# 1. Build the keep-map sidecar from the pruned repo's reap_plan.json + safetensors
python3 scripts/reap/build_reap_keepmap.py
#    -> /data/hipfire-models/reap_keepmap_162B_k144/{keep_by_layer.json, tid2eid_l{0,1,2}.i32}

# 2. (optional) Identity sidecar to validate the machinery is an exact no-op
python3 scripts/reap/build_keepall_sidecar.py
#    HIPFIRE_DEEPSEEK4_REAP_KEEPMAP=/data/hipfire-models/reap_keepall_256 ... must == full PPL

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
