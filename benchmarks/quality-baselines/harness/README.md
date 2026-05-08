# Harness — quant-quality eval scripts

This directory holds the tools that build / consume / aggregate the
KLD references and produce the result tables.

## Files

| File | Purpose | Status |
|---|---|---|
| `manifest.json`         | SHA-pinned index of BF16 reference dumps | schema only; `references` empty |
| `kldref_format.py`      | Reader/writer for the hipfire KLD ref format + per-sequence sidecar | done |
| `kld_reduce.py`         | Bootstrap CI + result-table emitter | done |
| `tokenizer_parity.py`   | Step 1.5 tokenizer-parity check | stub |
| `eval_gguf.sh`          | Step 7 GGUF-anchor wrapper | stub |
| `canary.md`             | 11-seq harness-output reproducibility fixture | structure only; sequences TBD |

## How to add a new quant variant

1. Make sure the BF16 reference for the model exists. If not, run
   `cargo run --release --example build_kld_ref -p hipfire-runtime -- ...`
   (see plan §"Reference dump methodology"). Upload to
   `hipfire-models/hipfire-eval-refs`. Add SHA + producer_cmd to
   `manifest.json`.

2. Run the candidate against the cached reference:

   - hipfire variants: `cargo run --release --example eval_hipfire \
        -p hipfire-runtime -- --variant <name> --ref <ref-path> \
        --slice ../slice/wikitext2-1024s-2048ctx.txt --arch <gfx>`

   - GGUF anchor variants: `./eval_gguf.sh <candidate.gguf> <variant-name>`

   Output: `<variant>__<arch>.kldseq` under
   `../results/<date>/per-seq/`.

3. Aggregate:

   ```
   python3 kld_reduce.py --result-dir ../results/<date>/per-seq/ \
                         --out-md   ../results/<date>/result-table.md \
                         --out-json ../results/<date>/result-data.json
   ```

4. Eyeball the markdown table; commit alongside the run's
   `2026-MM-DD-quant-pareto.md` write-up.

## Plan reference

`docs/plans/issue-113-quant-quality-eval.md` (rev-3.1) is the source of
truth for design decisions. Step 0 (read llama.cpp perplexity.cpp) is
already done; Step 1 (this skeleton) is what's being built right now.

## Pinned llama.cpp commit

`9dcf83552887bb898b4a98a5761361e504e31fc3` (master, 2026-05-08).

`build_kld_ref` (lands in Step 2) hard-checks this commit hash before
spawning `llama-perplexity`. If the user's llama.cpp build is from a
different commit, the format may have drifted and we'd produce a bad
reference; bail loudly.
