# Quant quality eval — KLD-primary harness

Tracking issues #113 (uniform MQ-family quality) and #116 (Lloyd
quality). Plan: `docs/plans/issue-113-quant-quality-eval.md`
(rev-3.3).

This directory holds the eval harness: slice + scripts +
canary fixture + reference manifest + result tables.

## Layout

```
benchmarks/quality-baselines/
├── README.md                    # this file
├── slice/                       # frozen prompt bytes
│   ├── README.md
│   ├── make_slice.sh            # generator from wikitext-2 train (uses .venv/bin/python3)
│   ├── slice.md5                # checksum tripwire
│   └── wikitext2-1024s-2048ctx.txt   # 10.5 MB committed fixture, md5 83b0205a…
├── harness/                     # the actual harness scripts + format readers
│   ├── README.md                # how-to-add-quant
│   ├── manifest.json            # SHA-pinned reference index
│   ├── kld_reduce.py            # bootstrap CI + result-table emitter (incl. PPL)
│   ├── kldref_format.py         # HFKLDR + HFKSEQ-v2 reader/writer
│   ├── tokenizer_parity.py      # Step 1.5 tokenizer-parity check
│   └── canary.md                # 11-seq fixture (expected KLDs land after Step 5)
├── refs/                        # BF16 ref blobs (gitignored)
│   └── .gitignore
└── results/                     # output tables + plots
    └── README.md
```

The producer / candidate-side binaries are Rust examples in
`crates/hipfire-runtime/examples/` — `build_kld_ref.rs`,
`eval_hipfire.rs`, `eval_gguf.rs`, `tokenize_slice.rs`. The harness
reaches into them via plain `cargo run --release --example <name>`
invocations; nothing in this directory needs to know their paths.

## Workflow (overview)

1. **One-time** — generate the slice via `make_slice.sh`, dump BF16
   references on gfx1151 (via `build_kld_ref.rs`), upload to
   `hipfire-models/hipfire-eval-refs`, fill `manifest.json` with
   sha256 + `hf_repo` + producer metadata.

2. **Per quant variant** — run `eval_hipfire` (hipfire candidates)
   or `eval_gguf` (GGUF candidates) against the cached reference.
   Output: a small `<variant>__<arch>.kldseq` file under
   `results/<date>/per-seq/`.

3. **Aggregate** — `kld_reduce.py` reads per-sequence-KLD files,
   bootstraps 95% CIs, emits the result table (markdown + JSON) with
   columns: variant, arch, n_chunks, mean KLD ± CI, p99 KLD, PPL.

## Status (2026-05-08)

- Plan: `docs/plans/issue-113-quant-quality-eval.md` (rev-3.3).
- Step 0 (read llama.cpp perplexity.cpp): **done**.
- Step 1 (skeleton + format readers + reducer): **done**.
- Step 1.5 (tokenizer parity): **done** — verdict: parity fails
  structurally but doesn't block (consumer reads token IDs from
  ref, not from re-tokenization). See plan §"Step 1.5 verdict".
- Step 1.6 (top-K residual sanity): **done** — top-K=256 confirmed.
- Step 2 (`build_kld_ref.rs`): **done**.
- Step 3 (`eval_hipfire.rs`): **done**.
- Step 4 (9B BF16 ref dump): **done** (2.48 GB, gfx1151, 53 min).
- Step 5 (first canary candidate, 9B mq4 vs 9B BF16 ref):
  **running on gfx1100**.
- Step 6 (27B BF16 ref dump): **running on gfx1151**.
- Step 7.A (`eval_gguf.rs`): **done**.
- Step 7.B (GGUF anchor candidate runs): **pending** Step 6 output.
- Steps 8 (DFlash τ) and 9 (write-up): **pending**.

## References

- llama.cpp pinned commit: `9dcf83552887bb898b4a98a5761361e504e31fc3`.
- HF refs repo: `hipfire-models/hipfire-eval-refs` (created
  2026-05-08; uploads pending Step 4–6 completion).
- Plan: `docs/plans/issue-113-quant-quality-eval.md`.
