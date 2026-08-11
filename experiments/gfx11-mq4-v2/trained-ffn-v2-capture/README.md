# Trained FFN-v2 teacher capture

This diagnostic captures one dense Qwen3.5/Qwen3.6 FFN layer at the production prefill boundary. It writes the residual stream immediately before FFN RMSNorm and immediately after the down-projection residual update. The offline teacher target is therefore `residual_out - residual_in`.

The path is default-off and intended only for dataset construction. It performs synchronous device-to-host copies and file I/O, and rejects graph-prefill and kernel profiling. It currently supports the batched dense FFN paths used by the Qwen3.6-27B MQ4 benchmark; it is not a serving feature.

## Environment contract

- `HIPFIRE_RDNA3_FFN_CAPTURE_DIR`: new output directory.
- `HIPFIRE_RDNA3_FFN_CAPTURE_LAYER`: one zero-based layer index.
- `HIPFIRE_RDNA3_FFN_CAPTURE_MAX_TOKENS`: token cap, default 8192.

Capture format v2 produces little-endian F16 `residual_in` and `ffn_delta` files. The delta is formed from downloaded F32 residuals before F16 serialization, avoiding cancellation from subtracting two separately rounded residual tensors. `tensor_manifest.json` describes tensor files and shapes. `run_manifest.json` adds the model SHA-256, HFQ metadata fingerprint, prompt SHA-256, and benchmark parameters.

## Reproducible smoke

```bash
GPU_ID=1 \
MODEL=$HOME/.hipfire/models/qwen3.6-27b.mq4 \
PROMPT_FILE=/path/to/fixed-training-corpus.txt \
PREFILL=256 \
CAPTURE_LAYER=0 \
CAPTURE_TOKENS=256 \
OUT_DIR=/tmp/hipfire-ffn-v2-capture-layer0 \
experiments/gfx11-mq4-v2/trained-ffn-v2-capture/run_capture.sh
```

If `PROMPT_FILE` is omitted, the repository README is used as a portable smoke fixture. Dataset construction must provide a fixed corpus explicitly. The script refuses to overwrite an existing output directory. Raw tensor files are experimental artifacts and must not be committed; use `/tmp` (the default) or the ignored local `captures/` directory inside this experiment.

## GPU1 smoke evidence

The initial 2026-08-11 v1 smoke used the 14,984,158,208-byte Qwen3.6-27B MQ4 artifact, layer 0, and 256 README tokens on a W7900/gfx1100. Its separately rounded residual files exposed why the v2 direct-delta format is required and must not be used as reconstruction quality evidence. The first-run throughput is intentionally not reported because model/kernel JIT and synchronous capture I/O dominate this diagnostic execution.
