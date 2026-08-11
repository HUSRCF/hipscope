# Trained FFN-v2 single-layer reconstruction

This experiment trains one reduced-width Qwen3.6-27B dense FFN outside the serving runtime. It consumes default-off residual captures from `trained-ffn-v2-capture`, initializes complete 256-channel groups from the official FP8 checkpoint, and writes a standalone BF16 student artifact plus an auditable manifest.

The group ranking is frozen before optimization and must come from `prune_dense_ffn_groups --selection-output`. This uses the exact production MQ4 gate/up/down weight-energy formula and deterministic group-ID tie break, including its execution-format weights. The trainer verifies that the selection and every capture name the same production model SHA-256. Held-out captures never participate in group selection.

The offline forward matches the Qwen3.5/Qwen3.6 norm contract used by hipfire: RMSNorm multiplies by `1 + checkpoint_weight`. The bias is recorded in the output manifest rather than inferred from the checkpoint.

Two teacher contracts are explicit:

- `captured-mq4` reconstructs the FFN delta produced by the retained production MQ4 runtime.
- `source-fp8` recomputes the full-width source FP8 FFN on the captured residual input. This is the stronger source-checkpoint teacher, but it is not a BF16 reference.

`source-fp8` fails closed when its audit differs from the captured production contract by more than 0.25 relative-L2. `--allow-source-mismatch` exists only for explicit diagnostics; it must not be used for promotion training without resolving the source/runtime contract difference.

Train and held-out captures should come from different fixed corpora. The output directory must not already exist.

Export the exact production selection once per target width:

```bash
cargo run --release -p hipfire-quantize --example prune_dense_ffn_groups -- \
  --input "$HOME/.hipfire/models/qwen3.6-27b.mq4" \
  --keep-groups 41 \
  --selection-output /path/to/qwen36-27b-41g-selection.json \
  --dry-run
```

```bash
HIP_VISIBLE_DEVICES=0 \
conda run -n UNI2 --no-capture-output python \
  experiments/gfx11-mq4-v2/trained-ffn-v2-reconstruction/train_layer.py \
  --source-dir /home/husrcf/Code/ProtBind/MTP/data/modelscope_downloads/Qwen/Qwen3.6-27B-FP8 \
  --selection-manifest /path/to/qwen36-27b-41g-selection.json \
  --train-capture /path/to/train-capture-layer0 \
  --heldout-capture /path/to/heldout-capture-layer0 \
  --output-dir /path/to/ffn-v2-layer0-41g \
  --layer 0 --keep-groups 41 --teacher source-fp8 \
  --train-tokens 4096 --heldout-tokens 1024 \
  --steps 200 --batch-size 16
```

`manifest.json` records the frozen groups, source shard SHA-256, optimizer contract, held-out relative-L2 before and after training, and peak allocated device memory. This artifact is not loadable by serving yet and must not be presented as an end-to-end quality or PP8192 result. Keep local trainer artifacts outside the checkout or under the ignored `outputs/` directory.

By default the script also evaluates up to 64 held-out tokens through the full-width source FP8 block and records its relative-L2 and RMS ratio against the captured production MQ4 delta. This audit distinguishes width-reduction error from a source/runtime contract mismatch before a long training run.
