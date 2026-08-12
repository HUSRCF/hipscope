# Up-A4 layer-isolation probe

- GPU: AMD Radeon Pro W7900 Dual Slot, gfx1100, GPU1
- Model: `qwen3.6-27b.mq4`
- Prompt: `docs/testINPUT.md`, 3375 tokens
- Compared position: 2047
- Hidden dimension: 5120
- Control: production group128 Q8 activation path
- Candidate: one FFN up projection using signed-A4 group128

The compact measurements are in `summary.tsv`. All eight candidates preserved
the first generated token, but each single-layer substitution produced
3.82-5.80% final-hidden relative L2 error. This is a sensitivity diagnostic,
not evidence that those layers are quality-safe.
