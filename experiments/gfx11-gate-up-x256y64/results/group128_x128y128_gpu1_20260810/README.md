# Balanced X128/Y128 Tile Probe

GPU: Radeon Pro W7900 (`gfx1100`), GPU1. Baseline and candidate were alternated for 15 pairs after warmup.

This exact probe applies the mature llama.cpp RDNA3 128-output x 128-token geometry to Hipfire's HFQ4-G256/group128 contract.

| Shape | Operation | X256/Y64 baseline | X128/Y128 | Relative |
|---|---|---:|---:|---:|
| M17408 K5120 N2048 | set (gate/up) | 4.7464 ms | 4.9694 ms | 0.9551x |
| M5120 K17408 N2048 | add (down/residual) | 4.8245 ms | 5.0060 ms | 0.9637x |

Both cases are exact (`max_abs=0`). The balanced geometry is a negative result for this format: reducing activation staging does not repay the larger HFQ4 weight tile and associated packed-weight expansion work. It remains standalone and is not production-routed.
