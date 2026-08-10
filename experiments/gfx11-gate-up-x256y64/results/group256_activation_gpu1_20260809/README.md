# Q8 group256 activation probes (GPU1)

Standalone W7900/gfx1100 probes only. Production dispatch was not changed.

## Results

| Variant | Shape | group128 baseline | Candidate | Speedup | Correctness |
| --- | --- | ---: | ---: | ---: | ---: |
| group256 direct | M17408/K5120/N2048 | 4.6338 ms | 4.9034 ms | 0.9450x | max abs 1.192e-7 |
| group256 direct | M512/K512/N256 | 0.0428 ms | 0.0415 ms | 1.0316x | max abs 1.192e-7 |
| group256 staged | M512/K512/N256 | 0.0428 ms | 0.0578 ms | 0.7401x | max abs 1.192e-7 |

The direct path spilled 21 VGPR values and used 84 bytes of private memory per thread at the real shape. The staged path added two 64-column activation slices to LDS and was already 26% slower at the short gate, so it was not run at the full shape.

## Decision

Do not route either variant into serving. Sharing one activation scale across 256 values did not offset direct-load spills at the real shape, while staging the packed activation slices added too much synchronization/LDS overhead.
