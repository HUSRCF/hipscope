# gfx11 X256/Y32 Group128 Negative Result

This standalone probe keeps the production 8-wave workgroup, 256-column
output span, Q8-group128 activations, and FP32 accumulation. It reduces the
row tile from 64 to 32 so each lane owns 32 instead of 64 outputs without
duplicating packed weight rows.

Hardware: Radeon Pro W7900 (`gfx1100`), selected with
`HIP_VISIBLE_DEVICES=1`.

| Shape | X256/Y64 row2 | X256/Y32 | Relative | Max abs |
| --- | ---: | ---: | ---: | ---: |
| M512/K512/N256 | 0.0417 ms | 0.0329 ms | **1.2691x** | 0 |
| M17408/K5120/N2048 | 4.7586 ms | 5.3287 ms | **0.8930x** | skipped after exact short check |

The full-set code object uses 236 VGPR, 27 SGPR, zero spills, and zero private
bytes per thread, versus 252 VGPR for production X256/Y64 row2. As with the
16-wave probe, the reduced accumulator footprint helps only the short shape.
At the real FFN shape, repeating activation-side work across twice as many row
tiles costs 10.70%, so this topology must not enter serving dispatch.
