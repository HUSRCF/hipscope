# gfx11 X256/Y64 Group128 16-Wave Negative Result

This standalone probe keeps the production X256/Y64 tile, group128 Q8
activation format, FP32 accumulation, and output layout. It doubles the
workgroup from 8 to 16 waves so each lane owns 32 rather than 64 outputs.

Hardware: Radeon Pro W7900 (`gfx1100`), selected with
`HIP_VISIBLE_DEVICES=1`.

| Shape | 8-wave row2 | 16-wave | Relative | Max abs |
| --- | ---: | ---: | ---: | ---: |
| M512/K512/N256 | 0.0425 ms | 0.0383 ms | **1.1097x** | 0 |
| M17408/K5120/N2048 | 4.6018 ms | 5.5687 ms | **0.8264x** | skipped after exact short check |

The code-object audit confirms that the accumulator reduction worked, but not
enough to offset the 512-thread workgroup cost:

| Full-set kernel | Workgroup | VGPR | SGPR | Spills | Private bytes/thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production row2 | 256 | 252 | 31 | 0 | 0 |
| Wave16 | 512 | 236 | 27 | 0 | 0 |

The short shape is misleading. At the production FFN shape, wave16 regresses
by 17.36%, so it remains standalone-only and must not enter serving dispatch.
