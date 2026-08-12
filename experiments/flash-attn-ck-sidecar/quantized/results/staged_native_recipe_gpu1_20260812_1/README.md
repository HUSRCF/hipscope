# FA4 gfx11 D256 component A/B

Ten alternating process pairs compare the pinned dense CK sidecar with the
FA4 D256 candidate at Q=2048, 24 query heads, 4 KV heads, and D=256 on Radeon
Pro W7900 / gfx1100. `raw.csv` retains every process result.

| K | pinned CK median | FA4 CK median | median ratio |
| ---: | ---: | ---: | ---: |
| 2,048 | 1.404482 ms | 1.281179 ms | 1.0962x |
| 4,096 | 3.364502 ms | 2.930388 ms | 1.1481x |
| 6,144 | 5.455183 ms | 4.734181 ms | 1.1523x |
| 8,192 | 7.869337 ms | 7.138237 ms | 1.1024x |

The aggregate paired median is 1.1268x, all 10 pairs are positive, and every
candidate row reports `max_abs=0` against its packed-route reference.
