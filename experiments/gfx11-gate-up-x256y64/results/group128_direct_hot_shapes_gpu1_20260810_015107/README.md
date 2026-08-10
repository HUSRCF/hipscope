# Exact Group128 Direct-Activation Hot-Shape Sweep

This standalone W7900 (`gfx1100`) sweep preserves the production group128 Q8
format but removes the full activation LDS tile. A non-unrolled K128-half loop
keeps the kernel spill-free (`140 VGPR`, `34 SGPR`, zero private bytes).

| Shape | Mode | LDS baseline | Direct activation | Relative | Max abs |
| --- | --- | ---: | ---: | ---: | ---: |
| gate/up M17408 K5120 | set | 4.9492 ms | 4.7759 ms | 1.0363x | 1.43e-6 |
| QKVZA M10240 K5120 | set | 2.9918 ms | 2.8915 ms | 1.0347x | 1.43e-6 |
| GDN out M6144 K5120 | set | 1.8185 ms | 1.7783 ms | 1.0226x | 1.43e-6 |
| attention QKV M12288 K5120 | set | 3.5379 ms | 3.4059 ms | 1.0388x | 1.43e-6 |
| FFN down M5120 K17408 | add | 5.0801 ms | 4.9435 ms | 1.0276x | 5.72e-6 |
| auxiliary down M5120 K6144 | add | 1.8616 ms | 1.8365 ms | 1.0136x | 1.91e-6 |

All standalone hotspots improve, but the production PP8192 A/B in
`../pp8192_group128_direct_ck_gpu1_20260810_015302/` regresses. The standalone
gain is therefore insufficient evidence for serving promotion.
