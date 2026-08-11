# X128/Y128 with retained quad-row loader on gfx1100

This standalone probe isolates activation-tile geometry from the packed-weight
loader. The candidate combines the balanced `X128/Y128` Wave32-WMMA geometry
with the retained group128 quad-row `u32x2` loader. Serving dispatch was not
changed.

Environment: AMD Radeon Pro W7900 (`gfx1100`), HIP 7.14, GPU1. The reference is
the production `X256/Y64` group128 quad-row path. Each full-shape result is the
median of 15 in-process alternating pairs after kernel warmup and a five-second
DPM warmup.

| Shape | Reference | Candidate | Relative | Correctness |
| --- | ---: | ---: | ---: | --- |
| gate/up set, `M=17408 K=5120 N=2048` | 4.4780 ms | 4.4981 ms | 0.9955x | bit-exact |
| down add, `M=5120 K=17408 N=2048` | 4.5610 ms | 4.6309 ms | 0.9849x | bit-exact |

The candidate code object is Wave32. Both full-set and full-add entrypoints use
228 VGPRs, zero VGPR/SGPR spills, and a zero-byte private segment. The negative
result is therefore not explained by scratch traffic. Reusing the retained
loader removes the confound in the earlier X128/Y128 comparison, but the taller
output tile still does not improve either production FFN shape.

This result is far below the MQ4-v2 `1.30x` admission threshold and closes the
balanced-tile activation-reuse direction for the current packed-MQ4 contract.
Do not add serving routing for this candidate. The temporary standalone
dispatch was removed after recording this rejection; the exact compile-time
configuration is preserved below.

Probe configuration:

```text
MMQ_X=128
MMQ_Y=128
MMQ_ROW_FRAGS=2
MMQ_COL_GROUPS=2
MMQ_PERM_NIBBLE=1
MMQ_Q8_GROUP128=1
MMQ_WEIGHT_QUAD_ROW_U32X2=1
```
