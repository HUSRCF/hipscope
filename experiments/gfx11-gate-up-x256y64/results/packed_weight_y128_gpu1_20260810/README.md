# Packed-weight Y128 probe

This standalone gfx1100 probe tested a single-output `X256/Y128`, 16-wave
workgroup that keeps HFQ4 weights packed in LDS and expands each WMMA fragment
in registers. It preserves the Q8-group128 activation contract.

| Shape | Retained X256/Y64 | Packed Y128 | Relative | Correctness |
| --- | ---: | ---: | ---: | ---: |
| M512/K512/N256 | 0.0417 ms | 0.0696 ms | 0.598x | exact |
| M17408/K5120/N2048 | 4.4993 ms | 8.1051 ms | 0.555x | exact |

The candidate compiles to 234 VGPR, 28 SGPR, zero scratch, and zero spills.
The regression therefore comes from repeatedly unpacking MQ4 at every fragment
consumer, not resource spilling. Keeping the expanded weight tile in LDS is
the better organization for the current single-output kernel.
