# Exact Q8 two-plane IU4 boundary

Device: Radeon Pro W7900, `gfx1100`, selected with `HIP_VISIBLE_DEVICES=1`.

The candidate preserves the production Q8 group128 quantization contract and
represents every activation exactly as `q8 = low_u4 + 16 * high_i4`. Packed MQ4
weights feed two native IU4 WMMA passes without nibble expansion.

## Instruction probe

```text
iu8_ms=0.369553
iu4_ms=0.235775
iu4_instruction_speedup=1.567397x
exact_q8_two_pass_estimate=0.783698x
```

## Correctness smoke

Shape `M512/K512/N256`:

```text
Q8 group128:       0.0430 ms
exact Q8 IU4:      0.0819 ms
speedup:           0.5252x
max_abs_vs_q8:     1.78813934e-7
relative_l2_vs_q8: 1.46542573e-7
cosine_vs_q8:      1.0000000000
```

## Full gate/up shape

Shape `M17408/K5120/N2048`, 11 alternating pairs:

```text
Q8 group128:       4.5042 ms
exact Q8 IU4:      9.7581 ms
speedup:           0.4616x
max_abs_vs_q8:     2.38418579e-6
relative_l2_vs_q8: 2.06147413e-7
cosine_vs_q8:      1.0000000000
```

The serial-row revision reduced resources from 13 to 4 VGPR spills but did not
improve latency. Forcing a runtime row loop removed more live state but slowed
the candidate to `0.2958x`. The retained experiment therefore uses the faster
unrolled revision.

## Conclusion

The exact decomposition is numerically valid, but two IU4 WMMA passes cost more
than the saved MQ4 expansion and expanded-weight LDS traffic on this gfx1100
shape. It remains standalone-only and must not be production routed.
