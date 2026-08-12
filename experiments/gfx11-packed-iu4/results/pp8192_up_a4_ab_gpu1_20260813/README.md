# PP8192 up-only signed-A4 paired A/B

This run isolates signed-A4 activation quantization to the FFN up projection on
the current gfx1100 production prefill recipe. The Q8 control and A4 candidate
were interleaved in AB/BA order for five pairs after separate prewarming.

```text
Q8 median:                 1192.9 tok/s
up-A4 median:              1217.4 tok/s
trimmed-median gain:       1.0205x (+2.05%)
pairwise-ratio median:     1.0195x
pairwise range:            1.0172x to 1.0228x
decode median:             33.0 tok/s for both
short token IDs:           identical in all five pairs
```

The run required the staged quantized CK sidecar and checked both the sidecar
and the requested up-only route in every arm. This is a performance result, not
a production-quality approval; the separate 20-case LongBench matrix changed
the token stream in five cases.
