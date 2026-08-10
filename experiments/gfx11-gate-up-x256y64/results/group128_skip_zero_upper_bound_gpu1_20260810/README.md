# Group128 Zero-Correction Upper Bound

This standalone timing-only ablation set every synthetic HFQ4 affine zero
metadata value to zero, then compiled a candidate that removed activation-sum
collection, zero correction, and the final per-group LDS barrier. Baseline and
candidate are therefore exactly equivalent for these inputs. Tests ran on
Radeon Pro W7900 (`gfx1100`), GPU1, with 15 alternating pairs after warmup.

| Shape | Mode | Baseline | Skip zero | Relative | Max abs | Mean abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gate/up `M17408 K5120 N2048` | set | 4.6734 ms | 4.6516 ms | 1.0047x | 0 | 0 |
| FFN down `M5120 K17408 N2048` | add | 4.7601 ms | 4.7714 ms | 0.9976x | 0 | 0 |

Removing the complete affine zero-correction path is worth less than 0.5% on
the dominant set shape and is neutral on the add shape. The real MQ4 artifact
stores arbitrary per-group minima, so production cannot remove this work
without changing the quantization contract. Do not implement a separate
low-rank correction kernel; its theoretical budget is already too small.
