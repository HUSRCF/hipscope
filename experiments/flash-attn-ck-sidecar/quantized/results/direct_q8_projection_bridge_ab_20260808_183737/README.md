# Direct MQ-Q8 projection bridge A/B

W7900 / gfx1100, ROCm 7.14, seven paired trials per row count. The baseline
materializes FP32 attention output, applies the sigmoid gate, performs the MQ
FWHT, and quantizes to the projection kernel's Q8_1 layout in separate passes.
The candidate reads CK's FP16 output and emits that Q8_1 layout directly.

| Rows | Baseline median | Fused median | Paired speedup median |
| ---: | ---: | ---: | ---: |
| 128 | `0.033771 ms` | `0.013864 ms` | `2.451x` |
| 512 | `0.098684 ms` | `0.022916 ms` | `4.273x` |
| 2,048 | `0.445144 ms` | `0.090569 ms` | `4.906x` |
| 8,192 | `2.589886 ms` | `0.624394 ms` | `4.150x` |

All 28 comparisons are byte-exact in the quantized payload and exact in the
four Q8_1 `d`/`sum` metadata groups: `q_mismatches=0`, `max_d_abs=0`, and
`max_sum_abs=0`. The gfx1100 code object reports wave32, 42 VGPR, 18 SGPR,
zero VGPR/SGPR spill, and zero fixed private storage for the fused kernel. This
establishes the bridge operator itself; model-level value is evaluated
separately by `run_direct_q8_model_ab.sh`.
