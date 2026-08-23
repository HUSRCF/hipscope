# Latest-beta full-stack W7900 PP8192 A/B

This result uses official `beta@80a572c8` plus the individually migrated gfx11 packed-MQ4 production stack. Both arms enable X256/Y64, permuted group128 quad-row weights, fused SwiGLU Q8 packing, FP16 FFN intermediates, and the group256 serial-row fallback. The CK arm additionally enables the staged Asym3-K/Q8-V attention sidecar.

| Mode | Process medians (tok/s) | Median (tok/s) |
| --- | --- | ---: |
| Full packed-MQ4, native attention | `756.1`, `731.4`, `748.4` | **748.4** |
| Full packed-MQ4, staged CK attention | `1225.8`, `1213.8`, `1221.5` | **1221.5** |

The paired speedup median is **1.6321x** with 3/3 positive pairs. Every pair produced identical greedy token IDs. Decode remained neutral at `35.0-35.2 tok/s`; these routes target prefill.

Each process executes three PP8192 prefill runs and reports its in-process median. Process order alternates, with a ten-second idle interval between arms. Reproduce with `scripts/bench-gfx11-ck-prefill-ab.sh`; the script carries the complete admitted production flag set.
