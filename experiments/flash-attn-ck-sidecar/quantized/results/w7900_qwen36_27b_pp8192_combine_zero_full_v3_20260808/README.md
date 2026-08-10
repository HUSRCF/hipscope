# W7900 PP8192 combined-zero production result

> Superseded by the barrier-safe v4 result. Independent review found that this
> candidate lacked a workgroup barrier before the next K group reused LDS.
> The measured output happened to match, but these numbers are retained only
> as development history and must not be used as production evidence.

This run explored the gfx11 HFQ4-G256 combined-zero correction in both aligned
`full_set` and `full_add` MMQ kernels. The workload is Qwen3.6-27B MQ4 on a
Radeon Pro W7900, Asym3 K plus Q8 V cache, PP8192, 2048-token chunks, and eight
decode tokens. Native and quantized-CK routes ran in alternating fresh
processes.

| route | raw prefill tok/s | median tok/s |
| --- | --- | ---: |
| native | 538.7, 605.3, 608.6 | 605.3 |
| CK plus combined-zero MMQ | 862.2, 851.5, 855.6 | **855.6** |

The CK route is stable across all three trials. Its median is 1.4135x the
native median, 5.88% above the earlier 808.1 tok/s CK median, and 1.76% above
the 840.8 tok/s result where combined-zero was active only in `full_set`.
Decode remains unchanged at 32.6-32.8 tok/s. The first native trial is a low
outlier and is retained in `results.tsv`; it does not affect the CK stability
or the comparison against the earlier CK medians.

The kernel-level A/B for aligned `full_add` measured 2.2114 ms legacy versus
2.0963 ms combined-zero, a 1.0549x speedup. Numerical error was
`max_abs=2.62e-6`, and the resource audit improved from 250 to 240 VGPR with
zero scratch for both variants. The corresponding `full_set` projection
matrix improved by 5.3%-5.5% across the model's M=5120/10240/12288/17408
shapes.

`real_prompt_ck.log` is a separate greedy correctness check using the
3369-token wrapped `docs/testINPUT.md` prompt and 256-token chunks. It emitted
the same 16 token IDs as the prior native and CK baselines:

```text
[248068, 198, 8160, 579, 264, 7047, 1817, 25,
 271, 16, 13, 220, 2972, 2014, 53983, 2570]
```

The optimization is limited to aligned gfx11 full kernels. The generic
non-aligned fallback and the separate gfx12 implementation are unchanged.
