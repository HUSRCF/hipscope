# W7900 PP8192 barrier-safe combined-zero result

This is the final production validation of the gfx11 HFQ4-G256 combined-zero
correction in both aligned `full_set` and `full_add` MMQ kernels. It includes a
workgroup barrier after the correction so all eight waves finish consuming
`tile_x` and `zero_sums` before the next K group reuses LDS.

The workload is Qwen3.6-27B MQ4 on a Radeon Pro W7900, Asym3 K plus Q8 V cache,
PP8192, 2048-token chunks, and eight decode tokens. Native and quantized-CK
routes ran in alternating fresh processes.

| route | raw prefill tok/s | median tok/s |
| --- | --- | ---: |
| native | 531.1, 608.1, 610.5 | 608.1 |
| CK plus barrier-safe combined-zero MMQ | 863.4, 850.5, 855.0 | **855.0** |

The CK median is 1.4060x the native median and 5.80% above the earlier 808.1
tok/s CK median. Decode remains unchanged at 32.7-32.9 tok/s. The first native
trial is a low outlier and is retained in `results.tsv`; all three CK runs stay
within 1.5% of their median.

Final gfx11 resource metadata for both `full_set` and `full_add` is 240 VGPR,
zero VGPR/SGPR spills, and zero private segment bytes. The aligned route uses
an additional 512 B of LDS. The generic non-aligned fallback and separate
gfx12 implementation are unchanged.

`real_prompt_ck.log` is a separate greedy correctness check using the
3369-token wrapped `docs/testINPUT.md` prompt and 256-token chunks. It emitted
the same 16 token IDs as the prior native and CK baselines:

```text
[248068, 198, 8160, 579, 264, 7047, 1817, 25,
 271, 16, 13, 220, 2972, 2014, 53983, 2570]
```

That run measured 634.4 prefill tok/s and 35.71 decode tok/s. It validates the
multi-chunk numerical path; it is not part of the PP8192 performance median.
