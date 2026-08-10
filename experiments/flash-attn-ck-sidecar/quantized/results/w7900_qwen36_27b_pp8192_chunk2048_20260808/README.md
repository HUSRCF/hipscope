# W7900 Qwen3.6-27B quantized CK runtime A/B

Hardware and workload: Radeon Pro W7900 / gfx1100, Qwen3.6-27B MQ4, Asym3 K
plus Q8 V cache, PP8192, prefill chunk 2048, and eight generated tokens. Each
route ran in a fresh process; order alternated across three trials.

Median result:

```text
native: 593.9 tok/s
CK:     808.1 tok/s
delta:  1.3607x (+36.07%)
```

`results.tsv` is the primary performance record. `meta.txt` records the
binary, model, and sidecar hashes. The two correctness logs use a separate
3369-token real prompt with 256-token chunks. Their `AR tokens` arrays are
identical:

```text
[248068, 198, 8160, 579, 264, 7047, 1817, 25,
 271, 16, 13, 220, 2972, 2014, 53983, 2570]
```

The correctness run measured 580.62 tok/s native and 635.93 tok/s CK. It
validates the multi-chunk bottom-right causal route; it is not used for the
headline performance median.

`rebuild_check.tsv` is a one-trial reproducibility check made after rebuilding
the default sidecar artifact with `build_quantized_sidecar.sh`; it measured
615.6 tok/s native and 829.9 tok/s CK. It confirms that the tracked build path,
not only the temporary tuning artifact, reproduces the target range.

[`rocprof_prefill_breakdown.md`](rocprof_prefill_breakdown.md) isolates the
CK-on PP8192 prefill window from a rocprof kernel trace. MQ4 GEMM occupies
79.31% of prefill wall time, rising to 80.42% when its Q8_1 input quantization
is included. It contains the next optimization priorities and Amdahl budget.
