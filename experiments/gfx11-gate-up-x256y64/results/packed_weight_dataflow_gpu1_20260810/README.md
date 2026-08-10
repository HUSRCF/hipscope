# Packed HFQ4 dataflow probes on gfx1100

Device: AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14.

Shape: `M=17408, K=5120, N=2048`, set output, 15 alternating pairs. The reference is the retained exact group128 `X256/Y64` kernel.

| Candidate | Reference ms | Candidate ms | Speedup | max_abs | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct global packed weight, N2 fragment reuse | 4.5262 | 10.7542 | 0.4209x | 7.78e-4 | Reject |
| Cooperative packed-LDS weight, Y64 | 4.5208 | 9.3749 | 0.4822x | 0 | Reject |

The direct-global prototype first exposed an affine metadata indexing bug: one scale/zero value had been reused for all eight accumulator rows in a lane. Loading metadata per accumulator row removed the large error. Its remaining small difference is expected because the diagnostic path retains FP32 metadata while the reference stages FP16 metadata.

The exact packed-LDS Y64 result isolates the dataflow tradeoff without changing the production tile topology. Reducing weight LDS by about 11 KiB does not offset repeated register-side nibble expansion in the WMMA loop. On this W7900 gate/up shape, the retained expanded-i8 LDS path is more than twice as fast, so neither candidate is suitable for that serving route.

Resource audit for the corrected direct-global full-set kernel: 256 VGPRs, 28 VGPR spills, 116 bytes private segment, wave32. This independently explains part of the regression and closes the direct packed-weight line.
