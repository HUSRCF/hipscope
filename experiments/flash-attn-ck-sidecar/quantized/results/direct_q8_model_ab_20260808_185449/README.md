# Qwen3.6-27B direct MQ-Q8 bridge A/B

Radeon Pro W7900 / gfx1100, ROCm 7.14, Qwen3.6-27B MQ4 with Asym3 K and Q8 V.
Five fresh-process pairs used alternating order, three PP8192 iterations per
process, a three-second DPM warmup, and an eight-token decode sanity check.
The baseline sidecar exposes the production CK attention route; the candidate
adds the byte-exact direct MQ-Q8 projection bridge.

| Mode | Prefill tok/s, raw | Median |
| --- | --- | ---: |
| Baseline | `690.2, 684.5, 682.3, 682.9, 670.8` | `682.9` |
| Direct MQ-Q8 | `685.7, 686.5, 681.1, 680.4, 663.4` | `681.1` |

Candidate / baseline is `0.9974x` (`-0.26%`). Decode remained within
`32.7-33.1 tok/s`. The bridge was confirmed active in every candidate log and
absent in every baseline log.

The standalone bridge is 2.45x-4.91x faster and byte-exact, but its absolute
saving is only about 31 ms across the 16 full-attention layers at PP8192. That
is below 0.3% of the roughly 12-second model prefill and did not produce a
measurable production gain. The automatic runtime route is therefore rejected;
the ABI prototype and benchmark remain as a documented boundary experiment.
