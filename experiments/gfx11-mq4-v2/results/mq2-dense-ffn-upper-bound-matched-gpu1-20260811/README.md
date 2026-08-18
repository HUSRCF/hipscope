# MQ2-Lloyd dense FFN matched upper-bound screen

This standalone screen maps the existing grouped MQ2-Lloyd FP16-WMMA kernel
to one expert and one routed slot per token. The cross-format admission result
uses same-process, alternating-order paired timings against the retained gfx11
MQ4 Wave32-WMMA set primitive.

```text
GPU: AMD Radeon Pro W7900 / gfx1100
HIP: 7.14
N: 2048
processes: 2
paired samples per process and shape: 7

shape                 MQ4 median range  MQ2 median range   MQ2 / MQ4
gate/up 17408x5120    4.7123-4.7829 ms  26.3854-26.7536 ms 0.1786-0.1788x
down     5120x17408   4.8208-4.8984 ms  26.2918-26.7198 ms 0.1833-0.1834x
```

Both controls use set mode. The zero-difference correctness rows in the raw
logs compare the MQ2 single-wave and four-wave implementations of the same
synthetic MQ2 artifact; they do not establish model-level MQ2 quality or
cross-format equality with MQ4. MQ2-Lloyd was already rejected separately on
model quality after text collapse and historical 9B perplexity around 2163.

Decision: reject the current MQ2-Lloyd implementation as an MQ4-v2 execution
shortcut. The paired result does not assign the slowdown to one unisolated
mechanism and does not claim that every two-bit format must be slow.

Reproduce:

```bash
GPU_ID=1 RUNS=2 \
  ./experiments/gfx11-mq4-v2/run_mq2_dense_ffn_upper_bound.sh
```
