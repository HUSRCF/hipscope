# gfx11 dense-FFN group-pruning probe

This experiment asks how much Qwen3.6-27B prefill can improve by reducing dense
FFN work while preserving hipfire's 256-channel MQ rotation boundary. It is an
experimental model transformation, not a numerically equivalent kernel change.

## Method

`prune_dense_ffn_groups` scores each complete 256-channel FFN group from the
quantized gate, up, and down weights, then rewrites matching gate/up rows and
down columns into a smaller HFQ artifact. The original model has 68 groups
(`intermediate_size=17408`). The runtime fast path remains opt-in through
`HIPFIRE_RDNA3_FFN_VARIABLE_WIDTH=1` and only admits 256-aligned widths.

Build and create an artifact:

```bash
cargo build --release -p hipfire-quantize --example prune_dense_ffn_groups
target/release/examples/prune_dense_ffn_groups \
  --input "$HOME/.hipfire/models/qwen3.6-27b.mq4" \
  --keep-groups 60 \
  --output /tmp/qwen3.6-27b-ffn60g.mq4
```

## W7900 results

GPU1, gfx1100, ROCm 7.14, Asym3 KV, quantized CK attention sidecar, PP8192,
seven rounds after a five-second DPM warmup:

| model/path | width | loaded size | PP8192 median | last round |
| --- | ---: | ---: | ---: | ---: |
| original fast path | 17408 | 13.955 GiB | 1214.9 tok/s | 1198.7 tok/s |
| 60/68, generic fallback | 15360 | 12.959 GiB | 1045.7 tok/s | 1041.9 tok/s |
| **60/68, variable-width fast path** | **15360** | **12.959 GiB** | **1303.0 tok/s** | **1287.4 tok/s** |

The valid fast-path comparison is **+7.25% median**. The generic result shows
that changing model width without preserving the production FFN dataflow is a
performance regression despite doing less arithmetic.

Basic greedy checks for the 60/68 artifact produced the correct result for
`17 * 23`, the capital of France, and a simple Python function. This is only a
functional gate; it is not a quality benchmark. On a fixed 512-token window
from `docs/testINPUT.md`, the original model scored `PPL=4.2070` while 60/68
scored `PPL=5.0505` (`+20.05%`). The 60/68 artifact is therefore a structural
performance bound, not a production candidate. More aggressive weight-only
selection also failed the basic gate: 51/68 answered raw `17 * 23 =` with
`439`, and 40/68 produced an incorrect multiplication result.

The complete timing matrix can be reproduced with `MODE=all`; use
`MODE=original`, `fallback`, or `pruned` to run one row. The converter rejects
non-canonical MQ4G256 payloads instead of silently applying its fixed 256-value
group geometry to incompatible tensors.

## Full trace

An unprofiled `rocprofv3` trace of 60/68 covered the prefill interval from the
first `embedding_q8_batched` dispatch through the fourth prefill
`gemv_hfq4g256` dispatch:

| component | GPU ms | prefill span |
| --- | ---: | ---: |
| gate/up FP16-output MQ4 | 2198.440 | 35.30% |
| down/residual group128 MQ4 | 1145.606 | 18.39% |
| group256 set MQ4 | 1045.567 | 16.79% |
| group256 residual MQ4 | 428.363 | 6.88% |
| GDN fast core | 477.950 | 7.67% |
| CK attention compute | 287.814 | 4.62% |
| all serialized kernels | 6167.888 | 99.02% |
| inter-dispatch gaps | 60.812 | 0.98% |
| **GPU span** | **6228.700** | **100%** |

The internal `HIPFIRE_PROFILE` table covered only 2089 ms of a 6292 ms wall
run because several external/fast kernels are not wrapped by its timers. That
untracked time is not startup or CPU overhead. The full trace rejects a host
submission optimization: the natural asynchronous path is already 99% covered
by GPU kernels.

At 1303 tok/s, reaching 1500 tok/s requires a 1.151x overall speedup. With
77.36% of the current wall in packed-MQ4, that requires about 1.204x across the
whole packed family. Restricting the optimization to gate/up/down requires
about 1.324x. Further width reduction is not admissible without activation
calibration and quality recovery.
