# gfx11 trained FFN-v2 admission plan

Status: direction gate. This is a new approximate-model profile, not a
transparent serving optimization and not a continuation of tile tuning.

## Why this direction exists

The retained Qwen3.6-27B prefill path uses Wave32 WMMA and is mature at about
1.12k tok/s, with a best controlled configuration near 1.19k tok/s. The
lossless MQ4-v2 admission matrix has already screened execution copies,
alternate packed layouts, row-scale contracts, smaller bit widths, current CK
A16 x I4, and a dense-FP16 rocBLAS ceiling. Dense FP16 reached only 1.138x on
gate/up and 1.092x on down. Applying that optimistic local ceiling to the whole
packed-MQ4 wall share projects only about 1.30k tok/s.

The active-width timing sweep establishes a different boundary. Exact
measurements show that 41/68 groups provide 1.6609x weighted FFN speedup and
project only 1477 tok/s from the 1189 tok/s controlled baseline. A five-pair
fresh-process ABBA test puts 39/68 at 1.7378x and projects 1501.33 tok/s.
Transparent pruning is not
admissible: the 41/68 dynamic oracle
increased perplexity by 9.25% on the short document and 52.79% on WikiText2.
The 60/68 static artifact reached 1303 tok/s but increased fixed-window
perplexity by 20.05%.

Therefore 1.5k tok/s is no longer a lossless-backend target. It requires a new
checkpoint contract with structured FFN work reduction and quality recovery
through reconstruction, distillation, or retraining.

## Model and resource boundary

The target language model has 64 layers, hidden size 5120, and FFN width
17408. Its three dense FFN matrices contain approximately:

```text
full layer:  3 * 5120 * 17408 = 267.39M parameters
41/68 layer: 3 * 5120 * 10496 = 161.22M parameters
39/68 layer: 3 * 5120 *  9984 = 153.35M parameters
full FFNs:   17.11B parameters across 64 layers
39/68 FFNs:   9.81B parameters across 64 layers
```

A conventional full-model AdamW run is not admitted on one W7900: optimizer
state for the pruned FFNs alone is far beyond device memory. A sequential
layerwise reconstruction probe is feasible. One 41/68 FFN block is about
322 MB in BF16. BF16 parameters and gradients, FP32 optimizer moments and an
optional master copy, one full-width FP8 teacher block, and bounded activation
batches are budgeted at no more than 8 GiB peak device memory. The four-layer
pilot trains layers sequentially rather than keeping four optimizer states
resident.

The available source artifact is the 29 GB Qwen3.6-27B FP8 checkpoint at:

```text
/home/husrcf/Code/ProtBind/MTP/data/modelscope_downloads/Qwen/Qwen3.6-27B-FP8
```

It stores one language layer per safetensors shard and exposes FP8 weights plus
inverse scales for gate, up, and down. This is suitable for one-layer-at-a-time
initialization. It is not a BF16 quality reference, so final admission still
requires comparison against the retained production model and, if available,
an official higher-precision reference.

## First probe: one-layer teacher reconstruction

The first implementation must remain outside the default serving route.

1. Capture real hidden input `x` and full-width teacher FFN output `y` for one
   selected layer from at least two corpora. Capture one layer per run and cap
   the dataset at 8192 tokens, approximately 160 MiB for FP16 `x` plus `y`.
2. Initialize a 41/68 screen and a 39/68 target student from the source FP8
   gate/up/down weights. Rank complete 256-channel groups with the existing
   deterministic gate/up/down weight-energy score in
   `prune_dense_ffn_groups`; write the selected group IDs to the manifest and
   freeze them before training. Held-out data must not select groups. The wider
   screen is an early quality-recovery gate; it is not the final performance
   profile. Any activation-aware or learned selector is a separately named
   ablation and cannot replace this initialization after seeing held-out data.
3. Optimize only the selected layer's student FFN to reconstruct `y`, using
   4096 training tokens, at least 1024 held-out tokens from a different corpus,
   and a fixed manifest of optimizer, learning-rate, step, and batch settings.
   Keep the surrounding model frozen and off device.
4. Quantize the reconstructed block to the existing MQ4 contract and repeat
   the held-out reconstruction check through the production-equivalent
   group128 Q8 activation, fused SwiGLU, rotated down-input, and residual
   dataflow. FP16 block reconstruction alone is not serving evidence.
5. Only after one layer passes, repeat on four representative layers: early,
   middle GDN, middle full-attention, and late. A full 64-layer conversion is
   not admitted before this gate.

The capture path must be diagnostic and default-off. It must identify the
model SHA-256, layer index, tokenizer/corpus fingerprint, tensor shape, dtype,
and token count in a manifest. Raw captures must not be committed.

## Admission and stop conditions

The one-layer probe proceeds only if all of the following hold:

- the measured 39/68 standalone result remains at least 1.70x for the
  gate/up/down weighted primitive aggregate;
- the 41/68 screen reduces held-out FFN-output relative L2 error by at least
  50% versus its untrained initialization, after which the 39/68 target must
  independently meet the same relative-improvement gate;
- the improvement survives MQ4 requantization;
- the trained artifact preserves the 256-channel group and runtime stride
  contracts;
- each sequential pilot layer remains below 8 GiB peak device memory and the
  complete four-layer pilot remains below 24 GPU-hours.

The full conversion is admitted only after all of these production gates pass:

- PP8192 uses at least five alternating fresh-process pairs, identical runtime
  flags and prompt bytes, and reports the paired median plus every raw round;
- the trained 39/68 artifact reaches at least 1450 tok/s at pilot admission and
  1500 tok/s before promotion;
- loaded VRAM and checkpoint bytes are reported and neither may exceed the
  retained 68/68 model;
- the fixed project corpus and WikiText2 use their existing matched-position
  evaluators and show no more than 2% PPL regression versus the retained MQ4
  baselines;
- LongBench-hard30 loses no correct answer, and the fixed GSM8K-100 and
  HumanEval+ suites each lose at most one passing item versus the retained MQ4
  baseline; prompts, decoding parameters, and scorer revisions are pinned in
  the result manifest;
- greedy token IDs, runtime route, and fallback counters are recorded; the
  candidate must use the same attention, GDN, and non-FFN projection routes,
  must add no generic FFN fallback, and must not increase any fallback counter
  relative to the retained baseline.

A looser quality target must be published as a separate approximate model
profile, never silently selected by the runtime.

Stop this direction if the one-layer MQ4 student fails to halve held-out
reconstruction error, or if the four-layer pilot shows compounding quality
loss that cannot meet the 2% PPL gate. In that case the measured lossless
backend ceiling should be reported as roughly 1.25-1.30k tok/s on this W7900,
and the 1.5k target should be retired rather than pursued through more tile or
layout variants.

## Explicit non-goals

- No serving default change before a separately identified checkpoint passes
  quality gates.
- No claim that the two rejected Wave32 WMMA probes reject WMMA generally.
- No further tile, barrier, LDS, or prefetch sweep under the retained MQ4
  contract without a mechanism that can exceed the measured dense roofline.
- No whole-model training job before the one-layer and four-layer gates pass.
