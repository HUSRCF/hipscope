# Exp #8: hipGraph multi-node launch-overhead microbench

**Date:** 2026-05-07
**Status:** PRE-REGISTRATION (criterion locked before treatment)

## Hypothesis under test

Per Kaden's design note: when emulating WMMA via meta-instruction macros that expand to a sequence of N small kernels, capturing that sequence as a multi-node hipGraph should amortize per-launch dispatch overhead so the cluster does not cost as much as N sequential native launches.

This is distinct from the PR3 verdict (single-kernel-per-graph capture) and the BC-250 monolithic PoC (whole-forward-pass single graph). The intermediate granularity (N small launches per graph, where N is a small integer like 4-16) is empirically untested on RDNA1.

## Lever

Wrap N tiny kernel launches as one captured hipGraph, replay N times in a loop, vs the same N native sequential launches in a loop, on identical conditions.

## Scenario

- Hardware: hipx, single RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1).
- Kernel: minimal-work kernel — single thread writes a single value to a buffer (`p[i] = (float)i;`). Block [1,1,1], grid [1,1,1]. Maximum exposure of launch overhead vs kernel work.
- N values to test: 4, 8, 16, 32, 64, 128 launches per "macro cluster."
- TRIALS per (N, mode): 1000 iterations to amortize timing noise.
- Synchronize after each iteration to measure end-to-end cost.

## Win criterion (pre-registered)

Multi-node graph form's per-cluster wall time is at least 25% faster than native sequential at N=16 or above. (At small N like N=4, graph capture overhead may dominate; at large N, amortization should clearly win if the hypothesis is correct.)

Specifically:
- WIN: `t_graph(N) / t_native(N) <= 0.75` at N=16, AND ≤ 0.6 at N=64.
- LOSS: `t_graph(N) / t_native(N) >= 1.05` at any N (graph slower).
- NO_CHANGE: between 0.75 and 1.05 at N=16; cluster amortization is real but small.

## Quality gate

Both forms must produce identical buffer contents post-launch (deterministic kernel writes are reproducible).

## Action on win

This validates the WMMA-emulation hypothesis architecturally. Ship a small note to the project memory; document that multi-node graph capture is a real lever for fixed-sequence kernel macros even on RDNA1, in contrast to the PR3 (single-kernel) and BC-250 (monolithic forward-pass) verdicts. Future WMMA-emulation kernel work should use this pattern.

## Action on loss / no-change

Confirm that the BC-250 + PR3 generalization extends to small-N multi-node graphs as well. Document. The structural cause (graph-boundary sync per atomic graph_launch + native burst-mode pipelining superiority) holds at all granularities tested. Don't pursue WMMA-emulation as a graph-amortized lever on RDNA1.

## Implementation note

A new example `crates/rdna-compute/examples/bench_graph_launch_overhead.rs` will be created. It uses the same graph capture pattern as `hip_graph_poc.rs`: `stream_begin_capture` → N `launch_kernel_blob` calls → `stream_end_capture` → `graph_instantiate` → loop of `graph_launch + stream_synchronize`. Kernarg blobs are allocated as `Box`-leaked to outlive the captured graph (per `hip_graph_poc.rs` documented gotcha).
