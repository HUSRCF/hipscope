# gfx11 MQ4-v2 experiments

This directory is reserved for execution-format experiments that satisfy
`docs/plans/gfx11-mq4-v2-execution-format.md`.

Do not add another tile-only variant of the retained group128 quad-row kernel.
Every candidate must state which numerical or execution-format contract it
changes, its resident-byte overhead, and the packed-MQ4 wall share it can
plausibly affect.

P0, a symmetric signed-int4 weight contract with no affine zero-point
correction, was rejected at 1.027x gate/up and 1.039x down. Existing HFQ4G128
is not such a contract. P1 is a bounded-correction IU4 activation path only if
the correction remains sparse and the combined kernel clears the performance
and quality gates.

The existing Q8_0 Wave32 WMMA backend was also rejected as a high-memory
execution copy: it used 2x the weight bytes and reached only 0.279x gate/up and
0.310x down relative to retained MQ4.

The first accepted probe must cover both production FFN shapes:

```text
gate/up: M=17408 K=5120 N=2048 set
down:    M=5120  K=17408 N=2048 add
```

Required output:

```text
reference_ms
candidate_ms
speedup
max_abs
relative_l2
cosine
resident_bytes_per_weight
dynamic_lds_bytes
vgpr_count
vgpr_spill_count
```

Candidates below 1.30x on either large FFN shape stop at standalone.

Passing the local speed threshold is necessary but not sufficient for routing.
Exact candidates must also match the retained output tolerance. Approximate
candidates must pass long-prompt generation and a task-level quality suite.
Every promoted candidate then needs a PP8192 ABBA test with identical routing
apart from the backend under test, plus an explicit execution-format memory
accounting. The retained backend remains the fallback and A/B reference.
