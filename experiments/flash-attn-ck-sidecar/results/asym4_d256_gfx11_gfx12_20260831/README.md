# Asym4 D256 CK validation

This record validates the optional Asym4-Givens/FWHT K plus Q8 V loader and
CK attention route. It does not claim an end-to-end speedup because the native
Asym4 prefill baseline on this revision fails before a valid A/B can complete.

## Raw ABI correctness

The same sidecar source was built separately for `gfx1100` and `gfx1201` and
run with `smoke_raw_abi`.

| GPU target | Cell | Max absolute error | Mean absolute error |
| --- | --- | ---: | ---: |
| `gfx1100` | Asym4-Givens D256 GQA causal | `5.862117e-05` | `1.000180e-05` |
| `gfx1100` | Asym4-FWHT D256 GQA causal | `6.847084e-05` | `1.012444e-05` |
| `gfx1201` | Asym4-Givens D256 GQA causal | `5.862117e-05` | `1.000134e-05` |
| `gfx1201` | Asym4-FWHT D256 GQA causal | `6.847084e-05` | `1.012253e-05` |

Both targets reported a 65,536-byte workspace for the smoke shape. Existing
dense, Q8, and Asym3 cells also passed in the same runs.

## Production-path smoke

Configuration: Radeon Pro W7900 (`gfx1100`), Qwen3.6-27B MQ4, Asym4 KV,
caller-owned 512 MiB transient workspace, CK sidecar enabled.

| Prompt | Runs | Prefill throughput | Result |
| ---: | ---: | --- | --- |
| 2048 | 1 | `860.0 tok/s` | CK route selected; prefill and decode completed |
| 8192 | 5 | `833.8, 825.1, 816.2, 808.2, 801.1 tok/s` | all prefill runs completed |

The PP8192 median was `816.2 tok/s`. The monotonic drift makes this a
stability result, not a performance claim.

## Native baseline blocker

With the CK route absent or forced off, the same production binary fails at
both PP2048 and PP8192 with `hipError 700` reported by the next H2D copy. A
binary built without the `flash-attn-ck` feature reproduces the PP2048 failure.
The CK-enabled PP2048 run completes and logs
`selected_asym4_givens_d256`, so the failure is outside the optional loader and
prevents a valid native-versus-CK performance comparison on this revision.
