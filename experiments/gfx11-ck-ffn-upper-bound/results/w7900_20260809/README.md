# W7900 result, 2026-08-09

Hardware was an AMD Radeon Pro W7900 (`gfx1100`). GPU1 was idle before the
run. Each process performs the CK example's internal warmup before timing.

| path | CK INT8 median | production packed-MQ4 estimate | apparent CK delta |
| --- | ---: | ---: | ---: |
| gate or up projection | 5.157 ms | 5.269 ms | -2.1% |
| down + residual projection | 5.270 ms | 5.471 ms | -3.7% |

Production estimates come from the PP8192 profile:

- gate/up: `2697.5 ms / 512 calls = 5.269 ms/call`
- down/residual: `1400.6 ms / 256 calls = 5.471 ms/call`

The comparison favors CK because the CK example does less work and uses an
unpacked INT8 weight matrix. Its small apparent lead is insufficient to cover
MQ4 unpacking/scales, the correct output type, and residual fusion. This is a
negative feasibility result for a generic CK replacement, not a claim that
the two kernels implement identical arithmetic.
