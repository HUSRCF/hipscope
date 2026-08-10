# Group256 Serial-Row Real-Prompt Quality Boundary

This test compared the production group128 path with the opt-in group256
serial-row path using the same 3371-token prompt and 128 requested greedy output
tokens on Radeon Pro W7900 (`gfx1100`), GPU1.

| Metric | Result |
| --- | ---: |
| Group128 output IDs | 129 |
| Group256 output IDs | 129 |
| Common prefix | 93 tokens |
| First differing output token | 93 |

The long common prefix shows that the route is numerically close, but the
eventual greedy divergence is sufficient to reject it as the default
production path. Its PP8192 performance result is retained only as evidence
that sharing one activation scale across 256 K elements reduces hot-kernel
overhead; an exact group128 implementation must preserve the original scaling
contract.
