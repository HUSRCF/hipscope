# Direct rotated-Q directional rejection

This experiment moved Qwen's FP32 Givens rotation into the CK Q buffer view,
eliminating the separate rotate-and-store kernel. Both GPUs had resident jobs,
so timings are suitable only for rejecting this direction, not for a positive
performance claim.

At Q=128/K=8192, three alternating pairs produced:

- separate Q bridge median: `2.01729631 ms`;
- direct rotated-Q median: `2.04782534 ms`;
- direct/separate: `1.0151x` (`1.51%` slower).

Both paths had the same `2.14353204e-05` maximum absolute error against the
native Asym3/Q8 attention output. Direct Q reduced private storage from 396 to
368 bytes and VGPR spills from 101 to 91, but increased SGPR use from 47 to 54.
The implementation and build switch were removed.
