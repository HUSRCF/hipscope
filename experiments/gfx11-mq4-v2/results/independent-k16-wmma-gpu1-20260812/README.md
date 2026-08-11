# Independent K16 WMMA probe on gfx1100

This standalone probe tested whether the two K16 operations used for each K32
MQ4 dot product could overlap better if they accumulated independently. The
candidate retained the production Q8-group128 and affine MQ4 contracts. It
computed two signed IU8 Wave32-WMMA results, then combined the two `i32`
accumulators before applying the unchanged FP32 scale and zero correction.

Device: AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. Each result is the
median of 15 in-process alternating pairs after three kernel warmups and a
five-second DPM warmup.

| Shape | Production | Independent K16 | Relative | Correctness |
| --- | ---: | ---: | ---: | --- |
| gate/set, `M=17408 K=5120 N=2048` | 4.4403 ms | 4.9968 ms | 0.8886x | bit-exact |
| down/add, `M=5120 K=17408 N=2048` | 4.5701 ms | 5.1118 ms | 0.8940x | bit-exact |

The candidate full-set code object remained Wave32 with 256 VGPRs, four VGPR
spills, 20 bytes of private storage per thread, and 27 SGPRs. The additional
temporary accumulator and vector integer merge therefore did not create a new
reported spill class, but their instruction/live-range cost exceeded any
benefit from removing the K16 accumulator dependency. The temporary kernel and
dispatch entry were removed after measurement; this architecture is closed.
