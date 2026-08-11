# Lane-owned MQ4 metadata probe on gfx1100

This standalone exact-math probe removed the repeated `half2(scale, zero)`
plane from the packed-weight LDS tile. Each Wave32 lane loaded the original
FP32 affine header for one output row, performed the same FP16 conversion as
the production path, and supplied the selected row metadata with
`ds_bpermute_b32`. Packed weight and Q8 activation payloads remained in LDS;
the quantization and FP32 accumulation contracts were unchanged.

Device: AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. Each result is the
median of 15 in-process alternating pairs after three kernel warmups and a
five-second DPM warmup.

| Shape | Production | Lane metadata | Relative | Correctness |
| --- | ---: | ---: | ---: | --- |
| gate/set, `M=17408 K=5120 N=2048` | 4.3312 ms | 5.6458 ms | 0.7672x | bit-exact |
| down/add, `M=5120 K=17408 N=2048` | 4.4222 ms | 5.8136 ms | 0.7607x | bit-exact |

The candidate full-set code object used Wave32, 256 VGPRs, three VGPR spills,
16 bytes of private storage per thread, and 26 SGPRs. Reducing the reported
spill count by one did not offset repeated header loads and per-output lane
permutations. The original LDS metadata plane is substantially faster; the
temporary kernel and dispatch entry were removed after measurement.
