# Chunked GDN F32 HIP performance gate

This result evaluates the existing source-level chunked F32 GDN PoC against
the sequential GPU oracle on Radeon Pro W7900/gfx1100. The command was:

```bash
HIP_VISIBLE_DEVICES=1 HIPFIRE_GDN_BENCH=1 \
  cargo run --release --features deltanet \
  --example gdn_chunk_parity -p rdna-compute
```

Correctness passed with max errors around `1.5e-7`. Performance did not pass:

| Tokens | Chunk | Sequential | Chunked | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0.0394 ms | 0.0895 ms | 0.44x |
| 8 | 8 | 0.0438 ms | 0.1283 ms | 0.34x |
| 16 | 16 | 0.0534 ms | 0.2766 ms | 0.19x |
| 32 | 16 | 0.0747 ms | 0.5314 ms | 0.14x |
| 32 | 32 | 0.0746 ms | 0.4734 ms | 0.16x |
| 128 | 32 | 0.2027 ms | 1.9093 ms | 0.11x |
| 256 | 32 | 0.3699 ms | 3.7586 ms | 0.10x |
| 512 | 32 | 0.7153 ms | 6.5822 ms | 0.11x |

The current implementation is retained as a correctness prototype only. A
production attempt would need a different execution decomposition rather than
further tuning of this launch/LDS/solve structure.
