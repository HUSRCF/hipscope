# gfx1100 Long Generation Bug

## Symptom
All Qwen3.5 models (9B, 27B) degenerate into word salad / synonym repetition
after ~50-100 tokens of continuous generation on the 7900 XTX (gfx1100).
Short responses (< 50 tokens) and multi-turn (short per-turn) are coherent.

## Not caused by
- ds_swizzle turbo changes (Q8 path doesn't use turbo_common)
- Weight quantization (happens on both Q4 and HFQ6)
- KV cache quantization (happens on Q8 KV)
- Sampling params (happens at temp=0, temp=0.3, with repeat penalty)
- Session changes to dispatch.rs or kernels.rs (Q8 path unaffected by changes)

## Found: DeltaNet gfx1100 requant uses roundf() instead of stochastic rounding
The file `gated_delta_net_q8.gfx1100.hip` uses deterministic `roundf()` for
Q8 requantization of the recurrent state. The generic kernel explicitly warns:
> "Deterministic roundf systematically crushes small values to 0, causing
> cumulative state corruption over hundreds of tokens."

Fixed by renaming the broken variant to `.broken` so the generic kernel
(with stochastic rounding) is used instead. This changed `!!!` output to
actual English but the synonym spiral persists.

## Remaining issue
Even with the generic kernel (stochastic rounding), long generation still
degenerates. This suggests a SECOND source of numerical drift on gfx1100,
likely in one of the other kernels (GEMV, RMSNorm, RoPE, SiLU, etc.)
that accumulates over the forward pass.

## Next steps to investigate
1. Compare GEMV output between gfx1100 and gfx1010 for identical input
2. Check if gfx1100's FP32 rounding mode differs from gfx1010
3. Profile logit entropy over generation steps — does it collapse?
4. Check if the gfx1200 (RDNA4) has the same issue
5. Try forcing FP32 DeltaNet state (no Q8 requant) via FP32_STATE=1 env var
