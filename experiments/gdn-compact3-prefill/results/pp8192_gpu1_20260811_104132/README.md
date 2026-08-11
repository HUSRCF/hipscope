# Qwen3.6-27B compact3 GDN prefill probe

This rejected probe specialized the existing Q8 fast recurrence for the model's
`16 Q/K heads : 48 value/state heads` contract. State head `h` read compact
Q/K head `h / 3`, eliminating the 3x Q/K repeat-interleave materialization.
The production state and output layouts were unchanged.

One warmed PP8192 process pair on W7900/gfx1100 produced identical 32-token
greedy output IDs:

| Path | Prefill | Decode |
| --- | ---: | ---: |
| expanded baseline | 1115.2 tok/s | 32.9 tok/s |
| compact3 | 1110.5 tok/s | 32.9 tok/s |

The candidate was `0.9958x` overall. This is a single warmed-process screen,
not a multi-pair median. An earlier `944.4 tok/s` candidate sample
was the first compile of the new source specialization and is excluded.

The overall candidate does not meet the 1.10x local admission threshold, so no
runtime option or production kernel is retained. The screen supports rejecting
this complete compact3 route; it does not isolate a single microarchitectural
cause.
