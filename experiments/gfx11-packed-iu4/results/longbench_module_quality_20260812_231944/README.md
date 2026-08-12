# Gate/up signed-A4 LongBench screening

This matrix compares the production Q8 activation path with gate-only and
up-only signed-A4 on 20 fixed long-context multiple-choice prompts. All modes
use the same model, asym3 KV cache, staged quantized CK sidecar, closed-think
assistant framing, and greedy decoding. The dataset SHA-256 is recorded in
`artifacts.sha256`.

```text
mode       accuracy    token-exact vs Q8    answer-state agreement vs Q8
Q8        8/20        20/20                20/20
gate-A4   7/20        16/20                17/20
up-A4     8/20        15/20                17/20
```

Two counting-heavy Q8 cases had not emitted a final choice at the initial
24-token cap. A targeted 128-token rerun still had no Q8 final choice, so those
rows are retained as truncated rather than assigned an inferred answer. Gate-A4
has one unambiguous net regression: case 12 changes the correct Q8 answer `D`
to incorrect `A`. Up-A4 ties the small-sample accuracy but changes five token
streams, so neither candidate is approved as a default production route.
