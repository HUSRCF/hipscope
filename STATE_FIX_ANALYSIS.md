# DeltaNet State Degradation Analysis

## Reference Implementation (flash-linear-attention)

Gate computation in `fla/layers/gated_deltanet.py:268`:
```python
g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
```

- `A_log.exp()` > 0
- `softplus(...)` > 0
- Negation → g **always ≤ 0**
- Therefore `exp(g) ∈ (0, 1]` — state always decays, never amplifies

## Hipfire Implementation

### Gate path (CORRECT)
1. `alpha_gate.hip`: computes `sp * (-exp(a_log))` → matches reference, always ≤ 0
2. `gated_delta_net_q8.hip:42`: `alpha = expf(gate)` → exp(negative) ∈ (0, 1] ✓

### Requant path (BUG)
`gated_delta_net_q8.hip:74-88`: After each token, the state S (128×128 per head)
is requantized f32 → int8 → stored. Next token dequantizes back.

**Each requantization loses precision:**
- Per-row max determines scale: `scale = max_abs / 127`
- Values near zero round to 0: `roundf(value / scale)` → 0 when |value| < scale/2
- After 500 tokens: cumulative error destroys small state elements

**Why this causes degradation at ~500 tokens:**
- The state matrix S has both large and small eigenvalue components
- Large components survive requantization (they dominate the scale)
- Small components get systematically crushed to zero by `roundf`
- After ~500 rounds: only the dominant components remain, the model loses
  fine-grained state information → outputs become repetitive/confused

## Fix: Stochastic Rounding

Replace deterministic `roundf` with stochastic rounding:
- `floor(x + uniform_random)` has E[result] = x (unbiased)
- Small values that would always round to 0 now have a probability
  proportional to their magnitude of rounding to ±1
- Eliminates systematic bias; errors cancel over many tokens
