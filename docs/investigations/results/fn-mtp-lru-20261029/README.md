# Native MTP with the exact Flash-Next expert LRU

This follow-up builds on frozen PR #1022 at `2c1c46365e9f5da9fd361dfccdeef0e8840c4846`.
The existing native MTP executor and expert-cache transport are extended with
one stable reader per bounded verify batch. No throughput result is claimed here.

Current status: 207 scoped product/regression CPU test executions and eight
Rust oracle controls pass. GPU/model qualification is pending; CPU oracles do
not qualify an execution backend.

`fn_mtp_lru_oracle` requires `lab,reference-parity`. Its default is plan-only;
`--cpu-controls` opens no GPU. Explicit collector execution uses tiny fixtures
and the production collector. The separate existing
`fn_expert_cache_publication --execute` exercises paired async transport.

The control (cache off) runs first, followed by cache on, in separate fresh
processes. The model arm uses the product loader, the same K7 resource reserve in both
cache modes, and real forced native K3 windows. It compares target R1/2/4/8
logits/state, native token IDs, completion/publication tickets, terminal repair
and prefix restore. Different head schedules are diagnostic; cache-on/off head
hashes are comparable only after their actual native schedules match.
See `EXACTNESS-POLICY.json` for the precise domain and exclusions.

The developer runner and independent receipt verifier are under `tools/`.
Run no-model controls before model arms, retain every failed attempt, and hold
the shared project GPU lock. The runner never retries or overwrites proof.
Its timings are instrumented and are not performance evidence.
