# gfx1201 DFlash S1-S6 E2E evidence

## Scope

- Commit: `2814cceb9` (`perf(dflash): port fa batch fusions to gfx1201`)
- Host: `X570`
- GPU: AMD Radeon AI PRO R9700 (`gfx1201`, 34.2 GB VRAM)
- HIP runtime: `7.14`
- Target KV: Q8, contiguous backend
- Candidate: S1-S6 defaults enabled on the validated gfx1201 path
- Control: all six fusions disabled with their existing `*_OFF=1` environment controls

The control disables:

```text
HIPFIRE_DN_SNAPSHOT_BULK_OFF=1
HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1
HIPFIRE_MQ_F16_PROJECTION_OFF=1
HIPFIRE_MQ_F16_RESIDUAL_OFF=1
HIPFIRE_GDN_PRE_FUSE_OFF=1
HIPFIRE_FA_BATCH_FUSE_OFF=1
```

## Production E2E results

### Qwen3.5-9B MQ4 + matched DFlash draft

The five-genre battery completed under candidate, control, and ordinary AR. Candidate and control DFlash transcripts were byte-identical. A multi-turn chain also completed with real prefix-cache hits (`456` and `601` cached tokens), and its candidate/control transcript was byte-identical.

An 8K NIAH fixture produced a 5,489-token request and recalled `mauve-velociraptor-7741` under both arms (`recall=1/1`, byte-identical output).

For the fixed 3,798-token prompt and 512-token response, after excluding each arm's first JIT-contaminated run:

| Metric | S1-S6 off | S1-S6 on | Delta |
|---|---:|---:|---:|
| Prefill median | 2,034.7 tok/s | 2,094.5 tok/s | +2.94% |
| Decode median | 65.5 tok/s | 67.9 tok/s | +3.66% |

Warm samples:

```text
off prefill: 2040.7, 2034.7, 2031.7 tok/s
on  prefill: 2107.7, 2089.4, 2094.5 tok/s
off decode : 65.6, 65.5, 65.4 tok/s
on  decode : 67.9, 67.9, 67.8 tok/s
```

### Qwen3.8-27B MQ4V2 + matched DFlash draft

Qwen3.8 uses an effort-native prompt contract. The valid no-think request therefore sets both `reasoning_effort=none` and `max_think_tokens=1`; using only the named `thinking=off` budget is rejected because it may leave an open think span at the generation cap.

The fixed 3,798-token prompt generated 512 tokens under both arms with byte-identical output and DFlash `tau=2.15`. A five-request same-daemon run then exercised prefix-cache restore: requests 2-5 each reused 2,304 prompt tokens and produced candidate/control transcripts that were byte-identical turn-for-turn.

Steady cached requests 2-5:

| Metric | S1-S6 off | S1-S6 on | Delta |
|---|---:|---:|---:|
| Prefill median | 646.3 tok/s | 656.5 tok/s | +1.59% |
| Decode median | 32.05 tok/s | 33.5 tok/s | +4.52% |

Samples:

```text
off cached prefill: 643.5, 646.1, 646.4, 647.2 tok/s
on  cached prefill: 661.4, 656.9, 656.1, 655.6 tok/s
off cached decode : 30.7, 32.1, 32.0, 32.1 tok/s
on  cached decode : 33.6, 33.5, 33.5, 33.5 tok/s
```

Repeated identical requests change the DFlash operating point after prefix-cache restore (`tau=2.15` on the cold-prefix request and `tau=0.91` on cached requests). These cached-request numbers must not be mixed with cold-prefix decode measurements.

## State-oracle boundary

`scripts/redline_daemon_harness.py --dflash-verify-shadow --pm4` currently fails on gfx1201 for both candidate and the S1-S6-disabled control. Both recorded-HIP and PM4 states diverge from ordinary HIP after the initial position. The control reported 347 failed checks versus 333 for the candidate. This therefore does not isolate a regression in the S1-S6 port; it is an upstream gfx1201 shadow-harness/capture baseline issue that needs separate investigation.

The production serving checks above remain positive: no crash, no empty response under the valid Qwen3.8 reasoning contract, correct NIAH recall, real cache reuse, and byte-identical candidate/control transcripts.

## Artifacts

`raw/` contains the harness JSON and daemon logs. HIPRTC code objects and per-home caches are intentionally excluded.
