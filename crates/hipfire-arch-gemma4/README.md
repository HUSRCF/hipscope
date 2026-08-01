# Gemma 4

This crate is the staged home for Gemma 4 text inference (`arch_id = 13`).

Current scope:

- parse Gemma 4 text configuration from an HFQ metadata envelope;
- recognize the exact E2B and E4B text topologies;
- reject unknown E-series shapes and malformed attention metadata;

Current non-scope:

- no quantizer route, loader `Carrier`, or serving dispatch is registered yet;
- no vision or audio tower execution;
- no speculative assistant for E4B;
- no Dense12B or MoE26B execution contract.

The next increment must land text-tower quantization together with the Carrier
and runtime types so no intermediate revision emits an unloadable `arch_id=13`
artifact.
