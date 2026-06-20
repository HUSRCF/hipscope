// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Static explainer text for Settings keys (5e). A plain, GPU-independent data
//! table: selecting a key in the Settings tab shows what it is, what it does,
//! its tradeoff, the default, when to use it, and any load-bearing interaction
//! (e.g. thinking↔dflash, kv_cache:auto = inherit). Authored as a const table so
//! it is testable without a GPU or a daemon and can be cross-checked against the
//! config defaults.

/// One key's help. Every field is a short, honest sentence (no marketing).
#[derive(Clone, Copy, Debug)]
pub struct KnobInfo {
    pub key: &'static str,
    /// Human-friendly title.
    pub title: &'static str,
    /// What it is (one line).
    pub summary: &'static str,
    /// What it does + the tradeoff (pros/cons).
    pub effect: &'static str,
    /// The default value, as a string (cross-checked against config defaults).
    pub default: &'static str,
    /// When to use / leave it.
    pub when: &'static str,
    /// A load-bearing interaction or caveat, if any.
    pub note: Option<&'static str>,
}

/// Help for `key`, if a curated entry exists.
pub fn knob_info(key: &str) -> Option<&'static KnobInfo> {
    KNOBS.iter().find(|k| k.key == key)
}

/// Curated, honest help for the Settings keys. Order is not significant (lookup
/// is by key). Defaults here mirror `config::defaults()` — see the cross-check
/// test below so the two can't silently drift.
pub const KNOBS: &[KnobInfo] = &[
    KnobInfo {
        key: "default_model",
        title: "Default model",
        summary: "The model serve pre-warms and chat uses when no model is named.",
        effect: "Picks which weights load. No quality/speed tradeoff by itself — it just selects the model.",
        default: "qwen3.5:9b",
        when: "Set it to the model you use most. Change it from the Models tab.",
        note: None,
    },
    KnobInfo {
        key: "max_seq",
        title: "Context (KV cache capacity)",
        summary: "Maximum tokens the KV cache is allocated to hold at load.",
        effect: "Larger = longer context, but more VRAM reserved up front. Smaller = less VRAM, shorter ceiling.",
        default: "32768",
        when: "Raise for long documents/agents; lower to fit a tight VRAM budget.",
        note: Some("Allocated at load — changing it takes effect on the next serve/restart."),
    },
    KnobInfo {
        key: "kv_cache",
        title: "KV cache precision",
        summary: "Precision/memory tradeoff for the attention key/value cache.",
        effect: "Lower precision (q8 → asym3 → asym2) saves VRAM and can speed decode, at some quality risk. Higher keeps fidelity.",
        default: "auto",
        when: "Leave on auto unless you need to fit a larger context or are A/B-testing precision.",
        note: Some("auto is the inherit sentinel — it resolves to the model's registry default_kv_mode if set, else q8 (the universal default; near-reference and DFlash-safe). A model can ship a compressed default via the registry; it's no longer guessed from the GPU arch."),
    },
    KnobInfo {
        key: "dflash_mode",
        title: "Spec decode (DFlash)",
        summary: "Speculative decoding: a small drafter proposes tokens the target verifies in batches.",
        effect: "Big decode speedup on predictable output (code, tool-calls, structured: ~4–5×). On free-form prose the acceptance rate is ~1, so it is a slight loss. Correctness is unchanged (the target verifies every token).",
        default: "off",
        when: "Turn on for code/agentic workloads with a drafter attached; leave off for open-ended chat.",
        note: Some("A thinking BUDGET routes DFlash to plain decode: the dispatch gate bails when max_think_tokens > 0 and the think block isn't closed (the default thinking-on path sets a budget). Turn thinking off — or set its budget to 0 — and use greedy to actually get spec-decode."),
    },
    KnobInfo {
        key: "thinking",
        title: "Thinking (reasoning block)",
        summary: "Whether reasoning models emit a hidden <think> block before the answer.",
        effect: "On = better reasoning on hard prompts, but more tokens/latency and it suppresses spec-decode. Off = faster, leaner, spec-decode eligible.",
        default: "on",
        when: "On for hard reasoning/math; off for throughput, code, or to enable DFlash.",
        note: Some("Interacts with dflash_mode: an active thinking budget (max_think_tokens > 0) makes DFlash fall back to plain decode. Turn thinking off to use spec-decode."),
    },
    KnobInfo {
        key: "prefill_compression",
        title: "Prefill compression (pflash)",
        summary: "Compresses a long prompt's KV during prefill using a drafter's importance scoring.",
        effect: "Cuts prefill time and KV footprint on long prompts. Costs a little quality on what it drops, and does nothing without a drafter.",
        default: "off",
        when: "auto = compress only past prefill_threshold tokens; always = every prefill; off = never.",
        note: Some("Needs a prefill_drafter (.hfq path). With no drafter set, compression is disabled — the TUI warns when you turn it on."),
    },
    KnobInfo {
        key: "prefill_drafter",
        title: "Prefill drafter",
        summary: "Path to the small drafter model (.hfq) that scores prompt importance for pflash.",
        effect: "Required for prefill_compression to engage. Empty disables compression entirely.",
        default: "",
        when: "Set to a drafter .hfq when using prefill compression; leave empty otherwise.",
        note: Some("An empty drafter disables prefill_compression regardless of its setting (the TUI warns when you enable compression without one)."),
    },
    KnobInfo {
        key: "prefill_threshold",
        title: "Prefill threshold",
        summary: "Token count above which auto-mode prefill compression kicks in.",
        effect: "Higher = only very long prompts compress; lower = compress sooner. Only relevant when prefill_compression=auto.",
        default: "32768",
        when: "Lower it if you want shorter prompts compressed too; raise to restrict to very long contexts.",
        note: None,
    },
    KnobInfo {
        key: "mtp_mode",
        title: "Multi-token prediction (MTP)",
        summary: "Uses a model's MTP head to predict several tokens per step where supported.",
        effect: "Can raise decode throughput on MTP-capable models with no correctness cost; a no-op on models without an MTP head.",
        default: "auto",
        when: "Leave on auto; force on/off only when A/B-testing throughput.",
        note: None,
    },
    KnobInfo {
        key: "mtp_k",
        title: "MTP depth (K)",
        summary: "How many tokens the MTP head proposes per step.",
        effect: "Higher K can accept more per step (faster) but risks lower acceptance on hard text. Only used when MTP is active.",
        default: "3",
        when: "Tune only when measuring MTP throughput; the default is a safe middle.",
        note: None,
    },
    KnobInfo {
        key: "flash_mode",
        title: "Flash attention mode",
        summary: "Controls the flash-attention path for the Q8 KV cache.",
        effect: "auto picks per arch; always/never force it. Mostly a performance knob; the asym KV modes are flash-only.",
        default: "auto",
        when: "Leave on auto unless debugging an attention path.",
        note: None,
    },
    KnobInfo {
        key: "mmq_screen",
        title: "MMQ screening",
        summary: "Mixed-precision matmul screening for quantized GEMMs.",
        effect: "auto enables it where it helps; a low-level performance knob.",
        default: "auto",
        when: "Leave on auto unless profiling kernels.",
        note: None,
    },
    KnobInfo {
        key: "cask",
        title: "CASK (TriAttention eviction)",
        summary: "KV-cache eviction policy that keeps a working set instead of the full context.",
        effect: "Frees KV VRAM on long contexts. Switches between m-fold and drop eviction; only actually evicts when a TriAttention sidecar is attached.",
        default: "false",
        when: "Enable only with a published sidecar and after validating output quality.",
        note: Some("m-fold eviction + DFlash can produce a repetition attractor; validate together. Eviction needs a sidecar path, not just this toggle."),
    },
    KnobInfo {
        key: "dflash_adaptive_b",
        title: "DFlash adaptive batch",
        summary: "Lets DFlash adapt its draft batch size to recent acceptance.",
        effect: "Can improve spec-decode throughput by sizing drafts to how well they're accepted. No correctness effect.",
        default: "true",
        when: "Leave on; only relevant when dflash_mode is active.",
        note: None,
    },
    KnobInfo {
        key: "prompt_normalize",
        title: "Prompt normalization",
        summary: "Collapses 3+ consecutive newlines to 2 before tokenizing.",
        effect: "Stabilizes tokenization so whitespace differences don't swing performance; no meaning change for normal text.",
        default: "true",
        when: "Leave on. Turn off only when raw repeated newlines are semantically load-bearing.",
        note: None,
    },
    KnobInfo {
        key: "default_chatml",
        title: "Default ChatML",
        summary: "Whether to apply a ChatML scaffold when a model has no embedded chat template.",
        effect: "Improves chat formatting for templateless models; harmless when a model carries its own template.",
        default: "true",
        when: "Leave on; turn off only for raw/base models where a ChatML scaffold hurts.",
        note: None,
    },
    KnobInfo {
        key: "chat_template",
        title: "Chat template",
        summary: "Path to a custom Jinja chat template (.j2/.jinja). Empty = model default.",
        effect: "Overrides how messages are rendered into the prompt. Wrong templates degrade output.",
        default: "",
        when: "Set only when you need a specific template; leave empty to use the model's own.",
        note: Some("Must point to a file that exists; an invalid path is rejected on save."),
    },
    KnobInfo {
        key: "temperature",
        title: "Temperature",
        summary: "Sampling randomness for generation.",
        effect: "Lower = more deterministic/repeatable; higher = more varied/creative but less reliable.",
        default: "0.3",
        when: "Low (≤0.3) for code/factual; higher (0.7–1.0) for brainstorming.",
        note: Some("Greedy (≈0) is what spec-decode runs at; sampling adds cost."),
    },
    KnobInfo {
        key: "top_p",
        title: "Top-p (nucleus)",
        summary: "Caps sampling to the smallest set of tokens whose probability sums to p.",
        effect: "Lower = safer, more focused; higher = more diverse. Interacts with temperature.",
        default: "0.8",
        when: "Leave near default; lower it to tighten outputs.",
        note: None,
    },
    KnobInfo {
        key: "host",
        title: "Serve endpoint",
        summary: "The bind address (and port) of the OpenAI-compatible serve API.",
        effect: "0.0.0.0 listens on all interfaces; 127.0.0.1 is local-only. Chat and API clients connect here.",
        default: "0.0.0.0",
        when: "Use 127.0.0.1 to keep serve private to this machine; 0.0.0.0 to expose it on the network.",
        note: Some("Paired with port (default 11435). Changes take effect on the next serve/restart."),
    },
    KnobInfo {
        key: "max_tokens",
        title: "Max tokens",
        summary: "Upper bound on tokens generated per response.",
        effect: "Caps response length and runaway generation. No quality effect within the cap.",
        default: "4096",
        when: "Raise for long outputs; lower to bound latency/cost.",
        note: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::writer::EDITABLE_FIELDS;

    #[test]
    fn lookup_returns_curated_entry() {
        let dflash = knob_info("dflash_mode").expect("dflash entry");
        assert_eq!(dflash.title, "Spec decode (DFlash)");
        assert!(knob_info("does_not_exist").is_none());
    }

    #[test]
    fn every_editable_field_has_help() {
        // 5e: selecting any editable key must show an explainer.
        for f in EDITABLE_FIELDS {
            assert!(
                knob_info(f.key).is_some(),
                "no explainer for editable key {}",
                f.key
            );
        }
    }

    #[test]
    fn defaults_match_config_defaults() {
        // STRICT: every knob's `default` must exist in config::defaults() AND
        // match it — no silent skip. config::defaults() is the TUI's mirror of bun
        // CONFIG_DEFAULTS (verified key-by-key), so this also pins knob defaults to
        // the real defaults. A knob without a config default fails loudly so it
        // can't drift unnoticed (this is exactly the gap that hid default_chatml).
        let cfg = crate::hipfire::config::ConfigState::defaults_for_test();
        for k in KNOBS {
            let expected = cfg.get(k.key).unwrap_or_else(|| {
                panic!(
                    "knob {} has no config default to cross-check — add it to config::defaults()",
                    k.key
                )
            });
            assert_eq!(
                &k.default, expected,
                "knob default for {} drifted from config defaults ({} vs {})",
                k.key, k.default, expected
            );
        }
    }

    #[test]
    fn load_bearing_interactions_are_documented() {
        // The two interactions this phase exists to surface.
        assert!(
            knob_info("dflash_mode").unwrap().note.unwrap().contains("thinking"),
            "dflash note must mention the thinking interaction"
        );
        assert!(
            knob_info("thinking").unwrap().note.unwrap().to_lowercase().contains("dflash")
                || knob_info("thinking").unwrap().note.unwrap().contains("spec-decode"),
            "thinking note must mention the dflash interaction"
        );
        assert!(
            knob_info("kv_cache").unwrap().note.unwrap().contains("inherit"),
            "kv_cache note must explain auto = inherit"
        );
    }
}
