//! The bring-up contract for a hipfire architecture. Implement this
//! trait in your arch crate (e.g. `hipfire-arch-qwen35`) to plug a
//! model into the runtime. Generation, sampling, eviction, spec
//! decode, paging, prompt framing, and EOS filtering all live in
//! the runtime crate; the arch contributes only the model-specific
//! pieces.
//!
//! Default impls cover the Qwen3.5 family conventions. Override only
//! what diverges for your arch.

use crate::hfq::HfqFile;
use rdna_compute::Gpu;

pub trait Architecture: Send + 'static {
    type Weights;
    type State;
    type Config: Clone + Send + 'static;

    fn arch_id() -> u32;
    fn name() -> &'static str;

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String>;
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String>;
    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String>;

    // Forward pass shapes are arch-specific; declare the surface but
    // don't constrain types in this trait — concrete arch crates
    // expose their own typed forward methods. The runtime's generic
    // generation loop holds an `impl Architecture`-bound model and
    // uses arch crate-specific call sites.
    //
    // Future PRs may tighten the forward signatures once we see what
    // the qwen35 / qwen35-vl / llama splits actually need. For PR 7
    // the trait is intentionally minimal — just enough scaffolding for
    // a canary arch crate to implement and the runtime to type-check.

    // Optional overrides — defaults assume Qwen3.5 family.
    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        LoopGuardOverrides::default()
    }
    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides::default()
    }
    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoopGuardOverrides {
    /// If `Some`, replace the env-derived n-gram threshold.
    pub ngram_threshold: Option<usize>,
    pub ngram_window: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SamplerOverrides {
    /// Tokens to add to `SamplerConfig::blocked_tokens` for this arch
    /// (e.g. arch-specific `<tool_call>` opener IDs).
    pub blocked_tokens: Vec<u32>,
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct PromptFrameOverrides {
    /// If `Some`, override the assistant prefix scheme. Use for
    /// non-ChatML or non-thinking-mode arches.
    pub raw: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct EosFilterOverrides {
    /// Byte sequences that signal end-of-turn for this arch.
    /// Examples: Gemma4's `<end_of_turn>` (when forward-ported).
    pub stop_at: Vec<Vec<u8>>,
    pub holdback_prefixes: Vec<Vec<u8>>,
    pub strip_think: Option<bool>,
}
