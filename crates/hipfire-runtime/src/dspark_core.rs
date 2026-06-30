//! Arch-agnostic DSpark spec-decode core. The drafter body (deepseek4 MoE/MLA
//! chain, qwen3 dense transformer) is the only arch-specific seam — see
//! [`DsparkBody`]. Everything else (main_proj ingest, markov head, confidence
//! head, window orchestration) lives here.
use rdna_compute::{Gpu, GpuTensor};

#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub noise_token_id: u32,
    /// false ⇒ DFlash heads-off path (no confidence truncation).
    pub enable_confidence: bool,
}

pub struct DsparkWeights {
    pub cfg: DsparkConfig,
    pub main_proj: Option<GpuTensor>, // [dim, target_layer_ids.len()*dim]
    pub main_norm: Option<GpuTensor>,
    pub markov_w1: Option<GpuTensor>, // [vocab, rank]; None when markov_rank==0
    pub markov_w2: Option<GpuTensor>, // [vocab, rank]
    pub confidence_proj: Option<GpuTensor>, // [1, dim+rank]; None when !enable_confidence
    pub confidence_bias: Option<GpuTensor>, // [1]; qwen3 has bias, deepseek4 None
}

/// The arch-specific seam: draft one window's block given the assembled
/// `main_hidden`, returning the per-slot post-final-norm hidden (`x_head`).
pub trait DsparkBody {
    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &DsparkWeights,
        main_hidden: &GpuTensor, // [target_layer_ids.len()*dim]
        seed: u32,
        position: usize,
        block: usize,
        x_head_out: &GpuTensor, // [block, dim] out
    ) -> Result<(), String>;
    fn block_size(&self) -> usize;
    fn free(self: Box<Self>, gpu: &mut Gpu);
}
