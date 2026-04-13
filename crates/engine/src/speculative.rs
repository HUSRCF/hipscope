//! Speculative decoding infrastructure for hipfire.
//!
//! Phase 1: holds target + draft model slots side-by-side on a single shared
//! `Gpu`. The actual speculative decode loop (draft → verify → accept) lives
//! in `spec_loop` once Phase 2 lands. For now, each slot just supports
//! independent forward passes so we can validate that loading two models at
//! once works and that both produce coherent output.
//!
//! Both slots share the same `Gpu` instance — HIP kernels run serialized on
//! the default stream, and the MQ rotation scratch buffers on `Gpu` are reused
//! across calls. This is correct as long as we never have two in-flight GEMVs
//! on different models sharing the same MQ scratch (which we won't, since
//! speculative decode serializes draft-generate then target-verify).

use crate::hfq::HfqFile;
use crate::llama::{self, KvCache};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::tokenizer::Tokenizer;
use hip_bridge::{DeviceBuffer, HipResult};
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// INT8 co-located K and V (default).
    Q8,
}

impl Default for KvMode {
    fn default() -> Self {
        KvMode::Q8
    }
}

/// Configuration for loading a single model slot.
#[derive(Debug, Clone)]
pub struct ModelSlotConfig {
    pub max_seq: usize,
    pub kv_mode: KvMode,
    pub repeat_window: usize,
    pub state_quant: qwen35::StateQuant,
}

impl Default for ModelSlotConfig {
    fn default() -> Self {
        Self {
            max_seq: 2048,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant: qwen35::StateQuant::Q8,
        }
    }
}

/// A single loaded Qwen3.5 model with its own KV cache, DeltaNet state, and
/// forward-pass scratch. The `Gpu` is borrowed, not owned — multiple slots
/// share one `Gpu` instance.
pub struct ModelSlot {
    pub name: String,
    pub hfq: HfqFile,
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    pub scratch: Qwen35Scratch,
    pub slot_config: ModelSlotConfig,
}

impl ModelSlot {
    /// Load a model from `path` into a slot. The caller-supplied `gpu` is used
    /// for all allocations. `name` is a human-readable label used in logs.
    pub fn load(
        gpu: &mut Gpu,
        path: &Path,
        name: impl Into<String>,
        slot_config: ModelSlotConfig,
    ) -> HipResult<Self> {
        let name = name.into();
        let hfq = HfqFile::open(path).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("open {} ({}): {}", path.display(), name, e))
        })?;
        let config = qwen35::config_from_hfq(&hfq).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("invalid Qwen3.5 config in {} ({})", path.display(), name))
        })?;
        let weights = qwen35::load_weights(&hfq, &config, gpu)?;

        let n_kv_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::FullAttention)
            .count();

        let kv_cache = KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            slot_config.max_seq,
        )?;

        let dn_state = DeltaNetState::new_with_quant(gpu, &config, slot_config.state_quant)?;
        let scratch = Qwen35Scratch::new(gpu, &config, slot_config.repeat_window)?;

        Ok(Self {
            name,
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config,
        })
    }

    /// Load the tokenizer from this slot's HFQ metadata. Each slot technically
    /// carries its own tokenizer; callers should validate that two slots'
    /// tokenizers are compatible via `Tokenizer::is_compatible_with` before
    /// sharing.
    pub fn load_tokenizer(&self) -> Option<Tokenizer> {
        Tokenizer::from_hfq_metadata(&self.hfq.metadata_json)
    }

    /// Single-token forward pass. Writes logits into `self.scratch.logits`.
    pub fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()> {
        qwen35::forward_scratch(
            gpu,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
        )
    }

    /// Reset the DeltaNet recurrent state and zero the KV write head.
    /// Does NOT shrink the KV allocation — callers track `seq_pos` separately.
    pub fn reset_state(&mut self, gpu: &mut Gpu) {
        for s in &self.dn_state.s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &self.dn_state.s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &self.dn_state.conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }
}

/// A pair of target + draft slots sharing one `Gpu` and one tokenizer.
///
/// Phase 1 just carries both slots. Phase 2+ adds the `spec_decode_step`
/// method for the verify-and-accept loop.
pub struct SpecPair {
    pub target: ModelSlot,
    pub draft: ModelSlot,
    pub tokenizer: Tokenizer,
}

impl SpecPair {
    /// Load target and draft from separate HFQ files on the same `Gpu`.
    /// Validates that the two models share a compatible tokenizer before
    /// returning — speculative decode requires identical vocab + token IDs.
    pub fn load(
        gpu: &mut Gpu,
        target_path: &Path,
        draft_path: &Path,
        target_cfg: ModelSlotConfig,
        draft_cfg: ModelSlotConfig,
    ) -> HipResult<Self> {
        let target = ModelSlot::load(gpu, target_path, "target", target_cfg)?;
        let draft = ModelSlot::load(gpu, draft_path, "draft", draft_cfg)?;

        let target_tok = target.load_tokenizer().ok_or_else(|| {
            hip_bridge::HipError::new(0, "target model has no tokenizer in HFQ metadata")
        })?;
        let draft_tok = draft.load_tokenizer().ok_or_else(|| {
            hip_bridge::HipError::new(0, "draft model has no tokenizer in HFQ metadata")
        })?;

        if target_tok.vocab_size() != draft_tok.vocab_size() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer mismatch: target vocab={}, draft vocab={}. \
                     Speculative decode requires identical vocabularies.",
                    target_tok.vocab_size(),
                    draft_tok.vocab_size()
                ),
            ));
        }

        // Sanity-check a round-trip on a common string — catches vocab-size
        // match but token-ID mismatch (different BPE merges producing same
        // vocab count).
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        let a = target_tok.encode(probe);
        let b = draft_tok.encode(probe);
        if a != b {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer merge rules diverge: target={:?}, draft={:?}. \
                     Speculative decode requires identical tokenization.",
                    &a, &b
                ),
            ));
        }

        Ok(Self {
            target,
            draft,
            tokenizer: target_tok,
        })
    }

    /// Run a minimal smoke test: 8 forward passes on each slot with a dummy
    /// token sequence, ensuring neither model crashes and the logits buffers
    /// contain finite values. Returns `(target_ok, draft_ok)`.
    pub fn smoke_test(&mut self, gpu: &mut Gpu) -> HipResult<(bool, bool)> {
        // Token ID 1 is a safe placeholder for both Qwen3 and Qwen3.5; the
        // smoke test only checks that the forward pass runs without crashing
        // and produces finite logits.
        let probe_token: u32 = 1;
        for pos in 0..8 {
            self.target.forward(gpu, probe_token, pos)?;
        }
        for pos in 0..8 {
            self.draft.forward(gpu, probe_token, pos)?;
        }
        let target_logits = gpu.download_f32(&self.target.scratch.logits)?;
        let draft_logits = gpu.download_f32(&self.draft.scratch.logits)?;
        let target_ok = target_logits.iter().take(1024).all(|x| x.is_finite());
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());

        // Reset both after the smoke test so the caller starts from a clean
        // state at seq_pos=0.
        self.target.reset_state(gpu);
        self.draft.reset_state(gpu);

        Ok((target_ok, draft_ok))
    }
}

/// Result of one speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of draft tokens accepted (0..=k).
    pub accepted: usize,
    /// Target's next-token prediction at the first rejection point (or after
    /// all drafted tokens if accepted == k). Appended to `committed`.
    pub bonus_token: u32,
    /// The full sequence of tokens the draft proposed this cycle.
    pub drafted: Vec<u32>,
    /// The tokens actually committed to both models: `drafted[..accepted]`
    /// followed by `bonus_token`. Always non-empty (length = accepted + 1).
    pub committed: Vec<u32>,
}

/// Backing storage for a DeltaNetState snapshot. Holds device buffers sized
/// to match the source state's tensors. Allocate once per slot, reuse across
/// all speculative cycles.
pub struct DeltaNetSnapshot {
    s_matrix_bufs: Vec<DeviceBuffer>,
    s_scale_bufs: Vec<DeviceBuffer>,
    conv_state_bufs: Vec<DeviceBuffer>,
}

impl DeltaNetSnapshot {
    /// Allocate backup buffers matching `state`'s shapes.
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState) -> HipResult<Self> {
        let mut s_matrix_bufs = Vec::with_capacity(state.s_matrices.len());
        for t in &state.s_matrices {
            s_matrix_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut s_scale_bufs = Vec::with_capacity(state.s_scales.len());
        for t in &state.s_scales {
            s_scale_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut conv_state_bufs = Vec::with_capacity(state.conv_states.len());
        for t in &state.conv_states {
            conv_state_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        Ok(Self {
            s_matrix_bufs,
            s_scale_bufs,
            conv_state_bufs,
        })
    }

    /// Copy live state → backup.
    pub fn save_from(&mut self, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        Ok(())
    }

    /// Copy backup → live state (rewinds the recurrent state to the snapshot point).
    pub fn restore_to(&self, state: &mut DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (src, dst) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        Ok(())
    }
}

/// Compute the DFlash target-layer extraction indices for a model of
/// `num_target_layers` layers. Matches the `build_target_layer_ids` function in
/// the DFlash reference implementation:
///
/// ```text
/// start = 1
/// end   = num_target_layers - 3        # 29 for num_target_layers=32
/// step  = (end - start) / (num_extract - 1)
/// layers[i] = round(start + i * step)  # for i in 0..num_extract
/// ```
///
/// For Qwen3.5-9B (32 layers) and 5 extraction layers this returns
/// `[1, 8, 15, 22, 29]`, matching the hard-coded indices in the HuggingFace
/// `z-lab/Qwen3.5-9B-DFlash` config.
pub fn dflash_extract_layer_ids(num_target_layers: usize, num_extract: usize) -> Vec<usize> {
    if num_extract == 0 { return Vec::new(); }
    if num_extract == 1 { return vec![1]; }
    let start: f32 = 1.0;
    let end: f32 = (num_target_layers as i32 - 3).max(1) as f32;
    let step = (end - start) / (num_extract as f32 - 1.0);
    (0..num_extract)
        .map(|i| (start + i as f32 * step).round() as usize)
        .collect()
}

/// Ring buffer holding the most recent `max_positions` of hidden state
/// extractions from the target model's forward pass. Each of the `extract_layers`
/// entries is a `[max_positions, hidden_dim]` f32 GPU tensor. `head` is the
/// position that the NEXT write will land at (0..max_positions). `written` is
/// the total cumulative number of writes, used to tell full vs partial buffer.
///
/// For DFlash, the draft model pulls a contiguous slice ending at the most
/// recent position to use as context KV input.
pub struct HiddenStateRingBuffer {
    pub layer_bufs: Vec<GpuTensor>,
    pub extract_layers: Vec<usize>,
    pub max_positions: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub written: usize,
}

impl HiddenStateRingBuffer {
    /// Allocate GPU ring buffer for `num_extract` target layers.
    pub fn new(
        gpu: &mut Gpu,
        num_target_layers: usize,
        num_extract: usize,
        hidden_dim: usize,
        max_positions: usize,
    ) -> HipResult<Self> {
        let extract_layers = dflash_extract_layer_ids(num_target_layers, num_extract);
        let mut layer_bufs = Vec::with_capacity(num_extract);
        for _ in 0..num_extract {
            layer_bufs.push(gpu.alloc_tensor(&[max_positions * hidden_dim], rdna_compute::DType::F32)?);
        }
        let _ = layer_bufs.len(); // silence unused in case of Vec field confusion
        Ok(Self {
            layer_bufs,
            extract_layers,
            max_positions,
            hidden_dim,
            head: 0,
            written: 0,
        })
    }

    /// If `target_layer_idx` matches one of the extraction layers, return the
    /// index into `layer_bufs`/`extract_layers` for that layer. Otherwise None.
    #[inline]
    pub fn extract_slot(&self, target_layer_idx: usize) -> Option<usize> {
        self.extract_layers.iter().position(|&l| l == target_layer_idx)
    }

    /// Copy `x` (shape `[hidden_dim]`) into the ring buffer slot for the given
    /// extraction layer at the CURRENT head position. Call once per extracted
    /// layer per forward pass, then `advance_head()` at the end of the forward
    /// to move to the next slot.
    pub fn write_at_head(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        x: &GpuTensor,
    ) -> HipResult<()> {
        let offset = self.head * self.hidden_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &self.layer_bufs[extract_idx].buf,
            offset,
            &x.buf,
            0,
            self.hidden_dim * 4,
        )
    }

    /// Advance the write head. Call once per forward pass, AFTER all layer
    /// extractions for this position have been written.
    #[inline]
    pub fn advance_head(&mut self) {
        self.head = (self.head + 1) % self.max_positions;
        self.written += 1;
    }

    /// Reset to empty (head=0, written=0). GPU buffers are not zeroed; stale
    /// data is simply unreadable because `written < max_positions`.
    pub fn reset(&mut self) {
        self.head = 0;
        self.written = 0;
    }
}

/// Single-pass argmax for token sampling. Not SIMD-optimized — the logit
/// vector is downloaded once per verify step so the CPU scan cost is
/// negligible relative to GEMV work.
#[inline]
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Aggregated metrics for a sequence of speculative decode steps.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Total number of speculative cycles run.
    pub cycles: usize,
    /// Total number of tokens committed (sum of committed.len() across cycles).
    pub committed_tokens: usize,
    /// Total number of draft tokens accepted (sum of `accepted`).
    pub accepted_tokens: usize,
    /// Per-cycle acceptance count histogram, indexed by accepted count
    /// (0..=k). `acceptance_hist[i]` = number of cycles where exactly `i`
    /// draft tokens were accepted.
    pub acceptance_hist: Vec<usize>,
}

impl SpecStats {
    pub fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            committed_tokens: 0,
            accepted_tokens: 0,
            acceptance_hist: vec![0; k + 1],
        }
    }

    pub fn record(&mut self, step: &SpecStepResult) {
        self.cycles += 1;
        self.committed_tokens += step.committed.len();
        self.accepted_tokens += step.accepted;
        if step.accepted < self.acceptance_hist.len() {
            self.acceptance_hist[step.accepted] += 1;
        }
    }

    /// Mean accepted draft tokens per cycle. This is τ from the Leviathan paper.
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.cycles as f32
        }
    }

    /// Mean committed tokens per cycle (tau + 1 on average, since each
    /// cycle always commits one bonus token).
    pub fn mean_committed(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.committed_tokens as f32 / self.cycles as f32
        }
    }
}

/// One speculative decode step (greedy, Leviathan verify-and-accept).
/// Operates on separate `target` and `draft` `ModelSlot` handles so the
/// caller can keep them owned in top-level variables.
///
/// Preconditions:
/// - Both `target.scratch.logits` and `draft.scratch.logits` contain the
///   logits for position `pos` (from the previous commit or prompt prefill).
/// - `target_snap` / `draft_snap` are preallocated via `DeltaNetSnapshot::new_for`.
/// - `k >= 1` is the speculation count.
///
/// Postconditions:
/// - Both slots' state advances to `pos + committed.len()`, and their
///   `scratch.logits` contain logits at the new position.
/// - Returns a `SpecStepResult` describing how many draft tokens were
///   accepted, the bonus token, and the full committed sequence.
///
/// Naive sequential verification: runs the target on each drafted token one
/// at a time. Phase 5 replaces the inner loop with a single batched prefill.
pub fn spec_step_greedy(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft: &mut ModelSlot,
    pos: usize,
    k: usize,
    target_snap: &mut DeltaNetSnapshot,
    draft_snap: &mut DeltaNetSnapshot,
) -> HipResult<SpecStepResult> {
    assert!(k >= 1, "speculation count k must be ≥ 1");

    // Snapshot both models' recurrent state at position `pos` so we can
    // rewind after verification and commit the final accepted prefix.
    target_snap.save_from(&target.dn_state, gpu)?;
    draft_snap.save_from(&draft.dn_state, gpu)?;

    // Target's current logits (at position `pos`) are used to verify
    // drafted[0]. Capture before anything trashes them.
    let target_logits_at_pos: Vec<f32> = gpu.download_f32(&target.scratch.logits)?;

    // Draft k tokens. drafted[0] samples from draft's current logits (which
    // are also for position `pos`). drafted[i] samples from the logits
    // produced by draft.forward(drafted[i-1], pos+i-1).
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    {
        let first_logits = gpu.download_f32(&draft.scratch.logits)?;
        drafted.push(argmax_u32(&first_logits));
    }
    for i in 0..k {
        draft.forward(gpu, drafted[i], pos + i)?;
        if i + 1 < k {
            let logits = gpu.download_f32(&draft.scratch.logits)?;
            drafted.push(argmax_u32(&logits));
        }
    }

    // Verification: run the target on each drafted token, collect logits.
    // target_mid_logits[i] = target's prediction at position pos+i+1.
    let mut target_mid_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        target.forward(gpu, drafted[i], pos + i)?;
        target_mid_logits.push(gpu.download_f32(&target.scratch.logits)?);
    }
    // Acceptance:
    //   drafted[0] verified by target_logits_at_pos  (logits at pos)
    //   drafted[i] (i >= 1) verified by target_mid_logits[i-1] (logits at pos+i)
    let mut accepted: usize = 0;
    if !target_logits_at_pos.is_empty()
        && argmax_u32(&target_logits_at_pos) == drafted[0]
    {
        accepted = 1;
        for i in 1..k {
            if argmax_u32(&target_mid_logits[i - 1]) == drafted[i] {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Bonus token = target's prediction at position pos+accepted.
    let bonus_logits: &[f32] = if accepted == 0 {
        &target_logits_at_pos
    } else {
        &target_mid_logits[accepted - 1]
    };
    let bonus_token = argmax_u32(bonus_logits);

    // Commit = accepted draft prefix + bonus.
    let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
    committed.extend_from_slice(&drafted[..accepted]);
    committed.push(bonus_token);

    // Restore both models' state and replay the committed sequence so both
    // slots end at `pos + committed.len()` with correct logits.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    draft_snap.restore_to(&mut draft.dn_state, gpu)?;
    for (i, &tok) in committed.iter().enumerate() {
        target.forward(gpu, tok, pos + i)?;
        draft.forward(gpu, tok, pos + i)?;
    }

    Ok(SpecStepResult {
        accepted,
        bonus_token,
        drafted,
        committed,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash-specific target-side verify
// ═══════════════════════════════════════════════════════════════════════════

/// Output of a DFlash target verify step.
pub struct DflashVerifyOutput {
    /// Target argmax token at each of the B positions. argmax_per_pos[i]
    /// is what the target would greedy-decode at absolute position
    /// `start_pos + i` given the preceding context plus `draft_tokens[0..i]`.
    pub argmax_per_pos: Vec<u32>,
    /// Full logits downloaded for every position, concatenated row-major
    /// as `[B * vocab_size]`. The spec step currently only needs argmax,
    /// but the full logits are kept so temp>0 rejection sampling (Phase
    /// 7+) can plug in without re-running the target.
    pub logits_per_pos: Vec<f32>,
}

/// Run the target on `draft_tokens` (length B) positions starting at
/// `start_pos`. Advances `target.kv_cache` and `target.dn_state` by B
/// positions. Writes B hidden-state rows into `hidden_rb` (ring head
/// advances B times). Returns downloaded logits + argmax per position.
///
/// MVP path: B sequential `forward_scratch_with_hidden` calls. Each
/// call quantizes K/V into the KV cache at the relevant position and
/// extracts hidden states at the configured dflash layers.
///
/// Phase 7 optimization target: a single batched forward over the B
/// positions (a `forward_prefill_batch_with_hidden` variant) so the
/// GEMM is one launch instead of B, and only one logits download.
pub fn verify_dflash_block(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let vocab = target.config.vocab_size;
    let mut logits_per_pos: Vec<f32> = Vec::with_capacity(b * vocab);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);

    for (i, &tok) in draft_tokens.iter().enumerate() {
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            tok,
            start_pos + i,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let row = gpu.download_f32(&target.scratch.logits)?;
        debug_assert_eq!(row.len(), vocab);
        argmax_per_pos.push(argmax_u32(&row));
        logits_per_pos.extend_from_slice(&row);
    }

    Ok(DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    })
}

/// Download extracted target hidden states for the most recent B positions
/// from `hidden_rb` and concat them into a flat `[B × num_extract × hidden]`
/// host vector in the order expected by `dflash::draft_forward` (per-position,
/// then per-extract-layer).
///
/// Caller typically slices this by `[0..accept_len+1]` of the position
/// dimension when appending to the cumulative target_hidden buffer used
/// by subsequent draft forwards.
///
/// MVP path: downloads all `num_extract × hidden × B` floats via
/// `gpu.download_f32` per layer (fine at block size 16 + 5 layers: ~2.6 MB
/// per verify). Optimizable in 0.1.7 with a GPU-side scatter kernel.
pub fn download_hidden_block(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    b: usize,
) -> HipResult<Vec<f32>> {
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let written = hidden_rb.written;

    // Figure out which ring positions hold the most recent B writes.
    // `head` points to where the NEXT write will land. After B advances,
    // the most recent B sit at ring slots (head - B) mod max_pos ..
    // (head - 1) mod max_pos.
    assert!(b <= written, "verify must have written at least B rows to ring buffer");
    let head = hidden_rb.head;
    let start_slot = (head + max_pos - b) % max_pos;

    // Download every extract-layer buffer once (small — ≤ max_pos rows).
    let mut layer_data: Vec<Vec<f32>> = Vec::with_capacity(num_extract);
    for buf in &hidden_rb.layer_bufs {
        layer_data.push(gpu.download_f32(buf)?);
    }

    // Rearrange into per-position-then-per-extract-layer order.
    let mut out: Vec<f32> = Vec::with_capacity(b * num_extract * hidden);
    for pi in 0..b {
        let slot = (start_slot + pi) % max_pos;
        for ext in 0..num_extract {
            let src_off = slot * hidden;
            out.extend_from_slice(&layer_data[ext][src_off..src_off + hidden]);
        }
    }

    debug_assert_eq!(out.len(), b * num_extract * hidden);
    Ok(out)
}
