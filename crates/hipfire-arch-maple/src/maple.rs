// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact name
//! (no rename). Mirrors the Cohere2-MoE loader (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) but reflects Maple's structure:
//!   * Standard pre-norm block: `input_layernorm` for attention,
//!     `post_attention_layernorm` for the MoE branch (NOT cohere2's parallel
//!     block with one shared norm).
//!   * **QK-norm**: per-head RMSNorm gammas of width `head_dim` on q and k.
//!   * **Every** layer is MoE — 256 experts, no dense prefix, no shared expert,
//!     no routing bias.
//!   * **Untied** lm_head (`lm_head.weight` is its own tensor), and the
//!     embedding is `model.word_embeddings.weight` — NOT `embed_tokens`.
//!
//! Expert weights ship pre-split (gate_proj/up_proj/down_proj); the loader
//! byte-fuses gate_proj‖up_proj into the per-expert `gate_up` blob the indexed
//! GEMV kernels expect. For Maple this fuse is always same-dtype (everything
//! ternary is qt=51), but the mismatch check is kept: a hand-assembled or
//! partially-requantized checkpoint would otherwise mis-read the up half.

use crate::config::MapleConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Maple's embedding tensor. Named `word_embeddings`, not `embed_tokens` —
/// looking up the conventional name fails at load with "tensor not found".
pub const EMBED_TENSOR_NAME: &str = "model.word_embeddings.weight";
/// Untied output head.
pub const LM_HEAD_TENSOR_NAME: &str = "lm_head.weight";
/// Final RMSNorm gamma.
pub const FINAL_NORM_TENSOR_NAME: &str = "model.norm.weight";

/// Which projection of an expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    fn suffix(self) -> &'static str {
        match self {
            ExpertProj::Gate => "gate_proj",
            ExpertProj::Up => "up_proj",
            ExpertProj::Down => "down_proj",
        }
    }
}

/// Name of one expert projection tensor.
///
/// 18,432 of these (256 experts × 3 projections × 24 layers). A naming slip
/// fails at load, not at build, so the shape is pinned by test.
pub fn expert_tensor_name(layer: usize, expert: usize, proj: ExpertProj) -> String {
    format!(
        "model.layers.{layer}.mlp.experts.{expert}.{}.weight",
        proj.suffix()
    )
}

/// Name of a layer's ROUTER. Note this is `mlp.gate.weight` — distinct from an
/// expert's `mlp.experts.N.gate_proj.weight`. Matching on "gate" alone
/// conflates them.
pub fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("maple: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Load a 1D/raw F16/BF16/F32/Q8 vector → F32 GpuTensor.
///
/// Used for RMSNorm gammas (per-layer, QK-norm, and final). Maple's converter
/// carries every norm as **BF16**, so the BF16 arm is the hot one here — an
/// F16-only loader would reject the model outright.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => dequant_q8_0(&data),
        16 => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => {
            return Err(format!(
                "maple: expected F16/BF16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("maple: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("maple: load_wt {name}: {e}"))
}

/// quant_type → DType. **qt=51 (`MQ2G256LloydU`) is the whole point of this
/// arch**: it is the unrotated MQ2-Lloyd sibling that carries Maple's native
/// ternary weights losslessly, and the dispatcher must NOT rotate x for it.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        1 => DType::F16,
        2 => DType::F32,
        16 => DType::BF16,
        3 => DType::Q8_0,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        19 => DType::MQ2G256Lloyd,
        51 => DType::MQ2G256LloydU,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ──────────────────────────── Weights ────────────────────────────

/// One MoE expert: fused gate(gate_proj)‖up(up_proj) and down(down_proj).
pub struct MapleExpert {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// 256-expert MoE FFN (softmax top-8 + renorm, no bias, no shared expert).
pub struct MapleMoeFfn {
    pub router: WeightTensor,           // mlp.gate.weight [n_exp, hidden]
    pub experts: Vec<MapleExpert>,      // per-expert buffers (owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

/// One-entry pointer tables for the dense attention projections, built ONCE at
/// load. Rebuilding them per call would add four allocations per layer per
/// token-chunk.
pub struct AttnPtrTables {
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
}

pub struct MapleLayerWeights {
    pub input_norm: GpuTensor,     // input_layernorm.weight [hidden]
    pub post_attn_norm: GpuTensor, // post_attention_layernorm.weight [hidden]
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    /// Per-head QK-norm gammas, width `head_dim` (NOT hidden). Applied to q/k
    /// BEFORE RoPE.
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// Single-expert device-pointer tables for `wq`/`wk`/`wv`/`wo`, feeding
    /// `batch::dense_qt51_gemm` for batched prefill.
    pub attn_ptr_tables: AttnPtrTables,
    pub moe: MapleMoeFfn,
}

pub struct MapleWeights {
    pub embed: GpuTensor,      // model.word_embeddings.weight (raw bytes)
    pub embed_dtype: DType,    // dtype of `embed` (drives the lookup path)
    pub final_norm: GpuTensor, // model.norm.weight (RMSNorm gamma)
    pub lm_head: WeightTensor, // UNTIED — lm_head.weight
    pub layers: Vec<MapleLayerWeights>,
}

impl MapleWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &MapleConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let head_dim = cfg.head_dim;

        // Embedding. There is no BF16 embedding-lookup kernel, and Maple's
        // converter carries `word_embeddings` as BF16, so widen to F32 on the
        // host at load and hand the F32 path a buffer it can actually read.
        // Costs ~620 MB extra over the BF16 bytes (151936 × 2048); the
        // alternative is a new HIP kernel for one lookup per token.
        let (eqt, embed_bytes) = read_tensor(hfq, EMBED_TENSOR_NAME)?;
        let (embed, embed_dtype) = match eqt {
            16 => {
                let widened: Vec<f32> = embed_bytes
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let t = gpu
                    .upload_f32(&widened, &[widened.len()])
                    .map_err(|e| format!("maple: upload embed (bf16→f32): {e:?}"))?;
                (t, DType::F32)
            }
            2 | 3 => {
                let t = gpu
                    .upload_raw(&embed_bytes, &[embed_bytes.len()])
                    .map_err(|e| format!("maple: upload embed: {e:?}"))?;
                (t, if eqt == 2 { DType::F32 } else { DType::Q8_0 })
            }
            other => {
                return Err(format!(
                    "maple: embed quant_type {other} has no lookup path (expected BF16, F32 or Q8)"
                ))
            }
        };
        // Untied: a separate lm_head tensor, not a second view of the embedding.
        let lm_head = load_wt(hfq, gpu, LM_HEAD_TENSOR_NAME, cfg.vocab_size, hidden)?;
        let final_norm = load_f32(hfq, gpu, FINAL_NORM_TENSOR_NAME, &[hidden])?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_norm = load_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let post_attn_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let wq = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?;
            let wk = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wv = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wo = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?;
            // QK-norm gammas are head_dim wide, not hidden.
            let q_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_norm.weight"),
                &[head_dim],
            )?;
            let k_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_norm.weight"),
                &[head_dim],
            )?;

            let attn_ptr_tables = AttnPtrTables {
                wq: crate::batch::upload_single_expert_ptr_table(gpu, &wq)?,
                wk: crate::batch::upload_single_expert_ptr_table(gpu, &wk)?,
                wv: crate::batch::upload_single_expert_ptr_table(gpu, &wv)?,
                wo: crate::batch::upload_single_expert_ptr_table(gpu, &wo)?,
            };

            let router = load_wt(hfq, gpu, &router_tensor_name(l), n_exp, hidden)?;
            let mut experts = Vec::with_capacity(n_exp);
            for e in 0..n_exp {
                let (qt_g, g) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Gate))?;
                let (qt_u, u) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Up))?;
                // gate_up is byte-fused and tagged with ONE dtype; a mixed pair
                // would mis-read the up half as qt_g. Refuse at load rather than
                // serve silently-wrong inference.
                if qt_g != qt_u {
                    return Err(format!(
                        "maple L{l}E{e}: gate/up dtype mismatch ({qt_g} vs {qt_u}) — cannot byte-fuse gate_up"
                    ));
                }
                let mut gate_up_bytes = g;
                gate_up_bytes.extend_from_slice(&u);
                let gate_up = wt_from_raw(gpu, qt_g, &gate_up_bytes, 2 * moe_inter, hidden)
                    .map_err(|e2| format!("maple: fuse gate_up L{l}E{e}: {e2}"))?;
                let (qt_d, d) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Down))?;
                let down = wt_from_raw(gpu, qt_d, &d, hidden, moe_inter)
                    .map_err(|e2| format!("maple: down L{l}E{e}: {e2}"))?;
                experts.push(MapleExpert { gate_up, down });
            }
            // Device pointer tables for the indexed-MoE GEMV kernels.
            let gu_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let dn_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("maple: alloc gu_ptrs: {e:?}"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("maple: alloc dn_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(|e| format!("maple: htod gu_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(|e| format!("maple: htod dn_ptrs: {e:?}"))?;

            layers.push(MapleLayerWeights {
                input_norm,
                post_attn_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                attn_ptr_tables,
                moe: MapleMoeFfn {
                    router,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                },
            });
        }

        Ok(MapleWeights {
            embed,
            embed_dtype,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleWeights {
            embed,
            embed_dtype: _,
            final_norm,
            lm_head,
            layers,
        } = self;
        let _ = gpu.free_tensor(embed);
        let _ = gpu.free_tensor(final_norm);
        lm_head.free_all(gpu);
        for layer in layers {
            layer.free_gpu(gpu);
        }
    }
}

impl MapleExpert {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleExpert { gate_up, down } = self;
        gate_up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl MapleMoeFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleMoeFfn {
            router,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
        } = self;
        router.free_all(gpu);
        for e in experts {
            e.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(expert_gate_up_ptrs);
        let _ = gpu.free_tensor(expert_down_ptrs);
    }
}

impl MapleLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleLayerWeights {
            input_norm,
            post_attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            attn_ptr_tables,
            moe,
        } = self;
        for t in [input_norm, post_attn_norm, q_norm, k_norm] {
            let _ = gpu.free_tensor(t);
        }
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        let AttnPtrTables {
            wq: pq,
            wk: pk,
            wv: pv,
            wo: po,
        } = attn_ptr_tables;
        for t in [pq, pk, pv, po] {
            let _ = gpu.free_tensor(t);
        }
        moe.free_gpu(gpu);
    }
}

// ──────────────────────────── State ────────────────────────────

/// Batched-prefill flash sub-batch size at full context — the trailing factor
/// of the `flash_partials` allocation (see the Cohere2 note; same kernel).
const FLASH_PREFILL_SUBBATCH: usize = 64;

/// Default KV window. Maple advertises `max_position_embeddings` 131072, but KV
/// is allocated up front; 32k is the same generous-but-not-maximal default the
/// other arches use, and the daemon honours a larger explicit `max_seq`.
const DEFAULT_MAX_SEQ: usize = 32_768;

/// Per-decode GPU scratch + KV cache (one slot per layer — every Maple layer is
/// attention). Buffers are eager-allocated.
pub struct MapleState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer,
    pub max_seq: usize,
    pub n_tokens: usize,

    pub h: GpuTensor,      // [hidden] residual stream
    pub normed: GpuTensor, // [hidden] pre-branch RMSNorm output

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // moe scratch
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k_top] i32-in-F32
    pub topk_weights: GpuTensor,  // [k_top]
    pub gate_batch: GpuTensor,    // [k_top*moe_inter]
    pub up_batch: GpuTensor,      // [k_top*moe_inter]
    pub act_batch: GpuTensor,     // [k_top*moe_inter] clamped SwiGLU output
    pub down_expanded: GpuTensor, // [k_top*hidden]

    // head
    pub final_norm_buf: GpuTensor,
    pub logits: GpuTensor,
    pub flash_partials: GpuTensor,
}

impl MapleState {
    pub fn new(gpu: &mut Gpu, cfg: &MapleConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(DEFAULT_MAX_SEQ);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &MapleConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;

        // The FWHT sign LUT is still required: the shared MoE helpers reference
        // it even though MQ2G256LloydU itself never rotates.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("maple: ensure_mq_signs: {e:?}"))?;

        let kv = KvCache::new_gpu_q8(
            gpu,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("maple: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("maple: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("maple: alloc {label}: {e:?}"))
        };

        Ok(MapleState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            normed: alloc(gpu, hidden, "normed")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, k * moe_inter, "up_batch")?,
            act_batch: alloc(gpu, k * moe_inter, "act_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            flash_partials: alloc(
                gpu,
                cfg.num_attention_heads
                    * max_seq.div_ceil(128)
                    * (2 + cfg.head_dim)
                    * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
        })
    }

    /// Reset for a fresh conversation: rewind the KV cursor AND zero the KV
    /// buffers. Maple is pure attention with no recurrent state, so the rewind
    /// alone is sufficient for correctness; zeroing makes the reset holistic.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("maple reset: clear kv: {e:?}"))?;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleState {
            kv,
            pos_buf,
            max_seq: _,
            n_tokens: _,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            act_batch,
            down_expanded,
            final_norm_buf,
            logits,
            flash_partials,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            act_batch,
            down_expanded,
            final_norm_buf,
            logits,
            flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_tensor_names_match_the_published_checkpoint() {
        // 18,432 expert tensors (256 experts x 3 x 24 layers). Getting the
        // naming wrong fails at load, not at build.
        assert_eq!(
            expert_tensor_name(7, 255, ExpertProj::Down),
            "model.layers.7.mlp.experts.255.down_proj.weight"
        );
        assert_eq!(
            expert_tensor_name(0, 0, ExpertProj::Gate),
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(
            expert_tensor_name(3, 12, ExpertProj::Up),
            "model.layers.3.mlp.experts.12.up_proj.weight"
        );
    }

    #[test]
    fn router_is_mlp_gate_not_an_expert_gate_proj() {
        // `mlp.gate.weight` (router) vs `mlp.experts.N.gate_proj.weight`
        // (expert). Conflating them loads a [256, 2048] router as an expert.
        assert_eq!(router_tensor_name(3), "model.layers.3.mlp.gate.weight");
        assert_ne!(
            router_tensor_name(3),
            expert_tensor_name(3, 0, ExpertProj::Gate)
        );
    }

    #[test]
    fn embedding_is_word_embeddings_not_embed_tokens() {
        assert_eq!(EMBED_TENSOR_NAME, "model.word_embeddings.weight");
        assert_ne!(EMBED_TENSOR_NAME, "model.embed_tokens.weight");
        // Untied: the head is its own tensor, not a second view of the embedding.
        assert_eq!(LM_HEAD_TENSOR_NAME, "lm_head.weight");
        assert_ne!(LM_HEAD_TENSOR_NAME, EMBED_TENSOR_NAME);
    }

    #[test]
    fn bf16_widening_matches_the_reference_bit_pattern() {
        // The converter carries norms/router/embeddings as BF16, so this is the
        // hot path in load_f32. BF16 is the top 16 bits of the f32.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x4049), f32::from_bits(0x40490000));
    }
}
