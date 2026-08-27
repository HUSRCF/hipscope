// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2-VL vision: SigLIP2-NaFlex tower + multi-modal projector.
//!
//! GPU kernels (all pre-existing, gfx1101-validated via qwen35-vl):
//! `gemm_f16[_wmma_mb8]`, `layernorm_batched`, `vit_attention_f32`,
//! `gelu_tanh_f32` (tower MLP), `add_inplace_f32`, `bias_add_f32`,
//! `transpose_f32`. The projector's exact-erf GELU runs on host between
//! two GPU linears (`libm::erff`; ACT2FN["gelu"] is torch.erf, NOT tanh).
//!
//! SigLIP2 has separate q/k/v projections; they are concatenated into a
//! single `[3h, h]` F16 weight + `[3h]` bias at LOAD time so the forward
//! can emit the packed `[n, 3h]` q|k|v buffer `vit_attention_f32` consumes
//! (one gemm per layer instead of three, and no strided pack pass).

use crate::config::VisionConfig;
use crate::image::{Prepared, SubImage};
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── GPU-side weights ────────────────────────────────────────────────────────

pub struct VisionLayerWeights {
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    /// Fused `[3h, h]` of stacked q/k/v row-major F16.
    pub qkv_w: GpuTensor,
    pub qkv_b: GpuTensor,
    pub proj_w: GpuTensor,
    pub proj_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub fc1_w: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor,
    pub fc2_b: GpuTensor,
}

pub struct VisionWeights {
    /// Patch embedding Linear rows `[1152, 768]` normalized to `(dy,dx,c)`
    /// input layout. Handles both serializations seen in the wild
    /// (Linear `[out, ps·ps·C]` or Conv `[out, C, ps, ps]`) per narrow-spec R1.
    pub patch_embed_w: GpuTensor,
    pub patch_embed_b: GpuTensor,
    /// Learned position table `[num_position_embeddings · hidden]` on CPU:
    /// every sub-image resizes a different slice out of it (small buffer).
    pub pos_embed: Vec<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub post_ln_w: GpuTensor,
    pub post_ln_b: GpuTensor,
    // ── projector ──
    /// `[2048, 4608]` F16 — input vector is the pixel-unshuffle block from
    /// [`pixel_unshuffle_token`] (columns-pair-first channel interleave).
    pub proj1_w: GpuTensor,
    pub proj1_b: GpuTensor,
    /// `[2048, 2048]` F16.
    pub proj2_w: GpuTensor,
    pub proj2_b: GpuTensor,
}

impl VisionWeights {
    /// Free every GPU buffer (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embed_w);
        let _ = gpu.free_tensor(self.patch_embed_b);
        for l in self.layers {
            for t in [
                l.norm1_w, l.norm1_b, l.qkv_w, l.qkv_b, l.proj_w, l.proj_b, l.norm2_w, l.norm2_b,
                l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b,
            ] {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(self.post_ln_w);
        let _ = gpu.free_tensor(self.post_ln_b);
        let _ = gpu.free_tensor(self.proj1_w);
        let _ = gpu.free_tensor(self.proj1_b);
        let _ = gpu.free_tensor(self.proj2_w);
        let _ = gpu.free_tensor(self.proj2_b);
    }
}

// ─── Tensor loading helpers (F16-in-artifact → F32 or F16 GPU) ──────────────

fn load_f32_cpu(hfq: &HfqFile, name: &str, n: usize) -> Vec<f32> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let mut vals: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("expected F16/F32 for {name}, got qt={other}"),
    };
    vals.truncate(n);
    vals
}

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let vals = load_f32_cpu(hfq, name, n);
    gpu.upload_f32(&vals, &[n])
}

fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        1 => gpu.upload_raw(data, &[n]),
        2 => {
            // F32 container → narrow to F16 at load (vision kernels are F16).
            let f16_bytes: Vec<u8> = data
                .chunks_exact(4)
                .take(n)
                .flat_map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        other => panic!("{name}: unsupported vision quant_type={other} (expected F16=1 or F32=2)"),
    }
}

/// Normalize the patch-embedding weight to `[out, ps·ps·C]` rows ordered for
/// input vectors laid out `(dy, dx, c)` (HF convert_image_to_patches order).
///
/// Artifact serializations handled:
/// - Linear `[1152, 768]`: already aligned with `(dy,dx,c)` — verbatim.
/// - Conv `[1152, C, kh, kw]`: kernel dims reorder to `(dy,dx,c)`.
fn patch_weight_rows(hfq: &HfqFile) -> (Vec<f32>, usize, usize) {
    const NAME: &str = "model.vision_tower.vision_model.embeddings.patch_embedding.weight";
    let (info, data) = hfq.tensor_data(NAME).unwrap_or_else(|| panic!("missing {NAME}"));
    let out = info.shape[0] as usize;
    let raw: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        q => panic!("{NAME}: unsupported quant_type={q}"),
    };
    match info.shape.len() {
        // Linear: [out, ps*ps*C] with rows (dy,dx,c) — HF processors store it
        // pre-flattened in exactly the pixel-vector order.
        2 => {
            assert_eq!(raw.len(), out * info.shape[1] as usize);
            (raw, out, info.shape[1] as usize)
        }
        // Conv: [out, C, kh, kw] → fold to (dy,dx,c).
        4 => {
            let c_n = info.shape[1] as usize;
            let kh = info.shape[2] as usize;
            let kw = info.shape[3] as usize;
            let in_dim = c_n * kh * kw;
            assert_eq!(raw.len(), out * in_dim);
            let mut folded = vec![0.0f32; raw.len()];
            for o in 0..out {
                for dy in 0..kh {
                    for dx in 0..kw {
                        for c in 0..c_n {
                            folded[o * in_dim + (dy * kw + dx) * c_n + c] =
                                raw[o * in_dim + c * kh * kw + dy * kw + dx];
                        }
                    }
                }
            }
            (folded, out, in_dim)
        }
        other => panic!(
            "{NAME}: unexpected rank {other} (want Linear [out,in] or Conv [out,C,kh,kw]) — \
             refusing to guess the layout"
        ),
    }
}

fn concat_qkv(hfq: &HfqFile, i: usize, hidden: usize) -> (Vec<u8>, Vec<f32>) {
    // Stack independent q/k/v F16 weights into one [3h, h] buffer so the
    // forward's single fused gemm produces the q|k|v-packed layout
    // `vit_attention_f32` expects. Biases concatenate likewise.
    let pfx = format!("model.vision_tower.vision_model.encoder.layers.{i}.self_attn");
    let mut w = Vec::with_capacity(9 * hidden * hidden);
    let mut b = Vec::with_capacity(3 * hidden);
    for part in ["q_proj", "k_proj", "v_proj"] {
        let (info, data) = hfq
            .tensor_data(&format!("{pfx}.{part}.weight"))
            .unwrap_or_else(|| panic!("missing {pfx}.{part}.weight"));
        assert_eq!(info.shape.len(), 2);
        assert_eq!(info.shape[0] as usize, hidden);
        assert_eq!(info.shape[1] as usize, hidden);
        assert_eq!(info.quant_type, 1, "expect F16 attention weights");
        w.extend_from_slice(data);
        b.extend(load_f32_cpu(hfq, &format!("{pfx}.{part}.bias"), hidden));
    }
    (w, b)
}

pub fn load_vision_weights(
    hfq: &HfqFile,
    cfg: &VisionConfig,
    gpu: &mut Gpu,
) -> HipResult<VisionWeights> {
    let h = cfg.hidden_size;

    match gpu.arch.as_str() {
        "gfx1100" | "gfx1101" | "gfx1102" => {}
        other => eprintln!(
            "  ⚠ vision tower not yet validated on {other}; results may differ from \
             the RDNA3-wave32 reference",
        ),
    }

    eprintln!("  loading LFM2-VL vision tower (GPU)...");
    let (pw_rows, out_rows, in_dim) = patch_weight_rows(hfq);
    assert_eq!(in_dim, cfg.patch_size * cfg.patch_size * cfg.num_channels);
    let pw_u16: Vec<u8> = pw_rows.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect();
    let patch_embed_w = gpu.upload_raw(&pw_u16, &[out_rows])?;
    let patch_embed_b = load_f32_gpu(
        hfq,
        gpu,
        "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        h,
    )?;
    let pos_embed = load_f32_cpu(
        hfq,
        "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        cfg.num_position_embeddings * h,
    );

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        if i % 9 == 0 {
            eprintln!("  loading vision block {i}/{}...", cfg.num_layers);
        }
        let p = format!("model.vision_tower.vision_model.encoder.layers.{i}");
        let (qkv_w, qkv_b) = concat_qkv(hfq, i, h);
        layers.push(VisionLayerWeights {
            norm1_w: load_f32_gpu(hfq, gpu, &format!("{p}.layer_norm1.weight"), h)?,
            norm1_b: load_f32_gpu(hfq, gpu, &format!("{p}.layer_norm1.bias"), h)?,
            qkv_w: gpu.upload_raw(&qkv_w, &[3 * h * h])?,
            qkv_b: gpu.upload_f32(&qkv_b, &[3 * h])?,
            proj_w: load_f16_gpu(hfq, gpu, &format!("{p}.self_attn.out_proj.weight"))?,
            proj_b: load_f32_gpu(hfq, gpu, &format!("{p}.self_attn.out_proj.bias"), h)?,
            norm2_w: load_f32_gpu(hfq, gpu, &format!("{p}.layer_norm2.weight"), h)?,
            norm2_b: load_f32_gpu(hfq, gpu, &format!("{p}.layer_norm2.bias"), h)?,
            fc1_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.fc1.weight"))?,
            fc1_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.fc1.bias"), cfg.mlp_dim)?,
            fc2_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.fc2.weight"))?,
            fc2_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.fc2.bias"), h)?,
        });
    }

    eprintln!("  loading post_layernorm + projector...");
    Ok(VisionWeights {
        patch_embed_w,
        patch_embed_b,
        pos_embed,
        layers,
        post_ln_w: load_f32_gpu(hfq, gpu, "model.vision_tower.vision_model.post_layernorm.weight", h)?,
        post_ln_b: load_f32_gpu(hfq, gpu, "model.vision_tower.vision_model.post_layernorm.bias", h)?,
        proj1_w: load_f16_gpu(hfq, gpu, "model.multi_modal_projector.linear_1.weight")?,
        proj1_b: load_f32_gpu(hfq, gpu, "model.multi_modal_projector.linear_1.bias", cfg.projector_hidden_size)?,
        proj2_w: load_f16_gpu(hfq, gpu, "model.multi_modal_projector.linear_2.weight")?,
        proj2_b: load_f32_gpu(hfq, gpu, "model.multi_modal_projector.linear_2.bias", cfg.out_hidden_size)?,
    })
}

// ─── CPU per-image precomputes (exact-index, unit-tested) ───────────────────

/// Bilinear resize of the learned `(K,K,D)` position table to `(gh, gw)`,
/// torch semantics: `F.interpolate(mode="bilinear", align_corners=False,
/// antialias=True)`. Upscale axes are plain two-tap bilinear sampling
/// (antialias is a no-op there); downscale axes widen to a normalized
/// triangle filter over the source footprint, matching torch's antialiased
/// path. Narrow-spec §2.1/R3 documents this as the accepted approximation
/// point (resize-filter residual, below model sensitivity).
pub fn resize_pos_embed(
    table: &[f32],
    k_side: usize,
    dim: usize,
    gh: usize,
    gw: usize,
) -> Vec<f32> {
    struct Taps {
        idx: Vec<usize>,
        w: Vec<f64>,
    }
    fn axis_taps(dst_len: usize, src_len: usize) -> Vec<Taps> {
        let scale = src_len as f64 / dst_len as f64;
        let aa = scale > 1.0;
        (0..dst_len)
            .map(|i| {
                // align_corners=False sample center
                let center = (i as f64 + 0.5) * scale - 0.5;
                if !aa {
                    // torch bilinear: indices are clamped but the split
                    // fraction is taken against the UNCLAMPED floor, so
                    // off-table centers replicate the border value.
                    let l0 = center.floor();
                    let frac = center - l0;
                    let lo_i = (l0 as i64).clamp(0, src_len as i64 - 1);
                    let hi_i = ((l0 as i64) + 1).clamp(0, src_len as i64 - 1);
                    Taps { idx: vec![lo_i as usize, hi_i as usize], w: vec![1.0 - frac, frac] }
                } else {
                    let support = scale;
                    let lo = ((center - support).floor().max(0.0)) as usize;
                    let hi = ((center + support).ceil()).min((src_len - 1) as f64) as usize;
                    let mut idx = Vec::new();
                    let mut w = Vec::new();
                    let inv = 1.0 / scale;
                    for s in lo..=hi {
                        let d = (s as f64 - center).abs();
                        let t = (1.0 - d * inv).max(0.0);
                        idx.push(s);
                        w.push(t);
                    }
                    let sum: f64 = w.iter().sum();
                    if sum > 0.0 {
                        for v in w.iter_mut() {
                            *v /= sum;
                        }
                    }
                    Taps { idx, w }
                }
            })
            .collect()
    }

    let th = axis_taps(gh, k_side);
    let tw = axis_taps(gw, k_side);

    let mut out = vec![0.0f32; gh * gw * dim];
    for oy in 0..gh {
        for ox in 0..gw {
            let mut acc = vec![0.0f64; dim];
            let mut total = 0.0f64;
            for (&sy, &wy) in th[oy].idx.iter().zip(&th[oy].w) {
                for (&sx, &wx) in tw[ox].idx.iter().zip(&tw[ox].w) {
                    let wt = wy * wx;
                    if wt == 0.0 {
                        continue;
                    }
                    let base = (sy * k_side + sx) * dim;
                    for d in 0..dim {
                        acc[d] += (table[base + d] as f64) * wt;
                    }
                    total += wt;
                }
            }
            let norm = if total > 0.0 { total } else { 1.0 };
            let dst = (oy * gw + ox) * dim;
            for d in 0..dim {
                out[dst + d] = (acc[d] / norm) as f32;
            }
        }
    }
    out
}

/// HF `Lfm2VlMultiModalProjector::pixel_unshuffle(f=2)` on features arranged
/// `(gh, gw, channels)` (patch grid after unpad). Token vector component
/// `t = di·(2·C) + dj·C + c` reads feature patch `(2br+di, 2bc+dj)[c]` —
/// i.e. columns pair FIRST (dj stride C), rows second (di stride 2C):
/// replicated verbatim from the pinned HF ops including their swapped dim
/// naming. `factor` is 2 for every published LFM2-VL checkpoint; other even
/// factors fall back to a generic strided formulation.
pub fn pixel_unshuffle_tokens(feat: &[f32], gh: usize, gw: usize, ch: usize, factor: usize) -> Vec<f32> {
    let blocks_y = gh / factor;
    let blocks_x = gw / factor;
    let out_ch = ch * factor * factor;
    let mut out = vec![0.0f32; blocks_y * blocks_x * out_ch];
    // Generic loop derived from HF's reshape/permute chain: token vector is
    // di-major, then dj, then channel (see narrow spec §1.4 derivation).
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            for di in 0..factor {
                for dj in 0..factor {
                    let src_row = by * factor + di;
                    let src_col = bx * factor + dj;
                    let src = (src_row * gw + src_col) * ch;
                    // column offset first: t = di*(ch*f) + dj*ch   (f=2 ⇒ verified
                    // against the reshape trace); generalize stride order:
                    let dst_off = (by * blocks_x + bx) * out_ch;
                    for c in 0..ch {
                        // di-major then dj then c for any factor ≥ 2
                        out[dst_off + (di * factor * ch) + (dj * ch) + c] = feat[src + c];
                    }
                }
            }
        }
    }
    out
}

/// Exact-GELU (`x·Φ(x)` with erf Φ) used by the projector between linear_1
/// and linear_2 — ACT2FN["gelu"], distinct from the tower's tanh approx.
pub fn gelu_exact_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x *= 0.5 * (1.0 + libm::erff(*x / std::f32::consts::SQRT_2));
    }
}

// ─── GPU forward ─────────────────────────────────────────────────────────────

/// Vision linear Y[n, out] = W_f16[out, in] @ X[n, in]^T + bias.
/// Same routing as qwen35-vl: WMMA row-major writer on wave32 arches,
/// naive gemm + transpose elsewhere.
fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: Option<&GpuTensor>,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
    if let Some(b) = bias {
        gpu.bias_add_f32(&y, b, n, out_dim)?;
    }
    Ok(y)
}

/// Encode one request image into projected text-space tokens, concatenated
/// across its sub-images (tiles row-major then thumbnail). Row count equals
/// [`Prepared::total_tokens`]; failure to match fails loud BEFORE splicing.
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &VisionConfig,
    prepared: &Prepared,
) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let heads = cfg.num_heads;
    let head_dim = cfg.head_dim;
    let k_side = (cfg.num_position_embeddings as f64).sqrt() as usize;
    assert_eq!(k_side * k_side, cfg.num_position_embeddings);

    let all_tokens: usize = prepared.total_tokens(cfg);
    let mut out = Vec::with_capacity(all_tokens * cfg.out_hidden_size);

    let t0 = std::time::Instant::now();
    for sub in &prepared.sub_images {
        out.extend(tower_and_project_sub_image(gpu, weights, cfg, sub, h, heads, head_dim, k_side)?);
    }
    eprintln!(
        "  vision done: {} sub-images, {} tokens × {} dims ({:.2}s)",
        prepared.sub_images.len(),
        all_tokens,
        cfg.out_hidden_size,
        t0.elapsed().as_secs_f32(),
    );
    Ok(out)
}

fn tower_and_project_sub_image(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &VisionConfig,
    sub: &SubImage,
    h: usize,
    heads: usize,
    head_dim: usize,
    k_side: usize,
) -> Result<Vec<f32>, String> {
    let patches = sub.patches(cfg);
    let n = sub.gh(cfg) * sub.gw(cfg);
    let patch_dim = patches.len() / n.max(1);
    let eps = cfg.norm_eps;

    // patch embed → [n, h]
    let x_patches = gpu.upload_f32(&patches, &[n * patch_dim]).map_err(ehip("upload patches"))?;
    let x = linear_f16(gpu, &weights.patch_embed_w, &x_patches, Some(&weights.patch_embed_b), h, patch_dim, n)
        .map_err(ehip("patch embed"))?;
    gpu.free_tensor(x_patches).map_err(ehip("free patches"))?;

    // position embed add
    let pos = resize_pos_embed(&weights.pos_embed, k_side, h, sub.gh(cfg), sub.gw(cfg));
    let pos_gpu = gpu.upload_f32(&pos, &[pos.len()]).map_err(ehip("upload pos"))?;
    gpu.add_inplace_f32(&x, &pos_gpu).map_err(ehip("pos add"))?;
    gpu.free_tensor(pos_gpu).map_err(ehip("free pos"))?;

    // encoder layers: pre-LN attn residual + pre-LN MLP residual
    for lw in &weights.layers {
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32).map_err(ehip("alloc ln1"))?;
        gpu.layernorm_batched(&x, &lw.norm1_w, &lw.norm1_b, &tmp, n, h, eps)
            .map_err(ehip("ln1"))?;
        let qkv = linear_f16(gpu, &lw.qkv_w, &tmp, Some(&lw.qkv_b), 3 * h, h, n)
            .map_err(ehip("qkv"))?;
        gpu.free_tensor(tmp).map_err(ehip("free ln1"))?;

        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32).map_err(ehip("alloc attn"))?;
        gpu.vit_attention_f32(&qkv, &attn_out, n, h, heads, head_dim)
            .map_err(ehip("vit_attention"))?;
        gpu.free_tensor(qkv).map_err(ehip("free qkv"))?;

        let proj = linear_f16(gpu, &lw.proj_w, &attn_out, Some(&lw.proj_b), h, h, n)
            .map_err(ehip("attn proj"))?;
        gpu.free_tensor(attn_out).map_err(ehip("free attn"))?;
        gpu.add_inplace_f32(&x, &proj).map_err(ehip("resid1"))?;
        gpu.free_tensor(proj).map_err(ehip("free proj"))?;

        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32).map_err(ehip("alloc ln2"))?;
        gpu.layernorm_batched(&x, &lw.norm2_w, &lw.norm2_b, &tmp2, n, h, eps)
            .map_err(ehip("ln2"))?;
        let fc1 = linear_f16(gpu, &lw.fc1_w, &tmp2, Some(&lw.fc1_b), cfg.mlp_dim, h, n)
            .map_err(ehip("fc1"))?;
        gpu.free_tensor(tmp2).map_err(ehip("free ln2"))?;
        gpu.gelu_tanh_f32(&fc1, &fc1, n * cfg.mlp_dim).map_err(ehip("gelu(tanh)"))?;
        let fc2 = linear_f16(gpu, &lw.fc2_w, &fc1, Some(&lw.fc2_b), h, cfg.mlp_dim, n)
            .map_err(ehip("fc2"))?;
        gpu.free_tensor(fc1).map_err(ehip("free fc1"))?;
        gpu.add_inplace_f32(&x, &fc2).map_err(ehip("resid2"))?;
        gpu.free_tensor(fc2).map_err(ehip("free fc2"))?;
    }

    // post_layernorm (final LN — no pooling head)
    let normed = gpu.alloc_tensor(&[n * h], DType::F32).map_err(ehip("alloc post-ln"))?;
    gpu.layernorm_batched(&x, &weights.post_ln_w, &weights.post_ln_b, &normed, n, h, eps)
        .map_err(ehip("post-ln"))?;
    gpu.free_tensor(x).map_err(ehip("free tower out"))?;

    // download for CPU rearranges (small buffers)
    let feats = gpu.download_f32(&normed).map_err(ehip("download tower"))?;
    gpu.free_tensor(normed).map_err(ehip("free post-ln"))?;
    gpu.hip.device_synchronize().map_err(ehip("post-tower sync"))?;

    // 2×2 pixel-unshuffle merge → [tok, 4608]
    let ds = cfg.downsample_factor;
    let merged = pixel_unshuffle_tokens(&feats, sub.gh(cfg), sub.gw(cfg), h, ds);
    let tok = merged.len() / (h * ds * ds);

    // projector linear_1 → erf-GELU (host, exact) → linear_2
    let m1_in = gpu.upload_f32(&merged, &[merged.len()]).map_err(ehip("upload merged"))?;
    let mid_gpu = linear_f16(
        gpu,
        &weights.proj1_w,
        &m1_in,
        Some(&weights.proj1_b),
        cfg.projector_hidden_size,
        h * ds * ds,
        tok,
    )
    .map_err(ehip("proj1"))?;
    gpu.free_tensor(m1_in).ok();
    let mut mid = gpu.download_f32(&mid_gpu).map_err(ehip("download proj1"))?;
    gpu.free_tensor(mid_gpu).ok();
    gelu_exact_inplace(&mut mid);
    let act = gpu.upload_f32(&mid, &[mid.len()]).map_err(ehip("re-upload act"))?;

    let y = linear_f16(gpu, &weights.proj2_w, &act, Some(&weights.proj2_b), cfg.out_hidden_size, cfg.projector_hidden_size, tok)
        .map_err(ehip("proj2"))?;
    gpu.free_tensor(act).ok();
    let result = gpu.download_f32(&y).map_err(ehip("download proj2"))?;
    gpu.free_tensor(y).ok();
    debug_assert_eq!(result.len(), tok * cfg.out_hidden_size);
    Ok(result)
}

fn ehip(phase: &'static str) -> impl Fn(hip_bridge::HipError) -> String {
    move |e| format!("lfm2-vl vision {phase}: {e:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity case: resizing the table to its own side must return it
    /// unchanged (center exactly on each source sample, weight 1).
    #[test]
    fn pos_resize_identity_at_native_side() {
        let k = 4usize;
        let d = 3usize;
        let table: Vec<f32> = (0..k * k * d).map(|i| i as f32).collect();
        let out = resize_pos_embed(&table, k, d, k, k);
        assert_eq!(out.len(), table.len());
        for (a, b) in out.iter().zip(&table) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// Upscale must interpolate: a 2×2 constant-value table resized to 3×3
    /// keeps the value everywhere (constant field invariant under bilinear).
    #[test]
    fn pos_resize_upscale_constant_field() {
        let k = 2usize;
        let d = 2usize;
        let mut table = vec![0.0f32; k * k * d];
        for r in 0..k {
            for c in 0..k {
                for dd in 0..d {
                    table[(r * k + c) * d + dd] = ((r * k + c) * 10 + dd) as f32;
                }
            }
        }
        let out = resize_pos_embed(&table, k, d, 4, 4);
        assert_eq!(out.len(), 16 * d);
        // Corner (0,0) samples center of cell (0,0): exact table value.
        assert_eq!(out[0], table[0]);
        assert_eq!(out[1], table[1]);
    }

    /// Antialiased downscale conserves local mean better than naive corner
    /// sampling: over a checkerboard table the coarse output between cells
    /// must land near the average, not snap to one cell.
    #[test]
    fn pos_resize_downscale_smooths() {
        let k = 8usize;
        let d = 1usize;
        let mut table = vec![0.0f32; k * k];
        for r in 0..k {
            for c in 0..k {
                table[r * k + c] = if (r + c) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let out = resize_pos_embed(&table, k, d, 2, 2);
        for v in &out {
            assert!(
                *v > 0.15 && *v < 0.85,
                "downscaled checkerboard should approach ~0.5, got {v}"
            );
        }
    }

    /// Locks the LFM2-VL projector interleave derived from HF's swapped-dim
    /// reshape/permute chain: columns pair first (stride C), rows second
    /// (stride 2C). Any future "obvious" reordering fails loudly here.
    #[test]
    fn pixel_unshuffle_index_contract() {
        let gh = 2;
        let gw = 2;
        let ch = 2;
        // feat[patch_row][patch_col][ch] = 100·row + 10·col + ch
        let mut feat = vec![0.0f32; gh * gw * ch];
        for r in 0..gh {
            for c in 0..gw {
                for cc in 0..ch {
                    feat[(r * gw + c) * ch + cc] = (100 * r + 10 * c + cc) as f32;
                }
            }
        }
        let out = pixel_unshuffle_tokens(&feat, gh, gw, ch, 2);
        assert_eq!(out.len(), 1 * ch * 4);
        // single block (br=bc=di=dj):
        assert_eq!(out[(0 * 2 * 2) + (0 * 2) + 0] as usize, 0); // (0,0,c0)
        assert_eq!(out[(0 * 2 * 2) + (1 * 2) + 0] as usize, 10); // dj=1 → col 1
        assert_eq!(out[(1 * 2 * 2) + (0 * 2) + 0] as usize, 100); // di=1 → row 1
        assert_eq!(out[(1 * 2 * 2) + (1 * 2) + 1] as usize, 111);
    }

    #[test]
    fn gelu_exact_known_values() {
        let mut v = vec![0.0f32, 1.0, -1.0];
        gelu_exact_inplace(&mut v);
        assert!(v[0].abs() < 1e-6);
        assert!((v[1] - 0.841_344_7).abs() < 1e-5); // Φ(1)=0.841344…
        assert!((v[2] + 0.158_655_2).abs() < 1e-5); // 1−Φ(1), negative input
    }
}
