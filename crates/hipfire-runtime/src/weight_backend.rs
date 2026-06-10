// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-agnostic weight loading: dequant primitives, HF tensor-name resolution,
//! and a `WeightBackend` trait abstracting HFQ vs ParoQuant on-disk formats.
//! Per-arch crates build their `load_layer` schema on top of this; the only
//! arch-varying knobs are the RMSNorm `+bias` and the name-candidate resolver.

use crate::llama::{f16_to_f32, KvCache, WeightTensor};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ── HF tensor-name resolution ───────────────────────────────────────────────

/// Candidate on-disk names for a logical tensor, covering the HF nested
/// vision-wrapper layout (`model.language_model.*`), the flat layout (`model.*`),
/// and the bare name, plus the `lm_head` special-case. Layout convention only —
/// not model-specific math — so any HF text tower can share it.
pub fn hf_name_candidates(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(4);
    let mut push = |s: String| {
        if !out.iter().any(|x| x == &s) {
            out.push(s);
        }
    };
    if name == "lm_head.weight" {
        push(name.to_string());
        push("model.language_model.lm_head.weight".to_string());
        push("model.lm_head.weight".to_string());
        return out;
    }
    if name.starts_with("model.") {
        push(name.to_string());
    } else {
        push(format!("model.language_model.{name}"));
        push(format!("model.{name}"));
        push(name.to_string());
    }
    out
}

/// Flat-only resolver for arches stored without the vision-wrapper nesting
/// (qwen2, llama). Tries `model.{name}` then bare.
pub fn flat_name_candidates(name: &str) -> Vec<String> {
    if name.starts_with("model.") {
        vec![name.to_string()]
    } else {
        vec![format!("model.{name}"), name.to_string()]
    }
}

// ── Layer-relative name builders ────────────────────────────────────────────

/// HFQ projection name: `layers.{layer}.{rel}.weight` (the backend's candidate
/// resolver then adds any layout prefix).
pub fn hfq_proj_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}.weight")
}
/// HFQ norm / raw-f32 name: `layers.{layer}.{rel}` (rel already carries `.weight`
/// where the on-disk tensor has it, e.g. `input_layernorm.weight`).
pub fn hfq_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}
/// PaRo projection base (augmentor appends `.qweight`/`.weight`): `{mp}.layers.{layer}.{rel}`.
pub fn paro_proj_name(mp: &str, layer: usize, rel: &str) -> String {
    format!("{mp}.layers.{layer}.{rel}")
}
/// PaRo norm/raw-f32 name for `paro_load_norm`/`paro_load_f32`, which prepend `mp`
/// THEMSELVES — so this is prefix-LESS: `layers.{layer}.{rel}`.
pub fn paro_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}

// ── Dequant primitives (bodies filled in Task 2) ────────────────────────────

/// Quant `data` → device `WeightTensor [m, k]`. Moved from
/// `hipfire-arch-qwen35::qwen35::load_weight_tensor_raw` (Task 2).
pub fn dequant_weight_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    match quant_type {
        6 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        15 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            assert!(
                k % 256 == 0,
                "HFP4G32 v1 lm_head has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            assert!(
                k % 256 == 0,
                "MFP4G32 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        2 => {
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        16 => {
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        _ => panic!("unsupported quant_type {quant_type} for dequant_weight_raw"),
    }
}

/// RMSNorm scale `data` → device `GpuTensor [shape]`, adding `bias` to every
/// element (`1.0` for qwen3.5/gemma, `0.0` for qwen2/llama/minimax). Moved from
/// `load_norm_weight` (Task 2), with the `+= 1.0` generalised to `+= bias`.
pub fn dequant_norm(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    shape: &[usize],
    bias: f32,
) -> HipResult<GpuTensor> {
    let mut f32_data: Vec<f32> = match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => panic!("expected F16/F32 for norm, got qt={quant_type}"),
    };
    for v in &mut f32_data {
        *v += bias;
    }
    gpu.upload_f32(&f32_data, shape)
}

/// Raw f16/f32 `data` → device `GpuTensor [n]` (no bias). Moved from
/// `load_any_as_f32` (Task 2).
pub fn dequant_f32(gpu: &mut Gpu, quant_type: u8, data: &[u8], n: usize) -> HipResult<GpuTensor> {
    let f32_data: Vec<f32> = match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => crate::llama::dequantize_q8_0(data, n),
        14 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let inv_s = 0.0625;
                for i in 0..256 {
                    group[i] *= inv_s * signs1[i];
                }
            }
            out
        }
        6 | 7 | 13 | 15 => {
            let is_6bit = quant_type == 15;
            let group_size: usize =
                if quant_type == 6 || quant_type == 13 || quant_type == 15 {
                    256
                } else {
                    128
                };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = quant_type == 13 || quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (
                    Some(KvCache::gen_fwht_signs(42, 256)),
                    Some(KvCache::gen_fwht_signs(1042, 256)),
                )
            } else {
                (None, None)
            };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_6bit {
                    for i in (0..group_size).step_by(4) {
                        let bo = off + 8 + (i / 4) * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        out.push(scale * ((b0 & 0x3F) as f32) + zero);
                        out.push(scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero);
                        out.push(scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero);
                        out.push(scale * (((b2 >> 2) & 0x3F) as f32) + zero);
                    }
                } else {
                    for i in 0..group_size {
                        let byte_idx = i / 2;
                        let byte_val = data[off + 8 + byte_idx];
                        let nibble = if i % 2 == 0 {
                            byte_val & 0xF
                        } else {
                            byte_val >> 4
                        };
                        out.push(scale * nibble as f32 + zero);
                    }
                }
                if is_mq && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    for i in 0..256 {
                        group[i] *= s2[i];
                    }
                    let mut stride = 1;
                    while stride < 256 {
                        let mut j = 0;
                        while j < 256 {
                            for k in 0..stride {
                                let a = group[j + k];
                                let b = group[j + k + stride];
                                group[j + k] = a + b;
                                group[j + k + stride] = a - b;
                            }
                            j += stride * 2;
                        }
                        stride <<= 1;
                    }
                    let scale_inv = 0.0625;
                    for i in 0..256 {
                        group[i] *= scale_inv * s1[i];
                    }
                }
            }
            out
        }
        8 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 200;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for i in (0..group_size).step_by(4) {
                    let byte_off = 8 + (i / 4) * 3;
                    let b0 = data[off + byte_off] as u32;
                    let b1 = data[off + byte_off + 1] as u32;
                    let b2 = data[off + byte_off + 2] as u32;
                    let q0 = (b0 & 0x3F) as f32;
                    let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                    let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                    let q3 = ((b2 >> 2) & 0x3F) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                }
            }
            out
        }
        11 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 104;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for chunk in 0..32 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        12 => {
            let group_size: usize = 128;
            let bytes_per_group: usize = 56;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for chunk in 0..16 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        20 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 112;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 8];
                for k in 0..8 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
                }
                let start = out.len();
                for chunk in 0..32 {
                    let bo = off + 16 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as usize;
                    let q1 = ((b0 >> 3) & 7) as usize;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as usize;
                    let q3 = ((b1 >> 1) & 7) as usize;
                    let q4 = ((b1 >> 4) & 7) as usize;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as usize;
                    let q6 = ((b2 >> 2) & 7) as usize;
                    let q7 = ((b2 >> 5) & 7) as usize;
                    out.push(cb[q0]);
                    out.push(cb[q1]);
                    out.push(cb[q2]);
                    out.push(cb[q3]);
                    out.push(cb[q4]);
                    out.push(cb[q5]);
                    out.push(cb[q6]);
                    out.push(cb[q7]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        19 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 72;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 4];
                for k in 0..4 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..64 {
                    let byte_val = data[off + 8 + i] as usize;
                    out.push(cb[byte_val & 3]);
                    out.push(cb[(byte_val >> 2) & 3]);
                    out.push(cb[(byte_val >> 4) & 3]);
                    out.push(cb[(byte_val >> 6) & 3]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        30 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 160;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 16];
                for k in 0..16 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..128 {
                    let byte_val = data[off + 32 + i] as usize;
                    out.push(cb[byte_val & 0xF]);
                    out.push(cb[(byte_val >> 4) & 0xF]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        17 | 18 => {
            let is_mq3 = quant_type == 17;
            let group_size: usize = 256;
            let bytes_per_group: usize = if is_mq3 { 104 } else { 72 };
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_mq3 {
                    for chunk in 0..32 {
                        let bo = off + 8 + chunk * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        let q0 = (b0 & 7) as f32;
                        let q1 = ((b0 >> 3) & 7) as f32;
                        let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                        let q3 = ((b1 >> 1) & 7) as f32;
                        let q4 = ((b1 >> 4) & 7) as f32;
                        let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                        let q6 = ((b2 >> 2) & 7) as f32;
                        let q7 = ((b2 >> 5) & 7) as f32;
                        out.push(scale * q0 + zero);
                        out.push(scale * q1 + zero);
                        out.push(scale * q2 + zero);
                        out.push(scale * q3 + zero);
                        out.push(scale * q4 + zero);
                        out.push(scale * q5 + zero);
                        out.push(scale * q6 + zero);
                        out.push(scale * q7 + zero);
                    }
                } else {
                    for i in 0..64 {
                        let byte_val = data[off + 8 + i] as u32;
                        out.push(scale * ((byte_val & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 2) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 4) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 6) & 3) as f32) + zero);
                    }
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        _ => panic!("unsupported quant_type {quant_type} for dequant_f32"),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_candidates_cover_both_layouts() {
        let c = hf_name_candidates("layers.0.self_attn.q_proj.weight");
        assert_eq!(c[0], "model.language_model.layers.0.self_attn.q_proj.weight");
        assert_eq!(c[1], "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(c[2], "layers.0.self_attn.q_proj.weight");
    }
    #[test]
    fn lm_head_special_case() {
        let c = hf_name_candidates("lm_head.weight");
        assert_eq!(c[0], "lm_head.weight");
        assert!(c.contains(&"model.language_model.lm_head.weight".to_string()));
    }
    #[test]
    fn flat_candidates_are_two() {
        assert_eq!(flat_name_candidates("layers.0.mlp.down_proj.weight"),
                   vec!["model.layers.0.mlp.down_proj.weight".to_string(),
                        "layers.0.mlp.down_proj.weight".to_string()]);
    }
    #[test]
    fn name_builders() {
        assert_eq!(hfq_proj_name(3, "self_attn.q_proj"), "layers.3.self_attn.q_proj.weight");
        assert_eq!(hfq_plain_name(3, "input_layernorm.weight"), "layers.3.input_layernorm.weight");
        assert_eq!(paro_proj_name("model.language_model", 0, "linear_attn.in_proj_qkv"),
                   "model.language_model.layers.0.linear_attn.in_proj_qkv");
        assert_eq!(paro_plain_name(0, "input_layernorm.weight"), "layers.0.input_layernorm.weight");
    }
}
