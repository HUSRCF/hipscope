// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

//! Real-checkpoint parity probe for DeepSeek-V4 native routed FP4 experts.
//!
//! Reads one official E2M1+UE8M0 expert directly from safetensors, losslessly
//! reframes it as HFP4G32, and compares:
//!   1. an independent CPU reference;
//!   2. the generic HFP4G32 GEMV correctness anchor;
//!   3. the gfx90a indexed wave64 gate/up and down kernels.

use hipfire_runtime::model_source::{open_model, ModelSource};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

struct Matrix {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    reference: Vec<f32>,
}

fn ue8m0(byte: u8) -> Result<f32, String> {
    match byte {
        255 => Err("reserved UE8M0 NaN scale".into()),
        0 => Ok(f32::from_bits(0x0040_0000)), // 2^-127
        exponent => Ok(f32::from_bits((exponent as u32) << 23)),
    }
}

fn load_matrix(source: &dyn ModelSource, weight_name: &str) -> Result<Matrix, String> {
    let (weight_info, weight) = source
        .tensor_data(weight_name)
        .ok_or_else(|| format!("missing {weight_name}"))?;
    if weight_info.dtype != "I8" || weight_info.shape.len() != 2 {
        return Err(format!(
            "{weight_name}: expected rank-2 I8, got {} {:?}",
            weight_info.dtype, weight_info.shape
        ));
    }
    let rows = weight_info.shape[0];
    let cols = weight_info.shape[1] * 2;
    if cols % 32 != 0 {
        return Err(format!("{weight_name}: logical K={cols} is not G32"));
    }
    let scale_name = format!(
        "{}.scale",
        weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| format!("bad weight name {weight_name}"))?
    );
    let (scale_info, scales) = source
        .tensor_data(&scale_name)
        .ok_or_else(|| format!("missing {scale_name}"))?;
    let blocks = cols / 32;
    if scale_info.dtype != "F8_E8M0" || scale_info.shape != [rows, blocks] {
        return Err(format!(
            "{scale_name}: expected F8_E8M0 [{rows}, {blocks}], got {} {:?}",
            scale_info.dtype, scale_info.shape
        ));
    }
    if weight.len() != rows * cols / 2 || scales.len() != rows * blocks {
        return Err(format!("{weight_name}: malformed payload lengths"));
    }

    let row_bytes = 16 + blocks * 17;
    let mut packed = vec![0_u8; rows * row_bytes];
    let mut reference = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let packed_row = &weight[row * cols / 2..(row + 1) * cols / 2];
        let out_row = &mut packed[row * row_bytes..(row + 1) * row_bytes];
        out_row[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // f16(1.0)
        out_row[4..6].copy_from_slice(&(blocks as u16).to_le_bytes());
        for block in 0..blocks {
            let exponent = scales[row * blocks + block];
            let scale = ue8m0(exponent)
                .map_err(|error| format!("{scale_name} row={row} block={block}: {error}"))?;
            let src = &packed_row[block * 16..(block + 1) * 16];
            let dst = 16 + block * 17;
            out_row[dst] = exponent;
            out_row[dst + 1..dst + 17].copy_from_slice(src);
            for (byte_idx, byte) in src.iter().copied().enumerate() {
                let col = block * 32 + byte_idx * 2;
                reference[row * cols + col] = E2M1[(byte & 0x0f) as usize] * scale;
                reference[row * cols + col + 1] = E2M1[(byte >> 4) as usize] * scale;
            }
        }
    }
    Ok(Matrix {
        rows,
        cols,
        packed,
        reference,
    })
}

fn deterministic_input(cols: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(cols);
    for _ in 0..cols {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push((state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32);
    }
    let rms = (out
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / cols as f64)
        .sqrt();
    for value in &mut out {
        *value = (f64::from(*value) / rms) as f32;
    }
    out
}

fn cpu_gemv(matrix: &Matrix, x: &[f32]) -> Vec<f32> {
    assert_eq!(matrix.cols, x.len());
    matrix
        .reference
        .chunks_exact(matrix.cols)
        .map(|row| {
            row.iter()
                .zip(x)
                .fold(0.0_f32, |sum, (&weight, &activation)| {
                    weight.mul_add(activation, sum)
                })
        })
        .collect()
}

fn metrics(label: &str, reference: &[f32], candidate: &[f32]) -> bool {
    assert_eq!(reference.len(), candidate.len());
    let mut error2 = 0.0_f64;
    let mut reference2 = 0.0_f64;
    let mut candidate2 = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference);
        let candidate = f64::from(candidate);
        let error = candidate - reference;
        error2 += error * error;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
        max_abs = max_abs.max(error.abs());
    }
    let nrmse = (error2 / reference2.max(f64::MIN_POSITIVE)).sqrt();
    let cosine = dot / (reference2 * candidate2).sqrt().max(f64::MIN_POSITIVE);
    let reference_rms = (reference2 / reference.len() as f64).sqrt();
    let candidate_rms = (candidate2 / candidate.len() as f64).sqrt();
    let pass = nrmse < 2.0e-4 && cosine > 0.999_999;
    println!(
        "[{status}] {label}: nrmse={nrmse:.7e} cosine={cosine:.9} max_abs={max_abs:.7e} ref_rms={reference_rms:.7e} got_rms={candidate_rms:.7e}",
        status = if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn upload_matrix(gpu: &mut Gpu, matrix: &Matrix) -> Result<GpuTensor, String> {
    gpu.upload_raw(&matrix.packed, &[matrix.packed.len()])
        .map_err(|error| format!("upload matrix: {error:?}"))
}

fn pointer_table(gpu: &mut Gpu, matrix: &GpuTensor) -> Result<GpuTensor, String> {
    let pointer = matrix.buf.as_ptr() as u64;
    gpu.upload_raw(&pointer.to_ne_bytes(), &[8])
        .map_err(|error| format!("upload pointer table: {error:?}"))
}

fn run_gate_up(gpu: &mut Gpu, w1: &Matrix, w3: &Matrix, x: &[f32]) -> Result<bool, String> {
    if w1.rows != w3.rows || w1.cols != w3.cols {
        return Err("w1/w3 shape mismatch".into());
    }
    let cpu_gate = cpu_gemv(w1, x);
    let cpu_up = cpu_gemv(w3, x);
    let mut combined = w1.packed.clone();
    combined.extend_from_slice(&w3.packed);
    let combined_matrix = Matrix {
        rows: w1.rows + w3.rows,
        cols: w1.cols,
        packed: combined,
        reference: Vec::new(),
    };
    let d_combined = upload_matrix(gpu, &combined_matrix)?;
    let d_ptrs = pointer_table(gpu, &d_combined)?;
    let d_indices = gpu
        .upload_raw(&0_i32.to_le_bytes(), &[1])
        .map_err(|error| format!("upload index: {error:?}"))?;
    let d_x = gpu
        .upload_f32(x, &[x.len()])
        .map_err(|error| format!("upload x: {error:?}"))?;
    let d_gate = gpu
        .zeros(&[w1.rows], DType::F32)
        .map_err(|error| format!("alloc gate: {error:?}"))?;
    let d_up = gpu
        .zeros(&[w3.rows], DType::F32)
        .map_err(|error| format!("alloc up: {error:?}"))?;
    gpu.gemv_hfp4g32_moe_gate_up_indexed_batched(
        &d_ptrs,
        &d_indices,
        &d_x,
        &d_gate,
        &d_up,
        w1.rows + w3.rows,
        w1.cols,
        1,
        1,
    )
    .map_err(|error| format!("wave64 gate/up: {error:?}"))?;
    let wave_gate = gpu
        .download_f32(&d_gate)
        .map_err(|error| format!("download gate: {error:?}"))?;
    let wave_up = gpu
        .download_f32(&d_up)
        .map_err(|error| format!("download up: {error:?}"))?;

    let d_generic_gate = gpu
        .zeros(&[w1.rows], DType::F32)
        .map_err(|error| format!("alloc generic gate: {error:?}"))?;
    let d_generic_up = gpu
        .zeros(&[w3.rows], DType::F32)
        .map_err(|error| format!("alloc generic up: {error:?}"))?;
    gpu.gemv_hfp4g32(&d_combined, &d_x, &d_generic_gate, w1.rows, w1.cols)
        .map_err(|error| format!("generic gate: {error:?}"))?;
    let up_view = d_combined.sub_offset(w1.packed.len(), w3.packed.len());
    gpu.gemv_hfp4g32(&up_view, &d_x, &d_generic_up, w3.rows, w3.cols)
        .map_err(|error| format!("generic up: {error:?}"))?;
    let generic_gate = gpu
        .download_f32(&d_generic_gate)
        .map_err(|error| format!("download generic gate: {error:?}"))?;
    let generic_up = gpu
        .download_f32(&d_generic_up)
        .map_err(|error| format!("download generic up: {error:?}"))?;

    Ok(metrics("w1 CPU vs generic", &cpu_gate, &generic_gate)
        & metrics("w1 CPU vs wave64", &cpu_gate, &wave_gate)
        & metrics("w1 generic vs wave64", &generic_gate, &wave_gate)
        & metrics("w3 CPU vs generic", &cpu_up, &generic_up)
        & metrics("w3 CPU vs wave64", &cpu_up, &wave_up)
        & metrics("w3 generic vs wave64", &generic_up, &wave_up))
}

fn run_down(gpu: &mut Gpu, w2: &Matrix, x: &[f32]) -> Result<bool, String> {
    let cpu = cpu_gemv(w2, x);
    let d_weight = upload_matrix(gpu, w2)?;
    let d_ptrs = pointer_table(gpu, &d_weight)?;
    let d_indices = gpu
        .upload_raw(&0_i32.to_le_bytes(), &[1])
        .map_err(|error| format!("upload index: {error:?}"))?;
    let d_weights = gpu
        .upload_f32(&[1.0], &[1])
        .map_err(|error| format!("upload route weight: {error:?}"))?;
    let d_x = gpu
        .upload_f32(x, &[x.len()])
        .map_err(|error| format!("upload x: {error:?}"))?;
    let d_wave = gpu
        .zeros(&[w2.rows], DType::F32)
        .map_err(|error| format!("alloc wave down: {error:?}"))?;
    let d_generic = gpu
        .zeros(&[w2.rows], DType::F32)
        .map_err(|error| format!("alloc generic down: {error:?}"))?;
    gpu.gemv_hfp4g32_moe_down_residual_scaled_indexed_batched(
        &d_ptrs, &d_indices, &d_weights, &d_x, &d_wave, w2.rows, w2.cols, 1, 1,
    )
    .map_err(|error| format!("wave64 down: {error:?}"))?;
    gpu.gemv_hfp4g32(&d_weight, &d_x, &d_generic, w2.rows, w2.cols)
        .map_err(|error| format!("generic down: {error:?}"))?;
    let wave = gpu
        .download_f32(&d_wave)
        .map_err(|error| format!("download wave down: {error:?}"))?;
    let generic = gpu
        .download_f32(&d_generic)
        .map_err(|error| format!("download generic down: {error:?}"))?;
    Ok(metrics("w2 CPU vs generic", &cpu, &generic)
        & metrics("w2 CPU vs wave64", &cpu, &wave)
        & metrics("w2 generic vs wave64", &generic, &wave))
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(|| {
        "usage: test_deepseek4_source_hfp4_moe_gfx90a MODEL_DIR [LAYER] [EXPERT]".to_string()
    })?;
    let layer = args
        .next()
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(0);
    let expert = args
        .next()
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(0);
    let source = open_model(Path::new(&model))?;
    let prefix = format!("layers.{layer}.ffn.experts.{expert}");
    println!("source={} prefix={prefix}", source.path().display());
    let w1 = load_matrix(source.as_ref(), &format!("{prefix}.w1.weight"))?;
    let w3 = load_matrix(source.as_ref(), &format!("{prefix}.w3.weight"))?;
    let w2 = load_matrix(source.as_ref(), &format!("{prefix}.w2.weight"))?;
    println!(
        "w1={:?} w3={:?} w2={:?}",
        [w1.rows, w1.cols],
        [w3.rows, w3.cols],
        [w2.rows, w2.cols]
    );

    let mut gpu = Gpu::init().map_err(|error| format!("Gpu::init: {error:?}"))?;
    println!("arch={}", gpu.arch);
    let x_gate = deterministic_input(w1.cols, 0x0731_0001);
    let x_down = deterministic_input(w2.cols, 0x0731_0002);
    let gate_up_ok = run_gate_up(&mut gpu, &w1, &w3, &x_gate)?;
    let down_ok = run_down(&mut gpu, &w2, &x_down)?;
    if !gate_up_ok || !down_ok {
        return Err("native DeepSeek-V4 routed FP4 parity failed".into());
    }
    println!("ALL PASS");
    Ok(())
}
