// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

//! Compare one DeepSeek V4 routed-expert MQ2-Lloyd or MQ4G256 tensor against
//! the corresponding official shipped FP4+UE8M0 tensor.
//!
//! This is a developer quality oracle, not a product runtime path. The
//! official checkpoint is itself FP4, so its output is a shipped-reference
//! floor rather than a BF16/full-precision reference.
//!
//! Usage:
//!   deepseek4_expert_quant_oracle MODEL.hfq TENSOR SOURCE_WEIGHT SOURCE_SCALE [SAMPLES]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use std::fs;
use std::path::Path;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Clone, Copy)]
struct Metrics {
    nrmse: f64,
    cosine: f64,
    max_abs: f64,
    ref_rms: f64,
}

fn gen_fwht_signs(seed: u32) -> [f32; 256] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        if (state >> 16) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    })
}

fn inverse_fwht_256(x: &mut [f32; 256], signs1: &[f32; 256], signs2: &[f32; 256]) {
    for i in 0..256 {
        x[i] *= signs2[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut base = 0;
        while base < 256 {
            for j in 0..stride {
                let a = x[base + j];
                let b = x[base + j + stride];
                x[base + j] = a + b;
                x[base + j + stride] = a - b;
            }
            base += 2 * stride;
        }
        stride *= 2;
    }
    for i in 0..256 {
        x[i] *= 0.0625 * signs1[i];
    }
}

fn dequant_official_fp4(
    weight: &[u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if cols % 32 != 0 || cols % 2 != 0 {
        return Err(format!(
            "official FP4 logical cols must be divisible by 32, got {cols}"
        ));
    }
    let expected_weight = rows * cols / 2;
    let expected_scale = rows * (cols / 32);
    if weight.len() != expected_weight || scale.len() != expected_scale {
        return Err(format!(
            "official FP4 fixture size mismatch: weight={} expected={} scale={} expected={}",
            weight.len(),
            expected_weight,
            scale.len(),
            expected_scale
        ));
    }
    let stored_cols = cols / 2;
    let scale_cols = cols / 32;
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let packed = weight[row * stored_cols + col / 2];
            let code = if col & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let exponent = scale[row * scale_cols + col / 32] as i32 - 127;
            out[row * cols + col] = E2M1[code as usize] * (exponent as f32).exp2();
        }
    }
    Ok(out)
}

fn dequant_mq2_lloyd(
    data: &[u8],
    n: usize,
    signs1: &[f32; 256],
    signs2: &[f32; 256],
) -> Result<Vec<f32>, String> {
    if n % 256 != 0 {
        return Err(format!(
            "MQ2-Lloyd element count must be divisible by 256, got {n}"
        ));
    }
    let expected = (n / 256) * 72;
    if data.len() != expected {
        return Err(format!(
            "MQ2-Lloyd byte size {} != expected {expected}",
            data.len()
        ));
    }
    let mut out = vec![0.0f32; n];
    for block in 0..n / 256 {
        let src = &data[block * 72..(block + 1) * 72];
        let codebook = std::array::from_fn::<_, 4, _>(|i| {
            f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))
        });
        let mut group = [0.0f32; 256];
        for i in 0..256 {
            let packed = src[8 + i / 4];
            let code = ((packed >> (2 * (i % 4))) & 3) as usize;
            group[i] = codebook[code];
        }
        inverse_fwht_256(&mut group, signs1, signs2);
        out[block * 256..(block + 1) * 256].copy_from_slice(&group);
    }
    Ok(out)
}

fn dequant_mq4g256(
    data: &[u8],
    n: usize,
    signs1: &[f32; 256],
    signs2: &[f32; 256],
) -> Result<Vec<f32>, String> {
    if n % 256 != 0 {
        return Err(format!(
            "MQ4G256 element count must be divisible by 256, got {n}"
        ));
    }
    let expected = (n / 256) * 136;
    if data.len() != expected {
        return Err(format!(
            "MQ4G256 byte size {} != expected {expected}",
            data.len()
        ));
    }
    let mut out = vec![0.0f32; n];
    for block in 0..n / 256 {
        let src = &data[block * 136..(block + 1) * 136];
        let scale = f32::from_bits(u32::from_le_bytes(src[0..4].try_into().unwrap()));
        let zero = f32::from_bits(u32::from_le_bytes(src[4..8].try_into().unwrap()));
        let mut group = [0.0f32; 256];
        for i in 0..256 {
            let packed = src[8 + i / 2];
            let code = if i & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            group[i] = scale * code as f32 + zero;
        }
        inverse_fwht_256(&mut group, signs1, signs2);
        out[block * 256..(block + 1) * 256].copy_from_slice(&group);
    }
    Ok(out)
}

fn metrics(reference: &[f32], candidate: &[f32]) -> Metrics {
    assert_eq!(reference.len(), candidate.len());
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut cand2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&r, &c) in reference.iter().zip(candidate) {
        let r = r as f64;
        let c = c as f64;
        let d = c - r;
        err2 += d * d;
        ref2 += r * r;
        cand2 += c * c;
        dot += r * c;
        max_abs = max_abs.max(d.abs());
    }
    Metrics {
        nrmse: (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt(),
        cosine: dot / (ref2 * cand2).sqrt().max(f64::MIN_POSITIVE),
        max_abs,
        ref_rms: (ref2 / reference.len() as f64).sqrt(),
    }
}

fn deterministic_input(k: usize, sample: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9u32 ^ (sample as u32).wrapping_mul(0x85eb_ca6b);
    let mut x = Vec::with_capacity(k);
    for _ in 0..k {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        x.push((state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32);
    }
    let rms = (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64).sqrt();
    for v in &mut x {
        *v = (*v as f64 / rms) as f32;
    }
    x
}

fn gemv(weight: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let w = &weight[row * cols..(row + 1) * cols];
        let mut sum = 0.0f64;
        for col in 0..cols {
            sum += w[col] as f64 * x[col] as f64;
        }
        out[row] = sum as f32;
    }
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ *byte as u64).wrapping_mul(0x1000_0000_01b3)
    })
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let hfq_path = args.next().ok_or_else(|| {
        "usage: deepseek4_expert_quant_oracle MODEL.hfq TENSOR SOURCE_WEIGHT SOURCE_SCALE [SAMPLES]"
            .to_string()
    })?;
    let tensor_name = args.next().ok_or_else(|| "missing TENSOR".to_string())?;
    let source_weight_path = args
        .next()
        .ok_or_else(|| "missing SOURCE_WEIGHT".to_string())?;
    let source_scale_path = args
        .next()
        .ok_or_else(|| "missing SOURCE_SCALE".to_string())?;
    let samples = args
        .next()
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("invalid SAMPLES: {e}"))
        })
        .transpose()?
        .unwrap_or(8);

    let hfq = HfqFile::open(Path::new(&hfq_path)).map_err(|e| e.to_string())?;
    let tensor = hfq
        .tensors()
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| format!("HFQ tensor '{tensor_name}' not found"))?;
    if !matches!(tensor.quant_type, 13 | 19) || tensor.shape.len() != 2 {
        return Err(format!(
            "HFQ tensor must be 2D MQ4G256 qt=13 or MQ2G256Lloyd qt=19, got qt={} shape={:?}",
            tensor.quant_type, tensor.shape
        ));
    }
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    let (_, hfq_bytes) = hfq
        .tensor_data_vec(&tensor_name)
        .ok_or_else(|| format!("cannot read HFQ tensor '{tensor_name}'"))?;
    let source_weight =
        fs::read(&source_weight_path).map_err(|e| format!("read {source_weight_path}: {e}"))?;
    let source_scale =
        fs::read(&source_scale_path).map_err(|e| format!("read {source_scale_path}: {e}"))?;

    let reference = dequant_official_fp4(&source_weight, &source_scale, rows, cols)?;
    let signs1 = gen_fwht_signs(42);
    let signs2 = gen_fwht_signs(1042);
    let candidate = match tensor.quant_type {
        13 => dequant_mq4g256(&hfq_bytes, rows * cols, &signs1, &signs2)?,
        19 => dequant_mq2_lloyd(&hfq_bytes, rows * cols, &signs1, &signs2)?,
        _ => unreachable!(),
    };
    if reference.iter().chain(&candidate).any(|v| !v.is_finite()) {
        return Err("non-finite dequantized weight detected".to_string());
    }

    let wm = metrics(&reference, &candidate);
    let mut ref_outputs = Vec::with_capacity(samples * rows);
    let mut cand_outputs = Vec::with_capacity(samples * rows);
    for sample in 0..samples {
        let x = deterministic_input(cols, sample);
        ref_outputs.extend(gemv(&reference, &x, rows, cols));
        cand_outputs.extend(gemv(&candidate, &x, rows, cols));
    }
    let om = metrics(&ref_outputs, &cand_outputs);

    println!("reference=official-shipped-fp4 (not BF16/full precision)");
    println!("hfq={hfq_path}");
    println!(
        "tensor={tensor_name} qt={} shape=[{rows},{cols}] samples={samples}",
        tensor.quant_type
    );
    println!(
        "source_weight={} bytes={} fnv64=0x{:016x}",
        source_weight_path,
        source_weight.len(),
        fnv1a64(&source_weight)
    );
    println!(
        "source_scale={} bytes={} fnv64=0x{:016x}",
        source_scale_path,
        source_scale.len(),
        fnv1a64(&source_scale)
    );
    println!(
        "hfq_tensor_bytes={} fnv64=0x{:016x}",
        hfq_bytes.len(),
        fnv1a64(&hfq_bytes)
    );
    println!(
        "weight nrmse={:.9e} cosine={:.9} max_abs={:.9e} ref_rms={:.9e}",
        wm.nrmse, wm.cosine, wm.max_abs, wm.ref_rms
    );
    println!(
        "gemv   nrmse={:.9e} cosine={:.9} max_abs={:.9e} ref_rms={:.9e}",
        om.nrmse, om.cosine, om.max_abs, om.ref_rms
    );
    Ok(())
}
