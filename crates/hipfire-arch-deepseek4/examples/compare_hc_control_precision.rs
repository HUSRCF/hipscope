//! Compare DeepSeek V4 mHC controls from source-F32 and HFQ-F16 weights.
//!
//! The residual stream input is a raw F32 dump from the real GPU forward.
//! This probe performs both weight paths with identical CPU reduction order,
//! isolating the error introduced by storing mHC parameters as F16.
//!
//! Usage:
//!   compare_hc_control_precision MODEL.hfq STREAMS_F32.bin SOURCE_F32.bin \
//!       PREFIX BASE_OFFSET FN_OFFSET SCALE_OFFSET

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use std::path::Path;

fn read_f16(hfq: &HfqFile, name: &str) -> Result<Vec<f64>, String> {
    let (info, bytes) = hfq
        .tensor_data_pread(name)
        .ok_or_else(|| format!("HFQ tensor not found: {name}"))?;
    if info.quant_type != 1 || bytes.len() % 2 != 0 {
        return Err(format!(
            "{name}: expected F16 quant_type=1, got qt={} bytes={}",
            info.quant_type,
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
        .collect())
}

fn read_f32_slice(bytes: &[u8], offset: usize, n: usize) -> Result<Vec<f64>, String> {
    let end = offset
        .checked_add(n * 4)
        .ok_or_else(|| "F32 source range overflow".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "F32 source range {offset}..{end} exceeds {} bytes",
            bytes.len()
        ));
    }
    Ok(bytes[offset..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .collect())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sinkhorn(mut matrix: Vec<f64>, eps: f64, iters: usize) -> Vec<f64> {
    for row in matrix.chunks_exact_mut(4) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
        for v in row {
            *v = (*v - max).exp() / sum + eps;
        }
    }
    for col in 0..4 {
        let sum: f64 = (0..4).map(|row| matrix[row * 4 + col]).sum::<f64>() + eps;
        for row in 0..4 {
            matrix[row * 4 + col] /= sum;
        }
    }
    for _ in 0..iters.saturating_sub(1) {
        for row in matrix.chunks_exact_mut(4) {
            let sum: f64 = row.iter().sum::<f64>() + eps;
            for v in row {
                *v /= sum;
            }
        }
        for col in 0..4 {
            let sum: f64 = (0..4).map(|row| matrix[row * 4 + col]).sum::<f64>() + eps;
            for row in 0..4 {
                matrix[row * 4 + col] /= sum;
            }
        }
    }
    matrix
}

fn report(label: &str, source: &[f64], local: &[f64]) {
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut source_sum_sq = 0.0f64;
    for (&a, &b) in source.iter().zip(local) {
        let err = (a - b).abs();
        max_abs = max_abs.max(err);
        sum_sq += err * err;
        source_sum_sq += a * a;
    }
    let n = source.len() as f64;
    let rms = (sum_sq / n).sqrt();
    let source_rms = (source_sum_sq / n).sqrt();
    println!(
        "{label}: max_abs={max_abs:.12e} rms={rms:.12e} nrmse={:.12e}",
        if source_rms == 0.0 {
            0.0
        } else {
            rms / source_rms
        }
    );
}

fn controls(x: &[f64], w: &[f64], base: &[f64], scale: &[f64]) -> Vec<f64> {
    let inv_rms = 1.0 / ((x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64) + 1e-6).sqrt();
    (0..24)
        .map(|ctrl| {
            let row = &w[ctrl * x.len()..(ctrl + 1) * x.len()];
            let dot: f64 = x.iter().zip(row).map(|(&a, &b)| a * b).sum();
            let segment = if ctrl < 4 {
                0
            } else if ctrl < 8 {
                1
            } else {
                2
            };
            dot * inv_rms * scale[segment] + base[ctrl]
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err("usage: compare_hc_control_precision MODEL.hfq STREAMS_F32.bin SOURCE_F32.bin PREFIX BASE_OFFSET FN_OFFSET SCALE_OFFSET".into());
    }
    let base_offset: usize = args[4].parse()?;
    let fn_offset: usize = args[5].parse()?;
    let scale_offset: usize = args[6].parse()?;
    let prefix = &args[3];

    let mut hfq = HfqFile::open_at_offset(Path::new(&args[0]), 0)?;
    hfq.drop_mmap();
    let local_base = read_f16(&hfq, &format!("{prefix}_base"))?;
    let local_fn = read_f16(&hfq, &format!("{prefix}_fn"))?;
    let local_scale = read_f16(&hfq, &format!("{prefix}_scale"))?;
    if local_base.len() != 24 || local_scale.len() != 3 || local_fn.len() % 24 != 0 {
        return Err(format!(
            "unexpected HC shapes: base={} fn={} scale={}",
            local_base.len(),
            local_fn.len(),
            local_scale.len()
        )
        .into());
    }
    let x_dim = local_fn.len() / 24;

    let streams_bytes = std::fs::read(&args[1])?;
    let x = read_f32_slice(&streams_bytes, 0, x_dim)?;
    let source_bytes = std::fs::read(&args[2])?;
    let source_base = read_f32_slice(&source_bytes, base_offset, 24)?;
    let source_fn = read_f32_slice(&source_bytes, fn_offset, 24 * x_dim)?;
    let source_scale = read_f32_slice(&source_bytes, scale_offset, 3)?;

    let source_c = controls(&x, &source_fn, &source_base, &source_scale);
    let local_c = controls(&x, &local_fn, &local_base, &local_scale);
    let source_pre: Vec<f64> = source_c[..4].iter().map(|&v| sigmoid(v) + 1e-6).collect();
    let local_pre: Vec<f64> = local_c[..4].iter().map(|&v| sigmoid(v) + 1e-6).collect();
    let source_post: Vec<f64> = source_c[4..8].iter().map(|&v| 2.0 * sigmoid(v)).collect();
    let local_post: Vec<f64> = local_c[4..8].iter().map(|&v| 2.0 * sigmoid(v)).collect();
    let source_comb = sinkhorn(source_c[8..24].to_vec(), 1e-6, 20);
    let local_comb = sinkhorn(local_c[8..24].to_vec(), 1e-6, 20);

    println!("prefix={prefix} x_dim={x_dim}");
    report("control", &source_c, &local_c);
    report("pre", &source_pre, &local_pre);
    report("post", &source_post, &local_post);
    report("comb", &source_comb, &local_comb);
    println!("source_pre={source_pre:?}");
    println!("local_pre={local_pre:?}");
    println!("source_post={source_post:?}");
    println!("local_post={local_post:?}");
    Ok(())
}
