//! Compare F32 checkpoint tensor slices with the corresponding F16 HFQ tensors.
//!
//! This is a read-only quality probe. It is intentionally generic: each source
//! argument names a tensor, a raw blob containing its original F32 bytes, and
//! the tensor's byte offset within that blob.
//!
//! Usage:
//!   compare_source_f32 MODEL.hfq TENSOR=BLOB@BYTE_OFFSET [...]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use std::path::Path;

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: compare_source_f32 MODEL.hfq TENSOR=BLOB@BYTE_OFFSET [...]")?;
    let specs: Vec<String> = args.collect();
    if specs.is_empty() {
        return Err("at least one TENSOR=BLOB@BYTE_OFFSET spec is required".into());
    }

    let mut hfq = HfqFile::open_at_offset(Path::new(&model), 0)?;
    hfq.drop_mmap();

    for spec in specs {
        let (name, source_spec) = spec
            .split_once('=')
            .ok_or_else(|| format!("invalid spec (missing '='): {spec}"))?;
        let (source_path, offset_text) = source_spec
            .rsplit_once('@')
            .ok_or_else(|| format!("invalid spec (missing '@'): {spec}"))?;
        let source_offset: usize = offset_text.parse()?;

        let (info, local_bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("HFQ tensor not found: {name}"))?;
        if info.quant_type != 1 {
            return Err(format!(
                "{name}: expected HFQ quant_type=1 (F16), got {}",
                info.quant_type
            )
            .into());
        }
        if local_bytes.len() % 2 != 0 {
            return Err(format!("{name}: odd F16 byte length {}", local_bytes.len()).into());
        }

        let n = local_bytes.len() / 2;
        let source_bytes = std::fs::read(source_path)?;
        let source_end = source_offset
            .checked_add(n * 4)
            .ok_or("source byte range overflow")?;
        if source_end > source_bytes.len() {
            return Err(format!(
                "{name}: source range {source_offset}..{source_end} exceeds {} bytes",
                source_bytes.len()
            )
            .into());
        }

        let mut abs_err = Vec::with_capacity(n);
        let mut sum_sq = 0.0f64;
        let mut source_sum_sq = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut max_source = 0.0f64;
        let mut nonfinite = 0usize;

        for i in 0..n {
            let local = f16_to_f32(u16::from_le_bytes([
                local_bytes[2 * i],
                local_bytes[2 * i + 1],
            ]));
            let p = source_offset + 4 * i;
            let source = f32::from_le_bytes(source_bytes[p..p + 4].try_into()?);
            if !local.is_finite() || !source.is_finite() {
                nonfinite += 1;
                continue;
            }
            let err = (local as f64 - source as f64).abs();
            abs_err.push(err);
            sum_sq += err * err;
            source_sum_sq += (source as f64) * (source as f64);
            sum_abs += err;
            max_abs = max_abs.max(err);
            max_source = max_source.max((source as f64).abs());
        }

        if abs_err.is_empty() {
            return Err(format!("{name}: no finite values to compare").into());
        }
        abs_err.sort_by(f64::total_cmp);
        let count = abs_err.len() as f64;
        let rms = (sum_sq / count).sqrt();
        let source_rms = (source_sum_sq / count).sqrt();
        let nrmse = if source_rms == 0.0 {
            0.0
        } else {
            rms / source_rms
        };

        println!("tensor={name}");
        println!(
            "  shape={:?} elems={} source={}@{} nonfinite={}",
            info.shape, n, source_path, source_offset, nonfinite
        );
        println!("  source_rms={source_rms:.12e} source_max={max_source:.12e}");
        println!(
            "  max_abs={max_abs:.12e} rms={rms:.12e} nrmse={nrmse:.12e} mean_abs={:.12e} p50={:.12e} p99={:.12e} p999={:.12e}",
            sum_abs / count,
            percentile(&abs_err, 0.50),
            percentile(&abs_err, 0.99),
            percentile(&abs_err, 0.999),
        );
    }

    Ok(())
}
