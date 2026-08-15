//! Check whether F32→F16 router-bias storage changes top-k expert selection.
//!
//! Usage:
//!   compare_router_bias_precision MODEL.hfq SCORES_F32.bin SOURCE_BIAS_F32.bin TENSOR [K]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use std::path::Path;

fn read_f32(bytes: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if bytes.len() < n * 4 {
        return Err(format!("need {} F32 bytes, got {}", n * 4, bytes.len()));
    }
    Ok(bytes[..n * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_all_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "F32 byte length {} is not divisible by 4",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn ranking(scores: &[f32], bias: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .zip(bias)
        .enumerate()
        .map(|(i, (&score, &b))| (i, score + b))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: compare_router_bias_precision MODEL.hfq SCORES_F32.bin SOURCE_BIAS_F32.bin TENSOR [K]".into());
    }
    let k: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(6);
    let tensor = &args[3];

    let mut hfq = HfqFile::open_at_offset(Path::new(&args[0]), 0)?;
    hfq.drop_mmap();
    let (info, local_bytes) = hfq
        .tensor_data_pread(tensor)
        .ok_or_else(|| format!("HFQ tensor not found: {tensor}"))?;
    if info.quant_type != 1 || local_bytes.len() % 2 != 0 {
        return Err(format!(
            "{tensor}: expected F16 qt=1, got qt={} bytes={}",
            info.quant_type,
            local_bytes.len()
        )
        .into());
    }
    let local_bias: Vec<f32> = local_bytes
        .chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let n = local_bias.len();
    let source_bias = read_f32(&std::fs::read(&args[2])?, n)?;
    let all_scores = read_all_f32(&std::fs::read(&args[1])?)?;
    if all_scores.len() % n != 0 {
        return Err(format!(
            "score count {} is not divisible by expert count {n}",
            all_scores.len()
        )
        .into());
    }
    let rows = all_scores.len() / n;
    let scores = &all_scores[..n];

    let source_rank = ranking(scores, &source_bias);
    let local_rank = ranking(scores, &local_bias);
    let source_top: Vec<usize> = source_rank[..k].iter().map(|x| x.0).collect();
    let local_top: Vec<usize> = local_rank[..k].iter().map(|x| x.0).collect();
    let changed = source_top.iter().filter(|i| !local_top.contains(i)).count();

    let mut changed_rows = 0usize;
    let mut order_changed_rows = 0usize;
    let mut min_source_margin = f32::INFINITY;
    let mut min_local_margin = f32::INFINITY;
    let mut first_changes = Vec::new();
    for (row_idx, row_scores) in all_scores.chunks_exact(n).enumerate() {
        let source = ranking(row_scores, &source_bias);
        let local = ranking(row_scores, &local_bias);
        let source_ids: Vec<usize> = source[..k].iter().map(|x| x.0).collect();
        let local_ids: Vec<usize> = local[..k].iter().map(|x| x.0).collect();
        if source_ids != local_ids {
            order_changed_rows += 1;
        }
        if source_ids.iter().any(|id| !local_ids.contains(id)) {
            changed_rows += 1;
            if first_changes.len() < 8 {
                first_changes.push((row_idx, source_ids, local_ids));
            }
        }
        min_source_margin = min_source_margin.min(source[k - 1].1 - source[k].1);
        min_local_margin = min_local_margin.min(local[k - 1].1 - local[k].1);
    }

    println!("tensor={tensor} experts={n} k={k} rows={rows}");
    println!(
        "changed_set_rows={changed_rows} ({:.4}%) order_changed_rows={order_changed_rows} ({:.4}%)",
        100.0 * changed_rows as f64 / rows as f64,
        100.0 * order_changed_rows as f64 / rows as f64,
    );
    println!("min_source_margin={min_source_margin:.12e}");
    println!("min_local_margin={min_local_margin:.12e}");
    for (row, source_ids, local_ids) in first_changes {
        println!("changed_row={row} source_top={source_ids:?} local_top={local_ids:?}");
    }
    println!("first_row:");
    println!("source_top={source_top:?}");
    println!("local_top={local_top:?}");
    println!("changed_from_source={changed}");
    println!(
        "source_margin_k_k1={:.12e}",
        source_rank[k - 1].1 - source_rank[k].1
    );
    println!(
        "local_margin_k_k1={:.12e}",
        local_rank[k - 1].1 - local_rank[k].1
    );
    println!("rank\texpert\tscore\tsource_bias\tlocal_bias\tsource_sel\tlocal_sel");
    for (rank, &(expert, source_sel)) in source_rank.iter().take(12).enumerate() {
        println!(
            "{}\t{}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}",
            rank + 1,
            expert,
            scores[expert],
            source_bias[expert],
            local_bias[expert],
            source_sel,
            scores[expert] + local_bias[expert],
        );
    }
    Ok(())
}
