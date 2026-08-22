// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview safetensors → HFQ convert (arch 15, qt=51 MQ2G256LloydU).
//!
//! A dedicated path rather than a `--format` flag threaded through the shared
//! `pipeline.rs`, following the `pipeline_deepseek.rs` precedent. Maple's
//! requirements do not fit the generic pipeline:
//!
//! * The carrier is decided by tensor NAME, not by a global format flag: the
//!   linear projections are natively ternary and pack exactly, while the
//!   router, embeddings, lm_head and norms are genuinely full-precision.
//! * A non-ternary linear must ABORT the convert. The generic pipeline's job is
//!   to quantize whatever it is given as well as it can; here, "as well as it
//!   can" is precisely the wrong behaviour — it would silently produce a lossy
//!   model that looks identical to a correct one until it generated garbage.
//! * 18,432 expert tensors (256 experts × 3 projections × 24 layers) mean RSS
//!   discipline matters; tensors are spilled as they are produced.
//!
//! Keeping this out of `pipeline.rs` also keeps PR #599's shared convert path
//! untouched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hipfire_quantize::float16::bf16_to_f32;
use hipfire_quantize::safetensors_file::SafetensorsFile;

use crate::hfq::{write_hfq, HfqTensor, QuantType, TensorSpill};
use crate::maple::{maple_tensor_policy, pack_maple_tensor, MapleTensorPolicy};

/// `arch_id` for Maple-Preview. See `docs/architecture-ids.md`.
pub(crate) const ARCH_ID_MAPLE: u32 = 15;

/// Group size for the MQ2-Lloyd container.
const GROUP_SIZE: u32 = 256;

/// Running totals, printed at the end and stamped into the HFQ metadata so a
/// model on disk can be told apart from a re-run with different code.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MapleConvertStats {
    pub ternary_tensors: usize,
    pub ternary_weights: u64,
    pub ternary_bytes: u64,
    pub high_precision_tensors: usize,
    pub high_precision_bytes: u64,
    pub min_nonzero_frac: f64,
    pub max_nonzero_frac: f64,
}

/// Widen a BF16 blob to f32.
fn bf16_blob_to_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 2 != 0 {
        return Err(format!("BF16 blob length {} is odd", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Convert one tensor according to its name policy.
///
/// Returns the `HfqTensor` plus whether it took the ternary path.
fn convert_tensor(
    name: &str,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
    stats: &mut MapleConvertStats,
) -> Result<(HfqTensor, bool), String> {
    if dtype != "BF16" {
        return Err(format!(
            "{name}: expected BF16 (Maple publishes dequantized bf16 masters), got {dtype}"
        ));
    }
    let hfq_shape: Vec<u32> = shape.iter().map(|&d| d as u32).collect();

    match maple_tensor_policy(name) {
        MapleTensorPolicy::Ternary => {
            // Row-major [M, K]; K is the LAST dim and is what the 256-blocks
            // run along, so it is the axis the row-scale invariant lives on.
            let k = *shape
                .last()
                .ok_or_else(|| format!("{name}: ternary tensor has no dimensions"))?;
            let vals = bf16_blob_to_f32(bytes)?;
            // Single validate+pack entry point. Re-implementing the K%256 and
            // per-row checks here would give two copies that drift.
            let (data, row_stats) =
                pack_maple_tensor(&vals, k).map_err(|e| format!("{name}: {e}"))?;

            stats.ternary_tensors += 1;
            stats.ternary_weights += vals.len() as u64;
            stats.ternary_bytes += data.len() as u64;
            if stats.ternary_tensors == 1 {
                stats.min_nonzero_frac = row_stats.nonzero_frac;
                stats.max_nonzero_frac = row_stats.nonzero_frac;
            } else {
                stats.min_nonzero_frac = stats.min_nonzero_frac.min(row_stats.nonzero_frac);
                stats.max_nonzero_frac = stats.max_nonzero_frac.max(row_stats.nonzero_frac);
            }

            Ok((
                HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ2G256LloydU,
                    shape: hfq_shape,
                    group_size: GROUP_SIZE,
                    data,
                    spilled_len: 0,
                },
                true,
            ))
        }
        MapleTensorPolicy::KeepHighPrecision => {
            // Carried verbatim as BF16. These are the router, embeddings,
            // lm_head and norms — measured genuinely full-precision on the
            // published checkpoint, so there is nothing to pack losslessly.
            stats.high_precision_tensors += 1;
            stats.high_precision_bytes += bytes.len() as u64;
            Ok((
                HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::BF16,
                    shape: hfq_shape,
                    group_size: 1,
                    data: bytes.to_vec(),
                    spilled_len: 0,
                },
                false,
            ))
        }
    }
}

/// Enumerate the shards of a safetensors directory, in index order.
fn shard_paths(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    if shards.is_empty() {
        return Err(format!(
            "no .safetensors files in {} — point --input at the model directory",
            dir.display()
        ));
    }
    shards.sort();
    Ok(shards)
}

/// Convert a Maple-Preview safetensors directory into a `.hfq`.
///
/// Shards are processed one at a time and their page cache dropped as they are
/// consumed, so peak RSS stays bounded even though the source is ~40 GB.
pub(crate) fn convert_maple_safetensors(
    input_dir: &Path,
    output: &Path,
    config_json: &str,
) -> Result<MapleConvertStats, String> {
    let shards = shard_paths(input_dir)?;
    eprintln!(
        "maple: converting {} shard(s) from {}",
        shards.len(),
        input_dir.display()
    );

    let spill_dir = output.parent().unwrap_or(Path::new("."));
    let mut spill = TensorSpill::new(spill_dir)
        .map_err(|e| format!("create spill file in {}: {e}", spill_dir.display()))?;

    // BTreeMap so the output tensor order is deterministic regardless of shard
    // iteration order or filesystem enumeration.
    let mut tensors: BTreeMap<String, HfqTensor> = BTreeMap::new();
    let mut stats = MapleConvertStats::default();

    for shard in &shards {
        let sf =
            SafetensorsFile::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let names: Vec<String> = sf.tensor_names().into_iter().map(String::from).collect();
        eprintln!(
            "maple:   {} — {} tensor(s)",
            shard.file_name().unwrap_or_default().to_string_lossy(),
            names.len()
        );

        for name in &names {
            let (meta, bytes) = sf
                .tensor_data(name)
                .ok_or_else(|| format!("{name}: vanished from {}", shard.display()))?;
            let (mut t, _is_ternary) =
                convert_tensor(name, &meta.dtype, &meta.shape, bytes, &mut stats)?;

            // Spill immediately: 18,432 expert tensors would otherwise all sit
            // in RSS until write_hfq.
            let len = t.data.len() as u64;
            spill
                .spill(&t.data)
                .map_err(|e| format!("{name}: spill: {e}"))?;
            t.data = Vec::new();
            t.spilled_len = len;

            if tensors.insert(name.clone(), t).is_some() {
                return Err(format!("{name}: duplicate tensor across shards"));
            }
            sf.drop_tensor_pages(name);
        }
    }

    spill.flush().map_err(|e| format!("flush spill: {e}"))?;

    let ordered: Vec<HfqTensor> = tensors.into_values().collect();
    if stats.ternary_tensors == 0 {
        return Err(
            "no ternary tensors found — this does not look like a Maple checkpoint".to_string(),
        );
    }

    // Provenance in the metadata, so a model on disk can be told apart from one
    // built by different code. Stale artifacts have burned this project before.
    let metadata = build_metadata(config_json, &stats)?;

    write_hfq(output, ARCH_ID_MAPLE, &metadata, &ordered, Some(&mut spill))
        .map_err(|e| format!("write {}: {e}", output.display()))?;
    spill.cleanup();

    eprintln!(
        "maple: {} ternary tensor(s), {} weights, {:.2} GiB packed ({:.3} bpw)",
        stats.ternary_tensors,
        stats.ternary_weights,
        stats.ternary_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (stats.ternary_bytes * 8) as f64 / stats.ternary_weights.max(1) as f64,
    );
    eprintln!(
        "maple: {} high-precision tensor(s), {:.2} GiB",
        stats.high_precision_tensors,
        stats.high_precision_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    eprintln!(
        "maple: per-tensor nonzero fraction {:.4}..{:.4}",
        stats.min_nonzero_frac, stats.max_nonzero_frac
    );
    Ok(stats)
}

fn build_metadata(config_json: &str, stats: &MapleConvertStats) -> Result<String, String> {
    let mut v: serde_json::Value = if config_json.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(config_json).map_err(|e| format!("parse config.json: {e}"))?
    };
    let obj = v
        .as_object_mut()
        .ok_or_else(|| "config.json is not an object".to_string())?;
    obj.insert(
        "hipfire_maple_provenance".to_string(),
        serde_json::json!({
            "carrier": "MQ2G256LloydU",
            "quant_type": QuantType::MQ2G256LloydU as u8,
            "exact": true,
            "rotation": "none",
            "ternary_tensors": stats.ternary_tensors,
            "ternary_weights": stats.ternary_weights,
            "nonzero_frac_min": stats.min_nonzero_frac,
            "nonzero_frac_max": stats.max_nonzero_frac,
        }),
    );
    serde_json::to_string(&v).map_err(|e| format!("serialize metadata: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn ternary(k: usize, s: f32) -> Vec<f32> {
        (0..k)
            .map(|i| match i % 3 {
                0 => -s,
                1 => 0.0,
                _ => s,
            })
            .collect()
    }

    #[test]
    fn ternary_projection_packs_as_qt51() {
        let vals = ternary(512, 0.03125);
        let bytes = bf16_bytes(&vals);
        let mut stats = MapleConvertStats::default();
        let (t, is_ternary) = convert_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[2, 256],
            &bytes,
            &mut stats,
        )
        .unwrap();
        assert!(is_ternary);
        assert_eq!(t.quant_type, QuantType::MQ2G256LloydU);
        assert_eq!(t.group_size, 256);
        assert_eq!(t.data.len(), 2 * 72);
        assert_eq!(stats.ternary_weights, 512);
    }

    #[test]
    fn router_is_carried_full_precision_not_packed() {
        // The router is dense. Packing it as ternary would abort the convert;
        // routing it here by name is what prevents that.
        let vals: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let bytes = bf16_bytes(&vals);
        let mut stats = MapleConvertStats::default();
        let (t, is_ternary) = convert_tensor(
            "model.layers.0.mlp.gate.weight",
            "BF16",
            &[1, 256],
            &bytes,
            &mut stats,
        )
        .unwrap();
        assert!(!is_ternary);
        assert_eq!(t.quant_type, QuantType::BF16);
        assert_eq!(stats.high_precision_tensors, 1);
    }

    #[test]
    fn a_dense_projection_aborts_the_convert() {
        // NEGATIVE CONTROL: if a tensor the policy calls ternary turns out not
        // to be, we must fail loudly. Silently emitting a lossy approximation
        // is the failure mode this whole path exists to prevent.
        let vals: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let bytes = bf16_bytes(&vals);
        let mut stats = MapleConvertStats::default();
        let err = convert_tensor(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "BF16",
            &[1, 256],
            &bytes,
            &mut stats,
        )
        .unwrap_err();
        assert!(err.contains("not ternary"), "got: {err}");
        assert!(
            err.contains("gate_proj"),
            "error must name the tensor: {err}"
        );
    }

    #[test]
    fn non_bf16_input_is_rejected() {
        let mut stats = MapleConvertStats::default();
        let err = convert_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "F32",
            &[1, 256],
            &[0u8; 1024],
            &mut stats,
        )
        .unwrap_err();
        assert!(err.contains("BF16"), "got: {err}");
    }

    #[test]
    fn k_not_divisible_by_256_is_rejected() {
        let vals = ternary(384, 0.03125);
        let bytes = bf16_bytes(&vals);
        let mut stats = MapleConvertStats::default();
        let err = convert_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[1, 384],
            &bytes,
            &mut stats,
        )
        .unwrap_err();
        assert!(err.contains("multiple of 256"), "got: {err}");
    }

    #[test]
    fn provenance_is_stamped_into_metadata() {
        let stats = MapleConvertStats {
            ternary_tensors: 3,
            ternary_weights: 100,
            min_nonzero_frac: 0.61,
            max_nonzero_frac: 0.62,
            ..Default::default()
        };
        let md = build_metadata(r#"{"model_type":"maple"}"#, &stats).unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();
        assert_eq!(v["model_type"], "maple");
        assert_eq!(v["hipfire_maple_provenance"]["quant_type"], 51);
        assert_eq!(v["hipfire_maple_provenance"]["exact"], true);
        assert_eq!(v["hipfire_maple_provenance"]["rotation"], "none");
        assert_eq!(v["hipfire_maple_provenance"]["ternary_tensors"], 3);
    }
}
