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

use std::collections::HashSet;
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

    // ORDER IS LOAD-BEARING. `write_hfq` reads the spill file SEQUENTIALLY in
    // the order of the slice it is handed — there is no per-tensor spill
    // offset — so the output list must stay in the exact order the tensors
    // were spilled. A `BTreeMap` here (alphabetical) silently hands each
    // tensor another tensor's bytes; the container still validates, and only
    // the weights are wrong.
    //
    // Determinism does not need the map: `shard_paths()` sorts the shards and
    // `tensor_names()` sorts within a shard, so production order is already
    // stable across runs. The set is only for duplicate detection.
    let mut tensors: Vec<HfqTensor> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
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

            if !seen.insert(name.clone()) {
                return Err(format!("{name}: duplicate tensor across shards"));
            }
            tensors.push(t);
            sf.drop_tensor_pages(name);
        }
    }

    spill.flush().map_err(|e| format!("flush spill: {e}"))?;

    // Spill order == list order; do NOT sort or reorder past this point.
    let ordered: Vec<HfqTensor> = tensors;
    if stats.ternary_tensors == 0 {
        return Err(
            "no ternary tensors found — this does not look like a Maple checkpoint".to_string(),
        );
    }

    // Provenance in the metadata, so a model on disk can be told apart from one
    // built by different code. Stale artifacts have burned this project before.
    let metadata = build_metadata(input_dir, config_json, &stats)?;

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

/// Build the HFQ `metadata_json` envelope.
///
/// The envelope shape is NOT free-form: `MapleConfig::from_metadata_json` reads
/// the source config from the `config` key and
/// `Tokenizer::from_hfq_metadata` reads the tokenizer from `tokenizer` (a
/// STRING holding tokenizer.json verbatim), with optional `tokenizer_config`
/// and `generation_config` siblings. Emitting the bare config.json here
/// produces a `.hfq` that converts cleanly and then fails at load with a
/// missing-config-wrapper error — so this mirrors `pipeline.rs`'s envelope
/// exactly and only ADDS the provenance key.
fn build_metadata(
    input_dir: &Path,
    config_json: &str,
    stats: &MapleConvertStats,
) -> Result<String, String> {
    let config: serde_json::Value = if config_json.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(config_json).map_err(|e| format!("parse config.json: {e}"))?
    };

    let read_json = |name: &str| -> Option<serde_json::Value> {
        std::fs::read_to_string(input_dir.join(name))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    };
    // tokenizer.json is carried as a STRING, not a nested object — that is what
    // `Tokenizer::from_hfq_metadata` expects.
    let tokenizer_str = std::fs::read_to_string(input_dir.join("tokenizer.json")).ok();
    if tokenizer_str.is_none() {
        eprintln!(
            "maple: warning: no tokenizer.json in {} — the .hfq will not be servable",
            input_dir.display()
        );
    }

    // Maple-Preview's published `tokenizer_config.json` carries NO
    // `chat_template` — the vendor ships the template baked into their GGUF
    // metadata instead. Mirror `pipeline.rs`'s sidecar rule: when a
    // `chat_template.jinja` sits next to the weights and tokenizer_config has
    // no template of its own, fold it in under `chat_template`. That is the key
    // `m.chat_template` is populated from, and it is what lets `generate_maple`
    // render the vendor frame (with `# Tools`) instead of the hand-rolled
    // ChatML fallback, which can never advertise tools.
    let tokenizer_config = {
        let mut tc = read_json("tokenizer_config.json");
        let jinja_path = input_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let has_template = tc
                .as_ref()
                .and_then(|v| v.get("chat_template"))
                .map(|v| !v.is_null())
                .unwrap_or(false);
            if !has_template {
                if let Ok(jinja) = std::fs::read_to_string(&jinja_path) {
                    let n = jinja.len();
                    let obj = tc.get_or_insert_with(|| serde_json::json!({}));
                    if let Some(map) = obj.as_object_mut() {
                        map.insert(
                            "chat_template".to_string(),
                            serde_json::Value::String(jinja),
                        );
                        eprintln!(
                            "  embedded chat_template.jinja into tokenizer_config ({n} bytes)"
                        );
                    }
                }
            }
        }
        tc
    };

    let metadata = serde_json::json!({
        "architecture": "maple",
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
        "generation_config": read_json("generation_config.json"),
        "hipfire_maple_provenance": {
            "carrier": "MQ2G256LloydU",
            "quant_type": QuantType::MQ2G256LloydU as u8,
            "exact": true,
            "rotation": "none",
            "ternary_tensors": stats.ternary_tensors,
            "ternary_weights": stats.ternary_weights,
            "nonzero_frac_min": stats.min_nonzero_frac,
            "nonzero_frac_max": stats.max_nonzero_frac,
        },
    });
    serde_json::to_string(&metadata).map_err(|e| format!("serialize metadata: {e}"))
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
        let md = build_metadata(
            Path::new("/nonexistent"),
            r#"{"model_type":"maple"}"#,
            &stats,
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();
        assert_eq!(v["config"]["model_type"], "maple");
        assert_eq!(v["hipfire_maple_provenance"]["quant_type"], 51);
        assert_eq!(v["hipfire_maple_provenance"]["exact"], true);
        assert_eq!(v["hipfire_maple_provenance"]["rotation"], "none");
        assert_eq!(v["hipfire_maple_provenance"]["ternary_tensors"], 3);
    }

    /// Write a minimal BF16 safetensors file: u64 header length, header JSON,
    /// then the tensor payloads in `tensors` order.
    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, shape, vals) in tensors {
            let start = payload.len();
            for v in vals {
                payload.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&payload);
        std::fs::write(path, out).unwrap();
    }

    /// REGRESSION: `write_hfq` reads the spill file sequentially in the order
    /// of the slice it is handed, so the convert must emit tensors in the order
    /// it spilled them. An earlier version collected them through a BTreeMap,
    /// which reordered alphabetically and handed each tensor another tensor's
    /// bytes — the container still validated, only the weights were wrong.
    ///
    /// The names are chosen so ALPHABETICAL order differs from PRODUCTION
    /// order (shard "a" holds `zz_*`, shard "b" holds `aa_*`). Under the bug
    /// the two tensors swap payloads; with equal-length tensors and identical
    /// values the test would pass vacuously, so the two carry DIFFERENT
    /// row scales and the check is value-based.
    #[test]
    fn spill_order_is_preserved_when_alphabetical_order_differs() {
        let dir = std::env::temp_dir().join("maple_spill_order_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Distinct scales => distinct packed bytes => a swap is detectable.
        let zz = ternary(512, 0.0625);
        let aa = ternary(512, 0.00390625);
        write_safetensors(
            &dir.join("model-00001-of-00002.safetensors"),
            &[("zz.self_attn.q_proj.weight", vec![1, 512], zz.clone())],
        );
        write_safetensors(
            &dir.join("model-00002-of-00002.safetensors"),
            &[("aa.self_attn.q_proj.weight", vec![1, 512], aa.clone())],
        );
        std::fs::write(dir.join("config.json"), r#"{"model_type":"maple"}"#).unwrap();

        let out = dir.join("out.hfq");
        convert_maple_safetensors(&dir, &out, r#"{"model_type":"maple"}"#).expect("convert");

        // Read the container back and check each tensor decodes to ITS OWN
        // values, not its neighbour's.
        let hfq = hipfire_runtime::hfq::HfqFile::open(&out).expect("open hfq");
        assert_eq!(hfq.arch_id, ARCH_ID_MAPLE);
        for (name, expect) in [
            ("zz.self_attn.q_proj.weight", &zz),
            ("aa.self_attn.q_proj.weight", &aa),
        ] {
            let (info, data) = hfq.tensor_data_vec(name).expect("tensor present");
            assert_eq!(info.quant_type, QuantType::MQ2G256LloydU as u8);
            let recon = crate::quant_mq::dequantize_mq2g256_lloyd_u_to_f32(&data, expect.len());
            let max_err = expect
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(max_err, 0.0, "{name} decoded to the wrong tensor's bytes");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The runtime reads the source config from `config` and the tokenizer from
    /// a `tokenizer` STRING. A bare config.json at the top level converts fine
    /// and then fails at load, so pin the envelope shape here rather than
    /// discovering it on a 40 GB round trip.
    #[test]
    fn metadata_is_a_loadable_envelope_not_a_bare_config() {
        let dir = std::env::temp_dir().join("maple_md_envelope_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"model":{"vocab":{}}}"#).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":false}"#,
        )
        .unwrap();

        let md = build_metadata(
            &dir,
            r#"{"model_type":"maple","hidden_size":2048}"#,
            &MapleConvertStats::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();

        assert_eq!(v["architecture"], "maple");
        assert_eq!(v["config"]["hidden_size"], 2048);
        // A STRING, not an object — Tokenizer::from_hfq_metadata does
        // `.as_str()` and silently finds nothing if this is nested JSON.
        assert!(v["tokenizer"].is_string(), "tokenizer must be a string");
        assert!(v["tokenizer"].as_str().unwrap().contains("vocab"));
        assert_eq!(v["tokenizer_config"]["add_bos_token"], false);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `chat_template.jinja` sidecar must land in the metadata envelope under
    /// `tokenizer_config.chat_template` — that is the ONLY key the runtime
    /// populates `LoadedModel::chat_template` from. Maple-Preview's HF
    /// `tokenizer_config.json` genuinely has no template (the vendor ships it in
    /// GGUF metadata), so without this the daemon falls back to a hand-rolled
    /// ChatML frame that can never advertise tools.
    #[test]
    fn chat_template_sidecar_lands_in_metadata() {
        let dir = std::env::temp_dir().join("maple_md_chat_template_test");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"model":{"vocab":{}}}"#).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":false}"#,
        )
        .unwrap();
        let template = "{%- if tools %}<tools>{{ tools }}</tools>{%- endif %}";
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();

        let md = build_metadata(
            &dir,
            r#"{"model_type":"maple"}"#,
            &MapleConvertStats::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();
        assert_eq!(v["tokenizer_config"]["chat_template"], template);
        // The sibling keys survive the fold-in.
        assert_eq!(v["tokenizer_config"]["add_bos_token"], false);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The sidecar is OPTIONAL: no `chat_template.jinja` is not an error, and it
    /// must not synthesize an empty/null template key that would make the
    /// runtime think a template exists.
    #[test]
    fn missing_chat_template_sidecar_is_not_an_error() {
        let dir = std::env::temp_dir().join("maple_md_no_chat_template_test");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"model":{"vocab":{}}}"#).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":false}"#,
        )
        .unwrap();

        let md = build_metadata(
            &dir,
            r#"{"model_type":"maple"}"#,
            &MapleConvertStats::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();
        assert_eq!(v["tokenizer_config"]["add_bos_token"], false);
        assert!(
            v["tokenizer_config"].get("chat_template").is_none(),
            "absent sidecar must not fabricate a chat_template key"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An existing `tokenizer_config.chat_template` WINS over the sidecar —
    /// same precedence as the generic pipeline. Guards against a stale sidecar
    /// silently overriding the checkpoint's own template on a future repack.
    #[test]
    fn tokenizer_config_template_wins_over_sidecar() {
        let dir = std::env::temp_dir().join("maple_md_template_precedence_test");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{"model":{"vocab":{}}}"#).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"FROM_CONFIG"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "FROM_SIDECAR").unwrap();

        let md = build_metadata(
            &dir,
            r#"{"model_type":"maple"}"#,
            &MapleConvertStats::default(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&md).unwrap();
        assert_eq!(v["tokenizer_config"]["chat_template"], "FROM_CONFIG");

        std::fs::remove_dir_all(&dir).ok();
    }
}
