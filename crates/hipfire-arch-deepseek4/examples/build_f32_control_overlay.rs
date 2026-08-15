// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build a tiny DeepSeek V4 overlay containing the source-checkpoint F32
//! control tensors that the model consumes as F32 at runtime.
//!
//! The official checkpoint is sharded, so this tool uses HTTP Range requests
//! to fetch only safetensors headers and selected tensor byte ranges. It never
//! downloads expert weights or rewrites the base HFQ.

use hipfire_runtime::hfq::HfqFile;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_BASE_URL: &str = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main";
const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
const QT_F32: u8 = 2;

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u32>,
    data_offsets: [u64; 2],
}

struct OverlayTensor {
    name: String,
    shape: Vec<u32>,
    data: Vec<u8>,
}

fn selected(name: &str, kind: &str) -> bool {
    let gate = name.ends_with(".ffn.gate.bias");
    let attention = name.ends_with(".attn.attn_sink") || name.ends_with(".compressor.ape");
    // Keep HC isolated from `all`: current inference kernels consume these
    // controls as F16, so an HC-F32 overlay is a measurement artifact until a
    // dtype-aware runtime path exists. Attaching it for inference would be
    // invalid even though the overlay container itself passes validation.
    let hc = name.starts_with("hc_") || name.contains(".hc_");
    match kind {
        "all" => gate || attention,
        "gate" => gate,
        "attention" => attention,
        "hc" => hc,
        _ => false,
    }
}

fn curl_range(url: &str, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let expected = end
        .checked_sub(start)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| format!("invalid range {start}-{end}"))? as usize;
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--max-time",
            "120",
            "--range",
            &format!("{start}-{end}"),
            url,
        ])
        .output()
        .map_err(|e| format!("failed to run curl for {url}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "curl range {start}-{end} failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    if output.stdout.len() != expected {
        return Err(format!(
            "range {start}-{end} for {url}: got {} bytes, expected {expected}",
            output.stdout.len()
        ));
    }
    Ok(output.stdout)
}

fn read_or_fetch(path: &Path, url: &str, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let expected = (end - start + 1) as usize;
    if let Ok(data) = fs::read(path) {
        if data.len() == expected {
            return Ok(data);
        }
    }
    let data = curl_range(url, start, end)?;
    fs::write(path, &data).map_err(|e| format!("write cache {}: {e}", path.display()))?;
    Ok(data)
}

fn load_shard_header(
    shard: &str,
    base_url: &str,
    cache: &Path,
) -> Result<(u64, HashMap<String, TensorHeader>), String> {
    let url = format!("{}/{}?download=true", base_url.trim_end_matches('/'), shard);
    let stem = shard.replace('/', "_");
    let len_path = cache.join(format!("{stem}.header-len"));
    let len_bytes = read_or_fetch(&len_path, &url, 0, 7)?;
    let header_len = u64::from_le_bytes(len_bytes.try_into().unwrap());
    let header_path = cache.join(format!("{stem}.header.json"));
    let header_bytes = read_or_fetch(&header_path, &url, 8, 7 + header_len)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&header_bytes)
        .map_err(|e| format!("parse header for {shard}: {e}"))?;
    let mut tensors = HashMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let header: TensorHeader = serde_json::from_value(value)
            .map_err(|e| format!("parse tensor {name} in {shard}: {e}"))?;
        tensors.insert(name, header);
    }
    Ok((header_len, tensors))
}

fn write_overlay(path: &Path, arch_id: u32, tensors: &[OverlayTensor]) -> Result<(), String> {
    let metadata = b"{}";
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
        let name = tensor.name.as_bytes();
        let name_len = u16::try_from(name.len())
            .map_err(|_| format!("tensor name too long: {}", tensor.name))?;
        index.extend_from_slice(&name_len.to_le_bytes());
        index.extend_from_slice(name);
        index.push(QT_F32);
        index.push(tensor.shape.len() as u8);
        for dim in &tensor.shape {
            index.extend_from_slice(&dim.to_le_bytes());
        }
        index.extend_from_slice(&0u32.to_le_bytes());
        index.extend_from_slice(&(tensor.data.len() as u64).to_le_bytes());
    }

    let metadata_offset = 32u64;
    let unaligned = metadata_offset + metadata.len() as u64 + index.len() as u64;
    let data_offset = (unaligned + 4095) & !4095;
    let mut file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(HFQ_MAGIC).map_err(|e| e.to_string())?;
    file.write_all(&HFQ_VERSION.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&arch_id.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&(tensors.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&metadata_offset.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&data_offset.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(metadata).map_err(|e| e.to_string())?;
    file.write_all(&index).map_err(|e| e.to_string())?;
    let pad = usize::try_from(data_offset - unaligned).unwrap();
    file.write_all(&vec![0; pad]).map_err(|e| e.to_string())?;
    for tensor in tensors {
        file.write_all(&tensor.data).map_err(|e| e.to_string())?;
    }
    file.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err(format!(
            "usage: {} BASE.hfq model.safetensors.index.json OUTPUT_DIR [BASE_URL]",
            args.first()
                .map(String::as_str)
                .unwrap_or("build_f32_control_overlay")
        ));
    }
    let base_path = PathBuf::from(&args[1]);
    let index_path = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    let base_url = args.get(4).map(String::as_str).unwrap_or(DEFAULT_BASE_URL);
    let kind = std::env::var("HIPFIRE_DS4_F32_CONTROL_KIND").unwrap_or_else(|_| "all".to_string());
    if !matches!(kind.as_str(), "all" | "gate" | "attention" | "hc") {
        return Err(format!(
            "HIPFIRE_DS4_F32_CONTROL_KIND={kind}: expected all, gate, attention, or hc"
        ));
    }
    fs::create_dir_all(&output_dir).map_err(|e| format!("create {}: {e}", output_dir.display()))?;
    let cache = output_dir.join("source-cache");
    fs::create_dir_all(&cache).map_err(|e| format!("create {}: {e}", cache.display()))?;

    let mut index_json = Vec::new();
    File::open(&index_path)
        .and_then(|mut f| f.read_to_end(&mut index_json))
        .map_err(|e| format!("read {}: {e}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_slice(&index_json)
        .map_err(|e| format!("parse {}: {e}", index_path.display()))?;
    let selected_map: BTreeMap<String, String> = index
        .weight_map
        .into_iter()
        .filter(|(name, _)| selected(name, &kind))
        .collect();
    if selected_map.is_empty() {
        return Err("official index contains no selected F32 control tensors".to_string());
    }

    let base = HfqFile::open_at_offset(&base_path, 0)
        .map_err(|e| format!("open base {}: {e}", base_path.display()))?;
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, shard) in selected_map {
        if base.find_tensor_info(&name).is_some() {
            by_shard.entry(shard).or_default().push(name);
        } else if name.starts_with("mtp.") {
            eprintln!("skip sidecar-only tensor absent from base: {name}");
        } else {
            return Err(format!(
                "selected trunk tensor absent from base HFQ: {name}"
            ));
        }
    }

    let mut overlay = Vec::new();
    for (shard, names) in by_shard {
        eprintln!("header: {shard} ({} selected tensors)", names.len());
        let (header_len, header) = load_shard_header(&shard, base_url, &cache)?;
        let url = format!("{}/{}?download=true", base_url.trim_end_matches('/'), shard);
        let data_base = 8 + header_len;
        for name in names {
            let source = header
                .get(&name)
                .ok_or_else(|| format!("{name} missing from {shard} header"))?;
            if source.dtype != "F32" {
                return Err(format!("{name}: source dtype {} is not F32", source.dtype));
            }
            let expected = source.shape.iter().try_fold(4u64, |bytes, &dim| {
                bytes
                    .checked_mul(dim as u64)
                    .ok_or_else(|| format!("{name}: shape byte size overflow"))
            })?;
            let data_len = source.data_offsets[1] - source.data_offsets[0];
            if data_len != expected {
                return Err(format!(
                    "{name}: safetensors byte range {data_len} != F32 shape bytes {expected}"
                ));
            }
            let base_info = base
                .find_tensor_info(&name)
                .ok_or_else(|| format!("{name}: not present in base HFQ"))?;
            if base_info.shape != source.shape {
                return Err(format!(
                    "{name}: official shape {:?} != base shape {:?}",
                    source.shape, base_info.shape
                ));
            }
            let cache_name = name.replace('/', "_");
            let cache_path = cache.join(format!("{cache_name}.f32"));
            let start = data_base + source.data_offsets[0];
            let end = data_base + source.data_offsets[1] - 1;
            let data = read_or_fetch(&cache_path, &url, start, end)?;
            eprintln!("  F32 {name} {:?} ({} bytes)", source.shape, data.len());
            overlay.push(OverlayTensor {
                name,
                shape: source.shape.clone(),
                data,
            });
        }
    }
    overlay.sort_by(|a, b| a.name.cmp(&b.name));
    let output = output_dir.join("overlay.hfq");
    write_overlay(&output, base.arch_id, &overlay)?;

    let attached = HfqFile::open(&base_path)
        .map_err(|e| format!("reopen base before attach validation: {e}"))?;
    let overlay_file = HfqFile::open_at_offset(&output, 0)
        .map_err(|e| format!("validate overlay {}: {e}", output.display()))?;
    let mut validation_base = attached;
    validation_base.attach_overlay(overlay_file)?;
    eprintln!(
        "wrote {}: {} source-F32 tensors (kind={kind}), {:.2} MiB; attach validation passed",
        output.display(),
        overlay.len(),
        fs::metadata(&output).map_err(|e| e.to_string())?.len() as f64 / 1048576.0
    );
    Ok(())
}
