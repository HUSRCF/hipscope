// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Fail-closed validation for Hugging Face sharded safetensors directories.
//!
//! `SafetensorsSource::open` currently discovers every `*.safetensors` file in
//! a directory.  EP directory loads therefore validate the canonical index
//! first and require the discovered file set to equal its shard whitelist.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const INDEX_FILE: &str = "model.safetensors.index.json";
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexedSafetensors {
    pub(crate) shard_count: usize,
    pub(crate) tensor_count: usize,
}

pub(crate) fn validate_indexed_safetensors_dir(dir: &Path) -> Result<IndexedSafetensors, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a model directory", dir.display()));
    }

    let index_path = dir.join(INDEX_FILE);
    let index_text = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| format!("invalid {}: {e}", index_path.display()))?;
    let weight_map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} has no object weight_map", index_path.display()))?;
    if weight_map.is_empty() {
        return Err(format!("{} has an empty weight_map", index_path.display()));
    }

    let mut tensor_to_shard = BTreeMap::<String, String>::new();
    let mut indexed_shards = BTreeSet::<String>::new();
    for (tensor, shard_value) in weight_map {
        if tensor == "__metadata__" || tensor.is_empty() {
            return Err(format!(
                "{} has invalid tensor name {tensor:?} in weight_map",
                index_path.display()
            ));
        }
        let shard = shard_value.as_str().ok_or_else(|| {
            format!(
                "{} weight_map[{tensor:?}] is not a shard filename",
                index_path.display()
            )
        })?;
        validate_plain_shard_name(shard, &index_path)?;
        tensor_to_shard.insert(tensor.clone(), shard.to_string());
        indexed_shards.insert(shard.to_string());
    }

    let mut discovered_shards = BTreeSet::<String>::new();
    for entry in
        std::fs::read_dir(dir).map_err(|e| format!("failed to list {}: {e}", dir.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to list {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("safetensors")) {
            continue;
        }
        let file_type = entry
            .file_type()
            .map_err(|e| format!("failed to inspect {}: {e}", path.display()))?;
        if !file_type.is_file() {
            return Err(format!(
                "refusing non-regular safetensors shard {}",
                path.display()
            ));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            format!(
                "safetensors shard filename is not valid UTF-8: {}",
                path.display()
            )
        })?;
        discovered_shards.insert(name);
    }

    let missing: Vec<&String> = indexed_shards.difference(&discovered_shards).collect();
    let extra: Vec<&String> = discovered_shards.difference(&indexed_shards).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "safetensors shard set does not match {} (missing={missing:?}, extra={extra:?}); refusing mixed/incomplete directory",
            index_path.display()
        ));
    }

    let mut seen = BTreeMap::<String, String>::new();
    for shard in &indexed_shards {
        let shard_path = dir.join(shard);
        let header = read_safetensors_header(&shard_path)?;
        for (tensor, descriptor) in header {
            if tensor == "__metadata__" {
                continue;
            }
            if !descriptor.is_object() {
                return Err(format!(
                    "{} tensor {tensor:?} has a non-object descriptor",
                    shard_path.display()
                ));
            }
            if let Some(previous) = seen.insert(tensor.clone(), shard.clone()) {
                return Err(format!(
                    "duplicate safetensors tensor {tensor:?} in shards {previous:?} and {shard:?}"
                ));
            }
            match tensor_to_shard.get(&tensor) {
                Some(expected) if expected == shard => {}
                Some(expected) => {
                    return Err(format!(
                        "tensor {tensor:?} is physically in {shard:?}, but {INDEX_FILE} assigns it to {expected:?}"
                    ));
                }
                None => {
                    return Err(format!(
                        "tensor {tensor:?} in shard {shard:?} is absent from {INDEX_FILE}"
                    ));
                }
            }
        }
    }

    let absent: Vec<&String> = tensor_to_shard
        .keys()
        .filter(|tensor| !seen.contains_key(*tensor))
        .collect();
    if !absent.is_empty() {
        return Err(format!(
            "{} lists tensors absent from their assigned shards: {absent:?}",
            index_path.display()
        ));
    }

    Ok(IndexedSafetensors {
        shard_count: indexed_shards.len(),
        tensor_count: tensor_to_shard.len(),
    })
}

fn validate_plain_shard_name(shard: &str, index_path: &Path) -> Result<(), String> {
    let path = Path::new(shard);
    if shard.is_empty()
        || path.is_absolute()
        || path.file_name() != Some(OsStr::new(shard))
        || path.extension() != Some(OsStr::new("safetensors"))
    {
        return Err(format!(
            "{} contains unsafe/non-safetensors shard name {shard:?}",
            index_path.display()
        ));
    }
    Ok(())
}

fn read_safetensors_header(
    path: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let mut file =
        File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| format!("failed to read {} header length: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(format!(
            "{} has invalid safetensors header length {header_len} for file size {file_len}",
            path.display()
        ));
    }
    let header_len: usize = header_len
        .try_into()
        .map_err(|_| format!("{} header length does not fit usize", path.display()))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("failed to read {} header: {e}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| format!("invalid safetensors header in {}: {e}", path.display()))?;
    header
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{} safetensors header is not an object", path.display()))
}

#[cfg(test)]
mod tests {
    use super::validate_indexed_safetensors_dir;
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(case: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "hipfire-loader-index-{case}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_shard(dir: &Path, name: &str, tensors: &[&str]) {
        let mut header = serde_json::Map::new();
        for tensor in tensors {
            header.insert(
                (*tensor).to_string(),
                json!({"dtype":"F16","shape":[0],"data_offsets":[0,0]}),
            );
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        bytes.resize((bytes.len() + 7) & !7, b' ');
        let mut file = Vec::with_capacity(8 + bytes.len());
        file.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(&bytes);
        std::fs::write(dir.join(name), file).unwrap();
    }

    fn write_index(dir: &Path, entries: &[(&str, &str)]) {
        let weight_map: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .map(|(tensor, shard)| ((*tensor).to_string(), json!(shard)))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn accepts_exact_index_whitelist() {
        let dir = TestDir::new("valid");
        write_shard(&dir.0, "model-1.safetensors", &["a"]);
        write_shard(&dir.0, "model-2.safetensors", &["b"]);
        write_index(
            &dir.0,
            &[("a", "model-1.safetensors"), ("b", "model-2.safetensors")],
        );

        let validated = validate_indexed_safetensors_dir(&dir.0).unwrap();
        assert_eq!(validated.shard_count, 2);
        assert_eq!(validated.tensor_count, 2);
    }

    #[test]
    fn rejects_unindexed_or_missing_shards() {
        let dir = TestDir::new("mixed");
        write_shard(&dir.0, "model-1.safetensors", &["a"]);
        write_shard(&dir.0, "old-model.safetensors", &["stale"]);
        write_index(
            &dir.0,
            &[("a", "model-1.safetensors"), ("b", "model-2.safetensors")],
        );

        let err = validate_indexed_safetensors_dir(&dir.0).unwrap_err();
        assert!(err.contains("missing"), "{err}");
        assert!(err.contains("extra"), "{err}");
    }

    #[test]
    fn rejects_duplicate_or_misassigned_tensor() {
        let dir = TestDir::new("duplicate");
        write_shard(&dir.0, "model-1.safetensors", &["a"]);
        write_shard(&dir.0, "model-2.safetensors", &["a", "b"]);
        write_index(
            &dir.0,
            &[("a", "model-1.safetensors"), ("b", "model-2.safetensors")],
        );

        let err = validate_indexed_safetensors_dir(&dir.0).unwrap_err();
        assert!(err.contains("duplicate safetensors tensor"), "{err}");
    }
}
