// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Writable config persistence for the TUI Settings tab.
//!
//! This is the Rust mirror of the bun CLI config writer
//! (`cli/index.ts` CONFIG_DEFAULTS + validateConfigValue + saveConfig). The two
//! editors MUST agree on key names, the enum allowlists, and the numeric ranges
//! so `hipfire config` (bun) and the rust TUI read/write the SAME
//! `~/.hipfire/config.json` without divergence.
//!
//! Persistence model (matches bun saveConfig):
//!   * read-modify-write the raw JSON object (preserve every other key)
//!   * validate before writing (reject out-of-allowlist enum / out-of-range
//!     numeric) — an invalid value is NEVER persisted
//!   * atomic write: temp file in the same dir + rename over the target

use std::{fs, path::Path};

use serde_json::Value;

/// The kinds of settings the TUI can mutate. Each variant carries the canonical
/// key name and, for enums, the allowlist of legal string values mirrored from
/// the bun `validateConfigValue`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    /// String enum: cycle through a fixed allowlist (mirrors bun `includes(...)`).
    Enum(&'static [&'static str]),
    /// Bool stored as JSON true/false (mirrors bun `typeof value === "boolean"`).
    Bool,
    /// Float in an inclusive range, persisted as a JSON number.
    Float { min: f64, max: f64 },
    /// Integer in an inclusive range, persisted as a JSON integer.
    Int { min: i64, max: i64 },
    /// Free string (e.g. chat_template path). Empty string = unset/default.
    /// `require_existing_file` mirrors bun's existsSync+isFile path check.
    FreeStr { require_existing_file: bool },
}

/// One editable setting: canonical key + how to validate/cycle it.
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub key: &'static str,
    pub kind: FieldKind,
}

// ── Allowlists mirrored 1:1 from cli/index.ts:355-409 ───────────────────────
const KV_CACHE: &[&str] = &[
    "auto", "q8", "asym4", "asym3", "asym2", "fwht4", "fwht3", "fwht2", "turbo", "turbo4",
    "turbo3", "turbo2",
];
const MTP_MODE: &[&str] = &["off", "on", "auto"];
const DFLASH_MODE: &[&str] = &["on", "off", "auto"];
const THINKING: &[&str] = &["on", "off"];
const FLASH_MODE: &[&str] = &["auto", "always", "never"];
const MMQ_SCREEN: &[&str] = &["off", "on", "auto"];
const PREFILL_COMPRESSION: &[&str] = &["off", "auto", "always"];

/// The set of fields the TUI Settings tab can edit. Order is the display order
/// in the editable-settings table.
pub const EDITABLE_FIELDS: &[FieldSpec] = &[
    FieldSpec { key: "kv_cache", kind: FieldKind::Enum(KV_CACHE) },
    FieldSpec { key: "mtp_mode", kind: FieldKind::Enum(MTP_MODE) },
    FieldSpec { key: "dflash_mode", kind: FieldKind::Enum(DFLASH_MODE) },
    FieldSpec { key: "thinking", kind: FieldKind::Enum(THINKING) },
    FieldSpec {
        key: "chat_template",
        kind: FieldKind::FreeStr { require_existing_file: true },
    },
    FieldSpec { key: "temperature", kind: FieldKind::Float { min: 0.0, max: 2.0 } },
    // top_p in bun is `> 0 && <= 1`; we approximate the open lower bound below
    // in `validate` (strict >0) — the spec's min is informational.
    FieldSpec { key: "top_p", kind: FieldKind::Float { min: 0.0, max: 1.0 } },
    FieldSpec { key: "max_tokens", kind: FieldKind::Int { min: 1, max: 131072 } },
    // KV cache capacity (tokens). Mirrors bun validateConfigValue `max_seq`:
    // int 512..=524288 (cli/index.ts:362). Backs the Easy "Context" row.
    FieldSpec { key: "max_seq", kind: FieldKind::Int { min: 512, max: 524288 } },
    // Extra fields surfaced read/write so the TUI matches the bun schema.
    FieldSpec { key: "flash_mode", kind: FieldKind::Enum(FLASH_MODE) },
    FieldSpec { key: "mmq_screen", kind: FieldKind::Enum(MMQ_SCREEN) },
    FieldSpec {
        key: "prefill_compression",
        kind: FieldKind::Enum(PREFILL_COMPRESSION),
    },
    FieldSpec { key: "mtp_k", kind: FieldKind::Int { min: 1, max: 10 } },
    // Bool fields (cycled true/false via Left/Right/Space).
    FieldSpec { key: "dflash_adaptive_b", kind: FieldKind::Bool },
    FieldSpec { key: "cask", kind: FieldKind::Bool },
    FieldSpec { key: "prompt_normalize", kind: FieldKind::Bool },
    FieldSpec { key: "default_chatml", kind: FieldKind::Bool },
];

/// Look up the spec for a key, if it is editable.
pub fn field_spec(key: &str) -> Option<&'static FieldSpec> {
    EDITABLE_FIELDS.iter().find(|f| f.key == key)
}

/// Validate a candidate value for `key`. Mirrors bun `validateConfigValue`.
/// Returns the canonical [`Value`] to persist, or `None` if invalid.
pub fn validate(key: &str, raw: &str) -> Option<Value> {
    let spec = field_spec(key)?;
    match spec.kind {
        FieldKind::Enum(allow) => {
            if allow.contains(&raw) {
                Some(Value::String(raw.to_string()))
            } else {
                None
            }
        }
        FieldKind::Bool => match raw {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        FieldKind::Float { min, max } => {
            let v: f64 = raw.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            // top_p has a strict lower bound (>0) in bun; enforce it.
            let lo_ok = if key == "top_p" { v > min } else { v >= min };
            if lo_ok && v <= max {
                serde_json::Number::from_f64(v).map(Value::Number)
            } else {
                None
            }
        }
        FieldKind::Int { min, max } => {
            let v: i64 = raw.parse().ok()?;
            if v >= min && v <= max {
                Some(Value::Number(v.into()))
            } else {
                None
            }
        }
        FieldKind::FreeStr {
            require_existing_file,
        } => {
            if raw.is_empty() {
                return Some(Value::String(String::new()));
            }
            if require_existing_file {
                let expanded = expand_tilde(raw);
                let p = Path::new(&expanded);
                if p.is_file() {
                    Some(Value::String(raw.to_string()))
                } else {
                    None
                }
            } else {
                Some(Value::String(raw.to_string()))
            }
        }
    }
}

fn expand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest).to_string_lossy().into_owned();
        }
    }
    s.to_string()
}

/// Read-modify-write a single key into the config JSON at `path`, preserving all
/// other keys, validating first, and writing atomically (temp + rename).
///
/// `default_model` and any other free key can be written via
/// [`write_raw_value`]; this entry point is for the allowlisted editable fields.
pub fn write_value(path: &Path, key: &str, raw: &str) -> Result<Value, WriteError> {
    let value = validate(key, raw).ok_or_else(|| WriteError::Invalid {
        key: key.to_string(),
        value: raw.to_string(),
    })?;
    write_raw_value(path, key, value.clone())?;
    Ok(value)
}

/// Read-modify-write an already-validated [`Value`] for `key`. Used for
/// `default_model` (selected in the Models tab) and the validated editable
/// fields. Preserves every other key; atomic temp+rename.
pub fn write_raw_value(path: &Path, key: &str, value: Value) -> Result<(), WriteError> {
    let mut obj = read_object(path)?;
    obj.insert(key.to_string(), value);
    write_object(path, &obj)
}

/// Read the config file as a JSON object. A missing file is treated as an empty
/// object (the bun writer stores only non-default keys, so an absent file is
/// the all-defaults state). A present-but-corrupt file is an error rather than
/// a silent clobber — we never overwrite keys we could not parse.
fn read_object(path: &Path) -> Result<serde_json::Map<String, Value>, WriteError> {
    match fs::read_to_string(path) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(serde_json::Map::new());
            }
            match serde_json::from_str::<Value>(trimmed) {
                Ok(Value::Object(map)) => Ok(map),
                Ok(_) => Err(WriteError::NotObject),
                Err(e) => Err(WriteError::Parse(e.to_string())),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(serde_json::Map::new()),
        Err(e) => Err(WriteError::Io(e.to_string())),
    }
}

/// Atomically write the JSON object to `path`: serialize, write to a temp file
/// in the same directory, then rename over the target (rename is atomic on the
/// same filesystem). Matches the bun writer's pretty-print + trailing newline.
fn write_object(path: &Path, obj: &serde_json::Map<String, Value>) -> Result<(), WriteError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| WriteError::Io(e.to_string()))?;
    }
    let mut body =
        serde_json::to_string_pretty(&Value::Object(obj.clone())).map_err(|e| WriteError::Io(e.to_string()))?;
    body.push('\n');

    let tmp = tmp_path(path);
    fs::write(&tmp, body.as_bytes()).map_err(|e| WriteError::Io(e.to_string()))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        WriteError::Io(e.to_string())
    })?;
    Ok(())
}

fn tmp_path(path: &Path) -> std::path::PathBuf {
    let pid = std::process::id();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.json".to_string());
    let tmp_name = format!(".{name}.tmp.{pid}");
    match path.parent() {
        Some(p) => p.join(tmp_name),
        None => std::path::PathBuf::from(tmp_name),
    }
}

#[derive(Debug)]
pub enum WriteError {
    Invalid { key: String, value: String },
    NotObject,
    Parse(String),
    Io(String),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Invalid { key, value } => {
                write!(f, "rejected invalid value for {key}: {value:?}")
            }
            WriteError::NotObject => write!(f, "config.json is not an object; refusing to write"),
            WriteError::Parse(e) => write!(f, "config.json parse error: {e}; refusing to write"),
            WriteError::Io(e) => write!(f, "config write error: {e}"),
        }
    }
}

impl std::error::Error for WriteError {}

/// Cycle an enum field's current value to the next (or previous) allowlist entry.
/// Returns the new value string, or `None` if the key is not a cyclable enum.
pub fn cycle_enum(key: &str, current: &str, forward: bool) -> Option<String> {
    let spec = field_spec(key)?;
    if let FieldKind::Enum(allow) = spec.kind {
        let idx = allow.iter().position(|v| *v == current).unwrap_or(0);
        let n = allow.len();
        let next = if forward {
            (idx + 1) % n
        } else {
            (idx + n - 1) % n
        };
        Some(allow[next].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "hipfire-tui-writer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = base.join(unique);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_valid_value_preserving_other_keys() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        // Seed with pre-existing keys including a non-editable one.
        let mut f = fs::File::create(&cfg).unwrap();
        write!(
            f,
            "{{\n  \"default_model\": \"qwen3.5:9b\",\n  \"port\": 11435,\n  \"kv_cache\": \"auto\"\n}}\n"
        )
        .unwrap();
        drop(f);

        // Write a valid kv_cache value.
        let v = write_value(&cfg, "kv_cache", "q8").expect("valid write");
        assert_eq!(v, Value::String("q8".into()));

        // Read back: value applied, JSON still valid, OTHER keys preserved.
        let raw = fs::read_to_string(&cfg).unwrap();
        let parsed: Value = serde_json::from_str(&raw).expect("valid JSON round-trip");
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("kv_cache").unwrap(), &Value::String("q8".into()));
        assert_eq!(
            obj.get("default_model").unwrap(),
            &Value::String("qwen3.5:9b".into())
        );
        assert_eq!(obj.get("port").unwrap().as_i64().unwrap(), 11435);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_bad_enum_without_persisting() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{\n  \"kv_cache\": \"auto\"\n}\n").unwrap();

        let err = write_value(&cfg, "kv_cache", "not_a_mode");
        assert!(err.is_err(), "bad enum must be rejected");

        // File unchanged: still the original valid value.
        let raw = fs::read_to_string(&cfg).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("kv_cache").unwrap(),
            &Value::String("auto".into())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_out_of_range_numeric() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();

        assert!(write_value(&cfg, "temperature", "5.0").is_err());
        assert!(write_value(&cfg, "max_tokens", "0").is_err());
        assert!(write_value(&cfg, "top_p", "0").is_err()); // strict >0
        assert!(write_value(&cfg, "mtp_k", "11").is_err());
        // max_seq mirrors bun int 512..=524288.
        assert!(write_value(&cfg, "max_seq", "511").is_err());
        assert!(write_value(&cfg, "max_seq", "524289").is_err());

        // Valid boundaries accepted.
        assert!(write_value(&cfg, "temperature", "2.0").is_ok());
        assert!(write_value(&cfg, "max_tokens", "131072").is_ok());
        assert!(write_value(&cfg, "top_p", "1").is_ok());
        // max_seq boundaries + the Easy "Context" default.
        assert!(write_value(&cfg, "max_seq", "512").is_ok());
        assert!(write_value(&cfg, "max_seq", "524288").is_ok());
        assert!(write_value(&cfg, "max_seq", "32768").is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn creates_file_when_missing() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        assert!(!cfg.exists());
        write_value(&cfg, "thinking", "off").expect("create + write");
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("thinking").unwrap(),
            &Value::String("off".into())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cycles_enum_values() {
        assert_eq!(cycle_enum("kv_cache", "auto", true).unwrap(), "q8");
        assert_eq!(cycle_enum("dflash_mode", "off", true).unwrap(), "auto");
        // wrap-around backward from first
        assert_eq!(cycle_enum("thinking", "on", false).unwrap(), "off");
        assert!(cycle_enum("temperature", "0.3", true).is_none());
    }

    #[test]
    fn bool_field_round_trips_and_rejects_garbage() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();
        let v = write_value(&cfg, "prompt_normalize", "false").expect("valid bool");
        assert_eq!(v, Value::Bool(false));
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("prompt_normalize").unwrap(),
            &Value::Bool(false)
        );
        assert!(write_value(&cfg, "cask", "yes").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_clobber_corrupt_file() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{ this is not json").unwrap();
        let err = write_value(&cfg, "kv_cache", "q8");
        assert!(matches!(err, Err(WriteError::Parse(_))));
        let _ = fs::remove_dir_all(&dir);
    }
}
