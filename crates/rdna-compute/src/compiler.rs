//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Compiles HIP kernel sources to code objects, with caching.
/// Tries pre-compiled blobs first (kernels/compiled/{arch}/), falls back to hipcc.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    precompiled_dir: Option<PathBuf>,
    has_hipcc: bool,
}

impl KernelCompiler {
    pub fn new(arch: &str) -> HipResult<Self> {
        let cache_dir = std::env::temp_dir().join("hipfire_kernels");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;

        // Probe for pre-compiled kernels: try exe-relative, then CWD-relative
        let precompiled_dir = std::env::current_exe().ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .map(|dir| dir.join("kernels").join("compiled").join(arch))
            .filter(|p| p.is_dir())
            .or_else(|| {
                let cwd_path = PathBuf::from("kernels/compiled").join(arch);
                if cwd_path.is_dir() { Some(cwd_path) } else { None }
            });

        if let Some(ref dir) = precompiled_dir {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }

        // Probe for hipcc once at init, not per-kernel
        let has_hipcc = Command::new("hipcc").arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
            has_hipcc,
        })
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Tries pre-compiled blob first (with hash validation), falls back to hipcc.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        // Hash source + arch for cache validation (used by both pre-compiled and runtime paths)
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        let src_hash = format!("{:016x}", hasher.finish());

        // Try pre-compiled .hsaco first, validating with a .hash sidecar file.
        // If hash is missing/mismatched AND hipcc is available, prefer recompilation.
        // If hipcc is unavailable (packaged install), use the blob as-is.
        // See: https://github.com/Kaden-Schutt/hipfire/issues/2
        if let Some(ref dir) = self.precompiled_dir {
            let precompiled = dir.join(format!("{name}.hsaco"));
            let hash_file = dir.join(format!("{name}.hash"));
            if precompiled.exists() {
                let hash_ok = hash_file.exists() && {
                    let stored = std::fs::read_to_string(&hash_file).unwrap_or_default();
                    stored.trim() == src_hash
                };
                if hash_ok {
                    self.compiled.insert(name.to_string(), precompiled);
                    return Ok(&self.compiled[name]);
                }
                // No valid hash — only reject if hipcc can recompile
                if !self.has_hipcc {
                    eprintln!("  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)");
                    eprintln!("           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes.");
                    self.compiled.insert(name.to_string(), precompiled);
                    return Ok(&self.compiled[name]);
                }
                eprintln!("  {name}: pre-compiled blob has no hash file, recompiling");
            }
        }

        // Fall back to runtime compilation via hipcc
        let src_path = self.cache_dir.join(format!("{name}.hip"));
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
        let hash_path = self.cache_dir.join(format!("{name}.hash"));

        let cache_valid = obj_path.exists() && hash_path.exists()
            && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

        if !cache_valid {
            std::fs::write(&src_path, source).map_err(|e| {
                hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
            })?;

            let _ = std::fs::remove_file(&obj_path);

            let output = Command::new("hipcc")
                .args([
                    "--genco",
                    &format!("--offload-arch={}", self.arch),
                    "-O3",
                    "-o",
                    obj_path.to_str().unwrap(),
                    src_path.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| {
                    hip_bridge::HipError::new(0, &format!("failed to run hipcc: {e}"))
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("hipcc compilation failed for {name}:\n{stderr}"),
                ));
            }

            let _ = std::fs::write(&hash_path, &src_hash);
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }
}
