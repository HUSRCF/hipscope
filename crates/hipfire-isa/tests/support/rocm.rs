// SPDX-License-Identifier: Apache-2.0
//! Shared ROCm oracle discovery. Only an explicit no-ROCm marker skips tests.
#![allow(dead_code, unused_macros)]
#[path = "../../../rocm.rs"]
mod resolver;
use std::path::PathBuf;
pub fn root() -> PathBuf { resolver::root() }
pub fn llvm(name: &str) -> PathBuf { resolver::llvm(name) }
pub fn hipcc() -> PathBuf { root().join("bin/hipcc") }
pub fn llvm_bin_dir() -> PathBuf { root().join("lib/llvm/bin") }
pub fn rocm_tool(name: &str) -> Option<PathBuf> {
    if resolver::tests_disabled() { return None; }
    let path = llvm(name);
    assert!(path.is_file(), "required ROCm tool {} missing; set HIPFIRE_TEST_REQUIRE_ROCM=0 only on no-ROCm hosts", path.display());
    Some(path)
}
macro_rules! require_rocm_tool {
    ($name:expr) => {
        match $crate::rocm::rocm_tool($name) {
            Some(path) => path,
            None => { eprintln!("skipping: explicitly disabled ROCm tool {}", $name); return; }
        }
    };
}
