// SPDX-License-Identifier: Apache-2.0
//! Shared ROCm root and explicit oracle-test opt-out.
#![allow(dead_code)]

pub fn root() -> std::path::PathBuf {
    std::env::var_os("ROCM_PATH").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/rocm"))
}

pub fn llvm(tool: &str) -> std::path::PathBuf {
    root().join("lib/llvm/bin").join(tool)
}

pub fn tests_disabled() -> bool {
    std::env::var("HIPFIRE_TEST_REQUIRE_ROCM").as_deref() == Ok("0")
}
