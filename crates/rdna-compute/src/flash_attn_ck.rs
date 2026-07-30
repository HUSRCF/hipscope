// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Optional dynamic loader for the FlashAttention CK sidecar experiment.
//!
//! This module only exposes the raw all-FP16 sidecar ABI. It deliberately does
//! not route hipfire attention calls or allocate conversion scratch. Callers
//! must opt in at build time, load an explicit library path, and provide device
//! buffers whose lifetimes cover the asynchronous launch.

use libloading::{Library, Symbol};
use std::error::Error;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};

pub const FLASH_ATTN_CK_ABI_VERSION: u32 = 1;
const ERROR_CAPACITY: usize = 512;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnCkDType {
    F16 = 1,
    Bf16 = 2,
}

/// Stable C layout shared with `hipfire_flash_attn_ck.h`.
///
/// Strides are measured in elements, not bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnCkFwdParams {
    pub abi_version: u32,
    pub struct_size: u32,

    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub out: *mut c_void,
    pub stream: *mut c_void,

    pub dtype: i32,
    pub batch: i32,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub nhead_q: i32,
    pub nhead_k: i32,
    pub head_dim: i32,
    pub causal: i32,

    pub softmax_scale: f32,

    pub stride_q: i64,
    pub stride_k: i64,
    pub stride_v: i64,
    pub stride_out: i64,
    pub nhead_stride_q: i64,
    pub nhead_stride_k: i64,
    pub nhead_stride_v: i64,
    pub nhead_stride_out: i64,
    pub batch_stride_q: i64,
    pub batch_stride_k: i64,
    pub batch_stride_v: i64,
    pub batch_stride_out: i64,
}

impl FlashAttnCkFwdParams {
    pub fn new() -> Self {
        Self {
            abi_version: FLASH_ATTN_CK_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            q: std::ptr::null(),
            k: std::ptr::null(),
            v: std::ptr::null(),
            out: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            dtype: FlashAttnCkDType::F16 as i32,
            batch: 0,
            seqlen_q: 0,
            seqlen_k: 0,
            nhead_q: 0,
            nhead_k: 0,
            head_dim: 0,
            causal: 0,
            softmax_scale: 0.0,
            stride_q: 0,
            stride_k: 0,
            stride_v: 0,
            stride_out: 0,
            nhead_stride_q: 0,
            nhead_stride_k: 0,
            nhead_stride_v: 0,
            nhead_stride_out: 0,
            batch_stride_q: 0,
            batch_stride_k: 0,
            batch_stride_v: 0,
            batch_stride_out: 0,
        }
    }
}

impl Default for FlashAttnCkFwdParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum FlashAttnCkError {
    Load {
        path: PathBuf,
        source: libloading::Error,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
    AbiVersion {
        expected: u32,
        actual: u32,
    },
    Call {
        operation: &'static str,
        status: i32,
        message: String,
    },
}

impl fmt::Display for FlashAttnCkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, source } => {
                write!(
                    f,
                    "load FlashAttention CK sidecar {}: {source}",
                    path.display()
                )
            }
            Self::Symbol { name, source } => {
                write!(f, "resolve FlashAttention CK symbol {name}: {source}")
            }
            Self::AbiVersion { expected, actual } => write!(
                f,
                "FlashAttention CK ABI mismatch: expected {expected}, found {actual}"
            ),
            Self::Call {
                operation,
                status,
                message,
            } => write!(
                f,
                "FlashAttention CK {operation} failed with status {status}: {message}"
            ),
        }
    }
}

impl Error for FlashAttnCkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Load { source, .. } | Self::Symbol { source, .. } => Some(source),
            Self::AbiVersion { .. } | Self::Call { .. } => None,
        }
    }
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type FwdFn = unsafe extern "C" fn(*const FlashAttnCkFwdParams, *mut c_char, usize) -> i32;

/// Loaded sidecar and its stable function table.
pub struct FlashAttnCk {
    _library: &'static Library,
    fwd_supported: FwdFn,
    fwd: FwdFn,
}

impl FlashAttnCk {
    /// Load one explicit sidecar path. No soname search or implicit fallback is
    /// performed, so enabling the Cargo feature alone cannot change execution.
    ///
    /// The loaded library is intentionally pinned for the rest of the process.
    /// HIP launches are asynchronous, so unloading the code object when this
    /// handle is dropped would be unsafe while any submitted work is pending.
    ///
    /// # Safety
    ///
    /// `path` must identify a trusted native library implementing the declared
    /// ABI. Loading a native library may execute constructors, and the resolved
    /// symbols are trusted to follow their declared function signatures.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, FlashAttnCkError> {
        let path = path.as_ref();
        let library =
            unsafe { Library::new(path.as_os_str()) }.map_err(|source| FlashAttnCkError::Load {
                path: path.to_path_buf(),
                source,
            })?;

        let (fwd_supported, fwd) = unsafe {
            let abi_version: Symbol<'_, AbiVersionFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_abi_version",
                "abi_version",
            )?;
            let fwd_supported: Symbol<'_, FwdFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_fwd_supported",
                "fwd_supported",
            )?;
            let fwd: Symbol<'_, FwdFn> = symbol(&library, b"hipfire_flash_attn_ck_fwd", "fwd")?;

            let actual = abi_version();
            if actual != FLASH_ATTN_CK_ABI_VERSION {
                return Err(FlashAttnCkError::AbiVersion {
                    expected: FLASH_ATTN_CK_ABI_VERSION,
                    actual,
                });
            }

            (*fwd_supported, *fwd)
        };
        let library = Box::leak(Box::new(library));
        Ok(Self {
            _library: library,
            fwd_supported,
            fwd,
        })
    }

    pub fn is_supported(&self, params: &FlashAttnCkFwdParams) -> Result<(), FlashAttnCkError> {
        self.call("support check", self.fwd_supported, params)
    }

    /// Launch the sidecar on the stream stored in `params`.
    ///
    /// # Safety
    ///
    /// All pointers in `params` must name device allocations with the declared
    /// shape and element strides. They must remain valid until the asynchronous
    /// operation on `params.stream` has completed.
    pub unsafe fn forward(&self, params: &FlashAttnCkFwdParams) -> Result<(), FlashAttnCkError> {
        self.call("forward", self.fwd, params)
    }

    fn call(
        &self,
        operation: &'static str,
        function: FwdFn,
        params: &FlashAttnCkFwdParams,
    ) -> Result<(), FlashAttnCkError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let status = unsafe { function(params, error.as_mut_ptr().cast::<c_char>(), error.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(FlashAttnCkError::Call {
                operation,
                status,
                message: error_message(&error),
            })
        }
    }
}

unsafe fn symbol<'library, T>(
    library: &'library Library,
    bytes: &[u8],
    name: &'static str,
) -> Result<Symbol<'library, T>, FlashAttnCkError> {
    library
        .get(bytes)
        .map_err(|source| FlashAttnCkError::Symbol { name, source })
}

fn error_message(buffer: &[u8]) -> String {
    let len = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..len]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn fwd_params_matches_c_abi_layout() {
        assert_eq!(std::mem::size_of::<FlashAttnCkFwdParams>(), 184);
        assert_eq!(std::mem::align_of::<FlashAttnCkFwdParams>(), 8);
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, q), 8);
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, dtype), 48);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, softmax_scale),
            80
        );
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, stride_q), 88);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, batch_stride_out),
            176
        );
    }

    #[test]
    fn defaults_publish_current_abi() {
        let params = FlashAttnCkFwdParams::default();
        assert_eq!(params.abi_version, FLASH_ATTN_CK_ABI_VERSION);
        assert_eq!(params.struct_size as usize, std::mem::size_of_val(&params));
    }

    #[test]
    fn missing_sidecar_is_recoverable() {
        let error = unsafe {
            FlashAttnCk::load(OsStr::new(
                "/definitely/missing/libhipfire_flash_attn_ck.so",
            ))
        }
        .err()
        .expect("missing sidecar should return an error");
        assert!(matches!(error, FlashAttnCkError::Load { .. }));
    }

    #[test]
    fn explicit_test_sidecar_loads_and_rejects_invalid_params() {
        let Ok(path) = std::env::var("HIPFIRE_FLASH_ATTN_CK_TEST_LIB") else {
            return;
        };
        let sidecar = unsafe { FlashAttnCk::load(path) }.expect("load explicit test sidecar");
        let error = sidecar
            .is_supported(&FlashAttnCkFwdParams::default())
            .expect_err("zero-shape parameters must be rejected");
        assert!(matches!(
            error,
            FlashAttnCkError::Call {
                operation: "support check",
                status: 1,
                ..
            }
        ));
    }
}
