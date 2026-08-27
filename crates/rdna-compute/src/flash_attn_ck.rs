// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Optional dynamic loader for the FlashAttention CK sidecar experiment.
//!
//! This module exposes the raw dense and quantized sidecar ABIs. It deliberately
//! does not route hipfire attention calls or allocate conversion scratch.
//! Callers must opt in at build time, load an explicit library path, and provide
//! device buffers whose lifetimes cover the asynchronous launch.

use libloading::{Library, Symbol};
use std::error::Error;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};

pub const FLASH_ATTN_CK_ABI_VERSION: u32 = 1;
pub const FLASH_ATTN_CK_QUANTIZED_ABI_VERSION: u32 = 1;
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

/// Stable C layout shared with `hipfire_flash_attn_ck_quantized.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnCkQuantizedPrefillParams {
    pub abi_version: u32,
    pub struct_size: u32,

    pub q: *const f32,
    pub packed_k: *const u8,
    pub packed_v: *const u8,
    pub out: *mut f32,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub cos_theta: *const f32,
    pub sin_theta: *const f32,
    pub stream: *mut c_void,

    pub softmax_scale: f32,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub nhead_q: i32,
    pub nhead_k: i32,
    pub head_dim: i32,
    pub causal: i32,
    pub k_row_stride_bytes: i32,
    pub v_row_stride_bytes: i32,
}

impl FlashAttnCkQuantizedPrefillParams {
    pub fn new() -> Self {
        Self {
            abi_version: FLASH_ATTN_CK_QUANTIZED_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            q: std::ptr::null(),
            packed_k: std::ptr::null(),
            packed_v: std::ptr::null(),
            out: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
            cos_theta: std::ptr::null(),
            sin_theta: std::ptr::null(),
            stream: std::ptr::null_mut(),
            softmax_scale: 0.0,
            seqlen_q: 0,
            seqlen_k: 0,
            nhead_q: 0,
            nhead_k: 0,
            head_dim: 0,
            causal: 0,
            k_row_stride_bytes: 0,
            v_row_stride_bytes: 0,
        }
    }
}

impl Default for FlashAttnCkQuantizedPrefillParams {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnCkQuantizedMqQ8Params {
    pub abi_version: u32,
    pub struct_size: u32,
    pub prefill: FlashAttnCkQuantizedPrefillParams,
    pub gate: *const f32,
    pub signs1: *const f32,
    pub signs2: *const f32,
    pub q8_1_out: *mut c_void,
}

impl FlashAttnCkQuantizedMqQ8Params {
    pub fn new(prefill: FlashAttnCkQuantizedPrefillParams) -> Self {
        Self {
            abi_version: FLASH_ATTN_CK_QUANTIZED_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            prefill,
            gate: std::ptr::null(),
            signs1: std::ptr::null(),
            signs2: std::ptr::null(),
            q8_1_out: std::ptr::null_mut(),
        }
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
type QuantizedWorkspaceFn = unsafe extern "C" fn(i32, i32, i32) -> usize;
type QuantizedStagedWorkspaceFn = unsafe extern "C" fn(i32, i32, i32, i32, i32) -> usize;
type QuantizedFwdFn =
    unsafe extern "C" fn(*const FlashAttnCkQuantizedPrefillParams, *mut c_char, usize) -> i32;
type QuantizedMqQ8Fn =
    unsafe extern "C" fn(*const FlashAttnCkQuantizedMqQ8Params, *mut c_char, usize) -> i32;

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

/// Loaded quantized-prefill sidecar and its stable function table.
pub struct FlashAttnCkQuantized {
    _library: &'static Library,
    workspace_bytes: QuantizedWorkspaceFn,
    prefill_supported: QuantizedFwdFn,
    prefill: QuantizedFwdFn,
    staged_workspace_bytes: Option<QuantizedStagedWorkspaceFn>,
    staged_supported: Option<QuantizedFwdFn>,
    staged_prefill: Option<QuantizedFwdFn>,
    asym4_staged_supported: Option<QuantizedFwdFn>,
    asym4_staged_prefill: Option<QuantizedFwdFn>,
    mq_q8_supported: Option<QuantizedMqQ8Fn>,
    prefill_mq_q8: Option<QuantizedMqQ8Fn>,
}

impl FlashAttnCkQuantized {
    /// Load one explicit quantized sidecar path without changing runtime dispatch.
    ///
    /// # Safety
    ///
    /// `path` must identify a trusted native library implementing the declared ABI.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, FlashAttnCkError> {
        let path = path.as_ref();
        let library =
            unsafe { Library::new(path.as_os_str()) }.map_err(|source| FlashAttnCkError::Load {
                path: path.to_path_buf(),
                source,
            })?;

        let (
            workspace_bytes,
            prefill_supported,
            prefill,
            staged_workspace_bytes,
            staged_supported,
            staged_prefill,
            asym4_staged_supported,
            asym4_staged_prefill,
            mq_q8_supported,
            prefill_mq_q8,
        ) = unsafe {
            let abi_version: Symbol<'_, AbiVersionFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_quantized_abi_version",
                "quantized_abi_version",
            )?;
            let workspace_bytes: Symbol<'_, QuantizedWorkspaceFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_quantized_prefill_workspace_bytes",
                "quantized_prefill_workspace_bytes",
            )?;
            let prefill_supported: Symbol<'_, QuantizedFwdFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_quantized_prefill_supported",
                "quantized_prefill_supported",
            )?;
            let prefill: Symbol<'_, QuantizedFwdFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_quantized_prefill",
                "quantized_prefill",
            )?;
            let mut staged_workspace_bytes = library
                .get::<QuantizedStagedWorkspaceFn>(
                    b"hipfire_flash_attn_ck_quantized_staged_workspace_bytes",
                )
                .ok()
                .map(|symbol| *symbol);
            let mut staged_supported = library
                .get::<QuantizedFwdFn>(b"hipfire_flash_attn_ck_quantized_staged_supported")
                .ok()
                .map(|symbol| *symbol);
            let mut staged_prefill = library
                .get::<QuantizedFwdFn>(b"hipfire_flash_attn_ck_quantized_staged_prefill")
                .ok()
                .map(|symbol| *symbol);
            if staged_workspace_bytes.is_none()
                || staged_supported.is_none()
                || staged_prefill.is_none()
            {
                staged_workspace_bytes = None;
                staged_supported = None;
                staged_prefill = None;
            }
            let mut asym4_staged_supported = library
                .get::<QuantizedFwdFn>(b"hipfire_flash_attn_ck_asym4_staged_supported")
                .ok()
                .map(|symbol| *symbol);
            let mut asym4_staged_prefill = library
                .get::<QuantizedFwdFn>(b"hipfire_flash_attn_ck_asym4_staged_prefill")
                .ok()
                .map(|symbol| *symbol);
            if asym4_staged_supported.is_none() || asym4_staged_prefill.is_none() {
                asym4_staged_supported = None;
                asym4_staged_prefill = None;
            }
            let mq_q8_supported = library
                .get::<QuantizedMqQ8Fn>(b"hipfire_flash_attn_ck_quantized_mq_q8_supported")
                .ok()
                .map(|symbol| *symbol);
            let prefill_mq_q8 = library
                .get::<QuantizedMqQ8Fn>(b"hipfire_flash_attn_ck_quantized_prefill_mq_q8")
                .ok()
                .map(|symbol| *symbol);
            let actual = abi_version();
            if actual != FLASH_ATTN_CK_QUANTIZED_ABI_VERSION {
                return Err(FlashAttnCkError::AbiVersion {
                    expected: FLASH_ATTN_CK_QUANTIZED_ABI_VERSION,
                    actual,
                });
            }

            (
                *workspace_bytes,
                *prefill_supported,
                *prefill,
                staged_workspace_bytes,
                staged_supported,
                staged_prefill,
                asym4_staged_supported,
                asym4_staged_prefill,
                mq_q8_supported,
                prefill_mq_q8,
            )
        };
        let library = Box::leak(Box::new(library));
        Ok(Self {
            _library: library,
            workspace_bytes,
            prefill_supported,
            prefill,
            staged_workspace_bytes,
            staged_supported,
            staged_prefill,
            asym4_staged_supported,
            asym4_staged_prefill,
            mq_q8_supported,
            prefill_mq_q8,
        })
    }

    pub fn workspace_bytes(&self, seqlen_q: i32, nhead_q: i32, head_dim: i32) -> usize {
        unsafe { (self.workspace_bytes)(seqlen_q, nhead_q, head_dim) }
    }

    pub fn has_staged_route(&self) -> bool {
        self.staged_workspace_bytes.is_some()
            && self.staged_supported.is_some()
            && self.staged_prefill.is_some()
    }

    pub fn staged_workspace_bytes(
        &self,
        seqlen_q: i32,
        seqlen_k: i32,
        nhead_q: i32,
        nhead_k: i32,
        head_dim: i32,
    ) -> Option<usize> {
        self.staged_workspace_bytes
            .map(|function| unsafe { function(seqlen_q, seqlen_k, nhead_q, nhead_k, head_dim) })
    }

    pub fn is_staged_supported(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        let function = self
            .staged_supported
            .ok_or_else(|| FlashAttnCkError::Call {
                operation: "staged support check",
                status: -1,
                message: "sidecar does not export the optional staged CK route".to_string(),
            })?;
        self.call("staged support check", function, params)
    }

    pub fn is_asym4_staged_supported(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        let function = self
            .asym4_staged_supported
            .ok_or_else(|| FlashAttnCkError::Call {
                operation: "Asym4 staged support check",
                status: -1,
                message: "sidecar does not export the optional Asym4 staged route".to_string(),
            })?;
        self.call("Asym4 staged support check", function, params)
    }

    /// Launch staged quantized prefill on the stream stored in `params`.
    ///
    /// # Safety
    ///
    /// Device pointers and workspace must satisfy the sidecar contract and
    /// remain valid until the asynchronous stream work completes.
    pub unsafe fn staged_prefill(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        let function = self.staged_prefill.ok_or_else(|| FlashAttnCkError::Call {
            operation: "staged prefill",
            status: -1,
            message: "sidecar does not export the optional staged CK route".to_string(),
        })?;
        self.call("staged prefill", function, params)
    }

    /// Launch Givens-Asym4 staged prefill on the stream stored in `params`.
    ///
    /// # Safety
    ///
    /// Device pointers and workspace must satisfy the sidecar contract and
    /// remain valid until the asynchronous stream work completes.
    pub unsafe fn asym4_staged_prefill(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        let function = self
            .asym4_staged_prefill
            .ok_or_else(|| FlashAttnCkError::Call {
                operation: "Asym4 staged prefill",
                status: -1,
                message: "sidecar does not export the optional Asym4 staged route".to_string(),
            })?;
        self.call("Asym4 staged prefill", function, params)
    }

    pub fn is_supported(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        self.call("quantized support check", self.prefill_supported, params)
    }

    /// Launch quantized prefill on the stream stored in `params`.
    ///
    /// # Safety
    ///
    /// Device pointers, packed layouts, and workspace must satisfy the sidecar
    /// contract and remain valid until the asynchronous stream work completes.
    pub unsafe fn prefill(
        &self,
        params: &FlashAttnCkQuantizedPrefillParams,
    ) -> Result<(), FlashAttnCkError> {
        self.call("quantized prefill", self.prefill, params)
    }

    pub fn has_mq_q8_bridge(&self) -> bool {
        self.mq_q8_supported.is_some() && self.prefill_mq_q8.is_some()
    }

    pub fn is_mq_q8_supported(
        &self,
        params: &FlashAttnCkQuantizedMqQ8Params,
    ) -> Result<(), FlashAttnCkError> {
        let function = self.mq_q8_supported.ok_or_else(|| FlashAttnCkError::Call {
            operation: "MQ-Q8 support check",
            status: -1,
            message: "sidecar does not export the optional MQ-Q8 bridge".to_string(),
        })?;
        self.call_mq_q8("MQ-Q8 support check", function, params)
    }

    pub unsafe fn prefill_mq_q8(
        &self,
        params: &FlashAttnCkQuantizedMqQ8Params,
    ) -> Result<(), FlashAttnCkError> {
        let function = self.prefill_mq_q8.ok_or_else(|| FlashAttnCkError::Call {
            operation: "MQ-Q8 prefill",
            status: -1,
            message: "sidecar does not export the optional MQ-Q8 bridge".to_string(),
        })?;
        self.call_mq_q8("MQ-Q8 prefill", function, params)
    }

    fn call(
        &self,
        operation: &'static str,
        function: QuantizedFwdFn,
        params: &FlashAttnCkQuantizedPrefillParams,
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

    fn call_mq_q8(
        &self,
        operation: &'static str,
        function: QuantizedMqQ8Fn,
        params: &FlashAttnCkQuantizedMqQ8Params,
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
    fn quantized_params_match_c_abi_layout() {
        assert_eq!(
            std::mem::size_of::<FlashAttnCkQuantizedPrefillParams>(),
            120
        );
        assert_eq!(std::mem::align_of::<FlashAttnCkQuantizedPrefillParams>(), 8);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedPrefillParams, q),
            8
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedPrefillParams, workspace_bytes),
            48
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedPrefillParams, softmax_scale),
            80
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedPrefillParams, v_row_stride_bytes),
            112
        );

        assert_eq!(std::mem::size_of::<FlashAttnCkQuantizedMqQ8Params>(), 160);
        assert_eq!(std::mem::align_of::<FlashAttnCkQuantizedMqQ8Params>(), 8);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedMqQ8Params, prefill),
            8
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedMqQ8Params, gate),
            128
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkQuantizedMqQ8Params, q8_1_out),
            152
        );
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

    #[test]
    fn explicit_quantized_sidecar_loads_and_checks_gate() {
        let Ok(path) = std::env::var("HIPFIRE_FLASH_ATTN_CK_QUANTIZED_TEST_LIB") else {
            return;
        };
        let sidecar = unsafe { FlashAttnCkQuantized::load(path) }
            .expect("load explicit quantized test sidecar");
        assert_eq!(sidecar.workspace_bytes(128, 24, 256), 3_145_728);

        if std::env::var_os("HIPFIRE_FLASH_ATTN_CK_EXPECT_STAGED").is_some() {
            assert!(
                sidecar.has_staged_route(),
                "explicit sidecar must export the complete staged route"
            );
        }

        if std::env::var_os("HIPFIRE_FLASH_ATTN_CK_EXPECT_MQ_Q8").is_some() {
            assert!(
                sidecar.has_mq_q8_bridge(),
                "explicit sidecar must export the optional MQ-Q8 bridge"
            );
        }

        let error = sidecar
            .is_supported(&FlashAttnCkQuantizedPrefillParams::default())
            .expect_err("zero-shape quantized parameters must be rejected");
        assert!(matches!(
            error,
            FlashAttnCkError::Call {
                operation: "quantized support check",
                status: 1,
                ..
            }
        ));
    }
}
