//! Parse .hsaco (AMD GPU code object) ELF files.
//!
//! .hsaco files are Clang offload bundles containing ELF64 binaries with:
//! - .text section: GPU ISA machine code
//! - .rodata section: Kernel descriptors (64 bytes each, V3 ABI)
//! - .note section: AMDGPU metadata
//!
//! Kernel Descriptor V3 layout (64 bytes):
//!   [0:3]   group_segment_fixed_size (u32) — LDS bytes
//!   [4:7]   private_segment_fixed_size (u32) — scratch bytes
//!   [8:15]  kernarg_size (u64)
//!   [16:23] kernel_code_entry_byte_offset (i64) — relative to KD start
//!   [24:47] reserved
//!   [48:51] compute_pgm_rsrc1 (u32) — VGPRs, SGPRs, float mode
//!   [52:55] compute_pgm_rsrc2 (u32) — LDS blocks, user SGPRs, etc.
//!   [56:57] kernel_code_properties (u16)
//!   [58:63] reserved
//!
//! Reference: https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor

use crate::{RedlineError, Result};

/// Parsed kernel metadata from an .hsaco file.
#[derive(Debug, Clone)]
pub struct KernelMeta {
    pub name: String,
    /// Absolute offset within ELF of the kernel code entry point
    pub code_offset: u64,
    /// Absolute offset within ELF of the kernel descriptor
    pub kd_offset: u64,
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub kernarg_size: u64,
}

impl KernelMeta {
    /// Decode VGPR count from pgm_rsrc1 (GFX10+: granularity 8)
    pub fn vgpr_count(&self) -> u32 {
        // GFX10: VGPR_COUNT = (pgm_rsrc1[5:0] + 1) * 8
        // But hipcc often reports in units of 4 for wave64
        ((self.pgm_rsrc1 & 0x3F) + 1) * 4
    }

    /// Decode SGPR count from pgm_rsrc1
    pub fn sgpr_count(&self) -> u32 {
        (((self.pgm_rsrc1 >> 6) & 0xF) + 1) * 8
    }
}

/// Parsed .hsaco code object.
pub struct HsacoModule {
    /// Raw ELF bytes (after extracting from offload bundle if needed)
    pub elf: Vec<u8>,
    pub text_offset: u64,
    pub text_size: u64,
    pub kernels: Vec<KernelMeta>,
}

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const EM_AMDGPU: u16 = 224;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;

impl HsacoModule {
    /// Parse an .hsaco file from bytes.
    pub fn from_bytes(mut data: Vec<u8>) -> Result<Self> {
        // Handle Clang offload bundle wrapper
        if data.len() > 24 && &data[0..24] == b"__CLANG_OFFLOAD_BUNDLE__" {
            if let Some(pos) = data.windows(4).position(|w| w == ELF_MAGIC) {
                data = data[pos..].to_vec();
            } else {
                return Err(RedlineError { code: -1, message: "offload bundle contains no ELF".into() });
            }
        }

        if data.len() < 64 || data[0..4] != ELF_MAGIC {
            return Err(RedlineError { code: -1, message: "not a valid ELF file".into() });
        }
        if u16_le(&data, 18) != EM_AMDGPU {
            return Err(RedlineError { code: -1, message: "not an AMDGPU ELF".into() });
        }

        let shoff = u64_le(&data, 40) as usize;
        let shentsize = u16_le(&data, 58) as usize;
        let shnum = u16_le(&data, 60) as usize;
        let shstrndx = u16_le(&data, 62) as usize;
        let shstr_offset = u64_le(&data, shoff + shstrndx * shentsize + 24) as usize;

        let mut text_offset = 0u64;
        let mut text_size = 0u64;
        let mut symtab_offset = 0usize;
        let mut symtab_size = 0usize;
        let mut symtab_entsize = 0usize;
        let mut symtab_link = 0usize;

        for i in 0..shnum {
            let base = shoff + i * shentsize;
            let name_off = u32_le(&data, base) as usize;
            let sh_type = u32_le(&data, base + 4);
            let sh_offset = u64_le(&data, base + 24);
            let sh_size = u64_le(&data, base + 32);
            let name = read_cstr(&data, shstr_offset + name_off);

            if name == ".text" {
                text_offset = sh_offset;
                text_size = sh_size;
            }
            if sh_type == SHT_SYMTAB {
                symtab_offset = sh_offset as usize;
                symtab_size = sh_size as usize;
                symtab_entsize = u64_le(&data, base + 56) as usize;
                symtab_link = u32_le(&data, base + 40) as usize;
            }
        }

        // Get string table for symbol names
        let strtab_offset = if symtab_link < shnum {
            u64_le(&data, shoff + symtab_link * shentsize + 24) as usize
        } else { 0 };

        // Find kernel descriptors: symbols ending in ".kd"
        let mut kernels = Vec::new();
        if symtab_entsize > 0 {
            let num_syms = symtab_size / symtab_entsize;
            for i in 0..num_syms {
                let base = symtab_offset + i * symtab_entsize;
                if base + symtab_entsize > data.len() { break; }
                let st_name = u32_le(&data, base) as usize;
                let st_value = u64_le(&data, base + 8);
                let name = read_cstr(&data, strtab_offset + st_name);

                if name.ends_with(".kd") {
                    // st_value is the ELF virtual address — for ET_DYN, this is the file offset
                    let kd_off = st_value as usize;
                    if kd_off + 64 <= data.len() {
                        // V3 kernel descriptor layout
                        let group_segment_size = u32_le(&data, kd_off);
                        let private_segment_size = u32_le(&data, kd_off + 4);
                        let kernarg_size = u64_le(&data, kd_off + 8);
                        let code_entry_rel = i64_le(&data, kd_off + 16);
                        let pgm_rsrc1 = u32_le(&data, kd_off + 48);
                        let pgm_rsrc2 = u32_le(&data, kd_off + 52);

                        let code_offset = (kd_off as i64 + code_entry_rel) as u64;
                        let kernel_name = name.trim_end_matches(".kd").to_string();

                        kernels.push(KernelMeta {
                            name: kernel_name,
                            code_offset,
                            kd_offset: kd_off as u64,
                            pgm_rsrc1,
                            pgm_rsrc2,
                            group_segment_size,
                            private_segment_size,
                            kernarg_size,
                        });
                    }
                }
            }
        }

        Ok(Self { elf: data, text_offset, text_size, kernels })
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| RedlineError { code: -1, message: format!("failed to read {path}: {e}") })?;
        Self::from_bytes(data)
    }
}

fn u16_le(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

fn u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

fn u64_le(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        data[off], data[off+1], data[off+2], data[off+3],
        data[off+4], data[off+5], data[off+6], data[off+7],
    ])
}

fn i64_le(data: &[u8], off: usize) -> i64 {
    i64::from_le_bytes([
        data[off], data[off+1], data[off+2], data[off+3],
        data[off+4], data[off+5], data[off+6], data[off+7],
    ])
}

fn read_cstr(data: &[u8], off: usize) -> String {
    let mut end = off;
    while end < data.len() && data[end] != 0 { end += 1; }
    String::from_utf8_lossy(&data[off..end]).into()
}
