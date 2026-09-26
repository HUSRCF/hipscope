use smallvec::SmallVec;
use thiserror::Error;

use crate::{cfg::{Body, InstId, Terminator}, descriptor::KernelDescriptor, effects::Effects, metadata::HsaKernelMetadata, operand::{Modifiers, Operand}, provenance::Provenance};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Arch { Gfx1010, Gfx1030, Gfx1100, Gfx1151, Gfx1201 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Setting { Any, On, Off }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target { pub arch: Arch, pub xnack: Setting, pub sramecc: Setting, pub abi_version: u8 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wave { Wave32, Wave64 }
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymbolId(pub String);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Family { Sop1, Sop2, Sopc, Sopk, Sopp, Smem, Vop1, Vop2, Vopc, Vop3, Vop3p, Vopd, Vinterp, Ds, Vmem, Export }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Opcode { pub family: Family, pub id: u16 }
impl Opcode {
    pub fn name(self, arch: Arch) -> Option<&'static str> { crate::isa::lookup_opcode(arch, self).map(|row| row.name) }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VmemForm { Global, Scratch, Flat, Buffer, Image }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Form { Sop1, Sop2, Sopc, Sopk, Sopp, Smem, Vop1, Vop2, Vopc, Vop3, Vop3p, Vopd, Vinterp, Ds, Vmem(VmemForm), Export }

/// Values not represented by semantic operands/modifiers. An honored field is never
/// canonicalized; an ignored field is validated against the table's benign set.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub enum FormFields {
    #[default] None,
    Vop3b { src2_unused: u16 },
    Bits { ignored: SmallVec<[NamedField; 4]>, honored: SmallVec<[NamedField; 4]> },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedField { pub name: &'static str, pub value: u32 }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inst {
    pub op: Opcode,
    pub form: Form,
    pub fields: FormFields,
    pub operands: SmallVec<[Operand; 6]>,
    pub mods: Modifiers,
    pub literal: Option<u32>,
    pub effects: Effects,
    pub prov: Provenance,
}
impl Inst {
    pub fn text(&self, arch: Arch) -> Result<String, ValidateError> {
        let row = crate::isa::lookup(arch, self.op, self.form)
            .ok_or(ValidateError::UnknownOpcode { op: self.op, form: self.form })?;
        let mut text = row.name.to_owned();
        if !self.operands.is_empty() {
            text.push(' ');
            for (index, operand) in self.operands.iter().enumerate() {
                if index != 0 { text.push_str(", "); }
                text.push_str(&operand.to_string());
            }
        }
        text.push_str(&self.mods.text_suffix());
        Ok(text)
    }
    pub fn validate(&self, arch: Arch) -> Result<(), ValidateError> {
        let row = crate::isa::lookup(arch, self.op, self.form)
            .ok_or(ValidateError::UnknownOpcode { op: self.op, form: self.form })?;
        for operand in &self.operands {
            operand.validate().map_err(|reason| ValidateError::Operand(reason))?;
        }
        if let FormFields::Vop3b { src2_unused } = self.fields {
            if src2_unused > 0x1ff || src2_unused == 0xff {
                return Err(ValidateError::DangerousFill { field: "src2_unused", value: src2_unused as u32 });
            }
            if !row.benign_src2.iter().any(|&x| x == src2_unused) {
                return Err(ValidateError::UnknownDontCare { field: "src2_unused", value: src2_unused as u32 });
            }
        }
        for field in self.fields.ignored() {
            let rule = row.fields.iter().find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnknownDontCare { field: field.name, value: field.value })?;
            if rule.class != crate::isa::FieldClass::Ignored || !rule.allowed.contains(&field.value) {
                return Err(ValidateError::UnknownDontCare { field: field.name, value: field.value });
            }
        }
        for field in self.fields.honored() {
            let rule = row.fields.iter().find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnmodeledField { field: field.name })?;
            if rule.class != crate::isa::FieldClass::Honored || field.value & !rule.mask != 0 {
                return Err(ValidateError::UnmodeledField { field: field.name });
            }
        }
        if matches!(self.form, Form::Smem) {
            for operand in &self.operands {
                if let Operand::Reg(reg) = operand {
                    if reg.kind == crate::reg::Kind::S && reg.len > 1 && reg.base % u16::from(reg.len) != 0 {
                        return Err(ValidateError::MisalignedSmemSdata { base: reg.base, len: reg.len });
                    }
                }
            }
        }
        Ok(())
    }
}
impl FormFields {
    pub fn ignored(&self) -> &[NamedField] { match self { Self::Bits { ignored, .. } => ignored, _ => &[] } }
    pub fn honored(&self) -> &[NamedField] { match self { Self::Bits { honored, .. } => honored, _ => &[] } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program { pub target: Target, pub kernels: Vec<Kernel>, pub envelope: Envelope }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel { pub symbol: SymbolId, pub wave: Wave, pub abi: Abi, pub body: Body, pub origin: KernelOrigin }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Abi {
    Hsa { descriptor: KernelDescriptor, metadata: HsaKernelMetadata },
    Raw { user_sgprs: Vec<UserSgprRole>, wave: Wave, lds_bytes: u32, sidecar_sha256: [u8; 32] },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UserSgprRole { PrivateSegmentBuffer, DispatchPtr, QueuePtr, KernargSegmentPtr, DispatchId, FlatScratchInit, PrivateSegmentSize }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelOrigin {
    Frontend { kind: Frontend, object_sha256: [u8; 32], entry_va: u64, size: u64 },
    Authored { builder_crate: String, version: String, git: String },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frontend { Hipcc, Triton, Aco, Builder, Ctor }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Envelope { pub image: Vec<u8>, pub bundled: bool }
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ValidateError {
    #[error("unknown opcode {op:?} / encoding form {form:?}")]
    UnknownOpcode { op: Opcode, form: Form },
    #[error("invalid operand: {0}")]
    Operand(String),
    #[error("unknown don't-care value {value:#x} for {field}")]
    UnknownDontCare { field: &'static str, value: u32 },
    #[error("field {field} has no known encoding rule")]
    UnmodeledField { field: &'static str },
    #[error("unused field {field} value {value:#x} is the literal selector or out of range")]
    DangerousFill { field: &'static str, value: u32 },
    #[error("SMEM SDATA s{base} length {len} must be aligned")]
    MisalignedSmemSdata { base: u16, len: u8 },
    #[error("invalid program layout: {0}")]
    Layout(String),
    #[error("kernel {0} overlaps another kernel in the source image")]
    KernelOverlap(String),
}
impl Program {
    pub fn validate(&self) -> Result<(), ValidateError> {
        let mut regions = Vec::new();
        for kernel in &self.kernels {
            let body = &kernel.body;
            let mut seen = vec![false; body.insts.len()];
            for &id in &body.layout {
                let Some(slot) = seen.get_mut(id.0) else { return Err(ValidateError::Layout("layout refers to missing slot".into())); };
                if *slot { return Err(ValidateError::Layout("layout repeats instruction".into())); }
                *slot = true;
                body.insts.get(id).ok_or_else(|| ValidateError::Layout("layout refers to a tombstone".into()))?
                    .validate(self.target.arch)?;
            }
            if body.insts.iter().any(|(id, _)| !seen[id.0]) { return Err(ValidateError::Layout("live instruction omitted from layout".into())); }
            let mut cursor = 0;
            for block in &body.blocks {
                if block.range.0 != cursor || block.range.1 <= cursor || block.range.1 > body.layout.len() {
                    return Err(ValidateError::Layout("blocks must partition layout into nonempty contiguous runs".into()));
                }
                if block.id.0 >= body.blocks.len() || body.blocks[block.id.0].id != block.id {
                    return Err(ValidateError::Layout("block ids must match block indices".into()));
                }
                for &succ in &block.succs {
                    if body.blocks.get(succ.0).is_none() { return Err(ValidateError::Layout("edge leaves CFG".into())); }
                }
                cursor = block.range.1;
            }
            if cursor != body.layout.len() { return Err(ValidateError::Layout("blocks do not cover layout".into())); }
            if !body.blocks.is_empty() && !body.blocks.iter().any(|b| b.term == Terminator::EndPgm) {
                return Err(ValidateError::Layout("missing s_endpgm terminator".into()));
            }
            if let KernelOrigin::Frontend { entry_va, size, .. } = &kernel.origin {
                let end = entry_va.checked_add(*size).ok_or_else(|| ValidateError::KernelOverlap(kernel.symbol.0.clone()))?;
                if regions.iter().any(|&(lo, hi)| *entry_va < hi && lo < end) {
                    return Err(ValidateError::KernelOverlap(kernel.symbol.0.clone()));
                }
                regions.push((*entry_va, end));
            }
        }
        Ok(())
    }
}
