//! ELF, bundle, descriptor and metadata lifter for typed AMDGPU programs.
pub mod bundle;
pub mod elf;
pub mod kd;
pub mod metadata;
pub mod text;

use peacemaker_ir::{inst::{Abi, Arch, Program}, state::Lifted};

#[derive(Clone, Debug, Default)]
pub struct Options;
#[derive(Debug, thiserror::Error)]
pub enum LiftError {
    #[error("unsupported architecture or instruction at byte offset {offset}: {reason}")]
    Rejected { offset: u64, reason: String },
}

pub fn lift_object(_bytes: &[u8], _options: Options) -> Result<Lifted<Program>, LiftError> {
    todo!("C2/C3/C4/C5 object lifting")
}
pub fn lift_text(_source: &str, _arch: Arch) -> Result<Lifted<Program>, LiftError> {
    todo!("C3b text lifting through assembled bytes")
}
pub fn lift_raw(_bytes: &[u8], _abi: Abi, _arch: Arch) -> Result<Lifted<Program>, LiftError> {
    todo!("C2/C3 raw byte lifting")
}
