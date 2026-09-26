//! C3: gfx1201 binary instruction decoder/encoder.
use crate::inst::Inst;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("unrecognized machine instruction at byte offset {offset}: {reason}")]
    Rejected { offset: usize, reason: String },
}

pub fn decode(_words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    todo!("C3: decode every declared gfx1201 instruction form")
}
pub fn encode(_inst: &Inst) -> Result<smallvec::SmallVec<[u32; 3]>, DecodeError> {
    todo!("C3: encode directly from typed fields, never provenance bytes")
}
