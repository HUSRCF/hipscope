//! RDNA3/RDNA3.5 codec: separate target opcode tables, shared typed field codec.
use smallvec::SmallVec;
use crate::inst::{Arch, Inst};
pub use super::gfx12::DecodeError;

pub fn decode(arch: Arch, words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::decode_for(arch, words)
}
pub fn encode(arch: Arch, inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    assert!(matches!(arch, Arch::Gfx1100 | Arch::Gfx1151));
    super::gfx12::encode_for(arch, inst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{isa, wait::Counter};

    #[test]
    fn target_rows_reencode_from_typed_fields() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let mut failures = Vec::new();
            for row in isa::table(arch) {
                let words: Vec<u32> = row.encoding.split_whitespace()
                    .map(|word| u32::from_str_radix(word, 16).unwrap()).collect();
                match decode(arch, &words).and_then(|(inst, used)| {
                    if used != words.len() { return Err(super::super::gfx12::DecodeError::Rejected { offset: 0, reason: "width mismatch".into() }); }
                    let output = encode(arch, &inst)?;
                    if output.as_slice() != words { return Err(super::super::gfx12::DecodeError::Rejected { offset: 0, reason: "word mismatch".into() }); }
                    Ok(())
                }) {
                    Ok(()) => {},
                    Err(error) => failures.push(format!("{} {words:08x?}: {error}", row.name)),
                }
            }
            assert!(failures.is_empty(), "{arch:?}:\n{}", failures.join("\n"));
        }
    }

    #[test]
    fn gfx11_waits_and_scalar_float_are_arch_specific() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let (inst, _) = decode(arch, &[0xbf89_0432]).unwrap();
            let wait = inst.mods.wait.as_ref().unwrap();
            assert_eq!(wait.per_counter[Counter::Vm as usize], Some(1));
            assert_eq!(wait.per_counter[Counter::Exp as usize], Some(2));
            assert_eq!(wait.per_counter[Counter::Lgkm as usize], Some(3));
            assert_eq!(encode(arch, &inst).unwrap().as_slice(), &[0xbf89_0432]);
            let (mut variant, _) = decode(arch, &[0xbf89_043a]).unwrap();
            assert_eq!(variant.mods.wait, inst.mods.wait);
            variant.prov.bytes = None;
            assert_eq!(encode(arch, &variant).unwrap().as_slice(), &[0xbf89_043a]);
        }
        assert!(decode(Arch::Gfx1100, &[0xa000_0201]).is_err());
        let (add, _) = decode(Arch::Gfx1151, &[0xa000_0201]).unwrap();
        assert_eq!(add.op.name(Arch::Gfx1151), Some("s_add_f32"));
    }
}
