//! C5: LDS segment and address facts for lifted kernels.
//!
//! Lifted code gets facts, never invented slots (core.md §2.7): every DS
//! access is described by an [`LdsAccess`] whose address is resolved by a
//! per-block walk of the address VGPR's def chain (`v_mov_b32` with an
//! inline/literal constant, `ds_*` offsets added in; `v_add_*` with one
//! constant side degrades to [`AddrFact::Bounded`]; anything through memory,
//! `v_readlane`, a loop-carried value, or outside the block is
//! [`AddrFact::Unknown`]). All lifted accesses attach to one [`Whole`]
//! segment; `max_end` is `None` when any access is `Unknown` — the
//! machine-readable record Fable's Q7 launch-contract check consumes
//! (M1 emits no `Obligation` object for it; see `LdsAnalysis`).

use thiserror::Error;

use crate::cfg::{Body, InstId};
use crate::effects::MemClass;
use crate::inst::{Arch, Inst};
use crate::lds::{AddrFact, LdsAccess, LdsFacts, LdsKind, LdsSegment, SegId, SegmentOrigin};
use crate::operand::{ImmField, InlineConst, Operand};
use crate::reg::Kind;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LdsError {
    #[error("LDS facts need a gfx11 or gfx12 opcode table (got {0:?})")]
    UnsupportedArch(Arch),
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
}

/// M1 LDS analysis is facts-only: `max_end == None` (some address
/// `Unknown`) is the record of the launch-contract obligation, not a
/// separate object. Dynamic LDS sizes come from the dispatch packet, never
/// from the object, so no lifted kernel can prove them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LdsAnalysis {
    pub facts: LdsFacts,
}

fn access_kind(class: MemClass) -> Option<LdsKind> {
    match class {
        MemClass::DsLoad => Some(LdsKind::Load),
        MemClass::DsStore => Some(LdsKind::Store),
        MemClass::DsAtomic { .. } => Some(LdsKind::Atomic),
        _ => None,
    }
}

/// Data width in bytes from the opcode suffix (`_b32` → 4 …).
fn access_bytes(name: &str) -> u32 {
    if name.contains("_b128") {
        16
    } else if name.contains("_b64") {
        8
    } else if name.contains("_b32") {
        4
    } else if name.contains("_b16") {
        2
    } else if name.contains("_b8") {
        1
    } else {
        0
    }
}

/// DS offset operands carried by the access itself.
fn own_offsets(inst: &Inst) -> u32 {
    inst.operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Imm(ImmField::DsOffset(n)) => Some(u32::from(*n)),
            Operand::Imm(ImmField::DsOffset0(n) | ImmField::DsOffset1(n)) => {
                Some(u32::from(*n))
            }
            _ => None,
        })
        .fold(0u32, |acc, n| acc.wrapping_add(n))
}

fn const_operand(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::Inline(InlineConst::Integer(n)) => Some(*n as u32),
        Operand::Literal(n) => Some(*n),
        _ => None,
    }
}

fn is_add(name: &str) -> bool {
    name == "v_add_nc_u32" || name == "v_add_u32" || name == "v_add_co_u32"
}

/// Resolve the address VGPR by walking its def chain backward inside one
/// layout range (the containing block, or the whole layout when blockless).
fn resolve(
    body: &Body,
    arch: Arch,
    range: (usize, usize),
    pos: usize,
    addr_base: u16,
    offsets: u32,
) -> AddrFact {
    for back in (range.0..pos).rev() {
        let id = body.layout[back];
        let Some(def) = body.insts.get(id) else {
            continue;
        };
        let defines = def
            .effects
            .defs
            .iter()
            .any(|reg| reg.kind == Kind::V && reg.base <= addr_base && addr_base < reg.base + u16::from(reg.len));
        if !defines {
            continue;
        }
        let name = def.op.name(arch).unwrap_or("");
        if name == "v_mov_b32_e32" {
            if let Some(value) = def.operands.get(1).and_then(const_operand) {
                return AddrFact::Const(value.wrapping_add(offsets));
            }
            return AddrFact::Unknown;
        }
        if is_add(name) {
            let mut constant = None;
            for operand in def.operands.iter().skip(1) {
                if let Operand::Reg(_) | Operand::Half(_, _) = operand {
                    continue;
                }
                if let Some(value) = const_operand(operand) {
                    constant = Some(value);
                }
            }
            if let Some(value) = constant {
                let base = value.wrapping_add(offsets);
                return AddrFact::Bounded { lo: base, hi: u32::MAX };
            }
        }
        return AddrFact::Unknown;
    }
    AddrFact::Unknown
}

fn range_of(body: &Body, pos: usize) -> (usize, usize) {
    body.blocks
        .iter()
        .find_map(|block| {
            let (start, end) = block.range;
            (pos >= start && pos < end).then_some(block.range)
        })
        .unwrap_or((0, body.layout.len()))
}

/// LDS facts for one kernel body.
pub fn analyze(body: &Body, arch: Arch) -> Result<LdsAnalysis, LdsError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(LdsError::UnsupportedArch(arch));
    }
    let whole = SegId(0);
    let mut facts = LdsFacts {
        segments: vec![LdsSegment { id: whole, range: None, origin: SegmentOrigin::Whole }],
        accesses: Vec::new(),
        max_end: Some(0),
    };
    let mut end = 0u32;
    let mut bounded = false;
    for (pos, id) in body.layout.iter().enumerate() {
        let Some(inst) = body.insts.get(*id) else {
            return Err(LdsError::DanglingInst { id: *id });
        };
        let Some(mem) = &inst.effects.mem else {
            continue;
        };
        let Some(kind) = access_kind(mem.class) else {
            continue;
        };
        let name = inst.op.name(arch).unwrap_or("");
        let bytes = access_bytes(name);
        let addr = match inst.effects.uses.first() {
            Some(crate::reg::RegRef { kind: Kind::V, base, .. }) => resolve(
                body,
                arch,
                range_of(body, pos),
                pos,
                *base,
                own_offsets(inst),
            ),
            _ => AddrFact::Unknown,
        };
        if matches!(addr, AddrFact::Unknown) {
            bounded = true;
        } else if let AddrFact::Const(base) = addr {
            end = end.max(base.wrapping_add(bytes));
        } else {
            bounded = true;
        }
        facts.accesses.push((LdsAccess { inst: *id, addr, bytes, kind }, whole));
    }
    facts.max_end = if bounded { None } else { Some(end) };
    Ok(LdsAnalysis { facts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    use crate::cfg::Body;
    use crate::inst::{Form, Inst, VmemForm};
    use crate::operand::Modifiers;
    use crate::provenance::Provenance;

    fn row(name: &str) -> (crate::inst::Opcode, Form) {
        let row = crate::isa::gfx12().iter().find(|row| row.name == name).expect(name);
        (row.op, row.form)
    }

    fn v(base: u16) -> Operand {
        Operand::Reg(crate::reg::RegRef { kind: Kind::V, base, len: 1 })
    }

    fn inst(name: &str, operands: Vec<Operand>) -> Inst {
        let (op, form) = row(name);
        Inst::from_parts(
            Arch::Gfx1201,
            op,
            form,
            crate::inst::FormFields::None,
            SmallVec::from_vec(operands),
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut body = Body::default();
        for inst in insts {
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        body
    }

    #[test]
    fn constant_store_resolves_with_own_offset() {
        let mov = inst("v_mov_b32_e32", vec![v(10), Operand::Literal(0x1000)]);
        let store = inst(
            "ds_store_b32",
            vec![v(10), v(11), Operand::Imm(ImmField::DsOffset(64))],
        );
        let body = body_of(vec![mov, store]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert_eq!(analysis.facts.accesses.len(), 1);
        assert_eq!(
            analysis.facts.accesses[0].0.addr,
            AddrFact::Const(0x1040),
            "{:?}",
            analysis.facts.accesses[0].0
        );
        assert_eq!(analysis.facts.max_end, Some(0x1044));
    }

    #[test]
    fn computed_address_is_unknown_with_no_max_end() {
        let store = inst("ds_store_b32", vec![v(10), v(11)]);
        let body = body_of(vec![store]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert_eq!(analysis.facts.accesses[0].0.addr, AddrFact::Unknown);
        assert_eq!(analysis.facts.max_end, None);
    }

    #[test]
    fn vmem_instructions_are_not_lds_accesses() {
        let load = inst(
            "global_load_b32",
            vec![
                v(1),
                Operand::Reg(crate::reg::RegRef { kind: Kind::V, base: 2, len: 2 }),
                Operand::Vmem(crate::operand::VmemToken::Off),
            ],
        );
        let body = body_of(vec![load]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert!(analysis.facts.accesses.is_empty());
        assert_eq!(analysis.facts.max_end, Some(0));
    }

    #[allow(dead_code)]
    fn form_shape(_form: VmemForm) {}
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::Arch;
    use crate::lds::AddrFact;
    use crate::passes::cfg::build_blocks;

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("CFG");
        body
    }

    /// KT48 uses dynamic LDS throughout: accesses exist, addresses are not
    /// block-local constants, so `max_end` is `None` — the machine-readable
    /// record of the launch-contract obligation (dynamic 49,936 B comes
    /// from the dispatch packet, never the object).
    #[test]
    fn kt48_lds_facts() {
        let body = kt48_body();
        let analysis = super::analyze(&body, Arch::Gfx1201).unwrap();
        assert!(!analysis.facts.accesses.is_empty());
        assert!(
            analysis.facts.accesses.iter().any(|(access, _)| access.addr == AddrFact::Unknown),
            "expected computed LDS addresses"
        );
        assert_eq!(analysis.facts.max_end, None);
        assert_eq!(analysis.facts.segments.len(), 1);
    }
}
