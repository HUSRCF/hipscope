//! C4: clause windows and path-sensitive delay-ALU facts.
//!
//! Clause rule (core.md §2.8, §5.2): an `s_clause N` window is the N+1
//! following layout instructions. Every member must live in the opener's
//! block (a window crossing a leader is a lift rejection), be a memory
//! instruction of one class (widths may mix: `s_load_b256` + `s_load_b128`
//! share `SmemLoad`), and contain no wait and no `s_endpgm`.
//!
//! Delay rule (RDNA4 ISA §5.8 + §16.5, `rdna4-instruction-set-architecture.pdf`
//! pp56/269, confirmed against pinned `llvm-mc`/`llvm-dis` sweeps):
//! `INSTID0 = SIMM16[3:0]` names the hazard for the next instruction,
//! `INSTSKIP = SIMM16[6:4]` selects the instruction carrying the second
//! dependency (`SAME` = same next instruction, `NEXT` = the one after,
//! `SKIP_N` = N further), `INSTID1 = SIMM16[10:7]` names its hazard. Each
//! INSTID counts backwards N previously issued VALU on the executed path
//! (branched-over instructions do not count; EXEC-skipped VALU do). Values:
//! 0 = NO_DEP, 1-4 = VALU_DEP_n, 5-7 = TRANS32_DEP_n, 8 = FMA_ACCUM_CYCLE_1
//! (reserved), 9-11 = SALU_CYCLE_1..3, 12-15 invalid. This pass assumes C3
//! stores those raw field values in `DelayAluHint { instid0, instskip,
//! instid1 }` (C3 confirmed the bit ranges).
//!
//! Resolution is exact inside one straight-line block. Anything needing a
//! cross-block walk (producer or consumer past the block edge) yields
//! `DelayStatus::Ambiguous`, never a rejection: hints are performance-only
//! and `NO_DEP` is always sound, so downstream edits touching the reach must
//! rewrite the hint to `NO_DEP` (core.md §2.8). Per-path enumeration across
//! joins is intentionally not attempted: the two paths into a hint may
//! disagree on producer distance, and `Ambiguous` is the honest fact.

use thiserror::Error;

use crate::cfg::{BlockId, Body, InstId};
use crate::effects::{Control, MemClass};
use crate::inst::Form;
use crate::operand::Modifiers;
use crate::passes::cfg::containing_block;

/// One validated `s_clause` window: opener, members in layout order, and
/// their shared memory class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClauseWindow {
    pub opener: InstId,
    pub block: BlockId,
    pub members: Vec<InstId>,
    pub class: MemClass,
}

/// Per-path delay fact. `Resolved` carries both dependency slots (each
/// `None` = NO_DEP) plus the two consumer positions (the instructions the
/// hint constrains); `consumers[1]` is what C6's window policy calls the
/// hint's "last target". `Ambiguous` means some path needs a cross-block
/// walk: touchers must rewrite to NO_DEP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayFact {
    pub hint: InstId,
    pub block: BlockId,
    pub status: DelayStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DelayStatus {
    Resolved { producers: [Option<InstId>; 2], consumers: [Option<InstId>; 2] },
    Ambiguous,
}

impl DelayFact {
    /// Constructor for the edit layer (C6): mark a hint ambiguous after a
    /// structural change without re-resolving.
    pub fn ambiguous(hint: InstId, block: BlockId) -> Self {
        Self { hint, block, status: DelayStatus::Ambiguous }
    }

    pub fn is_ambiguous(&self) -> bool {
        matches!(self.status, DelayStatus::Ambiguous)
    }

    /// Dependency slots when resolved (`None` slot = NO_DEP); `None` when
    /// ambiguous.
    pub fn producers(&self) -> Option<[Option<InstId>; 2]> {
        match self.status {
            DelayStatus::Resolved { producers, .. } => Some(producers),
            DelayStatus::Ambiguous => None,
        }
    }

    /// Constrained consumer positions when resolved; `None` when ambiguous.
    pub fn consumers(&self) -> Option<[Option<InstId>; 2]> {
        match self.status {
            DelayStatus::Resolved { consumers, .. } => Some(consumers),
            DelayStatus::Ambiguous => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct WindowFacts {
    pub clauses: Vec<ClauseWindow>,
    pub delays: Vec<DelayFact>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WindowError {
    #[error("body has no blocks; run passes::cfg::build_blocks first")]
    BlocksNotBuilt,
    #[error("s_clause at layout index {index} opens a window of {len} that leaves its block")]
    CrossesLeader { index: usize, len: usize },
    #[error("s_clause at layout index {index} has no decoded window length")]
    ClauseLengthMissing { index: usize },
    #[error("s_clause at layout index {index}: member at layout index {member} is not a memory instruction")]
    NonMemoryMember { index: usize, member: usize },
    #[error("s_clause at layout index {index}: mixed memory classes at layout index {member}")]
    MixedClass { index: usize, member: usize },
    #[error("s_clause at layout index {index}: wait inside clause window at layout index {member}")]
    WaitInsideClause { index: usize, member: usize },
    #[error("s_endpgm inside s_clause window opened at layout index {index}")]
    EndPgmInsideClause { index: usize },
    #[error("s_delay_alu at layout index {index} has no decoded hint fields")]
    HintMissing { index: usize },
}

/// VALU-issuing forms for INSTID lookback. Comparisons issue on the vector
/// ALU; SALU/DS/VMEM/SMEM/export do not.
fn is_valu(inst: &crate::inst::Inst) -> bool {
    matches!(
        inst.form,
        Form::Vop1 | Form::Vop2 | Form::Vop3 | Form::Vop3p | Form::Vopd | Form::Vopc | Form::Vinterp
    )
}

fn is_salu(inst: &crate::inst::Inst) -> bool {
    matches!(inst.form, Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Sopk | Form::Sopp)
}

/// Backward VALU distance named by an INSTID value: `None` = NO_DEP,
/// `Some(0)` = nearest SALU (cycle penalty, no VALU reach), `Some(n)` = n-th
/// previous VALU. Values 12-15 are invalid per the ISA and resolve to
/// `Unknown` (ambiguous, never reject).
enum IdNeed {
    None,
    Valu(usize),
    Salu,
    Unknown,
}

fn id_need(value: u8) -> IdNeed {
    match value {
        0 => IdNeed::None,
        1..=4 => IdNeed::Valu(usize::from(value)),
        // TRANS32 issues on vector pipes; counting it as VALU lookback is
        // the M1 approximation (documented: hardware counts TRANS only).
        5..=7 => IdNeed::Valu(usize::from(value - 4)),
        // Reserved FMA accumulator penalty: treat as one VALU back.
        8 => IdNeed::Valu(1),
        9..=11 => IdNeed::Salu,
        _ => IdNeed::Unknown,
    }
}

/// Forward consumer distance for an INSTSKIP value: `SAME` = the next
/// instruction, `NEXT` = the one after, `SKIP_N` = N further still.
/// Values 6-7 are invalid per the ISA.
fn skip_forward(value: u8) -> Option<usize> {
    match value {
        0..=5 => Some(usize::from(value)),
        _ => None,
    }
}

fn clause_length(mods: &Modifiers) -> Option<usize> {
    mods.clause.map(usize::from)
}

/// Validate every clause window and resolve every delay hint. Requires built
/// blocks. Clause violations reject; delay over-reach goes ambiguous.
pub fn check_windows(body: &Body) -> Result<WindowFacts, WindowError> {
    if body.blocks.is_empty() {
        return Err(WindowError::BlocksNotBuilt);
    }
    let pos_of: std::collections::HashMap<InstId, usize> =
        body.layout.iter().enumerate().map(|(pos, &id)| (id, pos)).collect();
    let mut facts = WindowFacts::default();
    for (pos, &id) in body.layout.iter().enumerate() {
        let inst = body.insts.get(id).expect("layout references live insts");
        match inst.effects.control {
            Control::Clause => {
                let len = clause_length(&inst.mods)
                    .ok_or(WindowError::ClauseLengthMissing { index: pos })?;
                let block = containing_block(body, id).expect("inst is laid out in a block");
                let mut members = Vec::with_capacity(len + 1);
                let mut class: Option<MemClass> = None;
                for k in 1..=len + 1 {
                    let mpos = pos + k;
                    let Some(&member) = body.layout.get(mpos) else {
                        return Err(WindowError::CrossesLeader { index: pos, len });
                    };
                    if containing_block(body, member) != Some(block) {
                        return Err(WindowError::CrossesLeader { index: pos, len });
                    }
                    let minst =
                        body.insts.get(member).expect("layout references live insts");
                    if matches!(minst.effects.control, Control::EndPgm) {
                        return Err(WindowError::EndPgmInsideClause { index: pos });
                    }
                    if matches!(minst.effects.control, Control::Wait) {
                        return Err(WindowError::WaitInsideClause { index: pos, member: mpos });
                    }
                    let Some(mem) = &minst.effects.mem else {
                        return Err(WindowError::NonMemoryMember { index: pos, member: mpos });
                    };
                    if let Some(first) = class {
                        if mem.class != first {
                            return Err(WindowError::MixedClass { index: pos, member: mpos });
                        }
                    } else {
                        class = Some(mem.class);
                    }
                    members.push(member);
                }
                facts.clauses.push(ClauseWindow {
                    opener: id,
                    block,
                    members,
                    class: class.expect("non-empty window has a class"),
                });
            }
            Control::Delay => {
                let hint = inst.mods.delay.ok_or(WindowError::HintMissing { index: pos })?;
                let block = containing_block(body, id).expect("inst is laid out in a block");
                facts.delays.push(resolve_delay(body, &pos_of, id, block, pos, hint));
            }
            _ => {}
        }
    }
    Ok(facts)
}

fn resolve_delay(
    body: &Body,
    pos_of: &std::collections::HashMap<InstId, usize>,
    hint: InstId,
    block: BlockId,
    pos: usize,
    raw: crate::operand::DelayAluHint,
) -> DelayFact {
    let ambiguous = DelayFact::ambiguous(hint, block);
    let (start, end) = body.blocks[block.0].range;
    let _ = pos_of;
    let debug_assert_pos = body.layout.get(pos) == Some(&hint);
    debug_assert!(debug_assert_pos);

    // Consumers first (forward): consumer0 is the next instruction, consumer1
    // is skip_forward further. Either past the block end means a cross-block
    // reach.
    let fwd = match skip_forward(raw.instskip) {
        Some(fwd) => fwd,
        None => return ambiguous,
    };
    if pos + 1 >= end {
        return ambiguous;
    }
    let consumer0 = body.layout[pos + 1];
    if pos + 1 + fwd >= end || start >= end {
        return ambiguous;
    }
    let consumer1 = body.layout[pos + 1 + fwd];

    // Producers (backward): walk within the block only.
    let mut producers: [Option<InstId>; 2] = [None, None];
    for (slot, value) in [raw.instid0, raw.instid1].iter().enumerate() {
        match id_need(*value) {
            IdNeed::None => {}
            IdNeed::Unknown => return ambiguous,
            IdNeed::Salu => {
                // Nearest previous SALU-form instruction (M1 approximation
                // of the SALU cycle penalty).
                let mut found = None;
                let mut j = pos;
                while j > start {
                    j -= 1;
                    let candidate = body.layout[j];
                    if is_salu(&body.insts.get(candidate).expect("laid out")) {
                        found = Some(candidate);
                        break;
                    }
                }
                // No SALU in-block before the hint: the penalty has no
                // in-block producer; treat as no dependency rather than
                // guessing across the edge.
                producers[slot] = found;
            }
            IdNeed::Valu(need) => {
                let mut seen = 0usize;
                let mut found = None;
                let mut j = pos;
                while j > start {
                    j -= 1;
                    let candidate = body.layout[j];
                    if is_valu(&body.insts.get(candidate).expect("laid out")) {
                        seen += 1;
                        if seen == need {
                            found = Some(candidate);
                            break;
                        }
                    }
                }
                match found {
                    Some(inst) => producers[slot] = Some(inst),
                    None => return ambiguous,
                }
            }
        }
    }
    DelayFact {
        hint,
        block,
        status: DelayStatus::Resolved {
            producers,
            consumers: [Some(consumer0), Some(consumer1)],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Arena, Cond};
    use crate::effects::{Effects, MemEffect, OrderType, SrcRead};
    use crate::inst::{Form, Inst};
    use crate::isa;
    use crate::operand::{DelayAluHint, ImmField, Operand};
    use crate::passes::cfg::build_blocks;
    use crate::provenance::Provenance;
    use crate::wait::CounterSet;

    fn row_op(name: &str, form: Form) -> crate::inst::Opcode {
        isa::gfx12()
            .iter()
            .find(|row| row.name == name && row.form == form)
            .unwrap_or_else(|| panic!("missing table row {name} {form:?}"))
            .op
    }

    fn mk(name: &str, form: Form) -> Inst {
        Inst {
            op: row_op(name, form),
            form,
            fields: Default::default(),
            operands: Default::default(),
            mods: Default::default(),
            literal: None,
            effects: Effects::default(),
            prov: Provenance::default(),
        }
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut arena: Arena<Inst> = Arena::new();
        let mut layout = Vec::new();
        for inst in insts {
            layout.push(arena.insert(inst));
        }
        Body { insts: arena, blocks: Vec::new(), layout }
    }

    fn mem(name: &str, form: Form, class: MemClass) -> Inst {
        let mut inst = mk(name, form);
        inst.effects.mem = Some(MemEffect {
            class,
            counters: CounterSet(0),
            in_order_type: OrderType::Load,
            src_read: SrcRead::AtIssue,
        });
        inst
    }

    fn clause_opener(len: u8) -> Inst {
        let mut inst = mk("s_clause", Form::Sopp);
        inst.effects.control = Control::Clause;
        inst.mods.clause = Some(len);
        inst
    }

    fn delay_hint(instid0: u8, instskip: u8, instid1: u8) -> Inst {
        let mut inst = mk("s_delay_alu", Form::Sopp);
        inst.effects.control = Control::Delay;
        inst.mods.delay = Some(DelayAluHint { instid0, instskip, instid1 });
        inst
    }

    fn valu() -> Inst {
        mk("v_add_nc_u32_e32", Form::Vop2)
    }

    fn branch_to(off: i16) -> Inst {
        let mut inst = mk("s_cbranch_scc1", Form::Sopp);
        inst.effects.control = Control::Branch { cond: Cond::Scc1 };
        inst.operands.push(Operand::Imm(ImmField::Sopp(off)));
        inst
    }

    fn endpgm() -> Inst {
        let mut inst = mk("s_endpgm", Form::Sopp);
        inst.effects.control = Control::EndPgm;
        inst
    }

    #[test]
    fn delay_past_either_block_edge_is_ambiguous() {
        // Producer walk past block start: hint needs 2 VALU back, only 1 in block.
        let mut body = body_of(vec![valu(), delay_hint(2, 0, 0), valu(), endpgm()]);
        build_blocks(&mut body).unwrap();
        let facts = check_windows(&body).unwrap();
        assert!(facts.delays[0].is_ambiguous());
        assert_eq!(facts.delays[0].producers(), None);

        // Consumer past block end: a branch targets the instruction right
        // after the hint, so the hint is alone in its block and its consumer
        // lives in the next one.
        // Layout: hint@0 valu@1 valu@2 branch@3(taken valu@1, off -3) endpgm@4.
        let mut back = mk("s_branch", Form::Sopp);
        back.effects.control = Control::Jump;
        back.operands.push(Operand::Imm(ImmField::Sopp(-3)));
        let mut body = body_of(vec![delay_hint(0, 0, 0), valu(), valu(), back, endpgm()]);
        build_blocks(&mut body).unwrap();
        assert_eq!(body.blocks[0].range, (0, 1));
        let facts = check_windows(&body).unwrap();
        assert!(facts.delays[0].is_ambiguous());
    }
    #[test]
    fn smem_clause_of_mixed_widths_validates() {
        // KT48's s_load_b256 + s_load_b128 clause: mixed widths, one class.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b256", Form::Smem, MemClass::SmemLoad),
            mem("s_load_b128", Form::Smem, MemClass::SmemLoad),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.clauses.len(), 1);
        assert_eq!(facts.clauses[0].class, MemClass::SmemLoad);
        assert_eq!(facts.clauses[0].members.len(), 2);
    }

    #[test]
    fn clause_window_crossing_a_target_rejects() {
        // Branch targets the second clause member: leaders split the window.
        // Layout: clause(1)@0 mem@1 cbr@2(taken mem@1... forward: taken endpgm@4) mem@3 endpgm@4.
        // Simpler: clause@0, mem@1, branch@2 taken->endpgm@4, mem@3, endpgm@4
        // with the branch targeting mem@3? No: crossing needs a leader INSIDE
        // the window. Target mem@1 from a later branch instead:
        // clause@0 mem@1 endpgm... layout: [clause(1), mem, branch(taken mem@1), mem2, endpgm]:
        // branch@2 next=3, target mem@1: off = 1-3 = -2. Leaders: 0, target 1,
        // after-branch 3. Window members @1,@2: @2 is branch (non-mem)... that
        // rejects NonMemory, not CrossesLeader. Make the window longer:
        // clause(2)@0 mem@1 mem@2 branch@3(taken mem@2): SMEM is 2 dwords, so
        // pcs are 0,1,3,5,6 and off = 3-6 = -3. Leaders: 0, 2, 4.
        let mut body = body_of(vec![
            clause_opener(2),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            branch_to(-3),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::CrossesLeader { index: 0, len: 2 })
        );
    }

    #[test]
    fn clause_rejects_wait_nonmem_mixedclass_and_endpgm() {
        // Wait inside.
        let mut wait = mk("s_wait_dscnt", Form::Sopp);
        wait.effects.control = Control::Wait;
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            wait,
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::WaitInsideClause { index: 0, member: 2 })
        );
        // Non-memory member.
        let mut body = body_of(vec![clause_opener(0), valu(), endpgm()]);
        build_blocks(&mut body).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::NonMemoryMember { index: 0, member: 1 })
        );
        // Mixed classes.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            mem("global_load_b32", Form::Vmem(crate::inst::VmemForm::Global), MemClass::VmemLoad),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::MixedClass { index: 0, member: 2 })
        );
        // s_endpgm inside.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::EndPgmInsideClause { index: 0 })
        );
    }

    #[test]
    fn delay_resolves_producers_and_consumers_in_block() {
        // valu@0 valu@1 hint@2(VALU_DEP_1, SAME, VALU_DEP_1) consumer@3 endpgm@4.
        let mut body = body_of(vec![
            valu(),
            valu(),
            delay_hint(1, 0, 1),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.delays.len(), 1);
        let delay = &facts.delays[0];
        assert_eq!(
            delay.producers(),
            Some([Some(body.layout[1]), Some(body.layout[1])])
        );
        assert_eq!(
            delay.consumers(),
            Some([Some(body.layout[3]), Some(body.layout[3])])
        );
    }

    #[test]
    fn delay_skip_selects_the_second_consumer() {
        // hint@1 SKIP_1: consumer1 is two ahead (p+1+2 with skip value 2).
        let mut body = body_of(vec![
            valu(),
            delay_hint(1, 2, 0),
            valu(),
            valu(),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body).unwrap();
        let facts = check_windows(&body).unwrap();
        let delay = &facts.delays[0];
        assert_eq!(
            delay.consumers(),
            Some([Some(body.layout[2]), Some(body.layout[4])])
        );
        assert_eq!(delay.producers(), Some([Some(body.layout[0]), None]));
    }

    #[test]
    fn no_dep_hint_resolves_empty() {
        let mut body = body_of(vec![valu(), delay_hint(0, 0, 0), valu(), endpgm()]);
        build_blocks(&mut body).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.delays[0].producers(), Some([None, None]));
    }
}
