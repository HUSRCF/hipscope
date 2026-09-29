//! C5: per-arch SGPR, VALU, and WMMA hazards.
//!
//! Whole-program hazard replay over typed streams. The gfx12 SGPR tracker
//! ports `hipfire-isa/src/hazard.rs`; gfx11 rules are grounded in ROCm LLVM
//! `GCNHazardRecognizer.cpp` / `AMDGPU.td` at compiler commit
//! `8f497e0992fb7513f7f78a6f6b6f1056c375e961`:
//! * `gfx12_sgpr`: SALU-def/VALU-use SGPR tracking. The state machine is
//!   `hazard.rs` verbatim (memory clears, tracked-pair marks, demand
//!   reporting with self-clear). Where the builder *emits* the demanded
//!   `s_wait_alu`, the whole-program replay *verifies* it: a demand at
//!   instruction `i` is satisfied by an `s_wait_alu` with the matching
//!   depctr field at 0 on the reaching path (current block plus the
//!   single-predecessor chain, stopping at memory clears). Absent on that
//!   path it is an obligation.
//! * `gfx11_wave32`: gfx1100 TRANS→VALU forwarding (`s_waitcnt_depctr
//!   va_vdst(0)`), both gfx11 targets' VCMPX→PERMLANE interlock, and WMMA
//!   chaining, using LLVM `AMDGPU.td` target feature sets and
//!   `GCNHazardRecognizer.cpp` (`fixVALUTransUseHazard`,
//!   `fixVcmpxPermlaneHazards`, `fixWMMAHazards`). The gfx11 feature sets
//!   include `NoDataDepHazard`: the older VMEM/VALU-SGPR NOP rule does not
//!   apply, and gfx1151 does not have `VALUTransUseHazard`.
//! * WMMA chaining: a `v_wmma_*`/`v_swmmac_*` whose A/B (or SWMMAC index)
//!   operands overlap the previous matrix instruction's destination needs an
//!   independent VALU or `v_nop` between them. Replay joins all predecessor
//!   states at CFG boundaries, including loop backedges.
//!
//! Depctr field positions are pinned by `llvm-mc -mcpu=gfx1201`: `sa_sdst`
//! is bit 0, `va_vcc` is bit 1, `va_sdst` is bits [11:9] of the `s_wait_alu`
//! simm16.

use crate::cfg::{Body, InstId};
use thiserror::Error;
use crate::effects::{Control, MemClass};
use crate::inst::{Arch, Inst, Wave};
use crate::operand::{ImmField, Operand};
use crate::reg::{Kind, RegRef};
use crate::state::{Obligation, ObligationKind};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HazardError {
    #[error("hazard tables cover gfx1100/gfx1151/gfx1201 wave32 only (got {0:?} {1:?})")]
    Unsupported(Arch, Wave),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipeline {
    Salu,
    Valu,
    Vmem,
    Smem,
    Ds,
}

fn pipe_of(inst: &Inst, arch: Arch) -> Pipeline {
    match &inst.effects.mem {
        Some(mem) => match mem.class {
            MemClass::VmemLoad | MemClass::VmemStore | MemClass::VmemAtomic { .. }
            | MemClass::FlatLoad | MemClass::FlatStore | MemClass::FlatAtomic { .. } => Pipeline::Vmem,
            MemClass::SmemLoad => Pipeline::Smem,
            _ => pipe_by_name(inst, arch),
        },
        None => pipe_by_name(inst, arch),
    }
}

fn pipe_by_name(inst: &Inst, arch: Arch) -> Pipeline {
    match inst.op.name(arch).unwrap_or("") {
        name if name.starts_with('s') => Pipeline::Salu,
        name if name.starts_with('v') => Pipeline::Valu,
        _ => Pipeline::Ds,
    }
}

/// Port of `hipfire-isa` `Gfx12Sgpr`: tracks SGPR pairs read by VALU and
/// written by SALU/VALU, demanding `depctr_*_sdst(0)` / `depctr_va_vcc(0)`
/// before a cross-pipe reread. Clears on vector/scalar memory.
#[derive(Clone, Debug)]
pub struct Gfx12Sgpr {
    tracked: [bool; 64],
    salu: [bool; 128],
    valu: [bool; 128],
    vcc_salu: bool,
    vcc_valu: bool,
}

impl Default for Gfx12Sgpr {
    fn default() -> Self {
        Self {
            tracked: [false; 64],
            salu: [false; 128],
            valu: [false; 128],
            vcc_salu: false,
            vcc_valu: false,
        }
    }
}

impl Gfx12Sgpr {
    pub fn clear_on_memory(&mut self) {
        self.salu.fill(false);
        self.valu.fill(false);
        self.vcc_salu = false;
        self.vcc_valu = false;
    }

    pub fn step(
        &mut self,
        pipe: Pipeline,
        uses: &[RegRef],
        defs: &[RegRef],
        vcc_use: bool,
        vcc_def: bool,
    ) -> Vec<&'static str> {
        if matches!(pipe, Pipeline::Vmem | Pipeline::Smem) {
            self.clear_on_memory();
            return vec![];
        }
        if !matches!(pipe, Pipeline::Salu | Pipeline::Valu) {
            return vec![];
        }
        let mut sa = false;
        let mut va = false;
        let mut vcc = false;
        let mut salu_read_tracked = false;
        for reg in uses {
            if reg.kind != Kind::S {
                continue;
            }
            for n in reg.base as usize..reg.base as usize + reg.len as usize {
                if self.tracked[n / 2] {
                    sa |= self.salu[n];
                    if pipe == Pipeline::Valu {
                        va |= self.valu[n];
                    } else {
                        salu_read_tracked = true;
                    }
                }
            }
        }
        if vcc_use {
            sa |= self.vcc_salu;
            if pipe == Pipeline::Valu {
                vcc |= self.vcc_valu;
            } else {
                self.vcc_valu = false;
            }
        }
        let mut waits = Vec::new();
        if sa {
            waits.push("depctr_sa_sdst(0)");
            self.salu.fill(false);
            self.vcc_salu = false;
        }
        if va {
            waits.push("depctr_va_sdst(0)");
            self.valu.fill(false);
        }
        if vcc {
            waits.push("depctr_va_vcc(0)");
            self.vcc_valu = false;
        }
        if salu_read_tracked {
            self.valu.fill(false);
        }
        for reg in uses {
            if reg.kind == Kind::S && pipe == Pipeline::Valu {
                for n in reg.base as usize..reg.base as usize + reg.len as usize {
                    self.tracked[n / 2] = true;
                }
            }
        }
        for reg in defs {
            if reg.kind == Kind::S {
                for n in reg.base as usize..reg.base as usize + reg.len as usize {
                    if self.tracked[n / 2] {
                        if pipe == Pipeline::Salu {
                            self.salu[n] = true;
                        } else {
                            self.valu[n] = true;
                        }
                    }
                }
            }
        }
        if vcc_def {
            self.vcc_salu = pipe == Pipeline::Salu;
            self.vcc_valu = pipe == Pipeline::Valu;
        }
        waits
    }
}

/// gfx1100 wave32 TRANS-use hazard. Primary evidence: LLVM release/22.x
/// `llvm/lib/Target/AMDGPU/AMDGPU.td` FeatureISAVersion11_0_Common
/// enables VALUTransUseHazard, unlike FeatureISAVersion11_5_Common.
/// `GCNHazardRecognizer.cpp::fixVALUTransUseHazard` tracks five intervening
/// VALUs / one TRANS and expires on VMEM, DS, EXP or depctr va_vdst(0).
/// Neither gfx11 target needs the older VALU-SGPR→VMEM five-NOP rule:
/// FeatureISAVersion11_Common enables NoDataDepHazard, which bypasses
/// `checkVMEMHazards` in GCNHazardRecognizer::PreEmitNoopsCommon.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Gfx11Hazards {
    trans_defs: Vec<(RegRef, u8, u8)>,
}
impl Gfx11Hazards {
    const TRANS: [&'static str; 7] =
        ["v_rcp_", "v_sqrt_", "v_rsq_", "v_sin_", "v_cos_", "v_exp_", "v_log_"];

    pub fn step(
        &mut self,
        pipe: Pipeline,
        mnemonic: &str,
        uses: &[RegRef],
        defs: &[RegRef],
    ) -> Vec<String> {
        let mut waits = Vec::new();
        let trans = pipe == Pipeline::Valu
            && Self::TRANS.iter().any(|prefix| mnemonic.starts_with(prefix));
        if matches!(pipe, Pipeline::Vmem | Pipeline::Ds) {
            self.trans_defs.clear();
        }
        if pipe == Pipeline::Valu
            && self.trans_defs.iter().any(|(def, valu_age, trans_age)| {
                *valu_age <= 5
                    && *trans_age <= 1
                    && uses.iter().any(|reg| reg.overlaps(*def))
            })
        {
            waits.push("s_waitcnt_depctr depctr_va_vdst(0)".into());
            self.trans_defs.clear();
        }
        if pipe == Pipeline::Valu {
            for (_, valu_age, trans_age) in &mut self.trans_defs {
                *valu_age = valu_age.saturating_add(1);
                if trans {
                    *trans_age = trans_age.saturating_add(1);
                }
            }
            self.trans_defs
                .retain(|(_, valu_age, trans_age)| *valu_age <= 5 && *trans_age <= 1);
            if trans {
                self.trans_defs.extend(
                    defs.iter().filter(|reg| reg.kind == Kind::V).map(|reg| (*reg, 0, 0)),
                );
            }
        }
        waits
    }

    /// A decoded depctr va_vdst(0) drains a preceding TRANS.
    pub fn wait_va_vdst(&mut self) { self.trans_defs.clear(); }

    /// At a CFG join any reaching, unresolved TRANS must remain tracked.
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for &(reg, va, trans) in &other.trans_defs {
            if let Some((_, old_va, old_trans)) = self.trans_defs.iter_mut().find(|(r, _, _)| *r == reg) {
                if va < *old_va { *old_va = va; changed = true; }
                if trans < *old_trans { *old_trans = trans; changed = true; }
            } else {
                self.trans_defs.push((reg, va, trans));
                changed = true;
            }
        }
        if changed { self.trans_defs.sort_unstable_by_key(|(reg, va, trans)| (*reg, *va, *trans)); }
        changed
    }
}

/// Demanded wait satisfied by an `s_wait_alu` in the layout stream?
/// Searches backward from `index` to the nearest preceding vector/scalar
/// memory instruction (exclusive). The SGPR tracker itself runs linearly
/// over layout and resets at vector/scalar memory, so both the tracked
/// definitions behind the demand and any wait satisfying it lie after that
/// reset; producers (the builder at push time, LLVM's hazard pass) emit the
/// wait adjacently on the using path. A wait older than the reset cannot
/// satisfy post-reset tracking, and anything past it is out of scope.
fn wait_present(body: &Body, arch: Arch, index: usize, demand: &str) -> bool {
    let matches = |inst: &Inst| {
        inst.op.name(arch) == Some("s_wait_alu")
            && inst.operands.iter().any(|operand| {
                let bits = match operand {
                    Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => *n as u16,
                    _ => return false,
                };
                match demand {
                    "depctr_sa_sdst(0)" => bits & 1 == 0,
                    "depctr_va_vcc(0)" => bits >> 1 & 1 == 0,
                    "depctr_va_sdst(0)" => bits >> 9 & 7 == 0,
                    _ => false,
                }
            })
    };
    for back in (0..index).rev() {
        let id = body.layout[back];
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        if matches(inst) {
            return true;
        }
        if matches!(pipe_of(inst, arch), Pipeline::Vmem | Pipeline::Smem) {
            return false;
        }
    }
    false
}

/// One demanded-but-absent `s_wait_alu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingWait {
    pub inst: InstId,
    pub wait: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HazardAnalysis {
    pub missing: Vec<MissingWait>,
    pub obligations: Vec<Obligation>,
}

/// Whole-program hazard replay for the supported ISA tables. RDNA3 gfx11
/// admits wave32 and wave64; gfx1201's validated SGPR model is wave32 only.
pub fn analyze(body: &Body, arch: Arch, wave: Wave) -> Result<HazardAnalysis, HazardError> {
    if arch == Arch::Gfx1201 && wave != Wave::Wave32 {
        return Err(HazardError::Unsupported(arch, wave));
    }
    let mut analysis = HazardAnalysis::default();
    match arch {
        Arch::Gfx1201 => run_sgpr(body, arch, &mut analysis),
        Arch::Gfx1100 | Arch::Gfx1151 => run_gfx11(body, arch, &mut analysis),
        _ => return Err(HazardError::Unsupported(arch, wave)),
    }
    run_wmma(body, arch, &mut analysis);
    Ok(analysis)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Gfx11State {
    trans: Gfx11Hazards,
    cmpx_pending: bool,
}

impl Gfx11State {
    fn join(&mut self, other: &Self) -> bool {
        let trans_changed = self.trans.join(&other.trans);
        let cmpx_changed = !self.cmpx_pending && other.cmpx_pending;
        self.cmpx_pending |= other.cmpx_pending;
        trans_changed || cmpx_changed
    }
}

fn gfx11_block(body: &Body, arch: Arch, block: &crate::cfg::Block,
    state: &mut Gfx11State, mut analysis: Option<&mut HazardAnalysis>) {
    for &id in &body.layout[block.range.0..block.range.1] {
        let Some(inst) = body.insts.get(id) else { continue };
        if matches!(inst.effects.control, Control::EndPgm) {
            *state = Gfx11State::default();
            continue;
        }
        let name = inst.op.name(arch).unwrap_or("");
        // ROCm LLVM GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards and
        // AMDGPU.td FeatureISAVersion11_Common enable this on both targets.
        // fixVcmpxPermlaneHazards: V_NOP does NOT break the chain; a real
        // intervening VALU (e.g. v_mov_b32) does.
        let perm_lane = name == "v_permlane16_b32" || name == "v_permlanex16_b32";
        if perm_lane && state.cmpx_pending {
            if let Some(analysis) = &mut analysis {
                analysis.missing.push(MissingWait { inst: id, wait: "v_mov_b32 <live vgpr>, <same vgpr>".into() });
                analysis.obligations.push(Obligation {
                    kind: ObligationKind::Hazard, insts: vec![id],
                    rule_id: "gfx11-vcmpx-permlane".into(),
                    text: "VCMPX→PERMLANE needs a real intervening VALU (LLVM GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards)".into(),
                });
            }
        }
        if pipe_of(inst, arch) == Pipeline::Valu && name != "v_nop_e32" && name != "v_nop_e64" {
            state.cmpx_pending = name.starts_with("v_cmpx_");
        }
        // gfx11 depctr encoding: va_vdst is SIMM16[15:12]; zero drains
        // preceding TRANS (LLVM AMDGPU gfx11_waitcnt_depctr.rst).
        if name == "s_waitcnt_depctr" && inst.operands.iter().any(|op| {
            matches!(op, Operand::Imm(ImmField::Sopp(n)) if (*n as u16 >> 12) & 15 == 0)
        }) {
            state.trans.wait_va_vdst();
        }
        if arch == Arch::Gfx1100 {
            for demand in state.trans.step(
                pipe_of(inst, arch), name, &inst.effects.uses, &inst.effects.defs,
            ) {
                if let Some(analysis) = &mut analysis {
                    analysis.missing.push(MissingWait { inst: id, wait: demand.clone() });
                    analysis.obligations.push(Obligation {
                        kind: ObligationKind::Hazard, insts: vec![id],
                        rule_id: "gfx1100-trans-valu".into(),
                        text: format!("{name} needs {demand} (LLVM GCNHazardRecognizer.cpp fixVALUTransUseHazard)"),
                    });
                }
            }
        }
    }
}

fn run_gfx11(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    // LLVM GCNHazardRecognizer::hasHazard walks predecessor blocks. Join all
    // reaching unresolved facts and revisit loops until their state settles;
    // replay once more from the fixed-point inputs to emit each site once.
    let mut inputs = vec![None; body.blocks.len()];
    if inputs.is_empty() { return; }
    inputs[0] = Some(Gfx11State::default());
    let mut pending = std::collections::BTreeSet::from([0]);
    while let Some(index) = pending.pop_first() {
        let Some(mut state) = inputs[index].clone() else { continue };
        let block = &body.blocks[index];
        gfx11_block(body, arch, block, &mut state, None);
        for successor in &block.succs {
            let next = &mut inputs[successor.0];
            let changed = match next {
                Some(existing) => existing.join(&state),
                None => { *next = Some(state.clone()); true },
            };
            if changed { pending.insert(successor.0); }
        }
    }
    for (block, input) in body.blocks.iter().zip(inputs) {
        if let Some(mut state) = input {
            gfx11_block(body, arch, block, &mut state, Some(analysis));
        }
    }
}

fn run_sgpr(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    let mut machine = Gfx12Sgpr::default();
    for (pos, id) in body.layout.iter().enumerate() {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        if matches!(inst.effects.control, Control::EndPgm) {
            machine = Gfx12Sgpr::default();
            continue;
        }
        let pipe = pipe_of(inst, arch);
        // The builder's only call site feeds `false, false` (`lib.rs:63`):
        // vcc tracking exists in the machine but no producer feeds it, and
        // deriving it from implicit effects false-positives on KT48 carry
        // chains and vcc branches (verified). Same for the test helper below.
        for demand in machine.step(
            pipe, &inst.effects.uses, &inst.effects.defs, false, false,
        ) {
            if !wait_present(body, arch, pos, demand) {
                analysis.missing.push(MissingWait { inst: *id, wait: demand.into() });
                analysis.obligations.push(Obligation {
                    kind: ObligationKind::Hazard,
                    insts: vec![*id],
                    rule_id: format!("sgpr-{}", demand.replace(['(', ')'], "")),
                    text: format!("SGPR read needs {demand} on the reaching path"),
                });
            }
        }
    }
}

/// ROCm LLVM GCNHazardRecognizer.cpp::fixWMMAHazards searches across
/// predecessor edges, retaining every reaching prior WMMA until an independent
/// VALU (including V_NOP) breaks the chain.
fn wmma_block(body: &Body, arch: Arch, block: &crate::cfg::Block,
    previous: &mut Vec<RegRef>, mut analysis: Option<&mut HazardAnalysis>) {
    for &id in &body.layout[block.range.0..block.range.1] {
        let Some(inst) = body.insts.get(id) else { continue };
        let name = inst.op.name(arch).unwrap_or("");
        if name.starts_with("v_wmma_") || name.starts_with("v_swmmac_") {
            let chained = previous.iter().any(|old| {
                inst.effects.uses.iter().take(2).any(|src| old.overlaps(*src))
                    || (arch == Arch::Gfx1201 && name.starts_with("v_swmmac_")
                        && inst.effects.uses.get(2).is_some_and(|src| old.overlaps(*src)))
            });
            if chained {
                if let Some(analysis) = &mut analysis {
                    analysis.obligations.push(Obligation {
                        kind: ObligationKind::Hazard, insts: vec![id],
                        rule_id: "wmma-chain".into(),
                        text: format!("{name} reuses a reaching matrix destination without an intervening VALU or v_nop"),
                    });
                }
            }
            previous.clear();
            if let Some(dst) = inst.effects.defs.first() { previous.push(*dst); }
        } else if pipe_of(inst, arch) == Pipeline::Valu {
            previous.clear();
        }
    }
}

fn run_wmma(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    let mut inputs: Vec<Option<Vec<RegRef>>> = vec![None; body.blocks.len()];
    if inputs.is_empty() { return; }
    inputs[0] = Some(Vec::new());
    let mut pending = std::collections::BTreeSet::from([0]);
    while let Some(index) = pending.pop_first() {
        let Some(mut state) = inputs[index].clone() else { continue };
        let block = &body.blocks[index];
        wmma_block(body, arch, block, &mut state, None);
        for successor in &block.succs {
            let next = &mut inputs[successor.0];
            let changed = match next {
                Some(existing) => {
                    let old_len = existing.len();
                    for &reg in &state {
                        if !existing.contains(&reg) { existing.push(reg); }
                    }
                    existing.len() != old_len
                }
                None => { *next = Some(state.clone()); true },
            };
            if changed { pending.insert(successor.0); }
        }
    }
    for (block, input) in body.blocks.iter().zip(inputs) {
        if let Some(mut state) = input {
            wmma_block(body, arch, block, &mut state, Some(analysis));
        }
    }
}

/// Block-local SGPR replay used by unit tests: returns demands in order.
#[cfg(test)]
pub fn sgpr_demands(body: &Body, arch: Arch) -> Vec<(InstId, Vec<&'static str>)> {
    let mut machine = Gfx12Sgpr::default();
    let mut out = Vec::new();
    for id in &body.layout {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        let pipe = pipe_of(inst, arch);
        let demands = machine.step(
            pipe, &inst.effects.uses, &inst.effects.defs, false, false,
        );
        if !demands.is_empty() {
            out.push((*id, demands));
        }
    }
    out
}

/// Ported rule self-checks plus WMMA chaining on hand-built streams.
#[cfg(test)]
mod tests {
    use super::*;

    /// Gfx12Sgpr tracks the builder's exact demand sequence.
    #[test]
    fn sgpr_machine_demands_depctr_on_cross_pipe_reread() {
        let mut machine = Gfx12Sgpr::default();
        let s = |base: u16, len: u8| RegRef { kind: Kind::S, base, len };
        // VALU reads s4 (track pair 2), SALU writes s4, VALU rereads.
        assert!(machine.step(Pipeline::Valu, &[s(4, 1)], &[], false, false).is_empty());
        assert!(machine.step(Pipeline::Salu, &[], &[s(4, 1)], false, false).is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, &[s(4, 1)], &[], false, false),
            vec!["depctr_sa_sdst(0)"]
        );
        // VALU-written SGPR reread by VALU needs va_sdst.
        let mut machine = Gfx12Sgpr::default();
        assert!(machine.step(Pipeline::Valu, &[s(8, 1)], &[], false, false).is_empty());
        assert!(machine.step(Pipeline::Valu, &[], &[s(8, 1)], false, false).is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, &[s(8, 1)], &[], false, false),
            vec!["depctr_va_sdst(0)"]
        );
        // Vector memory clears the tracker.
        let mut machine = Gfx12Sgpr::default();
        assert!(machine.step(Pipeline::Valu, &[s(4, 1)], &[], false, false).is_empty());
        assert!(machine.step(Pipeline::Salu, &[], &[s(4, 1)], false, false).is_empty());
        assert!(machine.step(Pipeline::Vmem, &[], &[], false, false).is_empty());
        assert!(machine.step(Pipeline::Valu, &[s(4, 1)], &[], false, false).is_empty());
    }

    /// LLVM AMDGPU.td gfx11 NoDataDepHazard excludes VALU-SGPR→VMEM NOPs;
    /// gfx1100's separately gated TRANS-use rule needs depctr va_vdst(0).
    #[test]
    fn gfx11_trans_forwarding_but_no_vmem_sgpr_nop() {
        let v = |base: u16| RegRef { kind: Kind::V, base, len: 1 };
        let s = |base: u16| RegRef { kind: Kind::S, base, len: 1 };
        let mut machine = Gfx11Hazards::default();
        assert!(machine.step(Pipeline::Valu, "v_rcp_f32", &[], &[v(10)]).is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, "v_add_f32", &[v(10)], &[v(11)]),
            vec!["s_waitcnt_depctr depctr_va_vdst(0)"]
        );
        machine.wait_va_vdst();
        assert!(machine.step(Pipeline::Valu, "v_add_f32", &[v(10)], &[v(11)]).is_empty());
        assert!(machine.step(Pipeline::Valu, "v_add_f32", &[], &[s(4)]).is_empty());
        assert!(machine.step(Pipeline::Vmem, "buffer_load_b32", &[s(4)], &[v(0)]).is_empty());
    }

    /// A TRANS at the loop tail reaches a VALU use at the head only through
    /// the backedge; a linear layout scan misses this obligation.
    #[test]
    fn gfx1100_trans_use_at_loop_head_tracks_backedge() {
        use smallvec::SmallVec;
        use crate::cfg::Body;
        use crate::inst::FormFields;
        use crate::operand::Modifiers;
        use crate::provenance::Provenance;
        let arch = Arch::Gfx1100;
        let v = |base| Operand::Reg(RegRef { kind: Kind::V, base, len: 1 });
        let inst = |name: &str, operands: Vec<Operand>| {
            let row = crate::isa::table(arch).iter().find(|row| row.name == name).expect(name);
            Inst::from_parts(
                arch, row.op, row.form, FormFields::None,
                SmallVec::from_vec(operands), Modifiers::default(),
                None, Provenance::default(),
            ).unwrap_or_else(|error| panic!("{name}: {error:?}"))
        };
        let mut body = Body::default();
        for instruction in [
            inst("v_add_f32_e32", vec![v(12), v(10), v(13)]),
            inst("v_rcp_f32_e32", vec![v(10), v(11)]),
            inst("s_cbranch_scc1", vec![Operand::Imm(ImmField::Sopp(-3))]),
            inst("s_endpgm", vec![]),
        ] {
            let id = body.insts.insert(instruction);
            body.layout.push(id);
        }
        crate::passes::cfg::build_blocks(&mut body, arch).unwrap();
        assert!(body.blocks[0].succs.contains(&crate::cfg::BlockId(0)));
        let found = analyze(&body, arch, Wave::Wave32).unwrap();
        assert!(found.obligations.iter().any(|obligation|
            obligation.rule_id == "gfx1100-trans-valu"
                && obligation.insts == vec![body.layout[0]]), "{found:?}");
        assert!(!analyze(&body, Arch::Gfx1151, Wave::Wave32).unwrap()
            .obligations.iter().any(|obligation| obligation.rule_id == "gfx1100-trans-valu"));
    }

    /// Non-runtime LLVM-MC conformance probe: gfx11 VCMPX→V_NOP→PERMLANE16
    /// still needs a real VALU, while VCMPX→V_MOV→PERMLANEX16 is safe.
    /// These are assembled machine words, not a claim about runtime JIT sites.
    #[test]
    fn gfx11_cmpx_permlane_probe_in_both_wave_sizes() {
        use crate::cfg::Body;
        let words = [
            0x7d28_0300, // v_cmpx_gt_f32_e32 v0, v1
            0x7e00_0000, // v_nop: LLVM does not count this as a separator
            0xd65b_0002, 0x0004_0101, // v_permlane16_b32 v2, v1, s0, s1
            0x7d28_0300, // another VCMPX
            0x7e00_0300, // v_mov_b32_e32 v0, v0: real VALU separator
            0xd65c_0002, 0x0004_0101, // v_permlanex16_b32 v2, v1, s0, s1
            0xbfb0_0000, // s_endpgm
        ];
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let mut body = Body::default();
            let mut offset = 0;
            while offset < words.len() {
                let (inst, width) = crate::codec::gfx12::decode_for(arch, &words[offset..]).unwrap();
                assert_eq!(crate::codec::gfx12::encode_for(arch, &inst).unwrap().as_slice(), &words[offset..offset + width]);
                body.layout.push(body.insts.insert(inst));
                offset += width;
            }
            crate::passes::cfg::build_blocks(&mut body, arch).unwrap();
            for wave in [Wave::Wave32, Wave::Wave64] {
                let analysis = analyze(&body, arch, wave).unwrap();
                let sites = analysis.obligations.iter().filter(|o| o.rule_id == "gfx11-vcmpx-permlane")
                    .map(|o| o.insts.as_slice()).collect::<Vec<_>>();
                assert_eq!(sites, vec![&body.layout[2..3]], "{arch:?}/{wave:?}: {analysis:?}");
            }
        }
    }
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::{Arch, Wave};
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

    /// KT48 carries no unsatisfied SGPR/WMMA hazard: hipcc emitted every
    /// depctr wait the machine demands, and no WMMA chain is unguarded.
    #[test]
    fn kt48_hazards_are_clean() {
        let body = kt48_body();
        let analysis = super::analyze(&body, Arch::Gfx1201, Wave::Wave32).unwrap();
        assert!(analysis.missing.is_empty(), "{:?}", analysis.missing);
        assert!(analysis.obligations.is_empty(), "{:?}", analysis.obligations);
    }
}
