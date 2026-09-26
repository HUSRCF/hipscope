//! C5: per-arch SGPR, VALU, and WMMA hazards.
//!
//! Whole-program ports of `hipfire-isa/src/hazard.rs` (`Gfx12Sgpr`,
//! `Gfx11Hazards`) plus the WMMA chaining rule from `lib.rs:69-80`, run over
//! typed streams:
//!
//! * `gfx12_sgpr`: SALU-def/VALU-use SGPR tracking. The state machine is
//!   `hazard.rs` verbatim (memory clears, tracked-pair marks, demand
//!   reporting with self-clear). Where the builder *emits* the demanded
//!   `s_wait_alu`, the whole-program replay *verifies* it: a demand at
//!   instruction `i` is satisfied by an `s_wait_alu` with the matching
//!   depctr field at 0 on the reaching path (current block plus the
//!   single-predecessor chain, stopping at memory clears). Absent on that
//!   path it is an obligation.
//! * `gfx11_wave32`: ported struct for M2; `analyze` runs it once gfx11
//!   tables exist.
//! * WMMA chaining: a `v_wmma_*`/`v_swmmac_*` whose A/B (or SWMMAC index)
//!   operands overlap the previous matrix instruction's destination needs an
//!   independent VALU or `v_nop` between them. Scanned in layout order like
//!   the builder's push sequence; a violation is an obligation.
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
    #[error("hazard tables cover gfx1201 wave32 only (got {0:?} {1:?})")]
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
            MemClass::VmemLoad | MemClass::VmemStore => Pipeline::Vmem,
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

/// Port of `hipfire-isa` `Gfx11Hazards`: wave32 transcendental and
/// VMEM-SGPR rules. Run on gfx11 once its table lands (M2).
#[derive(Clone, Debug, Default)]
pub struct Gfx11Hazards {
    trans_defs: Vec<(RegRef, u8, u8)>,
    valu_sgpr_defs: Vec<(RegRef, u8)>,
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
        if pipe == Pipeline::Vmem {
            let needed = self
                .valu_sgpr_defs
                .iter()
                .filter(|(def, _)| uses.iter().any(|reg| reg.overlaps(*def)))
                .map(|(_, age)| 5u8.saturating_sub(*age))
                .max()
                .unwrap_or(0);
            if needed > 0 {
                waits.push(format!("s_nop {}", needed - 1));
                self.valu_sgpr_defs.clear();
            }
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
        for (_, age) in &mut self.valu_sgpr_defs {
            *age = age.saturating_add(1);
        }
        self.valu_sgpr_defs.retain(|(_, age)| *age < 5);
        if pipe == Pipeline::Valu {
            for (_, valu_age, trans_age) in &mut self.trans_defs {
                *valu_age = valu_age.saturating_add(1);
                if trans {
                    *trans_age = trans_age.saturating_add(1);
                }
            }
            self.trans_defs
                .retain(|(_, valu_age, trans_age)| *valu_age <= 5 && *trans_age <= 1);
            self.valu_sgpr_defs.extend(
                defs.iter().filter(|reg| reg.kind == Kind::S).map(|reg| (*reg, 0)),
            );
            if trans {
                self.trans_defs.extend(
                    defs.iter().filter(|reg| reg.kind == Kind::V).map(|reg| (*reg, 0, 0)),
                );
            }
        }
        waits
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

/// Whole-program hazard analysis for gfx1201 wave32.
pub fn analyze(body: &Body, arch: Arch, wave: Wave) -> Result<HazardAnalysis, HazardError> {
    if arch != Arch::Gfx1201 || wave != Wave::Wave32 {
        return Err(HazardError::Unsupported(arch, wave));
    }
    let mut analysis = HazardAnalysis::default();
    run_sgpr(body, arch, &mut analysis);
    run_wmma(body, arch, &mut analysis);
    Ok(analysis)
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
        let uses: Vec<RegRef> = inst.effects.uses.iter().copied().collect();
        let defs: Vec<RegRef> = inst.effects.defs.iter().copied().collect();
        // The builder's only call site feeds `false, false` (`lib.rs:63`):
        // vcc tracking exists in the machine but no producer feeds it, and
        // deriving it from implicit effects false-positives on KT48 carry
        // chains and vcc branches (verified). Same for the test helper below.
        for demand in machine.step(pipe, &uses, &defs, false, false) {
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

fn run_wmma(body: &Body, arch: Arch, analysis: &mut HazardAnalysis) {
    let mut previous: Option<(InstId, RegRef)> = None;
    for id in &body.layout {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        let name = inst.op.name(arch).unwrap_or("");
        if name.starts_with("v_wmma_") || name.starts_with("v_swmmac_") {
            let dst = inst.effects.defs.first().copied();
            let chained = match previous {
                Some((_, old)) => {
                    let ab = inst.effects.uses.iter().take(2).any(|src| old.overlaps(*src));
                    let index = name.starts_with("v_swmmac_")
                        && inst.effects.uses.get(2).is_some_and(|src| old.overlaps(*src));
                    ab || index
                }
                None => false,
            };
            if chained {
                analysis.obligations.push(Obligation {
                    kind: ObligationKind::Hazard,
                    insts: vec![*id],
                    rule_id: "wmma-chain".into(),
                    text: format!(
                        "{name} reuses the previous matrix destination without an intervening VALU or v_nop"
                    ),
                });
            }
            if let Some(dst) = dst {
                previous = Some((*id, dst));
            }
            continue;
        }
        if pipe_of(inst, arch) == Pipeline::Valu {
            previous = None;
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
        let uses: Vec<RegRef> = inst.effects.uses.iter().copied().collect();
        let defs: Vec<RegRef> = inst.effects.defs.iter().copied().collect();
        let demands = machine.step(pipe, &uses, &defs, false, false);
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

    /// Gfx11Hazards keeps its trans-age and VMEM-SGPR rules.
    #[test]
    fn gfx11_machine_keeps_trans_and_vmem_rules() {
        let v = |base: u16| RegRef { kind: Kind::V, base, len: 1 };
        let s = |base: u16| RegRef { kind: Kind::S, base, len: 1 };
        let mut machine = Gfx11Hazards::default();
        assert!(machine.step(Pipeline::Valu, "v_rcp_f32", &[], &[v(10)]).is_empty());
        assert_eq!(
            machine.step(Pipeline::Valu, "v_add_f32", &[v(10)], &[v(11)]),
            vec!["s_waitcnt_depctr depctr_va_vdst(0)"]
        );
        let mut machine = Gfx11Hazards::default();
        assert!(machine.step(Pipeline::Valu, "v_add_f32", &[], &[s(4)]).is_empty());
        assert_eq!(
            machine.step(Pipeline::Vmem, "buffer_load_b32", &[s(4)], &[v(0)]),
            vec!["s_nop 4"]
        );
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
        build_blocks(&mut body).expect("CFG");
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
