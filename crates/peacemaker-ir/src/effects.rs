use smallvec::SmallVec;
use crate::{cfg::{BarrierKind, Cond}, inst::{Arch, Form, Opcode, ValidateError}, lds::LdsAccess, operand::{Operand, Special}, reg::RegRef, wait::{Counter, CounterSet}};
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImplicitSet { pub reads: u8, pub writes: u8 }
impl ImplicitSet {
    pub const SCC: u8 = 1 << 0;
    pub const VCC: u8 = 1 << 1;
    pub const EXEC: u8 = 1 << 2;
    pub const M0: u8 = 1 << 3;
    pub const MODE: u8 = 1 << 4;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemClass { VmemLoad, VmemStore, VmemAtomic { returns: bool }, DsLoad, DsStore, DsAtomic { returns: bool }, SmemLoad, SmemStore, Export, LdsDma, FlatLoad, FlatStore, FlatAtomic { returns: bool } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderType { Load, Store, Ds, Smem, Sample, Bvh, Exp, Mixed }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrcRead { AtIssue, Deferred }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemEffect { pub class: MemClass, pub counters: CounterSet, pub in_order_type: OrderType, pub src_read: SrcRead }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Control { #[default] None, Branch { cond: Cond }, Jump, EndPgm, Barrier(BarrierKind), Wait, Clause, Delay, Halt, Trap }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Effects { pub defs: SmallVec<[RegRef; 2]>, pub uses: SmallVec<[RegRef; 4]>, pub implicit: ImplicitSet, pub mem: Option<MemEffect>, pub control: Control, pub lds: Option<LdsAccess> }

impl Effects {
    /// Single table source for authoring and lifted instruction effects.
    /// Physical register widths come from decoded operands; implicit reads and
    /// writes, counter membership and opcode-specific accumulator uses come
    /// from the same gfx12 row.
    pub fn from_table(arch: Arch, op: Opcode, form: Form, operands: &[Operand]) -> Result<Self, ValidateError> {
        let row = crate::isa::lookup(arch, op, form)
            .ok_or(ValidateError::UnknownOpcode { op, form })?;
        let mut out = Self::default();
        let mut field_names = row.grammar.split(',').map(|s| s.split(':').next().unwrap_or(""));
        // CMPX encodes the fixed EXEC destination as VDST, but it is not an operand.
        if form == Form::Vop3 && row.name.starts_with("v_cmpx_") {
            field_names.next();
        }
        // VOPC's explicit printed `vcc_lo` destination is fieldless in XML.
        let prefix = usize::from(form == Form::Vopc && row.name.starts_with("v_cmp_")
            && matches!(operands.first(), Some(Operand::Special(Special::Vcc | Special::VccLo))));
        for operand in operands.iter().skip(prefix) {
            let name = field_names.next().unwrap_or("");
            let def = row.defs.split(',').any(|s| s == name);
            let used = row.uses.split(',').any(|s| s == name);
            match operand {
                Operand::Reg(reg) | Operand::Half(reg, _) => {
                    if def { out.defs.push(*reg); }
                    if used { out.uses.push(*reg); }
                }
                Operand::Special(special) => {
                    let bit = match special {
                        Special::Scc => ImplicitSet::SCC,
                        Special::Vcc | Special::VccLo | Special::VccHi => ImplicitSet::VCC,
                        Special::Exec | Special::ExecLo | Special::ExecHi => ImplicitSet::EXEC,
                        Special::M0 => ImplicitSet::M0,
                        _ => 0,
                    };
                    if def { out.implicit.writes |= bit; }
                    if used { out.implicit.reads |= bit; }
                }
                _ => {}
            }
        }
        for effect in row.implicit.split(',') {
            let (reg, mode) = effect.split_once(':').unwrap_or(("", ""));
            let bit = match reg { "scc" => ImplicitSet::SCC, "vcc" => ImplicitSet::VCC, "exec" => ImplicitSet::EXEC, "m0" => ImplicitSet::M0, "mode" => ImplicitSet::MODE, _ => continue };
            if mode.contains('r') { out.implicit.reads |= bit; }
            if mode.contains('w') { out.implicit.writes |= bit; }
        }
        let name = row.name;
        out.control = if name == "s_endpgm" { Control::EndPgm }
            else if name == "s_branch" { Control::Jump }
            else if name.starts_with("s_cbranch") {
                let cond = if name.ends_with("scc0") { Cond::Scc0 }
                    else if name.ends_with("scc1") { Cond::Scc1 }
                    else if name.ends_with("vccz") { Cond::Vccz }
                    else if name.ends_with("vccnz") { Cond::Vccnz }
                    else if name.ends_with("execnz") { Cond::Execnz } else { Cond::Execz };
                Control::Branch { cond }
            } else if name == "s_barrier_signal" { Control::Barrier(BarrierKind::Signal(crate::cfg::BarrierId(0))) }
            else if name == "s_barrier_wait" { Control::Barrier(BarrierKind::Wait) }
            else if name.starts_with("s_wait") { Control::Wait }
            else if name == "s_clause" { Control::Clause }
            else if name == "s_delay_alu" { Control::Delay }
            else { Control::None };
        if row.counter != "-" {
            let (counter, units) = row.counter.split_once(':').ok_or_else(|| ValidateError::Operand("malformed counter rule".into()))?;
            let counter = match counter {
                "Km" => Counter::Km, "Ds" => Counter::Ds, "Load" => Counter::Load,
                "Store" => Counter::Store, "Sample" => Counter::Sample, "Bvh" => Counter::Bvh,
                "Exp" => Counter::Exp, _ => return Err(ValidateError::Operand("unknown counter rule".into())),
            };
            if !matches!(units, "1" | "2") { return Err(ValidateError::Operand("unsupported counter unit weight".into())); }
            let mut counters = CounterSet::default();
            counters.insert(counter);
            let (class, order) = if name.starts_with("s_load") { (MemClass::SmemLoad, OrderType::Smem) }
                else if name.starts_with("ds_load") { (MemClass::DsLoad, OrderType::Ds) }
                else if name.starts_with("ds_store") { (MemClass::DsStore, OrderType::Ds) }
                else if name.starts_with("ds_") { (MemClass::DsAtomic { returns: !out.defs.is_empty() }, OrderType::Ds) }
                else if name.starts_with("global_load") || name.starts_with("buffer_load") { (MemClass::VmemLoad, OrderType::Load) }
                else if name.starts_with("global_store") || name.starts_with("buffer_store") { (MemClass::VmemStore, OrderType::Store) }
                else { return Err(ValidateError::Operand(format!("unclassified memory rule for {name}"))); };
            out.mem = Some(MemEffect { class, counters, in_order_type: order,
                src_read: if matches!(class, MemClass::DsStore) { SrcRead::Deferred } else { SrcRead::AtIssue } });
        }
        Ok(out)
    }
}
