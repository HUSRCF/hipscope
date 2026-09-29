//! C5: per-counter weighted wait replay across the CFG.
//!
//! CFG-aware port of `hipfire-isa/src/ledger_replay.rs` onto typed streams,
//! implementing the rule table of core.md §2.6:
//!
//! * Issue: one [`PendingEvent`] per memory instruction. Counter membership
//!   comes from `Effects.mem.counters` (the C1 table column); the per-counter
//!   unit weight comes from the same `isa/gfx12.tbl` row (`Km:1` vs `Km:2`).
//! * Wait `s_wait_X n`: retire the oldest events of counter X until at most
//!   `n` units remain. `Km` and `Store` are out of order and retire only at
//!   `n = 0`; gfx11 `Lgkm` retires partially when all outstanding events
//!   belong to the in-order LDS family, but not when SMEM shares the counter.
//!   `Load` retires partially only within one family (buffer vs global, the
//!   `ledger.rs` rule kept from `ledger_replay.rs:45-61`).
//! * Hazard: touching `defs` of a pending load (RAW/WAW), or redefining
//!   `src_locks` of a pending store (WAR), before retirement is a finding.
//!   Store locks are `Effects.uses` (the `ledger.rs` rule); the WAR def set
//!   mirrors `ledger_replay.rs:131-144` exactly (loads plus `v_*`,
//!   `s_mov*`, `s_add*`, `s_sub*` destinations) so the CFG-aware replay
//!   reproduces the linear replay 1:1 on code without joins.
//! * `s_wait_alu depctr_vm_vsrc(0)` clears VMEM store locks
//!   (`ledger_replay.rs:109-114`); every other `s_wait_alu` is replay-neutral.
//! * `s_endpgm` empties the state; barriers never retire.
//! * Join at a block entry is the maximal pending set keyed by issuing
//!   `InstId`: counters union, units per-counter maximum, satisfied
//!   intersection, defs/locks union (core.md §2.6).
//!
//! Depctr bit positions below are pinned by `llvm-mc -mcpu=gfx1201`
//! (single-field probes): `sa_sdst` = bit 0, `va_vcc` = bit 1,
//! `vm_vsrc` = bits [4:2], `va_sdst` = bits [11:9], `va_vdst` = bits [15:12].

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::cfg::{BlockId, Body, InstId};
use crate::effects::{Control, MemClass};
use crate::inst::{Arch, Inst};
use crate::operand::{ImmField, Operand};
use crate::reg::{Kind, RegRef, RegSet};
use crate::state::ObligationKind;
use crate::state::Obligation;
use crate::wait::{Counter, CounterSet, EventId, PendingEvent, WaitFact, WaitState, N};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WaitError {
    #[error("wait replay needs a gfx11 or gfx12 opcode table (got {0:?})")]
    UnsupportedArch(Arch),
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
    #[error("wait replay did not reach a fixpoint within {walks} block walks")]
    NoFixpoint { walks: usize },
}

/// Census of wait instructions by opcode, for T6.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaitCensus {
    /// Immediates of every `s_wait_kmcnt`, in layout order (KT48: seven 0x0).
    pub kmcnt: Vec<u8>,
    pub loadcnt: usize,
    pub dscnt: usize,
    pub loadcnt_dscnt: usize,
    pub storecnt: usize,
    pub alu: usize,
}

/// Full result of the replay: issued events (with unit weights), per-wait
/// facts, per-site obligations, and the converged state before each inst.
#[derive(Clone, Debug)]
pub struct WaitReplay {
    /// One entry per memory instruction, in first-issue order.
    pub events: Vec<PendingEvent>,
    pub facts: Vec<WaitFact>,
    pub obligations: Vec<Obligation>,
    /// Converged `WaitState` before each instruction (final pass).
    pub before: HashMap<InstId, WaitState>,
}

fn mem_name(class: MemClass) -> &'static str {
    match class {
        MemClass::VmemLoad => "vmem-load",
        MemClass::VmemStore => "vmem-store",
        MemClass::VmemAtomic { .. } => "vmem-atomic",
        MemClass::DsLoad => "ds-load",
        MemClass::DsStore => "ds-store",
        MemClass::DsAtomic { .. } => "ds-atomic",
        MemClass::SmemLoad => "smem-load",
        MemClass::SmemStore => "smem-store",
        MemClass::Export => "export",
        MemClass::LdsDma => "lds-dma",
        MemClass::FlatLoad => "flat-load",
        MemClass::FlatStore => "flat-store",
        MemClass::FlatAtomic { .. } => "flat-atomic",
    }
}

fn is_load(class: MemClass) -> bool {
    matches!(
        class,
        MemClass::VmemLoad
            | MemClass::DsLoad
            | MemClass::SmemLoad
            | MemClass::FlatLoad
            | MemClass::VmemAtomic { returns: true }
            | MemClass::DsAtomic { returns: true }
            | MemClass::FlatAtomic { returns: true }
    )
}

/// Memory classes the replay tracks. Atomics/image/sample rows have no M1
/// table rows; anything else decodable is replayed.
fn is_tracked(class: MemClass) -> bool {
    matches!(
        class,
        MemClass::VmemLoad
            | MemClass::VmemStore
            | MemClass::DsLoad
            | MemClass::DsStore
            | MemClass::SmemLoad
            | MemClass::SmemStore
            | MemClass::FlatLoad
            | MemClass::FlatStore
    )
}

/// Load-counter family for the in-order rule: buffer vs everything else,
/// from the encoding form (the `ledger.rs` family rule).
fn family_of(inst: &Inst) -> &'static str {
    match inst.form {
        crate::inst::Form::Vmem(crate::inst::VmemForm::Buffer) => "buffer",
        crate::inst::Form::Vmem(_) => "global",
        crate::inst::Form::Ds => "ds",
        crate::inst::Form::Smem => "smem",
        _ => "scalar",
    }
}

fn opcode_name(inst: &Inst, arch: Arch) -> Option<&'static str> {
    inst.op.name(arch)
}

/// WAR def set mirroring `ledger_replay.rs:131-144`: loads plus the ALU
/// prefixes the text replay credits with destinations.
fn war_defs(inst: &Inst, arch: Arch) -> Vec<RegRef> {
    if inst.effects.mem.is_some_and(|mem| is_load(mem.class)) {
        return inst.effects.defs.iter().copied().collect();
    }
    let name = opcode_name(inst, arch).unwrap_or("");
    if name.starts_with("v_")
        || name.starts_with("s_mov")
        || name.starts_with("s_add")
        || name.starts_with("s_sub")
    {
        inst.effects.defs.iter().copied().collect()
    } else {
        Vec::new()
    }
}

fn dwords(regs: &[RegRef]) -> Vec<(Kind, u16)> {
    let mut out = Vec::new();
    for reg in regs {
        for i in 0..reg.len {
            out.push((reg.kind, reg.base + u16::from(i)));
        }
    }
    out
}

fn regset_dwords(set: &RegSet) -> Vec<(Kind, u16)> {
    dwords(&set.0)
}

fn overlaps(a: &[(Kind, u16)], b: &[(Kind, u16)]) -> bool {
    a.iter().any(|x| b.contains(x))
}

fn explicit_regs(inst: &Inst) -> Vec<RegRef> {
    let mut out: Vec<RegRef> =
        inst.effects.defs.iter().chain(inst.effects.uses.iter()).copied().collect();
    for operand in &inst.operands {
        match operand {
            Operand::Reg(reg) => out.push(*reg),
            Operand::Half(reg, _) => out.push(*reg),
            _ => {}
        }
    }
    out
}

/// Raw SOPP simm16 bits of an `s_wait_alu`, if present.
fn wait_alu_bits(inst: &Inst) -> Option<u16> {
    inst.operands.iter().find_map(|operand| match operand {
        Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => Some(*n as u16),
        _ => None,
    })
}

fn is_wait_alu(inst: &Inst, arch: Arch) -> bool {
    opcode_name(inst, arch) == Some("s_wait_alu")
}

fn counter_at(index: usize) -> Counter {
    match index {
        0 => Counter::Load,
        1 => Counter::Store,
        2 => Counter::Ds,
        3 => Counter::Km,
        4 => Counter::Sample,
        5 => Counter::Bvh,
        6 => Counter::Exp,
        7 => Counter::Vm,
        8 => Counter::Vs,
        _ => Counter::Lgkm,
    }
}

/// Unit weight per counter from the table row (`Km:2` → 2 on Km).
fn unit_weights(inst: &Inst, arch: Arch) -> [u8; N] {
    let mut units = [0u8; N];
    let Some(row) = crate::isa::lookup(arch, inst.op, inst.form) else {
        return units;
    };
    for pair in row.counter.split(',') {
        let Some((counter, weight)) = pair.split_once(':') else {
            continue;
        };
        let slot = match counter {
            "Load" => Counter::Load,
            "Store" => Counter::Store,
            "Ds" => Counter::Ds,
            "Km" => Counter::Km,
            "Sample" => Counter::Sample,
            "Bvh" => Counter::Bvh,
            "Exp" => Counter::Exp,
            "Vm" => Counter::Vm,
            "Vs" => Counter::Vs,
            "Lgkm" => Counter::Lgkm,
            _ => continue,
        };
        units[slot as usize] = weight.parse().unwrap_or(1);
    }
    units
}

fn join_states(a: &WaitState, b: &WaitState) -> WaitState {
    let mut merged: HashMap<EventId, PendingEvent> = HashMap::new();
    for event in a.pending.iter().chain(&b.pending) {
        match merged.get_mut(&event.id) {
            None => {
                merged.insert(event.id, event.clone());
            }
            Some(prior) => {
                prior.counters = prior.counters.union(event.counters);
                for i in 0..N {
                    prior.units[i] = prior.units[i].max(event.units[i]);
                }
                prior.satisfied = prior.satisfied.intersection(event.satisfied);
                let mut defs = prior.defs.0.clone();
                for reg in &event.defs.0 {
                    if !defs.contains(reg) {
                        defs.push(*reg);
                    }
                }
                prior.defs = RegSet(defs);
                let mut locks = prior.src_locks.0.clone();
                for reg in &event.src_locks.0 {
                    if !locks.contains(reg) {
                        locks.push(*reg);
                    }
                }
                prior.src_locks = RegSet(locks);
            }
        }
    }
    let mut pending: Vec<PendingEvent> = merged.into_values().collect();
    pending.sort_by_key(|event| event.id.0);
    WaitState { pending }
}

struct Recorder {
    facts: Vec<WaitFact>,
    obligations: Vec<Obligation>,
    before: HashMap<InstId, WaitState>,
}

struct EventTable {
    ids: HashMap<InstId, EventId>,
    events: Vec<PendingEvent>,
    next: u64,
}

impl EventTable {
    fn id_of(&mut self, inst: InstId) -> EventId {
        if let Some(id) = self.ids.get(&inst) {
            return *id;
        }
        let id = EventId(self.next);
        self.next += 1;
        self.ids.insert(inst, id);
        id
    }
}

fn retire(
    state: &mut WaitState,
    body: &Body,
    counter: Counter,
    count: u8,
) -> Vec<EventId> {
    if count != 0 {
        if matches!(counter, Counter::Km | Counter::Store | Counter::Vs) {
            return Vec::new();
        }
        // LLVM SIInsertWaitcnts.cpp::counterOutOfOrder: pre-gfx12 LGKM
        // includes LDS and SMEM. SMEM reads can retire out of order; LDS
        // accesses (including atomics) complete in issue order (RDNA3 ISA
        // §9: FLAT LGKM path is in-order with DS). A positive lgkmcnt can
        // therefore retire the oldest LDS only if no SMEM/GDS/message event
        // remains on any reaching path (the joined pending set is maximal).
        if counter == Counter::Lgkm && state.pending.iter().any(|event| {
            event.counters.contains(counter) && !event.satisfied.contains(counter)
                && !matches!(event.class,
                    MemClass::DsLoad | MemClass::DsStore | MemClass::DsAtomic { .. })
        }) {
            return Vec::new();
        }
        if counter == Counter::Load {
            let mut families = HashSet::new();
            for event in state.pending.iter().filter(|e| e.counters.contains(counter)) {
                let inst = body.insts.get(event.inst).expect("replay event names a live inst");
                families.insert(family_of(inst));
            }
            if families.len() > 1 {
                return Vec::new();
            }
        }
    }
    let mut retired = Vec::new();
    loop {
        let units: u32 = state
            .pending
            .iter()
            .filter(|e| e.counters.contains(counter) && !e.satisfied.contains(counter))
            .map(|e| u32::from(e.units[counter as usize]))
            .sum();
        if units <= u32::from(count) {
            break;
        }
        let Some(position) =
            state.pending.iter().position(|e| e.counters.contains(counter) && !e.satisfied.contains(counter))
        else {
            break;
        };
        let event = &mut state.pending[position];
        event.satisfied.insert(counter);
        if event.satisfied == event.counters {
            retired.push(event.id);
            state.pending.remove(position);
        }
    }
    retired
}

fn reg_name((kind, n): (Kind, u16)) -> String {
    let prefix = match kind {
        Kind::V => "v",
        Kind::S => "s",
        Kind::Ttmp => "ttmp",
    };
    format!("{prefix}{n}")
}

/// One obligation per consuming instruction (the linear replay's unit):
/// a store-source WAR wins over a load-destination RAW when both hit.
fn record_finding(
    recorder: &mut Recorder,
    consumer: InstId,
    war_hits: &[PendingEvent],
    raw_hits: &[PendingEvent],
    consumer_touch: &[(Kind, u16)],
    consumer_war: &[(Kind, u16)],
) {
    if !war_hits.is_empty() {
        let first = &war_hits[0];
        let mut blamed: Vec<String> = Vec::new();
        for event in war_hits {
            for dword in regset_dwords(&event.src_locks) {
                if consumer_war.contains(&dword) && !blamed.contains(&reg_name(dword)) {
                    blamed.push(reg_name(dword));
                }
            }
        }
        let rule = match first.class {
            MemClass::DsStore => "wait-war-ds-store",
            MemClass::SmemStore => "wait-war-smem-store",
            _ => "wait-war-vmem-store",
        };
        recorder.obligations.push(Obligation {
            kind: ObligationKind::SrcReadTiming(first.class),
            insts: vec![consumer],
            rule_id: rule.into(),
            text: format!(
                "redefines {} source ({}) before {} retires",
                mem_name(first.class),
                blamed.join(","),
                match first.class {
                    MemClass::DsStore => "DScnt",
                    MemClass::SmemStore => "KMcnt",
                    _ => "STOREcnt",
                }
            ),
        });
        return;
    }
    if !raw_hits.is_empty() {
        let first = &raw_hits[0];
        let mut blamed: Vec<String> = Vec::new();
        for event in raw_hits {
            for dword in regset_dwords(&event.defs) {
                if consumer_touch.contains(&dword) && !blamed.contains(&reg_name(dword)) {
                    blamed.push(reg_name(dword));
                }
            }
        }
        recorder.obligations.push(Obligation {
            kind: ObligationKind::Hazard,
            insts: vec![consumer],
            rule_id: format!("wait-raw-{}", mem_name(first.class)),
            text: format!(
                "touches pending {} destination ({}) before it retires",
                mem_name(first.class),
                blamed.join(",")
            ),
        });
    }
}

fn inst_of(body: &Body, id: InstId) -> Result<&Inst, WaitError> {
    body.insts.get(id).ok_or(WaitError::DanglingInst { id })
}

/// Exits of one block walk. C4 blocks can hold several conditional
/// branches (conditional fall-through is not a leader), so the
/// fall-through exit alone is path-ambiguous: every mid-block conditional
/// branch contributes its own exit state, taken at the branch point.
struct BlockExits {
    fall: WaitState,
    branches: Vec<(BlockId, WaitState)>,
}

fn branch_target(inst: &Inst) -> Option<BlockId> {
    inst.operands.iter().find_map(|operand| match operand {
        Operand::Label(id) => Some(*id),
        _ => None,
    })
}

fn push_event(
    arch: Arch,
    state: &mut WaitState,
    events: &mut EventTable,
    id: InstId,
    inst: &Inst,
    mem: &crate::effects::MemEffect,
    record_issued: bool,
) {
    let id_event = events.id_of(id);
    let (defs, locks) = if is_load(mem.class) {
        (RegSet(inst.effects.defs.iter().copied().collect()), RegSet(Vec::new()))
    } else {
        (RegSet(Vec::new()), RegSet(inst.effects.uses.iter().copied().collect()))
    };
    let fresh = PendingEvent {
        id: id_event,
        inst: id,
        counters: mem.counters,
        units: unit_weights(inst, arch),
        satisfied: CounterSet::default(),
        class: mem.class,
        defs,
        src_locks: locks,
        in_order_type: mem.in_order_type,
    };
    match state.pending.iter_mut().find(|e| e.id == id_event) {
        Some(slot) => *slot = fresh.clone(),
        None => state.pending.push(fresh.clone()),
    }
    if record_issued && events.events.iter().all(|e| e.id != id_event) {
        events.events.push(fresh);
    }
}

/// Walk one block-range from `entry`. With `recorder` set, snapshots the
/// state before each instruction and records facts plus one obligation per
/// consuming instruction; otherwise a dry fixpoint step.
fn walk_block(
    body: &Body,
    arch: Arch,
    range: (usize, usize),
    entry: &WaitState,
    events: &mut EventTable,
    mut recorder: Option<&mut Recorder>,
) -> Result<BlockExits, WaitError> {
    let mut state = entry.clone();
    let mut branches = Vec::new();
    for index in range.0..range.1 {
        let id = body.layout[index];
        let inst = inst_of(body, id)?;
        if let Some(recorder) = recorder.as_mut() {
            recorder.before.insert(id, state.clone());
        }
        match inst.effects.control {
            Control::EndPgm => {
                state.pending.clear();
                continue;
            }
            Control::Wait => {
                if let Some(wait) = &inst.mods.wait {
                    let mut satisfied = HashSet::new();
                    for (counter_index, count) in wait.per_counter.iter().enumerate() {
                        if let Some(n) = count {
                            for retired in retire(&mut state, body, counter_at(counter_index), *n) {
                                satisfied.insert(retired);
                            }
                        }
                    }
                    if let Some(recorder) = recorder.as_mut() {
                        let mut list: Vec<EventId> = satisfied.into_iter().collect();
                        list.sort_by_key(|event| event.0);
                        recorder.facts.push(WaitFact { wait: id, satisfies: list });
                    }
                } else if is_wait_alu(inst, arch)
                    && wait_alu_bits(inst).is_some_and(|bits| bits >> 2 & 7 == 0)
                {
                    for event in state.pending.iter_mut() {
                        if event.counters.contains(Counter::Store) {
                            event.src_locks.0.clear();
                        }
                    }
                }
                continue;
            }
            Control::Jump => {
                if let Some(target) = branch_target(inst) {
                    branches.push((target, state.clone()));
                }
                state = WaitState::default();
                break;
            }
            Control::Branch { .. } => {
                if let Some(target) = branch_target(inst) {
                    branches.push((target, state.clone()));
                }
            }
            _ => {}
        }
        let touch = dwords(&explicit_regs(inst));
        let war = dwords(&war_defs(inst, arch));
        let mut war_hits: Vec<PendingEvent> = Vec::new();
        let mut raw_hits: Vec<PendingEvent> = Vec::new();
        for event in &state.pending {
            if overlaps(&regset_dwords(&event.src_locks), &war) {
                war_hits.push(event.clone());
            } else if overlaps(&regset_dwords(&event.defs), &touch) {
                raw_hits.push(event.clone());
            }
        }
        if let Some(recorder) = recorder.as_mut() {
            if !war_hits.is_empty() || !raw_hits.is_empty() {
                record_finding(recorder, id, &war_hits, &raw_hits, &touch, &war);
            }
        }
        if let Some(mem) = &inst.effects.mem {
            if is_tracked(mem.class)
                || (matches!(arch, Arch::Gfx1100 | Arch::Gfx1151)
                    && matches!(mem.class, MemClass::DsAtomic { .. }))
            {
                push_event(arch, &mut state, events, id, inst, mem, recorder.is_some());
            }
        }
    }
    Ok(BlockExits { fall: state, branches })
}

/// Blocks in layout order, or one synthetic straight-line block when the
/// body has no built CFG (synthetic streams).
fn ranges_of(body: &Body) -> Vec<(BlockId, (usize, usize), Vec<BlockId>)> {
    if body.blocks.is_empty() {
        return vec![(BlockId(0), (0, body.layout.len()), Vec::new())];
    }
    body.blocks.iter().map(|block| (block.id, block.range, block.preds.iter().copied().collect())).collect()
}

/// CFG-aware wait replay with fixpoint over the block graph.
pub fn replay(body: &Body, arch: Arch) -> Result<WaitReplay, WaitError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(WaitError::UnsupportedArch(arch));
    }
    for id in &body.layout {
        if body.insts.get(*id).is_none() {
            return Err(WaitError::DanglingInst { id: *id });
        }
    }
    let ranges = ranges_of(body);
    let entries = fixpoint(body, arch, &ranges)?;
    // Final recording pass in layout order.
    let mut events = EventTable { ids: HashMap::new(), events: Vec::new(), next: 0 };
    let mut recorder = Recorder {
        facts: Vec::new(),
        obligations: Vec::new(),
        before: HashMap::new(),
    };
    for ((_, range, _), entry) in ranges.iter().zip(&entries) {
        walk_block(body, arch, *range, entry, &mut events, Some(&mut recorder))?;
    }
    Ok(WaitReplay {
        events: events.events,
        facts: recorder.facts,
        obligations: recorder.obligations,
        before: recorder.before,
    })
}

/// Block entry states: the least fixpoint of "entry = join of the incoming
/// exits" from empty states, solved with a worklist so a block is re-walked
/// only when its entry changes and a changed exit only revisits its own
/// successors. Each block's exit edges (fall-through, and every mid-block
/// branch the walk reaches) do not depend on the entry state, so the edge
/// lists come from the first walk.
fn fixpoint(body: &Body, arch: Arch, ranges: &[(BlockId, (usize, usize), Vec<BlockId>)]) -> Result<Vec<WaitState>, WaitError> {
    let count = ranges.len();
    let index: HashMap<BlockId, usize> = ranges.iter().enumerate().map(|(i, (id, _, _))| (*id, i)).collect();
    let mut first_at: HashMap<usize, usize> = HashMap::new();
    for (i, (_, range, _)) in ranges.iter().enumerate() {
        first_at.entry(range.0).or_insert(i);
    }
    let mut scratch_events = EventTable { ids: HashMap::new(), events: Vec::new(), next: 0 };
    let mut entries = vec![WaitState::default(); count];
    let mut exits = Vec::with_capacity(count);
    for (_, range, _) in ranges {
        exits.push(walk_block(body, arch, *range, &WaitState::default(), &mut scratch_events, None)?);
    }
    // Incoming edges per block: true fall-through first (p ends where b
    // begins and p's last instruction can fall through; C4 `preds`
    // conflate these with mid-block conditional-branch targets, which the
    // walk reports as branch exits instead), then branch exits.
    let mut incoming: Vec<Vec<(usize, Option<usize>)>> = vec![Vec::new(); count];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (p, (_, range, _)) in ranges.iter().enumerate() {
        let falls = range.0 != range.1 && body.insts.get(body.layout[range.1 - 1])
            .is_some_and(|last| !matches!(last.effects.control, Control::Jump | Control::EndPgm));
        if let (true, Some(&b)) = (falls, first_at.get(&range.1)) {
            incoming[b].push((p, None));
            succs[p].push(b);
        }
    }
    for (p, exit) in exits.iter().enumerate() {
        for (k, (target, _)) in exit.branches.iter().enumerate() {
            if let Some(&b) = index.get(target) {
                incoming[b].push((p, Some(k)));
                if !succs[p].contains(&b) { succs[p].push(b); }
            }
        }
    }
    let mut work: std::collections::BTreeSet<usize> = (0..count).collect();
    let budget = count.saturating_mul(1000).max(1000);
    let mut walks = 0usize;
    while let Some(b) = work.pop_first() {
        let mut joined: Option<WaitState> = None;
        for &(p, edge) in &incoming[b] {
            let state = match edge { None => &exits[p].fall, Some(k) => &exits[p].branches[k].1 };
            joined = Some(match joined { None => state.clone(), Some(prior) => join_states(&prior, state) });
        }
        let joined = joined.unwrap_or_default();
        if joined == entries[b] { continue; }
        walks += 1;
        if walks > budget { return Err(WaitError::NoFixpoint { walks }); }
        exits[b] = walk_block(body, arch, ranges[b].1, &joined, &mut scratch_events, None)?;
        entries[b] = joined;
        work.extend(succs[b].iter().copied());
    }
    Ok(entries)
}

/// Count wait instructions by opcode for T6.
pub fn census(body: &Body, arch: Arch) -> Result<WaitCensus, WaitError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(WaitError::UnsupportedArch(arch));
    }
    let mut out = WaitCensus::default();
    for id in &body.layout {
        let inst = inst_of(body, *id)?;
        let name = opcode_name(inst, arch).unwrap_or("");
        match name {
            "s_wait_kmcnt" => {
                let count = inst
                    .mods
                    .wait
                    .as_ref()
                    .and_then(|wait| wait.per_counter[Counter::Km as usize])
                    .unwrap_or(0);
                out.kmcnt.push(count);
            }
            "s_wait_loadcnt" => out.loadcnt += 1,
            "s_wait_dscnt" => out.dscnt += 1,
            "s_wait_loadcnt_dscnt" | "s_wait_storecnt_dscnt" => out.loadcnt_dscnt += 1,
            "s_wait_storecnt" => out.storecnt += 1,
            "s_wait_alu" => out.alu += 1,
            _ => {}
        }
    }
    Ok(out)
}
// C5 wait-replay tests: typed re-expressions of the linear
// `ledger_replay.rs:161-230` suite plus the KT48 T6 gate.

#[cfg(test)]
mod c5_tests {
    use smallvec::SmallVec;

    use crate::cfg::Body;
    use crate::effects::MemClass;
    use crate::inst::{Arch, Form, FormFields, Inst, Opcode};
    use crate::operand::{ImmField, Modifiers, Operand, VmemToken};
    use crate::passes::cfg::build_blocks;
    use crate::provenance::Provenance;
    use crate::reg::{Kind, RegRef};
    use crate::state::ObligationKind;
    use crate::wait::Counter;

    use super::{census, replay};

    fn table(name: &str) -> (Opcode, Form) {
        let row = crate::isa::gfx12().iter().find(|row| row.name == name).expect(name);
        (row.op, row.form)
    }

    fn mk(name: &str, operands: Vec<Operand>, mods: Modifiers) -> Inst {
        let (op, form) = table(name);
        Inst::from_parts(
            Arch::Gfx1201,
            op,
            form,
            FormFields::None,
            SmallVec::from_vec(operands),
            mods,
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
    }

    fn mi(name: &str, operands: Vec<Operand>) -> Inst {
        mk(name, operands, Modifiers::default())
    }
    fn v(base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind: Kind::V, base, len })
    }

    fn s(base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind: Kind::S, base, len })
    }

    fn wait_mods(load: Option<u8>, ds: Option<u8>, km: Option<u8>, store: Option<u8>) -> Modifiers {
        let mut wait = crate::wait::WaitImm::default();
        wait.per_counter[Counter::Load as usize] = load;
        wait.per_counter[Counter::Ds as usize] = ds;
        wait.per_counter[Counter::Km as usize] = km;
        wait.per_counter[Counter::Store as usize] = store;
        Modifiers { wait: Some(wait), ..Modifiers::default() }
    }

    fn sopp(n: i16) -> Operand {
        Operand::Imm(ImmField::Sopp(n))
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut body = Body::default();
        for inst in insts {
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        body
    }

    fn buffer_load(dst: u16, addr: u16) -> Inst {
        mi("buffer_load_b32",
            vec![v(dst, 1), v(addr, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
        )
    }

    fn vadd(dst: u16, a: u16, b: u16) -> Inst {
        mi("v_add_f32_e32", vec![v(dst, 1), v(a, 1), v(b, 1)])
    }

    fn ok(body: &Body) -> bool {
        replay(body, Arch::Gfx1201).unwrap().obligations.is_empty()
    }

    /// On gfx11 a positive lgkmcnt retires the oldest LDS access when no
    /// SMEM shares LGKM. With mixed outstanding SMEM the same wait cannot
    /// establish which earlier DS load has completed.
    #[test]
    fn gfx11_lgkm_partial_wait_respects_lds_order_and_smem_mixing() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let inst = |name: &str, operands: Vec<Operand>| {
                let row = crate::isa::table(arch).iter().find(|row| row.name == name).expect(name);
                Inst::from_parts(
                    arch, row.op, row.form, FormFields::None,
                    SmallVec::from_vec(operands), Modifiers::default(),
                    None, Provenance::default(),
                ).unwrap_or_else(|error| panic!("{arch:?} {name}: {error:?}"))
            };
            let ds = |dst| inst("ds_load_b32", vec![
                v(dst, 1), v(9, 1), Operand::Imm(ImmField::DsOffset(0)),
            ]);
            let wait = crate::codec::gfx11::decode(arch, &[0xbf89_0432]).unwrap().0;
            assert_eq!(wait.mods.wait.as_ref().unwrap().per_counter[Counter::Lgkm as usize], Some(3));
            let consumer = || inst("v_add_f32_e32", vec![v(20, 1), v(1, 1), v(21, 1)]);
            let ds_only = body_of(vec![ds(1), ds(2), ds(3), ds(4), wait.clone(), consumer()]);
            assert!(replay(&ds_only, arch).unwrap().obligations.is_empty(), "{arch:?}: oldest LDS must retire");
            let ds_later = body_of(vec![ds(1), ds(2), ds(3), ds(4), wait.clone(),
                inst("v_add_f32_e32", vec![v(20, 1), v(2, 1), v(21, 1)])]);
            assert!(replay(&ds_later, arch).unwrap().obligations.iter()
                .any(|obligation| obligation.rule_id == "wait-raw-ds-load"));
            let atomic = inst("ds_min_i32", vec![
                v(30, 1), v(31, 1), Operand::Imm(ImmField::DsOffset(0)),
            ]);
            let atomics_in_order = body_of(vec![atomic, ds(1), ds(2), ds(3), wait.clone(),
                inst("v_mov_b32_e32", vec![v(31, 1), v(21, 1)])]);
            assert!(replay(&atomics_in_order, arch).unwrap().obligations.is_empty(),
                "{arch:?}: oldest LDS atomic has retired before reusing its data source");
            let smem = inst("s_load_b32", vec![
                s(20, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0)),
            ]);
            let mixed = body_of(vec![ds(1), smem, ds(2), ds(3), ds(4), wait, consumer()]);
            assert!(replay(&mixed, arch).unwrap().obligations.iter()
                .any(|obligation| obligation.rule_id == "wait-raw-ds-load"),
                "{arch:?}: SMEM makes partial lgkmcnt unable to prove LDS retirement");
        }
    }

    /// Linear `load_wait_uses_only_retired_destinations`, typed.
    #[test]
    fn load_wait_retires_oldest_first() {
        let legal = body_of(vec![
            buffer_load(0, 8),
            buffer_load(1, 8),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(2, 0, 3),
        ]);
        assert!(ok(&legal));
        let tight = body_of(vec![
            buffer_load(0, 8),
            buffer_load(1, 8),
            mk("s_wait_loadcnt", vec![sopp(2)], wait_mods(Some(2), None, None, None)),
            vadd(2, 0, 3),
        ]);
        assert!(!ok(&tight));
    }

    /// Linear `combined_wait_keeps_young_vmem_pending`, typed.
    #[test]
    fn combined_wait_keeps_young_vmem_pending() {
        let stream = body_of(vec![
            buffer_load(0, 8),
            mi("ds_load_b32", vec![v(1, 1), v(9, 1), Operand::Imm(ImmField::DsOffset(0))]),
            mk(
                "s_wait_loadcnt_dscnt",
                vec![sopp(0x100)],
                wait_mods(Some(1), Some(0), None, None),
            ),
            vadd(2, 1, 3),
        ]);
        assert!(ok(&stream));
        let mut late = stream;
        let tail = mi("v_add_f32_e32", vec![v(2, 1), v(0, 1), v(3, 1)]);
        let id = late.insts.insert(tail);
        late.layout.push(id);
        assert!(!ok(&late));
    }

    /// Linear `asymmetric_combined_wait_decodes_load_then_ds`, typed.
    #[test]
    fn asymmetric_combined_wait() {
        let mut insts = Vec::new();
        for i in 0..8 {
            insts.push(buffer_load(i, 20));
        }
        for i in 0..4 {
            insts.push(mi("ds_load_b32",
                vec![v(8 + i, 1), v(21, 1), Operand::Imm(ImmField::DsOffset(0))],
            ));
        }
        insts.push(mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x703)],
            wait_mods(Some(7), Some(3), None, None),
        ));
        insts.push(vadd(22, 0, 8));
        assert!(ok(&body_of(insts.clone())));
        // LOADcnt 3 leaves v0..v4 pending: use of v0 faults.
        let mut weak = insts.clone();
        let at = weak.len() - 2;
        weak[at] = mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x307)],
            wait_mods(Some(3), Some(7), None, None),
        );
        assert!(!ok(&body_of(weak)));
        // DScnt 7 retires nothing: use of v8 faults.
        let mut weak_ds = insts.clone();
        let at = weak_ds.len() - 2;
        weak_ds[at] = mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x707)],
            wait_mods(Some(7), Some(7), None, None),
        );
        assert!(!ok(&body_of(weak_ds)));
    }

    /// Linear `store_lock_requires_storecnt_before_redefinition`, typed.
    #[test]
    fn store_lock_needs_storecnt() {
        let source = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("v_mov_b32_e32", vec![v(0, 1), v(5, 1)]),
        ]);
        assert!(!ok(&source));
        let drained = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mk("s_wait_storecnt", vec![sopp(0)], wait_mods(None, None, None, Some(0))),
            mi("v_mov_b32_e32", vec![v(0, 1), v(5, 1)]),
        ]);
        assert!(ok(&drained));
        // Reading (not redefining) a store source is fine.
        let read = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            vadd(2, 0, 3),
        ]);
        assert!(ok(&read));
    }

    /// Linear `smem_load_requires_kmcnt_before_scalar_use`, typed.
    #[test]
    fn smem_load_needs_kmcnt() {
        let source = body_of(vec![
            mi("s_load_b32", vec![s(4, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0))]),
            mi("s_add_co_u32", vec![s(5, 1), s(4, 1), s(6, 1)]),
        ]);
        assert!(!ok(&source));
        let drained = body_of(vec![
            mi("s_load_b32", vec![s(4, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0))]),
            mk("s_wait_kmcnt", vec![sopp(0)], wait_mods(None, None, Some(0), None)),
            mi("s_add_co_u32", vec![s(5, 1), s(4, 1), s(6, 1)]),
        ]);
        assert!(ok(&drained));
    }

    /// Linear `vm_vsrc_releases_store_sources…`, typed (VALU move stands in
    /// for the text test's `v_writelane_b32`, which has no M1 table row).
    #[test]
    fn vm_vsrc_releases_store_sources() {
        let store = || {
            mi("global_store_b128",
                vec![v(9, 2), v(10, 4), s(4, 2), Operand::Vmem(VmemToken::Off)],
            )
        };
        let blocked = body_of(vec![store(), mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)])]);
        assert!(!ok(&blocked));
        // depctr_vm_vsrc(0) == simm16 0xFF83 (llvm-mc pinned).
        let released = body_of(vec![
            store(),
            mk("s_wait_alu", vec![sopp(0xFF83u16 as i16)], Modifiers::default()),
            mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)]),
        ]);
        assert!(ok(&released));
        // Any other depctr does not release store sources.
        let other = body_of(vec![
            store(),
            mk("s_wait_alu", vec![sopp(0xFF9Eu16 as i16)], Modifiers::default()),
            mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)]),
        ]);
        assert!(!ok(&other));
    }

    /// Linear `mixed_vmem_loads_require_zero_wait`, typed.
    #[test]
    fn mixed_vmem_loads_need_zero_wait() {
        let mixed = body_of(vec![
            mi("buffer_load_b32",
                vec![v(0, 2), v(8, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("global_load_b64", vec![v(2, 1), v(8, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(!ok(&mixed));
        let mut zero = body_of(vec![
            mi("buffer_load_b32",
                vec![v(0, 2), v(8, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("global_load_b64", vec![v(2, 1), v(8, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(0)], wait_mods(Some(0), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(ok(&zero));
        // Same-family loads still retire partially: oldest of two globals.
        zero.layout.pop();
        let tail = vadd(3, 0, 4);
        let _ = tail;
        let same = body_of(vec![
            mi("global_load_b64", vec![v(0, 1), v(8, 2), s(4, 2)]),
            mi("global_load_b64", vec![v(1, 2), v(9, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(ok(&same));
    }

    // ---- KT48 T6 gate ----

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
        assert_eq!(body.layout.len(), 1696);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("KT48 CFG builds");
        body
    }

    #[test]
    fn kt48_wait_census() {
        let body = kt48_body();
        let census = census(&body, Arch::Gfx1201).unwrap();
        assert_eq!(census.kmcnt.len(), 7, "seven s_wait_kmcnt");
        assert!(census.kmcnt.iter().all(|count| *count == 0), "all kmcnt 0x0");
        assert_eq!(census.loadcnt, 16);
        assert_eq!(census.dscnt, 28);
        assert_eq!(census.loadcnt_dscnt, 3);
        assert_eq!(census.alu, 142);
    }

    /// T6 wait gate, corrected count. The linear text replay reports 122
    /// findings on KT48; the sound CFG-aware replay reports 124, all of the
    /// same single class. Decomposition (verified against
    /// `ledger_replay::replay_hazards` run on the canonical text of the same
    /// 1,696 instructions):
    /// * 120 findings are common to both replays;
    /// * 2 linear findings (the `ds_load_b128` at layout 917 and the use at
    ///   925) come from the linear walk's infeasible layout fall-through
    ///   across the unconditional `s_branch` at 915 (dword-pc target 1434 =
    ///   layout 967; layout 916 is reachable only via a later block, whose
    ///   exit has the DS stores retired). They are linear false positives;
    /// * 4 findings here (956, 957, 973, 974) are feasible first-iteration
    ///   paths the linear walk misses: the unconditional jump at 915 enters
    ///   the loop at 967 without executing the `s_wait_dscnt` waits at
    ///   928-949, so the `ds_store_2addr_b64` sources outstanding since 782
    ///   are still locked when 956/957 redefine v134/v135 and 973/974
    ///   redefine v130-133/v169-172. Same phenomenon as the other 120, same
    ///   disposition (diagnostic until the DS-source pin).
    #[test]
    fn kt48_replay_obligations_are_ds_store_war() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        assert_eq!(replay.obligations.len(), 124, "120 linear-common + 4 first-iteration sites");
        for obligation in &replay.obligations {
            assert_eq!(obligation.kind, ObligationKind::SrcReadTiming(MemClass::DsStore));
            assert_eq!(obligation.rule_id, "wait-war-ds-store");
            assert_eq!(obligation.insts.len(), 1);
        }
        // Every finding redefines a source of a ds_store_2addr_b64.
        for obligation in &replay.obligations {
            let consumer = obligation.insts[0];
            let inst = body.insts.get(consumer).unwrap();
            let mut touch = Vec::new();
            for reg in inst.effects.defs.iter().chain(inst.effects.uses.iter()) {
                for i in 0..reg.len {
                    touch.push((reg.kind, reg.base + u16::from(i)));
                }
            }
            let blamed = replay.events.iter().any(|event| {
                event.class == MemClass::DsStore
                    && body.insts.get(event.inst).unwrap().op.name(Arch::Gfx1201)
                        == Some("ds_store_2addr_b64")
                    && event.src_locks.0.iter().any(|reg| {
                        (0..reg.len).any(|i| touch.contains(&(reg.kind, reg.base + u16::from(i))))
                    })
            });
            assert!(blamed, "finding at {consumer:?} names no ds_store_2addr_b64 source");
        }
    }
    /// Path-sensitivity pin: layout positions 956/957/973/974 are
    /// obligations (feasible first-iteration paths via the 915 jump that
    /// skip the 928-949 waits), while 917/925 are not (reachable only from
    /// a later block; the 915->916 layout fall-through never executes).
    #[test]
    fn kt48_path_sensitive_sites() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        let positions: Vec<usize> = replay
            .obligations
            .iter()
            .map(|obligation| {
                body.layout.iter().position(|id| *id == obligation.insts[0]).unwrap()
            })
            .collect();
        for expected in [956, 957, 973, 974] {
            assert!(positions.contains(&expected), "missing feasible-path site {expected}");
        }
        for infeasible in [917, 925] {
            assert!(!positions.contains(&infeasible), "infeasible-path site {infeasible} flagged");
        }
    }

    #[test]
    fn kt48_kmcnt_units_and_retirement() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        let mut smem: Vec<_> =
            replay.events.iter().filter(|event| event.class == MemClass::SmemLoad).collect();
        assert_eq!(smem.len(), 6, "KT48 has six SMEM loads");
        smem.sort_by_key(|event| body.layout.iter().position(|id| *id == event.inst));
        // Layout order is [2,1,1,1,2,2]; the gate's [1,1,1,2,2,2] is the
        // same multiset grouped by width (3x b32 at 1 unit, 2x b128 + 1x
        // b256 at 2 units per RDNA4 ISA 5.7.1).
        let mut units: Vec<u8> =
            smem.iter().map(|event| event.units[Counter::Km as usize]).collect();
        units.sort();
        assert_eq!(units, vec![1, 1, 1, 2, 2, 2]);
        for event in &smem {
            let name = body.insts.get(event.inst).unwrap().op.name(Arch::Gfx1201).unwrap_or("?");
            let expected = if name.contains("b32") {
                1
            } else {
                assert!(name.contains("b128") || name.contains("b256"), "{name}");
                2
            };
            assert_eq!(event.units[Counter::Km as usize], expected, "{name}");
        }
        // Every SMEM event is retired by an s_wait_kmcnt 0x0.
        for event in &smem {
            let retired = replay.facts.iter().any(|fact| {
                fact.satisfies.contains(&event.id)
                    && body.insts.get(fact.wait).unwrap().op.name(Arch::Gfx1201)
                        == Some("s_wait_kmcnt")
            });
            assert!(retired, "smem event {:?} never retired by kmcnt 0x0", event.inst);
        }
    }

    #[test]
    fn kt48_mutated_kmcnt_produces_obligation() {
        let body = kt48_body();
        let wait_ids: Vec<_> = body
            .layout
            .iter()
            .filter(|id| {
                body.insts.get(**id).unwrap().op.name(Arch::Gfx1201) == Some("s_wait_kmcnt")
            })
            .copied()
            .collect();
        assert_eq!(wait_ids.len(), 7);
        let mut produced = false;
        for wait in wait_ids {
            let mut mutated = body.clone();
            let inst = mutated.insts.get_mut(wait).unwrap();
            inst.mods.wait.as_mut().unwrap().per_counter[Counter::Km as usize] = Some(1);
            let replay = replay(&mutated, Arch::Gfx1201).unwrap();
            if replay.obligations.iter().any(|obligation| {
                matches!(obligation.kind, ObligationKind::Hazard)
                    && obligation.rule_id == "wait-raw-smem-load"
            }) {
                produced = true;
                break;
            }
        }
        assert!(produced, "kmcnt 0x0 -> 0x1 must strand an s_load destination");
    }

    /// A pending load reaches its consumer across any number of blocks. The
    /// entry fixpoint used to re-walk every block per round with a
    /// 1000-round cap, so state advanced one block per round: this chain
    /// was never reached (and a 4485-block kernel took hours).
    #[test]
    fn pending_load_crosses_a_long_block_chain() {
        let mut insts = vec![buffer_load(0, 8)];
        insts.extend((0..1500).map(|_| mi("s_cbranch_scc0", vec![sopp(0)])));
        insts.push(vadd(2, 0, 3));
        insts.push(mi("s_endpgm", vec![]));
        let mut body = body_of(insts);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert!(body.blocks.len() > 1500);
        let consumer = body.layout[body.layout.len() - 2];
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        assert!(replay.obligations.iter().any(|o| o.rule_id == "wait-raw-vmem-load" && o.insts == [consumer]),
            "{:?}", replay.obligations);
    }
}
