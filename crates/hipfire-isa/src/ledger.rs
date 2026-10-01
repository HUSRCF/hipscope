use crate::{
    arch::Arch,
    insn::{Instruction, MemoryClass},
    reg::RegRef,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Counter { Load, Store, Ds, Km, Vm, Vs, Lgkm, Exp }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFamily { Buffer, Global, Ds, Smem, Export }

#[derive(Clone, Debug)]
pub struct Pending {
    pub id: u64,
    pub counter: Counter,
    pub defs: Vec<RegRef>,
    pub src_locks: Vec<RegRef>,
    pub in_order: bool,
    pub family: MemoryFamily,
    /// Pending on only some of the control paths that joined here.
    pub maybe: bool,
}

#[derive(Clone, Debug, Serialize)]
pub enum Reason { Use { reg: String }, Waw { reg: String }, War { reg: String }, Barrier, LoopFixpoint }

#[derive(Clone, Debug, Serialize)]
pub struct WaitProof { pub pc_index: usize, pub insn: String, pub counter: Counter, pub count: u8, pub reason: Reason }

#[derive(Clone, Debug, Default)]
pub struct Ledger { pending: Vec<Pending>, next_id: u64 }

impl Ledger {
    pub fn record(&mut self, arch: Arch, insn: &Instruction) {
        let Some(class) = insn.memory else { return };
        let (counter, in_order, load) = match (arch.gfx12(), class) {
            (true, MemoryClass::VmemLoad) => (Counter::Load, true, true),
            (true, MemoryClass::VmemStore) => (Counter::Store, true, false),
            (true, MemoryClass::DsLoad) => (Counter::Ds, true, true),
            (true, MemoryClass::DsStore) => (Counter::Ds, true, false),
            (true, MemoryClass::SmemLoad) => (Counter::Km, false, true),
            (false, MemoryClass::VmemLoad) => (Counter::Vm, true, true),
            (false, MemoryClass::VmemStore) => (Counter::Vs, true, false),
            (false, MemoryClass::DsLoad) => (Counter::Lgkm, true, true),
            (false, MemoryClass::DsStore) => (Counter::Lgkm, true, false),
            (false, MemoryClass::SmemLoad) => (Counter::Lgkm, false, true),
            (_, MemoryClass::Export) => (Counter::Exp, false, false),
        };
        let family = match class {
            MemoryClass::VmemLoad | MemoryClass::VmemStore if insn.mnemonic().starts_with("buffer_") => MemoryFamily::Buffer,
            MemoryClass::VmemLoad | MemoryClass::VmemStore => MemoryFamily::Global,
            MemoryClass::DsLoad | MemoryClass::DsStore => MemoryFamily::Ds,
            MemoryClass::SmemLoad => MemoryFamily::Smem,
            MemoryClass::Export => MemoryFamily::Export,
        };
        self.pending.push(Pending {
            id: self.next_id, counter,
            defs: if load { insn.defs.clone() } else { vec![] },
            src_locks: if load { vec![] } else { insn.uses.clone() },
            in_order, family, maybe: false,
        });
        self.next_id += 1;
    }

    pub fn required(&self, insn: &Instruction) -> Vec<(Counter, u8, Reason)> {
        let mut waits: Vec<(Counter, u8, Reason)> = Vec::new();
        for (index, pending) in self.pending.iter().enumerate() {
            let reason = pending.defs.iter().find_map(|def| {
                insn.uses.iter().find(|use_| def.overlaps(**use_))
                    .map(|_| Reason::Use { reg: def.to_string() })
                    .or_else(|| insn.defs.iter().find(|write| def.overlaps(**write))
                        .map(|_| Reason::Waw { reg: def.to_string() }))
            }).or_else(|| pending.src_locks.iter().find_map(|lock| {
                insn.defs.iter().find(|write| lock.overlaps(**write))
                    .map(|_| Reason::War { reg: lock.to_string() })
            }));
            let Some(reason) = reason else { continue };
            let peers = self.pending.iter().filter(|p| p.counter == pending.counter);
            let ordered = pending.in_order && peers.clone().all(|p| p.in_order && !p.maybe && p.family == pending.family);
            let younger = self.pending.iter().skip(index + 1).filter(|p| p.counter == pending.counter).count();
            let count = if ordered && younger <= 63 { younger as u8 } else { 0 };
            if let Some(wait) = waits.iter_mut().find(|(counter, _, _)| *counter == pending.counter) {
                if count < wait.1 { *wait = (pending.counter, count, reason) }
            } else {
                waits.push((pending.counter, count, reason));
            }
        }
        waits
    }

    /// An operation retires once `count` younger operations of its counter
    /// are certainly issued after it (without joined `maybe` operations:
    /// all but the youngest `count`).
    pub fn wait(&mut self, counter: Counter, count: u8) {
        let mut younger = 0usize;
        let mut keep = vec![true; self.pending.len()];
        for (index, p) in self.pending.iter().enumerate().rev() {
            if p.counter != counter { continue }
            if younger >= usize::from(count) { keep[index] = false }
            if !p.maybe { younger += 1 }
        }
        let mut keep = keep.into_iter();
        self.pending.retain(|_| keep.next().unwrap_or(true));
    }

    /// Join the ledger of another control path into this point. Equal
    /// shapes need the same waits on both paths, so this one stands.
    /// Otherwise every operation pending on either path stays pending, and
    /// one pending on only one path is `maybe`: waits on its counter drain
    /// to zero (`required`) and a partial wait retires only what is
    /// certain (`wait`), which holds on both paths.
    pub fn join(&mut self, other: &Ledger) {
        if self.shape() == other.shape() { return }
        let mut merged: Vec<Pending> = self.pending.iter().cloned().map(|mut p| {
            p.maybe |= !other.pending.iter().any(|o| o.id == p.id);
            p
        }).collect();
        for o in &other.pending {
            match merged.iter_mut().find(|p| p.id == o.id) {
                Some(p) => p.maybe |= o.maybe,
                None => merged.push(Pending { maybe: true, ..o.clone() }),
            }
        }
        merged.sort_by_key(|p| p.id);
        self.pending = merged;
        self.next_id = self.next_id.max(other.next_id);
    }
    /// Continue from another path's ledger without reusing an id this one
    /// issued (ids stay unique across the paths of one program).
    pub fn resume(&mut self, at: Ledger) {
        let next_id = self.next_id.max(at.next_id);
        *self = at;
        self.next_id = next_id;
    }

    pub fn drain(&mut self) -> Vec<(Counter, u8, Reason)> {
        let mut waits = Vec::new();
        for pending in &self.pending {
            if !waits.iter().any(|(counter, _, _)| *counter == pending.counter) {
                waits.push((pending.counter, 0, Reason::Barrier));
            }
        }
        for (counter, _, _) in &waits { self.wait(*counter, 0) }
        waits
    }

    /// After `s_wait_alu depctr_vm_vsrc(0)` every issued VMEM store has read
    /// its source registers: pending stores lock nothing and define nothing,
    /// so they leave the ledger (completion order is never inferred for them).
    pub fn release_store_sources(&mut self) {
        self.pending.retain(|p| !matches!(p.counter, Counter::Store | Counter::Vs));
    }

    /// An LDS store not yet retired (DScnt on gfx12, LGKMcnt on gfx11): a
    /// barrier that publishes LDS must drain it first.
    pub fn pending_stores(&self) -> bool {
        self.pending.iter().any(|p| matches!(p.counter, Counter::Ds | Counter::Lgkm)
            && p.family == MemoryFamily::Ds && !p.src_locks.is_empty())
    }
    pub fn is_empty(&self) -> bool { self.pending.is_empty() }
    /// Id of the most recently recorded operation.
    pub fn last_id(&self) -> Option<u64> { self.next_id.checked_sub(1) }
    /// Operation `id` has not been retired by a wait.
    pub fn is_pending(&self, id: u64) -> bool { self.pending.iter().any(|p| p.id == id) }
    /// Pending operations without their ids: two ledgers with equal shapes
    /// require the same waits for every later instruction.
    pub fn shape(&self) -> Vec<(Counter, Vec<RegRef>, Vec<RegRef>, bool, MemoryFamily, bool)> {
        self.pending.iter().map(|p| (p.counter, p.defs.clone(), p.src_locks.clone(), p.in_order, p.family, p.maybe)).collect()
    }

    pub fn wait_instruction(arch: Arch, waits: &[(Counter, u8, Reason)]) -> Result<Vec<(Counter, u8, String, Reason)>, String> {
        let mut out = Vec::new();
        for (counter, count, reason) in waits {
            let text = match (arch.gfx12(), counter) {
                (true, Counter::Load) => format!("s_wait_loadcnt {count:#x}"),
                (true, Counter::Store) => format!("s_wait_storecnt {count:#x}"),
                (true, Counter::Ds) => format!("s_wait_dscnt {count:#x}"),
                (true, Counter::Km) => format!("s_wait_kmcnt {count:#x}"),
                (false, Counter::Vm) => format!("s_waitcnt vmcnt({count})"),
                (false, Counter::Vs) => format!("s_waitcnt_vscnt null, {count:#x}"),
                (false, Counter::Lgkm) => format!("s_waitcnt lgkmcnt({count})"),
                (_, Counter::Exp) => format!("s_waitcnt expcnt({count})"),
                _ => return Err("counter/architecture mismatch".into()),
            };
            out.push((*counter, *count, text, reason.clone()));
        }
        Ok(out)
    }
}
