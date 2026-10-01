//! A reference backend for tests and documentation: it records instruction
//! text and runs a strict slot machine. Unlike the production builder it
//! never inserts a drain on its own: a barrier with an LDS store pending is
//! an error, so a program that passes here got every drain from the types.

use crate::backend::{Backend, EventId, SlotTransition};
use crate::target::LdsCounter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    Publishing,
    Published,
    Reading,
}

/// `Trace`'s state at a forward branch.
pub struct TraceFork(Vec<(String, Slot)>, Vec<u64>);

#[derive(Clone, Debug)]
pub struct Trace {
    arch: &'static str,
    /// Emitted instructions and labels, in order.
    pub text: Vec<String>,
    slots: Vec<(String, Slot)>,
    stores: Vec<u64>,
    next: u64,
    signalled: Option<Vec<SlotTransition>>,
}

impl Trace {
    pub fn new(arch: &'static str) -> Self {
        Self { arch, text: Vec::new(), slots: Vec::new(), stores: Vec::new(), next: 0, signalled: None }
    }
    /// Instructions whose mnemonic is `m`.
    pub fn count(&self, m: &str) -> usize {
        self.text.iter().filter(|t| t.split_whitespace().next() == Some(m)).count()
    }
    /// A raw (non-LDS, non-barrier) instruction.
    pub fn emit(&mut self, t: impl Into<String>) {
        self.text.push(t.into());
    }
    fn transitions(&mut self, ts: &[SlotTransition], apply: bool) -> Result<(), String> {
        if !self.stores.is_empty() {
            return Err("barrier with an LDS store pending".into());
        }
        for t in ts {
            let (id, from, to) = match *t {
                SlotTransition::Retire(id) => (id, Slot::Reading, Slot::Free),
                SlotTransition::Ready(id) => (id, Slot::Publishing, Slot::Published),
            };
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if slot.1 != from {
                return Err(format!("{t:?}: {} is {:?}", slot.0, slot.1));
            }
            if apply {
                slot.1 = to;
            }
        }
        Ok(())
    }
}

impl Backend for Trace {
    type Insn = String;
    fn arch_name(&self) -> &'static str {
        self.arch
    }
    /// Trace has no untyped LDS or barrier entry points to close.
    fn seal(&mut self) {}
    fn position(&self) -> usize {
        self.text.len()
    }
    fn lds_slot(&mut self, name: &str, base: u32, len: u32) -> Result<usize, String> {
        let _ = (base, len);
        self.slots.push((name.into(), Slot::Free));
        Ok(self.slots.len() - 1)
    }
    fn lds_relayout(&mut self) -> Result<(), String> {
        if let Some((name, s)) = self.slots.iter().find(|(_, s)| *s != Slot::Free) {
            return Err(format!("{name} is {s:?} at relayout"));
        }
        Ok(())
    }
    fn ds_store(&mut self, slots: &[usize], insn: String) -> Result<EventId, String> {
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if !matches!(slot.1, Slot::Free | Slot::Publishing) {
                return Err(format!("{} cannot be stored while {:?}", slot.0, slot.1));
            }
            slot.1 = Slot::Publishing;
        }
        self.emit(insn);
        self.stores.push(self.next);
        self.next += 1;
        Ok(EventId(self.next - 1))
    }
    fn ds_load(&mut self, slots: &[usize], insn: String) -> Result<(), String> {
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if !matches!(slot.1, Slot::Published | Slot::Reading) {
                return Err(format!("{} is not published", slot.0));
            }
            slot.1 = Slot::Reading;
        }
        self.emit(insn);
        Ok(())
    }
    fn lds_store_pending(&self, event: EventId) -> bool {
        self.stores.contains(&event.0)
    }
    fn drain_lds_stores(&mut self, counter: LdsCounter) -> Result<(), String> {
        self.emit(match counter {
            LdsCounter::Lgkm => "s_waitcnt lgkmcnt(0)",
            LdsCounter::Ds => "s_wait_dscnt 0x0",
        });
        self.stores.clear();
        Ok(())
    }
    fn barrier(&mut self, transitions: &[SlotTransition]) -> Result<(), String> {
        self.transitions(transitions, true)?;
        self.emit("s_barrier");
        Ok(())
    }
    fn barrier_signal(&mut self, transitions: &[SlotTransition]) -> Result<(), String> {
        if self.signalled.is_some() {
            return Err("split barrier already in flight".into());
        }
        self.transitions(transitions, false)?;
        self.signalled = Some(transitions.to_vec());
        self.emit("s_barrier_signal -1");
        Ok(())
    }
    fn barrier_wait(&mut self) -> Result<(), String> {
        let ts = self.signalled.take().ok_or("barrier wait without signal")?;
        self.transitions(&ts, true)?;
        self.emit("s_barrier_wait 0xffff");
        Ok(())
    }
    fn label(&mut self, name: &str) -> Result<(), String> {
        if self.text.iter().any(|t| t.strip_suffix(':') == Some(name)) {
            return Err(format!("duplicate label {name}"));
        }
        self.emit(format!("{name}:"));
        Ok(())
    }
    fn scalar_compare(&mut self, insn: String) -> Result<(), String> {
        if !(insn.starts_with("s_cmp") || insn.starts_with("s_bitcmp")) {
            return Err(format!("{insn} does not define SCC"));
        }
        self.emit(insn);
        Ok(())
    }
    fn branch_scc1(&mut self, target: &str) -> Result<(), String> {
        self.emit(format!("s_cbranch_scc1 {target}"));
        Ok(())
    }
    fn branch(&mut self, target: &str) -> Result<(), String> {
        self.emit(format!("s_branch {target}"));
        Ok(())
    }
    type Fork = TraceFork;
    fn fork(&self) -> Self::Fork {
        TraceFork(self.slots.clone(), self.stores.clone())
    }
    fn resume(&mut self, TraceFork(slots, stores): Self::Fork) {
        self.slots = slots;
        self.stores = stores;
    }
    fn joins(&self, other: &Self::Fork) -> bool {
        self.stores.len() == other.1.len()
    }
    fn lds_peer_stores(&mut self, slots: &[usize]) -> Result<(), String> {
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if slot.1 != Slot::Free {
                return Err(format!("{} cannot be stored by other waves while {:?}", slot.0, slot.1));
            }
            slot.1 = Slot::Publishing;
        }
        Ok(())
    }
    fn exec_from_scc(&mut self) -> Result<(), String> {
        self.emit("s_cselect_b32 exec_lo, -1, 0");
        Ok(())
    }
    fn exec_all(&mut self) -> Result<(), String> {
        self.emit("s_mov_b32 exec_lo, -1");
        Ok(())
    }
    fn loop_(&mut self, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String> {
        let entry = self.stores.len();
        let mut first = self.clone();
        first.label(head)?;
        let start = first.text.len();
        body(&mut first)?;
        if first.stores.len() != entry {
            return Err("loop back edge leaves a different number of LDS stores pending than its entry".into());
        }
        let mut second = first.clone();
        second.text.truncate(start - 1);
        second.label(head)?;
        body(&mut second)?;
        if second.text[start..] != first.text[start..] {
            return Err("loop body did not reach a fixed point".into());
        }
        *self = first;
        Ok(())
    }
}
