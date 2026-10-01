//! `Builder` as the lowering backend of the typed core
//! (`peacemaker_author::Workgroup<T, Builder>`). Every request maps onto the
//! builder call a hand-written kernel would make, in the same order, so a
//! ported kernel emits the same bytes and the same proof; the builder's
//! ledger, slot machine, hazards and loop fixpoint stay the second evaluator.
use crate::{Builder, MemoryScope, insn::Instruction, ledger::Counter};
use peacemaker_author::{Backend, EventId, LdsCounter, SlotTransition};

impl Backend for Builder {
    type Insn = Instruction;
    fn arch_name(&self) -> &'static str { self.spec.arch.name() }
    fn seal(&mut self) { self.typed = true }
    fn position(&self) -> usize { self.program.instructions.len() }
    fn lds_slot(&mut self, name: &str, base: u32, len: u32) -> Result<usize, String> { self.lds.add(name, base, len) }
    fn lds_relayout(&mut self) -> Result<(), String> { self.lds.relayout() }
    fn ds_store(&mut self, slots: &[usize], insn: Instruction) -> Result<EventId, String> {
        let (&first, rest) = slots.split_first().ok_or("LDS store without a slot")?;
        let old = self.lds.clone();
        // Every further slot the instruction writes takes its Publishing step first.
        let result = rest.iter().try_for_each(|&slot| self.lds.store(slot)).and_then(|()| self.ds_store_slot(first, insn));
        if result.is_err() { self.lds = old }
        result?;
        self.ledger.last_id().map(EventId).ok_or_else(|| "LDS store left no ledger entry".into())
    }
    fn ds_load(&mut self, slots: &[usize], insn: Instruction) -> Result<(), String> {
        let (&first, rest) = slots.split_first().ok_or("LDS load without a slot")?;
        let old = self.lds.clone();
        let result = rest.iter().try_for_each(|&slot| self.lds.load(slot)).and_then(|()| self.ds_load_slot(first, insn));
        if result.is_err() { self.lds = old }
        result
    }
    fn lds_store_pending(&self, event: EventId) -> bool { self.ledger.is_pending(event.0) }
    fn drain_lds_stores(&mut self, counter: LdsCounter) -> Result<(), String> {
        // The drain `barrier` inserted before the typed core owned it.
        self.wait(match counter { LdsCounter::Lgkm => Counter::Lgkm, LdsCounter::Ds => Counter::Ds }, 0)
    }
    fn barrier(&mut self, transitions: &[SlotTransition]) -> Result<(), String> { self.full_barrier(transitions, MemoryScope::LdsOnly) }
    fn barrier_signal(&mut self, transitions: &[SlotTransition]) -> Result<(), String> { self.signal(transitions) }
    fn barrier_wait(&mut self) -> Result<(), String> { self.arrive() }
    fn label(&mut self, name: &str) -> Result<(), String> { Builder::label(self, name) }
    fn scalar_compare(&mut self, insn: Instruction) -> Result<(), String> {
        let m = insn.mnemonic();
        if !(m.starts_with("s_cmp") || m.starts_with("s_bitcmp")) { return Err(format!("{m} does not define SCC")) }
        self.push(insn)
    }
    fn branch_scc1(&mut self, target: &str) -> Result<(), String> { self.push(Instruction::new(format!("s_cbranch_scc1 {target}"), vec![], vec![])) }
    fn exec_from_scc(&mut self) -> Result<(), String> { self.push(Instruction::new("s_cselect_b32 exec_lo, -1, 0", vec![], vec![])) }
    fn exec_all(&mut self) -> Result<(), String> { self.push(Instruction::new("s_mov_b32 exec_lo, -1", vec![], vec![])) }
    fn loop_(&mut self, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String> { self.fixpoint_loop(head, body) }
}
