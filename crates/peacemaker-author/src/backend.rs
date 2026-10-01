//! The lowering seam. The typed core decides *what* happens (which LDS
//! slots a store touches, which transitions a barrier carries, when a store
//! is drained); a backend turns each request into instructions and runs the
//! value-level checks behind it (wait ledger, LDS slot machine, hazards,
//! loop fixpoint). `hipfire-isa`'s `Builder` is the production backend;
//! `crate::trace::Trace` is a small reference backend for tests.

use crate::target::LdsCounter;

/// Build-time identity of one issued memory operation (a backend ledger id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventId(pub u64);

/// One LDS slot transition carried by a barrier (the builder's
/// `lds::Transition`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotTransition {
    /// Every wave finished reading the slot: it may be rewritten.
    Retire(usize),
    /// The slot's drained stores become visible to every wave.
    Ready(usize),
}

pub trait Backend: Sized {
    /// A raw, register-allocated instruction.
    type Insn;
    /// Architecture name, compared with `Target::NAME`.
    fn arch_name(&self) -> &'static str;
    /// The typed core owns LDS, barriers and loops from now on: the
    /// backend's own untyped entry points for them must refuse.
    fn seal(&mut self);
    /// Program position (instructions emitted so far, waits included).
    fn position(&self) -> usize;

    fn lds_slot(&mut self, name: &str, base: u32, len: u32) -> Result<usize, String>;
    /// End the slot layout; every slot must be Free.
    fn lds_relayout(&mut self) -> Result<(), String>;
    /// One DS store touching every slot in `slots`; returns its event.
    fn ds_store(&mut self, slots: &[usize], insn: Self::Insn) -> Result<EventId, String>;
    /// One DS load reading every slot in `slots`.
    fn ds_load(&mut self, slots: &[usize], insn: Self::Insn) -> Result<(), String>;
    /// The LDS store `event` has not been retired by any wait yet.
    fn lds_store_pending(&self, event: EventId) -> bool;
    /// Wait until every issued LDS store has completed (count 0: stores
    /// retire out of order with respect to the same counter's loads).
    fn drain_lds_stores(&mut self, counter: LdsCounter) -> Result<(), String>;

    /// Full barrier carrying `transitions`.
    fn barrier(&mut self, transitions: &[SlotTransition]) -> Result<(), String>;
    /// gfx12 split barrier: signal.
    fn barrier_signal(&mut self, transitions: &[SlotTransition]) -> Result<(), String>;
    /// gfx12 split barrier: wait for the signal in flight.
    fn barrier_wait(&mut self) -> Result<(), String>;

    fn label(&mut self, name: &str) -> Result<(), String>;
    /// Issue a scalar compare (`s_cmp*`, `s_bitcmp*`) that defines SCC.
    fn scalar_compare(&mut self, insn: Self::Insn) -> Result<(), String>;
    /// `s_cbranch_scc1 target`.
    fn branch_scc1(&mut self, target: &str) -> Result<(), String>;
    /// EXEC = SCC ? all lanes : none.
    fn exec_from_scc(&mut self) -> Result<(), String>;
    /// EXEC = all lanes.
    fn exec_all(&mut self) -> Result<(), String>;
    /// Emit `body` once as a loop at `head`; the backend verifies that its
    /// wait state reaches a fixed point across the back edge.
    fn loop_(&mut self, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String>;
}
