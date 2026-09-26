use smallvec::SmallVec;
use crate::{cfg::{BarrierKind, Cond}, lds::LdsAccess, reg::RegRef, wait::CounterSet};
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImplicitSet { pub reads: u8, pub writes: u8 }
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
