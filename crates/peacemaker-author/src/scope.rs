//! Execution scopes, condition handles and loops (design §4.6-§4.7).
//!
//! - `Workgroup`: workgroup-uniform control. Barriers, LDS layout,
//!   `loop_carried`, `wg_skip_if`, kernel exits and the wave-role `handoff`
//!   live here. It dereferences to `Wave`.
//! - `Wave`: one wave's program. LDS stores/loads, waits, raw ISA, and
//!   wave-uniform scopes (`skip_if`, `exec_if`, `if_else`, `loop_until`)
//!   whose bodies see only a `Wave`, so a barrier under wave- or
//!   lane-dependent control does not compile.
//!
//! A `Workgroup` owns its backend for one session (`backend::Auth`): the
//! backend refuses a second `Workgroup::new`, so a wave scope cannot
//! rebuild a workgroup scope from `Wave::isa`, and raw access through
//! `isa` cannot reach the authorized LDS, barrier, branch and loop entry
//! points.
//!
//! Every place two control paths meet (a skip target, an `if_else` join)
//! joins the backend state of every path into it
//! (`Backend::join`): waits and hazard guards hold on each path, and paths
//! that disagree on LDS ownership are refused.
//!
//! Conditions are affine handles: `Uniform<Scc>` (wave-uniform, from any
//! scalar compare) and `WgUniform<Scc>` (workgroup-uniform). A handle is
//! consumed by exactly one branch and refused if SCC was redefined since
//! its compare. In M0 register values are pinned (hand register plans keep
//! byte identity), so the workgroup uniformity of a compare's operands is
//! the caller's claim at `Workgroup::scmp_wg_uniform`, not a derivation.

use crate::backend::{Auth, Backend, SlotTransition};
use crate::lds::{AllFree, LdsRegion, Lowering, Published, Ring, State, StoreTarget, Transitions, Free, Slots, Writing, ready};
use crate::target::{SplitBarrier, Target, WaitModel};
use crate::wait::{Drained, Event, LdsWrite, Pending, Pendings};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// The scalar condition code.
pub enum Scc {}

/// A wave-uniform condition.
#[must_use = "a condition must be consumed by a branch"]
pub struct Uniform<C> {
    at: usize,
    _p: PhantomData<fn() -> C>,
}
/// A workgroup-uniform condition: every wave of the workgroup branches the
/// same way, so the branch may skip or repeat a barrier.
#[must_use = "a condition must be consumed by a branch"]
pub struct WgUniform<C> {
    at: usize,
    _p: PhantomData<fn() -> C>,
}
/// The exit of a `Wave::loop_until` loop.
pub struct LoopExit {
    label: String,
}
/// A kernel exit: its label followed by `s_endpgm`. `Workgroup::exit_if`
/// branches to it; `Workgroup::end` places it, after which nothing more may
/// be emitted, and the backend refuses to finish a kernel that never placed
/// it.
#[must_use = "a kernel exit must be placed with `Workgroup::end`"]
pub struct End {
    label: String,
}
impl End {
    pub fn label(&self) -> &str {
        &self.label
    }
}
fn fresh<B: Backend>(b: &B, at: usize) -> Result<(), String> {
    if b.position() != at {
        return Err("SCC was redefined between its compare and the branch that consumes it".into());
    }
    Ok(())
}

mod sealed {
    pub trait Sealed {}
    /// Only this crate can respawn carried state.
    pub struct Token;
}
use sealed::Token;

/// State carried around a loop or across a join. The loop body is emitted
/// twice (the backend's fixpoint check), so the core respawns the
/// back-edge state for the second emission; nothing else can duplicate a
/// token or region.
pub trait Carried: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn respawn(&self, token: &Token) -> Self;
}
impl sealed::Sealed for () {}
impl Carried for () {
    fn respawn(&self, _: &Token) -> Self {}
}
impl<R, S: State> sealed::Sealed for LdsRegion<R, S> {}
impl<R, S: State> Carried for LdsRegion<R, S> {
    fn respawn(&self, _: &Token) -> Self {
        LdsRegion::new(self.slots, self.last)
    }
}
impl<R, C: State, N: State> sealed::Sealed for Ring<R, C, N> {}
impl<R, C: State, N: State> Carried for Ring<R, C, N> {
    fn respawn(&self, _: &Token) -> Self {
        self.duplicate()
    }
}
impl<W: WaitModel, E: Event> sealed::Sealed for Pending<W, E> {}
impl<W: WaitModel, E: Event> Carried for Pending<W, E> {
    fn respawn(&self, _: &Token) -> Self {
        Pending::new(self.last)
    }
}
impl<E: Event> sealed::Sealed for Drained<E> {}
impl<E: Event> Carried for Drained<E> {
    fn respawn(&self, _: &Token) -> Self {
        Drained::new(self.last)
    }
}
macro_rules! carried_tuple {
    ($($t:ident),+) => {
        impl<$($t: Carried),+> sealed::Sealed for ($($t,)+) {}
        impl<$($t: Carried),+> Carried for ($($t,)+) {
            #[allow(non_snake_case)]
            fn respawn(&self, token: &Token) -> Self {
                let ($($t,)+) = self;
                ($($t.respawn(token),)+)
            }
        }
    };
}
carried_tuple!(A);
carried_tuple!(A, B);
carried_tuple!(A, B, C);
carried_tuple!(A, B, C, D);
carried_tuple!(A, B, C, D, E);
carried_tuple!(A, B, C, D, E, F);
carried_tuple!(A, B, C, D, E, F, G);
carried_tuple!(A, B, C, D, E, F, G, H);

/// One wave of a kernel on target `T`, lowered by backend `B`.
pub struct Wave<'b, T: Target, B: Backend> {
    b: &'b mut B,
    auth: Auth,
    _t: PhantomData<T>,
}

/// Workgroup-uniform scope of a kernel; dereferences to its `Wave`.
pub struct Workgroup<'b, T: Target, B: Backend> {
    wave: Wave<'b, T, B>,
}
impl<'b, T: Target, B: Backend> Deref for Workgroup<'b, T, B> {
    type Target = Wave<'b, T, B>;
    fn deref(&self) -> &Self::Target {
        &self.wave
    }
}
impl<T: Target, B: Backend> DerefMut for Workgroup<'_, T, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wave
    }
}

/// A split barrier in flight: its regions are unreachable until
/// `Workgroup::wait_arrived` hands them back in their new states.
#[must_use = "a signalled barrier must be waited"]
pub struct Arrived<X> {
    x: X,
}

impl<'b, T: Target, B: Backend> Wave<'b, T, B> {
    /// The backend for raw instructions (ALU, VMEM, SMEM, crossbar). Its
    /// LDS, barrier, branch and loop entry points need an `Auth` this scope
    /// never hands out, and a `Workgroup` cannot be rebuilt over it.
    pub fn isa(&mut self) -> &mut B {
        self.b
    }
    /// A `Free` region or ring buffer as `Writing` with an empty token, so a
    /// sequence of stores can be folded by passing the pair back.
    pub fn begin_write<X: StoreTarget<T::Waits>>(&mut self, x: X) -> X::Out {
        x.begin()
    }
    /// One LDS store into `x`; returns the writing state and its token.
    pub fn ds_store<X: StoreTarget<T::Waits>>(&mut self, x: X, insn: B::Insn) -> Result<X::Out, String> {
        let e = self.b.ds_store(&self.auth, x.slots().ids(), insn)?;
        Ok(x.stored(e))
    }
    /// One LDS store covering two regions (one instruction, two slots).
    pub fn ds_store2<X: StoreTarget<T::Waits>, Y: StoreTarget<T::Waits>>(
        &mut self,
        x: X,
        y: Y,
        insn: B::Insn,
    ) -> Result<(X::Out, Y::Out), String> {
        let (sx, sy) = (x.slots(), y.slots());
        let mut ids = [0usize; 8];
        let n = sx.ids().len() + sy.ids().len();
        ids[..sx.ids().len()].copy_from_slice(sx.ids());
        ids[sx.ids().len()..n].copy_from_slice(sy.ids());
        let e = self.b.ds_store(&self.auth, &ids[..n], insn)?;
        Ok((x.stored(e), y.stored(e)))
    }
    /// One LDS load from a published region. Its destination registers
    /// are awaited by the backend on first use.
    pub fn ds_load<R>(&mut self, r: &LdsRegion<R, Published>, insn: B::Insn) -> Result<(), String> {
        self.b.ds_load(&self.auth, r.slots.ids(), insn)
    }
    /// One LDS load from a ring's current buffer.
    pub fn ds_load_cur<R, N: State>(&mut self, r: &Ring<R, Published, N>, insn: B::Insn) -> Result<(), String> {
        let slots: Slots = r.cur_slots().ok_or("read of a ring buffer no barrier has published")?;
        self.b.ds_load(&self.auth, slots.ids(), insn)
    }
    /// Wait for a token's operations; emits the counter wait the backend
    /// still needs (none when an earlier wait retired them).
    pub fn wait<E: Event>(&mut self, p: Pending<T::Waits, E>) -> Result<Drained<E>, String> {
        if p.last.is_some_and(|e| self.b.lds_store_pending(e)) {
            self.b.drain_lds_stores(&self.auth, <T::Waits as WaitModel>::LDS)?;
        }
        Ok(Drained::new(p.last))
    }
    /// Wait for several tokens with one counter wait.
    pub fn wait_all<P: Pendings<T::Waits>>(&mut self, p: P) -> Result<P::Drained, String> {
        let mut pending = false;
        let b = &*self.b;
        p.each_event(&mut |e| pending |= b.lds_store_pending(e));
        if pending {
            self.b.drain_lds_stores(&self.auth, <T::Waits as WaitModel>::LDS)?;
        }
        Ok(p.drained())
    }
    /// A scalar compare (`s_cmp*`, `s_bitcmp*`): SCC is wave-uniform.
    pub fn scmp(&mut self, insn: B::Insn) -> Result<Uniform<Scc>, String> {
        self.b.scalar_compare(&self.auth, insn)?;
        Ok(Uniform { at: self.b.position(), _p: PhantomData })
    }
    /// Branch over `body` to `target` when `cond` holds. Both paths reach
    /// `target` with `state`'s type, and the backend joins the branch
    /// point's state into the body's; the body cannot reach a barrier.
    pub fn skip_if<S>(
        &mut self,
        cond: Uniform<Scc>,
        target: &str,
        state: S,
        body: impl FnOnce(&mut Self, S) -> Result<S, String>,
    ) -> Result<S, String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, target)?;
        let skipped = self.b.fork();
        let state = body(self, state)?;
        self.b.join(&self.auth, skipped)?;
        self.b.label(&self.auth, target)?;
        Ok(state)
    }
    /// Run `body` with EXEC = all lanes if `cond`, else none, then restore
    /// EXEC. Every instruction still issues, so state flows straight
    /// through; the body cannot reach a barrier.
    pub fn exec_if<R>(&mut self, cond: Uniform<Scc>, body: impl FnOnce(&mut Self) -> Result<R, String>) -> Result<R, String> {
        fresh(self.b, cond.at)?;
        self.b.exec_from_scc(&self.auth)?;
        let out = body(self)?;
        self.b.exec_all(&self.auth)?;
        Ok(out)
    }
    /// A loop at `head` whose back edge is `s_branch head`; it is left only
    /// by `break_if` branches to `exit`, which labels the code after it. The
    /// body is wave scope (no barrier) and carries no typed state; the
    /// backend checks its wait state reaches a fixed point.
    pub fn loop_until(
        &mut self,
        head: &str,
        exit: &str,
        body: impl for<'x> Fn(&mut Wave<'x, T, B>, &LoopExit) -> Result<(), String>,
    ) -> Result<(), String> {
        let x = LoopExit { label: exit.into() };
        let auth = &self.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            let mut w = Wave { b, auth: auth.reenter(), _t: PhantomData };
            body(&mut w, &x)?;
            w.b.branch(&w.auth, head)
        };
        self.b.loop_(&self.auth, head, &emit)?;
        self.b.label(&self.auth, exit)
    }
    /// Leave the enclosing `loop_until` when `cond` holds.
    pub fn break_if(&mut self, cond: Uniform<Scc>, exit: &LoopExit) -> Result<(), String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, &exit.label)
    }
    /// Two-way branch: when `cond` holds, jump to `else_label` and run
    /// `els` from the state of the branch point; otherwise run `then`,
    /// which ends with `s_branch join`. The caller places `join` right
    /// after (the else arm falls through to it). The backend joins the
    /// arms' states there; neither arm can reach a barrier.
    pub fn if_else(
        &mut self,
        cond: Uniform<Scc>,
        else_label: &str,
        join: &str,
        then: impl FnOnce(&mut Self) -> Result<(), String>,
        els: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, else_label)?;
        let at = self.b.fork();
        then(self)?;
        self.b.branch(&self.auth, join)?;
        let then_end = self.b.fork();
        self.b.resume(&self.auth, at)?;
        self.b.label(&self.auth, else_label)?;
        els(self)?;
        self.b.join(&self.auth, then_end).map_err(|e| format!("the arms cannot join at {join}: {e}"))
    }
    /// A label (branch target) in straight-line code.
    pub fn label(&mut self, name: &str) -> Result<(), String> {
        self.b.label(&self.auth, name)
    }
}

impl<'b, T: Target, B: Backend> Workgroup<'b, T, B> {
    /// Take ownership of a backend built for `T`. From here on the
    /// backend's untyped LDS, barrier and loop entry points refuse, and so
    /// does a second `Workgroup::new` over it.
    pub fn new(b: &'b mut B) -> Result<Self, String> {
        if b.arch_name() != T::NAME {
            return Err(format!("backend targets {}, kernel is typed for {}", b.arch_name(), T::NAME));
        }
        let auth = Auth::mint();
        b.seal(&auth)?;
        Ok(Self { wave: Wave { b, auth, _t: PhantomData } })
    }
    /// Declare one LDS region (one backend slot; ids follow declaration order).
    pub fn lds<R: 'static>(&mut self, name: &str, base: u32, len: u32) -> Result<LdsRegion<R, Free>, String> {
        if base.checked_add(len).is_none_or(|end| end > T::LDS_BYTES) {
            return Err(format!("LDS region {name} [{base}, +{len}) exceeds {} bytes on {}", T::LDS_BYTES, T::NAME));
        }
        Ok(LdsRegion::new(Slots::one(self.wave.b.lds_slot(&self.wave.auth, name, base, len)?), None))
    }
    /// End the slot layout so a later phase can carve its own. Takes every
    /// region (all `Free`); the backend refuses if any slot is still live.
    pub fn relayout<X: AllFree>(&mut self, regions: X) -> Result<(), String> {
        drop(regions);
        self.wave.b.lds_relayout(&self.wave.auth)
    }
    fn lower<X: Transitions>(x: &X) -> Result<Lowering, String> {
        let mut l = Lowering::new();
        x.lower(&mut l)?;
        Ok(l)
    }
    /// One workgroup barrier carrying the transitions `x`.
    pub fn barrier<X: Transitions>(&mut self, x: X) -> Result<X::After, String> {
        let l = Self::lower(&x)?;
        self.wave.b.barrier(&self.wave.auth, l.list())?;
        Ok(x.after())
    }
    /// Split barrier, first half: this wave has arrived; its regions are
    /// held until `wait_arrived`.
    pub fn signal<X: Transitions>(&mut self, x: X) -> Result<Arrived<X>, String>
    where
        T: SplitBarrier,
    {
        let l = Self::lower(&x)?;
        self.wave.b.barrier_signal(&self.wave.auth, l.list())?;
        Ok(Arrived { x })
    }
    /// Split barrier, second half.
    pub fn wait_arrived<X: Transitions>(&mut self, a: Arrived<X>) -> Result<X::After, String>
    where
        T: SplitBarrier,
    {
        self.wave.b.barrier_wait(&self.wave.auth)?;
        Ok(a.x.after())
    }
    /// A scalar compare whose operands the caller asserts are
    /// workgroup-uniform (kernel arguments, workgroup ids, constants and
    /// counters derived only from them).
    pub fn scmp_wg_uniform(&mut self, insn: B::Insn) -> Result<WgUniform<Scc>, String> {
        self.wave.b.scalar_compare(&self.wave.auth, insn)?;
        Ok(WgUniform { at: self.wave.b.position(), _p: PhantomData })
    }
    /// Branch over `body` (which may hold barriers) to `target` when
    /// `cond` holds; both paths reach `target` with `state`'s type, and the
    /// backend joins the branch point's state into the body's.
    pub fn wg_skip_if<S>(
        &mut self,
        cond: WgUniform<Scc>,
        target: &str,
        state: S,
        body: impl FnOnce(&mut Self, S) -> Result<S, String>,
    ) -> Result<S, String> {
        fresh(self.wave.b, cond.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, target)?;
        let skipped = self.wave.b.fork();
        let state = body(self, state)?;
        self.wave.b.join(&self.wave.auth, skipped)?;
        self.wave.b.label(&self.wave.auth, target)?;
        Ok(state)
    }
    /// Reserve the kernel exit `label` (placed later by `end`).
    pub fn exit(&mut self, label: &str) -> Result<End, String> {
        self.wave.b.reserve_exit(&self.wave.auth, label)?;
        Ok(End { label: label.into() })
    }
    /// Leave the kernel when `cond` holds: skipping every later barrier is
    /// legal only for the whole workgroup, and only to the kernel's exit.
    pub fn exit_if(&mut self, cond: WgUniform<Scc>, end: &End) -> Result<(), String> {
        fresh(self.wave.b, cond.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, &end.label)
    }
    /// Place the kernel exit (`end:` then `s_endpgm`). Nothing may follow.
    pub fn end(&mut self, end: End) -> Result<(), String> {
        self.wave.b.end_program(&self.wave.auth, &end.label)
    }
    /// A loop at `head`: `body` receives the carried state and returns the
    /// same type plus the workgroup-uniform condition of its back edge
    /// (`s_cbranch_scc1 head`). Pending tokens and region states are part
    /// of `S`, so a store that reaches the loop head undrained, where a
    /// barrier needs `Drained`, is a type error. The backend still checks
    /// the exact wait state reaches a fixed point; the loop's only exit is
    /// the fall-through of its back edge.
    pub fn loop_carried<S: Carried>(
        &mut self,
        head: &str,
        state: S,
        body: impl for<'x> Fn(&mut Workgroup<'x, T, B>, S) -> Result<(S, WgUniform<Scc>), String>,
    ) -> Result<S, String> {
        let input = RefCell::new(Some(state));
        let first: RefCell<Option<S>> = RefCell::new(None);
        let auth = &self.wave.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            let entry = input.borrow_mut().take();
            let s = match entry {
                Some(s) => s,
                None => first.borrow().as_ref().ok_or("loop re-emitted before its first emission finished")?.respawn(&Token),
            };
            let mut wg = Workgroup { wave: Wave { b, auth: auth.reenter(), _t: PhantomData } };
            let (out, back) = body(&mut wg, s)?;
            fresh(wg.wave.b, back.at)?;
            wg.wave.b.branch_scc1(&wg.wave.auth, head)?;
            let mut f = first.borrow_mut();
            if f.is_none() {
                *f = Some(out);
            }
            Ok(())
        };
        self.wave.b.loop_(&self.wave.auth, head, &emit)?;
        first.into_inner().ok_or_else(|| format!("loop {head} emitted no body"))
    }
    /// Wave-role LDS handoff. Waves where `readers` holds branch to
    /// `reader_label`; the others (writers) run `write`, publish `region`
    /// at one barrier and leave the kernel at `end`. Readers resume from
    /// the state of the branch, meet the writers at a barrier of their own
    /// and continue with the region `Published`. Each wave crosses exactly
    /// one barrier here and the writers cross none later, so the
    /// wave-dependent branch over a barrier is sound.
    pub fn handoff<R: 'static>(
        &mut self,
        readers: Uniform<Scc>,
        reader_label: &str,
        end: &str,
        region: LdsRegion<R, Free>,
        write: impl FnOnce(&mut Wave<'_, T, B>, LdsRegion<R, Free>) -> Result<(LdsRegion<R, Writing>, Pending<T::Waits, LdsWrite<R>>), String>,
    ) -> Result<LdsRegion<R, Published>, String> {
        fresh(self.wave.b, readers.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, reader_label)?;
        let at = self.wave.b.fork();
        let slots = region.slots;
        let (written, pending) = write(&mut self.wave, region)?;
        let drained = self.wave.wait(pending)?;
        let _writers_leave = self.barrier((ready(written, drained),))?;
        self.wave.b.branch(&self.wave.auth, end)?;
        self.wave.b.resume(&self.wave.auth, at)?;
        self.wave.b.label(&self.wave.auth, reader_label)?;
        self.wave.b.lds_peer_stores(&self.wave.auth, slots.ids())?;
        let ts: Vec<SlotTransition> = slots.ids().iter().map(|&id| SlotTransition::Ready(id)).collect();
        self.wave.b.barrier(&self.wave.auth, &ts)?;
        Ok(LdsRegion::new(slots, None))
    }
}
