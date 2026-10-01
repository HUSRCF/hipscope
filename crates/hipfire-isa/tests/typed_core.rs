//! The typed core over the builder: a publish/read/rotate double buffer
//! written against `peacemaker_author` emits exactly what the same program
//! emits through the untyped builder (instructions, waits and barrier
//! transitions), and a sealed builder refuses untyped LDS and barrier calls
//! and a second `Workgroup`. A skip target keeps the waits of the path that
//! skipped.
use hipfire_isa::{Arch, Builder, KernelSpec, KernargLayout, RegPlan};
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::lds::Transition;
use hipfire_isa::reg::{Kind, Live, RegRef};
use peacemaker_author::{Gfx1100, Gfx1201, Ring, Target, Workgroup, prime, rotate};

fn probe(arch: Arch) -> Builder {
    let mut plan = RegPlan::new(16, 8).unwrap();
    for (name, n) in [("a", 0), ("b", 1), ("addr", 2), ("c", 3)] { plan.v::<1>(name, n, Live::Whole).unwrap(); }
    for (name, n) in [("x", 5), ("y", 6)] { plan.s::<1>(name, n, Live::Whole).unwrap(); }
    Builder::new(KernelSpec {
        kernel_id: "probe".into(), variant: "default".into(), arch, symbol: "probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 64, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, plan)
}
fn v(n: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len: 1 } }
fn store(o: u32) -> Instruction { Instruction::new(format!("ds_store_b32 v2, v0 offset:{o}"), vec![], vec![v(2), v(0)]).memory(MemoryClass::DsStore) }
fn load(o: u32) -> Instruction { Instruction::new(format!("ds_load_b32 v1, v2 offset:{o}"), vec![v(1)], vec![v(2)]).memory(MemoryClass::DsLoad) }
fn mul() -> Instruction { Instruction::new("v_mul_f32_e32 v3, v1, v1", vec![v(3)], vec![v(1)]) }
/// `v1 = global[v2]`: pending on VMcnt/LOADcnt until awaited.
fn gload() -> Instruction { Instruction::new("global_load_b32 v1, v2, s[0:1]", vec![v(1)], vec![v(2)]).memory(MemoryClass::VmemLoad) }
fn cmp(text: &str) -> Instruction { Instruction::new(text, vec![], vec![]) }
/// The instructions after label `l`.
fn after(b: &Builder, l: &str) -> Vec<String> {
    let t = text(b);
    t[t.iter().position(|i| *i == format!("{l}:")).unwrap() + 1..].to_vec()
}

/// Two stores into buffer 0, publish; read it while staging buffer 1; rotate.
fn untyped(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let s0 = b.lds.add("B0", 0, 256).unwrap();
    let s1 = b.lds.add("B1", 256, 256).unwrap();
    b.ds_store(s0, store(0)).unwrap();
    b.ds_store(s0, store(4)).unwrap();
    b.barrier(&[Transition::Ready(s0)]).unwrap();
    b.ds_load(s0, load(0)).unwrap();
    b.ds_store(s1, store(256)).unwrap();
    b.push(mul()).unwrap();
    b.barrier(&[Transition::Retire(s0), Transition::Ready(s1)]).unwrap();
    b.ds_load(s1, load(256)).unwrap();
    b
}
enum Buf {}
fn typed<T: Target>(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    let (b0, b1) = (wg.lds::<Buf>("B0", 0, 256).unwrap(), wg.lds::<Buf>("B1", 256, 256).unwrap());
    let st = wg.ds_store(Ring::new(b0, b1), store(0)).unwrap();
    let (ring, pending) = wg.ds_store(st, store(4)).unwrap();
    let drained = wg.wait(pending).unwrap();
    let (ring,) = wg.barrier((prime(ring, drained),)).unwrap();
    wg.ds_load_cur(&ring, load(0)).unwrap();
    let (ring, pending) = wg.ds_store(ring, store(256)).unwrap();
    wg.isa().push(mul()).unwrap();
    let drained = wg.wait(pending).unwrap();
    let (ring,) = wg.barrier((rotate(ring, drained),)).unwrap();
    wg.ds_load_cur(&ring, load(256)).unwrap();
    b
}
fn text(b: &Builder) -> Vec<String> { b.program.instructions.iter().map(|i| i.text.clone()).collect() }

#[test]
fn typed_double_buffer_emits_the_untyped_program() {
    for (arch, t) in [(Arch::Gfx1100, typed::<Gfx1100>(Arch::Gfx1100)), (Arch::Gfx1201, typed::<Gfx1201>(Arch::Gfx1201))] {
        let u = untyped(arch);
        assert_eq!(text(&t), text(&u), "{arch:?}");
        let tr = |b: &Builder| b.barriers.iter().map(|p| (p.pc_index, p.transitions.clone())).collect::<Vec<_>>();
        assert_eq!(tr(&t), tr(&u), "{arch:?}");
        let waits = |b: &Builder| b.waits.iter().map(|w| (w.pc_index, w.insn.clone())).collect::<Vec<_>>();
        assert_eq!(waits(&t), waits(&u), "{arch:?}");
    }
    // The drains are real: one per publishing barrier.
    assert_eq!(text(&untyped(Arch::Gfx1100)).iter().filter(|t| *t == "s_waitcnt lgkmcnt(0)").count(), 2);
}

#[test]
fn sealed_builder_refuses_untyped_lds_barriers_and_loops() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds.add("S", 0, 256).unwrap();
    let wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    drop(wg);
    assert!(b.ds_store(slot, store(0)).is_err());
    assert!(b.barrier(&[]).is_err());
    assert!(b.loop_(".Lx", |_| Ok(())).is_err());
    // Nor does it take a second Workgroup.
    assert!(Workgroup::<Gfx1100, Builder>::new(&mut b).is_err());
    // A backend for another architecture is refused outright.
    let mut b = probe(Arch::Gfx1201);
    assert!(Workgroup::<Gfx1100, Builder>::new(&mut b).is_err());
}

/// The reviewer's escalation counterexample on the production backend: a
/// wave-only body rebuilding a `Workgroup` from `isa` to reach a barrier.
#[test]
fn a_wave_scope_cannot_rebuild_its_workgroup() {
    let mut b = probe(Arch::Gfx1100);
    let mut wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    let first_wave = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    let err = wg.skip_if(first_wave, ".Lskip", (), |w, ()| Workgroup::<Gfx1100, Builder>::new(w.isa()).map(drop)).unwrap_err();
    assert!(err.contains("already owned"), "{err}");
    assert!(b.barriers.is_empty());
}

/// A skip target joins the skipped path: a load the body awaited is still
/// pending on the path that skipped it.
#[test]
fn skip_target_keeps_the_skipped_paths_pending_load() {
    let mut b = probe(Arch::Gfx1100);
    let mut wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    wg.isa().push(gload()).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.skip_if(c, ".Lskip", (), |w, ()| w.isa().push(mul())).unwrap();
    wg.isa().push(mul()).unwrap();
    assert_eq!(after(&b, ".Lskip"), ["s_waitcnt vmcnt(0)", "v_mul_f32_e32 v3, v1, v1"]);
}

