//! Kernel exits, skip joins and scope ownership on the reference backend.
use peacemaker_author::{ready, trace::Trace, Gfx1100, Gfx1151, Workgroup};

enum Gate {}

/// A kernel exit is placed only by `Workgroup::end`, and a kernel that never
/// places it does not finish.
#[test]
fn exits_are_placed_by_end_only() {
    let mut b = Trace::new("gfx1100");
    let mut wg = Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let outside = wg.scmp_wg_uniform("s_cmp_ge_u32 s2, s3".into()).unwrap();
    wg.exit_if(outside, &end).unwrap();
    assert!(wg.label(".Lend").is_err());
    drop(end);
    assert!(b.finish().is_err());
}

/// The reviewer's escalation counterexample: a wave-only body rebuilding a
/// `Workgroup` over the backend it reaches through `isa` (to place a
/// barrier under wave-dependent control) is refused.
#[test]
fn a_wave_scope_cannot_rebuild_its_workgroup() {
    let mut b = Trace::new("gfx1100");
    let mut wg = Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap();
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into()).unwrap();
    let err = wg
        .skip_if(first_wave, ".Lskip", (), |w, ()| Workgroup::<Gfx1100, Trace>::new(w.isa()).map(drop))
        .unwrap_err();
    assert!(err.contains("already owned"), "{err}");
    assert_eq!(b.count("s_barrier"), 0);
}

/// A store the skipped body made stays pending past the skip target: the
/// continuation cannot re-carve the LDS as if the body never ran.
#[test]
fn skip_target_joins_a_store_of_the_body() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let odd = wg.scmp("s_bitcmp1_b32 s19, 7".into()).unwrap();
    wg.skip_if(odd, ".Lskip", (), |w, ()| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop)).unwrap();
    let err = wg.relayout((other,)).unwrap_err();
    assert!(err.contains("gate is Publishing"), "{err}");
}

/// Paths that disagree on LDS ownership (published on one, untouched on the
/// other) do not join.
#[test]
fn skip_join_refuses_disagreeing_lds_ownership() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let skip = wg.scmp_wg_uniform("s_cmp_eq_u32 s2, 0".into()).unwrap();
    let err = wg
        .wg_skip_if(skip, ".Lskip", (), |wg, ()| {
            let (gate, pending) = wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
            let drained = wg.wait(pending)?;
            wg.barrier((ready(gate, drained),)).map(drop)
        })
        .unwrap_err();
    assert!(err.contains("gate Published on one and Free on the other"), "{err}");
}

/// Raw access through `isa` cannot emit what the typed core owns.
#[test]
fn raw_trace_refuses_lds_barriers_and_branches() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    for t in ["ds_store_b32 v1, v2", "s_barrier", "s_branch .Lx", "s_cbranch_scc1 .Lx", "s_endpgm"] {
        assert!(wg.isa().raw(t).is_err(), "{t}");
    }
    wg.isa().raw("v_mov_b32 v1, 0").unwrap();
}

/// `Forward` targets are placed once, after every branch to them, and
/// inside their block: no branch re-enters code.
#[test]
fn forward_branches_only_go_forward() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let err = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Lx")?;
            f.place(w, ".Lx")?;
            f.goto(w, ".Lx")
        })
        .unwrap_err();
    assert!(err.contains("branches only go forward"), "{err}");
    let err = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Ly")
        })
        .unwrap_err();
    assert!(err.contains(".Ly is never placed"), "{err}");
}

/// A target joins every branch to it: a slot stored only on the path of
/// the middle branch is still being written at the target.
#[test]
fn forward_target_joins_a_store_of_any_branch() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let _gate = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Lout")?;
            let (gate, pending) = w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
            let c = w.scmp("s_cmp_eq_u32 s2, 1".into())?;
            f.branch_if(w, c, ".Lout")?;
            let _drained = w.wait(pending)?;
            f.place(w, ".Lout")?;
            Ok(gate)
        })
        .unwrap();
    assert!(wg.relayout((other,)).unwrap_err().contains("gate is Publishing"));
}

/// `exit_unless` continues at its target from the branch point: the
/// body's store leaves the kernel with it. An exit something branches to
/// takes no tail.
#[test]
fn exit_unless_target_starts_from_the_branch() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let c = wg.scmp_wg_uniform("s_cmp_eq_u32 s2, 0".into()).unwrap();
    wg.exit_unless(c, ".Lgo", &end, |wg| wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop)).unwrap();
    // Every slot is Free at the target: the path that stored left the kernel.
    wg.relayout((other,)).unwrap();
    let err = wg.end_with(end, |w| w.isa().raw("v_mov_b32 v1, 0")).unwrap_err();
    assert!(err.contains("takes no tail"), "{err}");
    assert_eq!(b.text[..4], ["s_cmp_eq_u32 s2, 0", "s_cbranch_scc1 .Lgo", "ds_store_b128 v1, v[2:5]", "s_branch .Lend"]);
}
