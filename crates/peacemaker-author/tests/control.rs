//! Wave-role handoff and two-way branches on the reference backend.
use peacemaker_author::{trace::Trace, Gfx1151, Workgroup};

enum Gate {}

/// Writers drain before their barrier and leave; readers resume from the
/// branch point, meet them at their own barrier and may then read.
#[test]
fn handoff_publishes_to_the_reading_waves() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into()).unwrap();
    let gate = wg
        .handoff(readers, ".Lup", ".Lend", gate, |w, gate| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()))
        .unwrap();
    wg.ds_load(&gate, "ds_load_b128 v[8:11], v1".into()).unwrap();
    assert_eq!(
        b.text,
        [
            "s_cmp_ge_u32 s4, 2",
            "s_cbranch_scc1 .Lup",
            "ds_store_b128 v1, v[2:5]",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "s_branch .Lend",
            ".Lup:",
            "s_barrier",
            "ds_load_b128 v[8:11], v1",
        ]
    );
}

/// The arms of `if_else` must reach the join with the same pending stores.
#[test]
fn if_else_refuses_arms_that_join_with_different_waits() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let odd = wg.scmp("s_bitcmp1_b32 s19, 7".into()).unwrap();
    let err = wg
        .if_else(odd, ".Lodd", ".Ljoin", |w| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop), |_| Ok(()))
        .unwrap_err();
    assert!(err.contains(".Ljoin"), "{err}");
}
