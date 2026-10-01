//! Scope escapes as rustc errors: the lowering seam without an `Auth`, and a
//! forged or minted `Auth`. Rebuilding a `Workgroup` from wave scope, which
//! the types cannot see, is refused at run time (`tests/control.rs`).
#[test]
fn scope_escapes_are_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/backend_seam_in_wave_scope.rs");
    t.compile_fail("tests/ui/forged_auth.rs");
    t.compile_fail("tests/ui/minted_auth.rs");
}
