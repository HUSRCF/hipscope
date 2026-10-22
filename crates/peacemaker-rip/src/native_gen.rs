//! The production generators as natives (a child module of [`super`]).
//!
//! Two layers, both over `peacemaker_isa` and adding no typestate surface of
//! their own:
//!
//! - Coarse: `Iu4Gemm::emit(Iu4Spec { fold, tile, cacc, epi, act, arch })`,
//!   `Iu4Gemm::emit_module(fold, tile, cacc, act, arch)`,
//!   `MoeSym::emit(MoeSymSpec { arch, kind, nt, rr, codec })`,
//!   `MoeSym::emit_module(arch)` and `MoeSym::emit_module_codec(arch, codec)`
//!   return what the Rust functions return, and `Spec::validate` stays the
//!   gate. The `Spec` enums are enum values (`Fold::K128`, `Tile::T256x128x16`,
//!   `Cacc::Two`, `Epi::GateUpSilu`, `ALayout::Slab`, `WeightCodec::Mq3I4`,
//!   `MoeKind::GateUp`).
//! - Blocks: `Iu4Gen::new(spec)` and the `iu4::*` natives run the typed phase
//!   blocks of `iu4_gemm::blocks` inside one `Driver::typed` call each: the
//!   native takes the four LDS rings from the driver (`Lend::take`), runs the
//!   Rust block on the typed `Workgroup`, and gives them back
//!   (`Lend::give_back`) to the same driver slots, so the driver's own loop
//!   and branch rule covers script loops around blocks. `Lds` is the array of
//!   the four `RingId`s; blocks return it for single-assignment style.

use super::{
    arch_of, check_args, done, done_with, driver, nat, nat_of, ring_of, spec_value, str_of, u8_of,
    usize_of, Nat,
};
use crate::interp::{Cx, Interp, R};
use crate::value::Value;
use peacemaker_author::runtime::{Exit, Lend, RingId};
use peacemaker_author::{Free, Published};
use peacemaker_isa::kernels::iu4_gemm::{
    self as iu4, blocks, ALayout, Cacc, Epi, Fold, FreeRings, Gen, Lds, MetaD, MetaS, PayA, PayW,
    Tile, Wg,
};
use peacemaker_isa::kernels::qwen4_moe_sym as moe;
use peacemaker_isa::Emitted;
use std::cell::RefCell;
use std::rc::Rc;

type Res<T> = Result<T, String>;

const FNS: &[&str] = &[
    "Iu4Gemm::emit",
    "Iu4Gemm::emit_module",
    "MoeSym::emit",
    "MoeSym::emit_module",
    "MoeSym::emit_module_codec",
    "Iu4Gen::new",
    "iu4::declare_lds",
    "iu4::prologue",
    "iu4::kloop",
    "iu4::block",
    "iu4::block_stepping",
    "iu4::step_descriptor",
    "iu4::trip_end",
    "iu4::trips_cmp",
    "iu4::epilogue",
    "iu4::gdn_epilogue",
    "iu4::label",
    "iu4::tail_label",
];

pub(super) fn is_fn(name: &str) -> bool {
    FNS.contains(&name)
}

pub(super) fn is_type(n: &str) -> bool {
    matches!(
        n,
        "Fold"
            | "Tile"
            | "Cacc"
            | "Epi"
            | "ALayout"
            | "MoeKind"
            | "WeightCodec"
            | "Iu4Spec"
            | "MoeSymSpec"
            | "Iu4Gemm"
            | "MoeSym"
            | "Iu4Gen"
    )
}

pub(super) fn is_struct(n: &str) -> bool {
    matches!(n, "Iu4Spec" | "MoeSymSpec")
}

/// Enum values: `Value::Enum(type, variant, [])`, exactly like a script's own
/// unit enums.
const ENUMS: &[(&str, &[&str])] = &[
    ("Fold", &["K128", "K256Shared", "K256Pow2"]),
    ("Tile", &["T128x128x8", "T256x128x16"]),
    ("Cacc", &["One", "Two"]),
    (
        "Epi",
        &[
            "Set",
            "Add",
            "AddSigmoid",
            "GateUpSilu",
            "GateUpSiluBf16",
            "QkvzaGdn",
        ],
    ),
    ("ALayout", &["Token", "Slab"]),
    ("MoeKind", &["GateUp", "Down"]),
    ("WeightCodec", &["Mq4Sym", "Mq3I4", "E8pI4"]),
];

pub(super) fn assoc_const<'a>(ty: &str, item: &str) -> Option<Value<'a>> {
    if let Some((t, vs)) = ENUMS.iter().find(|(t, _)| *t == ty) {
        let v = vs.iter().find(|v| **v == item)?;
        return Some(Value::Enum(Rc::from(*t), Rc::from(*v), vec![]));
    }
    if ty == "iu4" {
        return Some(Value::str(match item {
            "ENTRY" => iu4::ENTRY,
            "K_BEGIN" => iu4::K_BEGIN,
            "K_LOOP" => iu4::K_LOOP,
            "K_LOOP_END" => iu4::K_LOOP_END,
            "EPI" => iu4::EPI,
            "END" => iu4::END,
            "GDN" => iu4::GDN,
            _ => return None,
        }));
    }
    None
}

// ---- Spec conversion ----------------------------------------------------------

fn variant_of<'a>(it: &Interp<'a>, v: &Value<'a>, ty: &str) -> Res<String> {
    match it.deref(v.clone()) {
        Value::Enum(t, var, p) if &*t == ty && p.is_empty() => Ok(var.to_string()),
        o => Err(format!("expected a {ty}, found {}", o.type_name())),
    }
}

fn unknown(ty: &str, var: &str) -> String {
    format!("`{ty}::{var}` is not a variant of {ty}")
}

fn fold_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Fold> {
    match variant_of(it, v, "Fold")?.as_str() {
        "K128" => Ok(Fold::K128),
        "K256Shared" => Ok(Fold::K256Shared),
        "K256Pow2" => Ok(Fold::K256Pow2),
        o => Err(unknown("Fold", o)),
    }
}
fn tile_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Tile> {
    match variant_of(it, v, "Tile")?.as_str() {
        "T128x128x8" => Ok(Tile::T128x128x8),
        "T256x128x16" => Ok(Tile::T256x128x16),
        o => Err(unknown("Tile", o)),
    }
}
fn cacc_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Cacc> {
    match variant_of(it, v, "Cacc")?.as_str() {
        "One" => Ok(Cacc::One),
        "Two" => Ok(Cacc::Two),
        o => Err(unknown("Cacc", o)),
    }
}
fn epi_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Epi> {
    match variant_of(it, v, "Epi")?.as_str() {
        "Set" => Ok(Epi::Set),
        "Add" => Ok(Epi::Add),
        "AddSigmoid" => Ok(Epi::AddSigmoid),
        "GateUpSilu" => Ok(Epi::GateUpSilu),
        "GateUpSiluBf16" => Ok(Epi::GateUpSiluBf16),
        "QkvzaGdn" => Ok(Epi::QkvzaGdn),
        o => Err(unknown("Epi", o)),
    }
}
fn layout_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<ALayout> {
    match variant_of(it, v, "ALayout")?.as_str() {
        "Token" => Ok(ALayout::Token),
        "Slab" => Ok(ALayout::Slab),
        o => Err(unknown("ALayout", o)),
    }
}
fn kind_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<moe::Kind> {
    match variant_of(it, v, "MoeKind")?.as_str() {
        "GateUp" => Ok(moe::Kind::GateUp),
        "Down" => Ok(moe::Kind::Down),
        o => Err(unknown("MoeKind", o)),
    }
}
fn codec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<moe::WeightCodec> {
    match variant_of(it, v, "WeightCodec")?.as_str() {
        "Mq4Sym" => Ok(moe::WeightCodec::Mq4Sym),
        "Mq3I4" => Ok(moe::WeightCodec::Mq3I4),
        "E8pI4" => Ok(moe::WeightCodec::E8pI4),
        o => Err(unknown("WeightCodec", o)),
    }
}

type Fields<'a> = Vec<(Rc<str>, Value<'a>)>;

fn fields_of<'a>(it: &Interp<'a>, v: &Value<'a>, name: &str) -> Res<Fields<'a>> {
    match it.deref(v.clone()) {
        Value::Struct(n, fs) if &*n == name => Ok(fs),
        o => Err(format!(
            "expected a `{name} {{ .. }}`, found {}",
            o.type_name()
        )),
    }
}
fn field<'a>(fs: &mut Fields<'a>, name: &str, k: &str) -> Res<Value<'a>> {
    let i = fs
        .iter()
        .position(|(f, _)| &**f == k)
        .ok_or_else(|| format!("`{name}` is missing the field `{k}` (every field is required)"))?;
    Ok(fs.remove(i).1)
}
fn no_more(fs: &Fields, name: &str) -> Res<()> {
    match fs.first() {
        Some((k, _)) => Err(format!("`{name}` has no field `{k}`")),
        None => Ok(()),
    }
}

fn iu4_spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<iu4::Spec> {
    let mut fs = fields_of(it, v, "Iu4Spec")?;
    let n = "Iu4Spec";
    let spec = iu4::Spec {
        fold: fold_of(it, &field(&mut fs, n, "fold")?)?,
        tile: tile_of(it, &field(&mut fs, n, "tile")?)?,
        cacc: cacc_of(it, &field(&mut fs, n, "cacc")?)?,
        epi: epi_of(it, &field(&mut fs, n, "epi")?)?,
        act: layout_of(it, &field(&mut fs, n, "act")?)?,
        arch: arch_of(it, &field(&mut fs, n, "arch")?)?,
    };
    no_more(&fs, n)?;
    Ok(spec)
}

fn enum_val<'a>(ty: &str, var: &str) -> Value<'a> {
    Value::Enum(Rc::from(ty), Rc::from(var), vec![])
}

fn iu4_spec_value<'a>(s: iu4::Spec) -> Value<'a> {
    let f = |k: &str, v: Value<'a>| (Rc::<str>::from(k), v);
    Value::Struct(
        "Iu4Spec".into(),
        vec![
            f("fold", enum_val("Fold", &format!("{:?}", s.fold))),
            f("tile", enum_val("Tile", &format!("{:?}", s.tile))),
            f("cacc", enum_val("Cacc", &format!("{:?}", s.cacc))),
            f("epi", enum_val("Epi", &format!("{:?}", s.epi))),
            f("act", enum_val("ALayout", &format!("{:?}", s.act))),
            f("arch", nat(Nat::Arch(s.arch))),
        ],
    )
}

fn moe_spec_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<moe::Spec> {
    let mut fs = fields_of(it, v, "MoeSymSpec")?;
    let n = "MoeSymSpec";
    let spec = moe::Spec {
        arch: arch_of(it, &field(&mut fs, n, "arch")?)?,
        kind: kind_of(it, &field(&mut fs, n, "kind")?)?,
        nt: u8_of(it, &field(&mut fs, n, "nt")?, "MoeSymSpec.nt")?,
        rr: u8_of(it, &field(&mut fs, n, "rr")?, "MoeSymSpec.rr")?,
        codec: codec_of(it, &field(&mut fs, n, "codec")?)?,
    };
    no_more(&fs, n)?;
    Ok(spec)
}

fn emitted_value<'a>(e: Emitted) -> Value<'a> {
    nat(Nat::Emitted(Rc::new(e)))
}

/// A module result as `(Vec<Emitted>, text, proof)`; the proof is its JSON.
fn module_value<'a>(
    (emitted, text, proof): (Vec<Emitted>, String, iu4::ModuleProof),
) -> Res<Value<'a>> {
    let proof = serde_json::to_string(&proof).map_err(|e| format!("module proof: {e}"))?;
    Ok(Value::Tuple(vec![
        Value::Array(emitted.into_iter().map(emitted_value).collect()),
        Value::str(text),
        Value::str(proof),
    ]))
}

// ---- argument conversion ------------------------------------------------------

fn gen_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Rc<Gen>> {
    match nat_of(it, v) {
        Some(Nat::Iu4Gen(g)) => Ok(g),
        _ => Err("expected an Iu4Gen (`Iu4Gen::new(Iu4Spec { .. })?`)".into()),
    }
}
fn wg_arg<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<()> {
    match nat_of(it, v) {
        Some(Nat::Wg) => Ok(()),
        _ => Err("the first argument is the workgroup scope `wg`".into()),
    }
}
fn exit_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<Exit> {
    match nat_of(it, v) {
        Some(Nat::Exit(e)) => Ok(e),
        _ => Err("expected a kernel Exit from `wg.exit(..)`".into()),
    }
}
fn bool_of<'a>(it: &Interp<'a>, v: &Value<'a>, what: &str) -> Res<bool> {
    match it.deref(v.clone()) {
        Value::Bool(b) => Ok(b),
        o => Err(format!("{what}: expected a bool, found {}", o.type_name())),
    }
}
/// The four ring ids of an `Lds`, in A, W, DS, SZ order.
fn lds_of<'a>(it: &Interp<'a>, v: &Value<'a>) -> Res<[RingId; 4]> {
    let Value::Array(xs) = it.deref(v.clone()) else {
        return Err("expected the four LDS rings (the array `declare_lds` returned)".into());
    };
    let ids: Vec<RingId> = xs.iter().map(|x| ring_of(it, x)).collect::<Res<_>>()?;
    ids.try_into()
        .map_err(|_| "an Lds is exactly four rings (A, W, DS, SZ)".to_string())
}
fn lds_val<'a>(ids: [RingId; 4]) -> Value<'a> {
    Value::Array(ids.into_iter().map(|g| nat(Nat::Ring(g))).collect())
}

fn take_free(lend: &mut Lend<'_>, ids: [RingId; 4]) -> Res<FreeRings> {
    Ok((
        lend.take::<PayA, Free, Free>(ids[0])?,
        lend.take::<PayW, Free, Free>(ids[1])?,
        lend.take::<MetaD, Free, Free>(ids[2])?,
        lend.take::<MetaS, Free, Free>(ids[3])?,
    ))
}
fn take_lds(lend: &mut Lend<'_>, ids: [RingId; 4]) -> Res<Lds> {
    Ok((
        lend.take::<PayA, Published, Free>(ids[0])?,
        lend.take::<PayW, Published, Free>(ids[1])?,
        lend.take::<MetaD, Published, Free>(ids[2])?,
        lend.take::<MetaS, Published, Free>(ids[3])?,
    ))
}
fn give_lds(lend: &mut Lend<'_>, ids: [RingId; 4], (a, w, d, s): Lds) -> Res<()> {
    lend.give_back(ids[0], a)?;
    lend.give_back(ids[1], w)?;
    lend.give_back(ids[2], d)?;
    lend.give_back(ids[3], s)
}

/// Run `body` as one `Driver::typed` call on the current driver.
fn typed<'a, T>(
    cx: &mut Cx<'_>,
    body: impl FnOnce(&mut Wg<'_>, &mut Lend<'_>) -> Res<T>,
) -> R<'a, Res<T>> {
    let mut body = Some(body);
    let mut out: Option<T> = None;
    let r = driver(cx)?.typed(&mut |wg, lend| {
        let body = body
            .take()
            .ok_or_else(|| "a typed body ran twice".to_string())?;
        out = Some(body(wg, lend)?);
        Ok(())
    });
    Ok(r.map(|()| {
        out.take()
            .expect("a typed body that returned Ok produced its value")
    }))
}

// ---- natives --------------------------------------------------------------------

pub(super) fn call_fn<'a>(
    it: &Interp<'a>,
    name: &str,
    a: &[Value<'a>],
    cx: &mut Cx<'_>,
) -> Option<R<'a>> {
    is_fn(name).then(|| call(it, name, a, cx))
}

fn call<'a>(it: &Interp<'a>, name: &str, a: &[Value<'a>], cx: &mut Cx<'_>) -> R<'a> {
    let need = |n: usize| check_args(name, a, n);
    match name {
        "Iu4Gemm::emit" => {
            need(1)?;
            Ok(done_with(iu4::emit(iu4_spec_of(it, &a[0])?), emitted_value))
        }
        "Iu4Gemm::emit_module" => {
            need(5)?;
            let (fold, tile, cacc) = (
                fold_of(it, &a[0])?,
                tile_of(it, &a[1])?,
                cacc_of(it, &a[2])?,
            );
            let (act, arch) = (layout_of(it, &a[3])?, arch_of(it, &a[4])?);
            let m = iu4::emit_module(fold, tile, cacc, act, arch).and_then(module_value);
            Ok(done_with(m, |v| v))
        }
        "MoeSym::emit" => {
            need(1)?;
            Ok(done_with(moe::emit(moe_spec_of(it, &a[0])?), emitted_value))
        }
        "MoeSym::emit_module" => {
            need(1)?;
            let m = moe::emit_module(arch_of(it, &a[0])?).and_then(module_value);
            Ok(done_with(m, |v| v))
        }
        "MoeSym::emit_module_codec" => {
            need(2)?;
            let (arch, codec) = (arch_of(it, &a[0])?, codec_of(it, &a[1])?);
            let m = moe::emit_module_codec(arch, codec).and_then(module_value);
            Ok(done_with(m, |v| v))
        }
        "Iu4Gen::new" => {
            need(1)?;
            let g = Gen::new(iu4_spec_of(it, &a[0])?);
            Ok(done_with(g, |g| nat(Nat::Iu4Gen(Rc::new(g)))))
        }
        "iu4::label" => {
            need(1)?;
            Ok(Value::str(blocks::label(&str_of(it, &a[0], "iu4::label")?)))
        }
        "iu4::tail_label" => {
            need(1)?;
            Ok(Value::str(blocks::tail_label(usize_of(
                it,
                &a[0],
                "iu4::tail_label",
            )?)))
        }
        "iu4::declare_lds" => {
            need(2)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let ids = typed(cx, |wg, lend| {
                let (pa, pw, pd, ps) = blocks::declare_lds(wg, &g)?;
                Ok([
                    lend.give(pa, "iu4.A"),
                    lend.give(pw, "iu4.W"),
                    lend.give(pd, "iu4.DS"),
                    lend.give(ps, "iu4.SZ"),
                ])
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::prologue" => {
            need(4)?;
            wg_arg(it, &a[0])?;
            let (g, exit, ids) = (gen_of(it, &a[1])?, exit_of(it, &a[2])?, lds_of(it, &a[3])?);
            let ids = typed(cx, |wg, lend| {
                let end = lend.end(&exit)?;
                let free = take_free(lend, ids)?;
                let lds = blocks::prologue(wg, &g, &end, free)?;
                give_lds(lend, ids, lds)?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::kloop" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, ids) = (gen_of(it, &a[1])?, lds_of(it, &a[2])?);
            let ids = typed(cx, |wg, lend| {
                let lds = take_lds(lend, ids)?;
                let lds = blocks::kloop(wg, &g, lds)?;
                give_lds(lend, ids, lds)?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::block" | "iu4::block_stepping" => {
            let stepping = name == "iu4::block_stepping";
            need(if stepping { 9 } else { 8 })?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            let tag = str_of(it, &a[2], "block tag")?;
            let h = usize_of(it, &a[3], "block plane")?;
            let fetch_next = bool_of(it, &a[4], "fetch_next")?;
            let end_label = str_of(it, &a[5], "end_label")?;
            let entry_label = bool_of(it, &a[6], "entry_label")?;
            let (step, last) = if stepping {
                (Some(usize_of(it, &a[7], "step descriptor")?), 8)
            } else {
                (None, 7)
            };
            let ids = lds_of(it, &a[last])?;
            let ids = typed(cx, |wg, lend| {
                let lds = take_lds(lend, ids)?;
                let lds = match step {
                    None => {
                        blocks::block(wg, &g, &tag, h, fetch_next, &end_label, entry_label, lds)
                    }
                    Some(s) => blocks::block_stepping(
                        wg,
                        &g,
                        &tag,
                        h,
                        fetch_next,
                        &end_label,
                        entry_label,
                        s,
                        lds,
                    ),
                }?;
                give_lds(lend, ids, lds)?;
                Ok(ids)
            })?;
            Ok(done_with(ids, lds_val))
        }
        "iu4::step_descriptor" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, which) = (gen_of(it, &a[1])?, usize_of(it, &a[2], "descriptor")?);
            Ok(done(blocks::step_descriptor(driver(cx)?.isa(), &g, which)))
        }
        "iu4::trip_end" => {
            need(2)?;
            wg_arg(it, &a[0])?;
            let g = gen_of(it, &a[1])?;
            Ok(done(typed(cx, |wg, _| blocks::trip_end(wg, &g))?))
        }
        "iu4::trips_cmp" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, cmp) = (gen_of(it, &a[1])?, str_of(it, &a[2], "trips_cmp")?);
            let c = typed(cx, |wg, lend| {
                Ok(lend.wg_cond(blocks::trips_cmp(wg, &g, &cmp)?))
            })?;
            Ok(done_with(c, |c| nat(Nat::WgCond(c))))
        }
        "iu4::epilogue" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, exit) = (gen_of(it, &a[1])?, exit_of(it, &a[2])?);
            Ok(done(typed(cx, |wg, lend| {
                let end = lend.end(&exit)?;
                blocks::epilogue(wg, &g, &end)
            })?))
        }
        "iu4::gdn_epilogue" => {
            need(3)?;
            wg_arg(it, &a[0])?;
            let (g, ids) = (gen_of(it, &a[1])?, lds_of(it, &a[2])?);
            Ok(done(typed(cx, |wg, lend| {
                let lds = take_lds(lend, ids)?;
                blocks::gdn_epilogue(wg, &g, lds)?;
                // The block consumed the rings (it re-lays the LDS out).
                ids.iter().try_for_each(|&id| lend.consumed(id))
            })?))
        }
        n => Err(format!("native function `{n}` is not implemented").into()),
    }
}

/// Methods of an `Iu4Gen`: `plan()`, `kernel_spec()`, `spec()`, `reg(name)`.
pub(super) fn method<'a>(
    it: &Interp<'a>,
    g: &Rc<Gen>,
    name: &str,
    a: &[Value<'a>],
) -> R<'a, Option<Value<'a>>> {
    let need = |n: usize| check_args(name, a, n);
    Ok(match name {
        "plan" => {
            need(0)?;
            Some(done_with(g.plan(), |p| {
                nat(Nat::Plan(Rc::new(RefCell::new(p))))
            }))
        }
        "kernel_spec" => {
            need(0)?;
            Some(spec_value(&g.kernel_spec()))
        }
        "spec" => {
            need(0)?;
            Some(iu4_spec_value(g.spec()))
        }
        "reg" => {
            need(1)?;
            Some(Value::option(
                g.reg(&str_of(it, &a[0], "reg")?).map(Value::u8),
            ))
        }
        _ => None,
    })
}
