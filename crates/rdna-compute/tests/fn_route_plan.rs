// SPDX-License-Identifier: Apache-2.0
// Flash-Next walked route plan: CPU-only planner tests. No compiler process or HIP.
//
// `route_plans/qwen3.8-flash-next.gfx1201.tsv` (the canonical monolithic header)
// and `route_plans/qwen3.8-flash-next-base.mq4-xts.gfx1201.tsv` (the split header:
// the canonical header minus the 128 PLE ngram_embedding shard tensors) are
// generated from module-load
// audits of the real dispatch (`scripts/route_plan_walk.py`). These tests check
// that every walked kernel is planned (walk ⊆ plan), that every planned HIP
// module is the release pack's object with the walked source (plan ⊆ pack),
// and that every load the walks did not cover refuses (lazy loading) instead
// of planning partially. Older Flash-Next traces (jit-g0, kernel-trace) ran
// other policies (Q8 QSA, host-mapped experts, an earlier dispatch) and are
// not a closure oracle for this route.
use rdna_compute::kernel_registry::{
    entries, route_entries, route_layout_sha256, walked_route_plans, KernelSpecRoute,
    KernelTensorMeta, PlannedKernel, RegistryError, RouteKernelInput, RouteKernelPlan,
    WalkedRoutePlan,
};
use rdna_compute::{DType, FeatureFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(FIXTURES).join(name)).unwrap()).unwrap()
}

/// `hipfire-runtime` `loader_api::kernel_dtype_for_quant_type` for the tags a
/// Flash-Next header carries.
fn dtype(tag: u64) -> DType {
    match tag {
        3 => DType::Q8_0,
        16 => DType::BF16,
        44 => DType::MQ4G256V2,
        47 => DType::MQ6G256V2,
        53 => DType::MQ4G128V2,
        54 => DType::Raw,
        other => panic!("header fixture quant tag {other} has no test mapping"),
    }
}

/// The canonical Flash-Next GPTQ3 header (`jit-g0/header-flash-next.json`,
/// header SHA-256 8703bb16…, the artifact the gfx1201 walks loaded).
fn flash_next_header() -> Vec<(String, DType, Vec<usize>)> {
    fixture("jit-g0/header-flash-next.json")["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_owned(),
                dtype(t["quant_tag"].as_u64().unwrap()),
                t["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect()
}

fn metas(header: &[(String, DType, Vec<usize>)]) -> Vec<KernelTensorMeta<'_>> {
    header
        .iter()
        .map(|(name, dtype, shape)| KernelTensorMeta {
            name,
            dtype: *dtype,
            shape,
        })
        .collect()
}

fn builtin_flags(arch: &str) -> FeatureFlags {
    let resolved = hipfire_config::resolve(std::iter::empty()).unwrap();
    let config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
    FeatureFlags::from_process_config(arch, &config)
}

/// Every walked Flash-Next plan for an arch (one per header layout).
fn walked_plans(arch: &str) -> Vec<&'static WalkedRoutePlan> {
    walked_route_plans()
        .unwrap()
        .iter()
        .filter(|plan| plan.model_arch == 16 && plan.arch == arch)
        .collect()
}

/// The walked plan whose own header layout is `layout`.
fn walked(arch: &str, layout: &str) -> &'static WalkedRoutePlan {
    let mut plans = walked_plans(arch)
        .into_iter()
        .filter(|plan| plan.layout == layout);
    let plan = plans
        .next()
        .unwrap_or_else(|| panic!("no walked Flash-Next plan for {arch} layout {layout}"));
    assert!(
        plans.next().is_none(),
        "two walked Flash-Next plans for {arch} layout {layout}"
    );
    plan
}

/// Header layout of the split MQ4 XTS artifact (the walked split plan).
const SPLIT_LAYOUT: &str = "65811e8a696c6b68d25a0c5bf9f971cef40caf60f47a83498a41297a15597c5c";
const NGRAM_SHARDS: usize = 128;

fn ngram_shard_name(shard: usize) -> String {
    format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight")
}

/// The split header: the canonical header minus exactly the ngram shard family.
fn split_header(canonical: &[(String, DType, Vec<usize>)]) -> Vec<(String, DType, Vec<usize>)> {
    let removed: Vec<String> = (0..NGRAM_SHARDS).map(ngram_shard_name).collect();
    for name in &removed {
        assert!(
            canonical.iter().any(|(n, _, _)| n == name),
            "canonical header lacks {name}"
        );
    }
    let split: Vec<_> = canonical
        .iter()
        .filter(|(name, _, _)| !removed.contains(name))
        .cloned()
        .collect();
    assert_eq!(canonical.len() - split.len(), NGRAM_SHARDS);
    assert!(split
        .iter()
        .all(|(name, _, _)| !name.contains("ngram_embedding.shard_")));
    split
}

/// One header layout a walked Flash-Next plan covers.
struct Layout {
    name: &'static str,
    header: Vec<(String, DType, Vec<usize>)>,
    sha: String,
}

/// Both walked layouts: the canonical monolithic header and the split header.
fn layouts() -> Vec<Layout> {
    let canonical = flash_next_header();
    let split = split_header(&canonical);
    [("canonical", canonical), ("split", split)]
        .into_iter()
        .map(|(name, header)| {
            let sha = route_layout_sha256(&metas(&header));
            Layout { name, header, sha }
        })
        .collect()
}

/// The walked axes, each at its first walked value (what the loader resolves
/// for the default route).
fn walked_axes(plan: &WalkedRoutePlan) -> Vec<(&'static str, String)> {
    plan.axes
        .iter()
        .map(|(name, values)| (*name, values[0].to_owned()))
        .collect()
}

struct Load<'a> {
    arch: &'a str,
    tensors: &'a [KernelTensorMeta<'a>],
    max_seq: usize,
    spec: KernelSpecRoute,
    flags: &'a FeatureFlags,
    axes: &'a [(&'a str, String)],
}

fn plan(load: &Load<'_>) -> Result<RouteKernelPlan, RegistryError> {
    let axis = |name: &str| {
        load.axes
            .iter()
            .find(|(n, _)| *n == name)
            .map_or("", |(_, v)| v.as_str())
    };
    route_entries(&RouteKernelInput {
        arch: load.arch,
        model_arch: 16,
        tensors: load.tensors,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: 0,
        max_seq: load.max_seq,
        prefill_chunk_rows: 0,
        kv_k: axis("qsa"),
        kv_v: axis("qsa"),
        state_quant: axis("gdn"),
        spec: load.spec,
        host_mapped_experts: false,
        retained_decode: true,
        hipcc_extra_flags: &load.flags.hipcc_extra_flags,
        flags: load.flags,
        route_axes: load.axes,
    })
}

/// `symbol -> (module, kind)` of a plan; one owner per symbol.
fn bindings(plan: &RouteKernelPlan) -> BTreeMap<&'static str, (&'static str, bool)> {
    let mut out = BTreeMap::new();
    for entry in &plan.entries {
        for &symbol in entry.symbols() {
            let embedded = matches!(entry, PlannedKernel::Embedded { .. });
            assert!(
                out.insert(symbol, (entry.module(), embedded)).is_none(),
                "symbol {symbol} planned twice"
            );
        }
    }
    out
}

/// AR and native-MTP plans of one layout at `max_seq` 16,384.
fn gfx1201_plans(layout: &Layout) -> (RouteKernelPlan, RouteKernelPlan) {
    let tensors = metas(&layout.header);
    let walked = walked("gfx1201", &layout.sha);
    let flags = builtin_flags("gfx1201");
    let axes = walked_axes(walked);
    let load = |spec| Load {
        arch: "gfx1201",
        tensors: &tensors,
        max_seq: 16_384,
        spec,
        flags: &flags,
        axes: &axes,
    };
    let ar = plan(&load(KernelSpecRoute::Ar))
        .unwrap_or_else(|e| panic!("{} AR route plan: {e:?}", layout.name));
    let mtp = plan(&load(KernelSpecRoute::NativeMtp))
        .unwrap_or_else(|e| panic!("{} native-MTP route plan: {e:?}", layout.name));
    (ar, mtp)
}

/// Each header layout resolves to exactly one walked plan: the canonical header
/// and the split header (canonical minus the 128 ngram shards) have distinct
/// layouts, plus the independently walked split mq2 header.
#[test]
fn walked_layouts_are_the_canonical_and_split_flash_next_headers() {
    let layouts = layouts();
    assert_eq!(layouts[0].header.len(), 1325);
    assert_eq!(layouts[1].header.len(), 1197);
    assert_eq!(layouts[1].sha, SPLIT_LAYOUT);
    assert_ne!(layouts[0].sha, layouts[1].sha);
    let plans = walked_plans("gfx1201");
    assert_eq!(
        plans.len(),
        layouts.len() + 1,
        "canonical layouts plus split mq2"
    );
    assert!(plans.iter().any(|plan| {
        plan.layout == "92d2c376d98535639df92432b6255c8de577ac43f57a5e9f32fbc71c970d471a"
    }));
    for layout in &layouts {
        assert_eq!(
            walked("gfx1201", &layout.sha).layout,
            layout.sha,
            "{}",
            layout.name
        );
    }
    assert!(plans.iter().all(|plan| plan.family == plans[0].family));
}

/// walk ⊆ plan: every (module, symbol) the walked dispatch bound is planned
/// under its route with the current registry recipe (module identity, walked
/// symbol set, current flags/profile/source), and nothing else is, for each
/// header layout's own plan. The checked-in row digests are the historical
/// walk-time capture provenance and are never compared here; regeneration is
/// the existing `scripts/route_plan_walk.py` + `scripts/route_plan_build.py`
/// generator, not a hash alias or a partial plan.
#[test]
fn every_walked_kernel_is_planned() {
    for layout in &layouts() {
        let (ar, mtp) = gfx1201_plans(layout);
        let walked = walked("gfx1201", &layout.sha);
        let inventory = entries("gfx1201", &builtin_flags("gfx1201").hipcc_extra_flags).unwrap();
        for (plan, native_mtp) in [(&ar, false), (&mtp, true)] {
            let rows: Vec<_> = walked
                .rows
                .iter()
                .filter(|row| row.native_mtp == native_mtp)
                .collect();
            assert_eq!(
                plan.entries.len(),
                rows.len(),
                "{} native_mtp={native_mtp}",
                layout.name
            );
            for row in rows {
                let entry = plan
                    .entries
                    .iter()
                    .find(|entry| entry.module() == row.module)
                    .unwrap_or_else(|| {
                        panic!("{}: walked {} is not planned", layout.name, row.module)
                    });
                assert_eq!(
                    entry.symbols(),
                    &row.symbols[..],
                    "{} {}",
                    layout.name,
                    row.module
                );
                if let PlannedKernel::Hip(e) = entry {
                    let live = inventory
                        .iter()
                        .find(|live| live.module == row.module)
                        .unwrap_or_else(|| {
                            panic!("{}: registry has no walked {}", layout.name, row.module)
                        });
                    assert_eq!(
                        e.source(),
                        live.source(),
                        "{} {} is not the current registry source",
                        layout.name,
                        row.module
                    );
                    assert_eq!(
                        e.flags, live.flags,
                        "{} {} flags are not the current registry recipe",
                        layout.name, row.module
                    );
                    assert_eq!(
                        e.scheduler_profile, live.scheduler_profile,
                        "{} {} profile is not the current registry recipe",
                        layout.name, row.module
                    );
                } else {
                    let sha = match entry {
                        PlannedKernel::Embedded { image, .. } => {
                            format!("{:x}", Sha256::digest(image))
                        }
                        PlannedKernel::Hip(_) => unreachable!(),
                    };
                    assert_eq!(sha, row.sha256, "{} {}", layout.name, row.module);
                }
                assert_eq!(
                    row.embedded,
                    matches!(entry, PlannedKernel::Embedded { .. }),
                    "{} {}",
                    layout.name,
                    row.module
                );
            }
        }
        // The MTP route falls back to AR, so its plan carries every AR binding.
        let (ar, mtp) = (bindings(&ar), bindings(&mtp));
        for (symbol, owner) in &ar {
            assert_eq!(
                mtp.get(symbol),
                Some(owner),
                "{}: AR binding {symbol} missing from the MTP plan",
                layout.name
            );
        }
    }
}

/// plan ⊆ pack: every planned HIP module is an installer-inventory entry (what
/// `scripts/build-kernel-pack.sh` ships) with the planned source, recipe and
/// every planned symbol in its index, so an installed pack satisfies the whole
/// plan with zero compiles. Both layouts' plans.
#[test]
fn every_planned_hip_module_is_in_the_release_pack() {
    let flags = builtin_flags("gfx1201");
    let pack = entries("gfx1201", &flags.hipcc_extra_flags).unwrap();
    for layout in &layouts() {
        let (ar, mtp) = gfx1201_plans(layout);
        let mut hip = 0;
        for entry in ar.entries.iter().chain(&mtp.entries) {
            let PlannedKernel::Hip(planned) = entry else {
                continue;
            };
            let packed = pack
                .iter()
                .find(|e| e.module == planned.module)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: planned {} is not in the gfx1201 pack",
                        layout.name, planned.module
                    )
                });
            assert_eq!(
                packed.source(),
                planned.source(),
                "{} {}",
                layout.name,
                planned.module
            );
            assert_eq!(
                packed.flags, planned.flags,
                "{} {}",
                layout.name, planned.module
            );
            assert_eq!(
                packed.scheduler_profile, planned.scheduler_profile,
                "{} {}",
                layout.name, planned.module
            );
            for symbol in planned.symbols {
                assert!(
                    packed.symbols.contains(symbol),
                    "{}: pack index of {} lacks {symbol}",
                    layout.name,
                    planned.module
                );
            }
            hip += 1;
        }
        assert!(hip > 0, "{}", layout.name);
    }
}

/// Loads with policies the walks did not cover get no plan (the caller keeps
/// lazy loading); a kernel the walks missed on a planned load warns and loads
/// (`scratch::route_load_tests`). Checked against each layout's own plan.
#[test]
fn unwalked_flash_next_policies_keep_lazy_loading() {
    let flags = builtin_flags("gfx1201");
    for layout in &layouts() {
        let tensors = metas(&layout.header);
        let walked = walked("gfx1201", &layout.sha);
        let axes = walked_axes(walked);
        let base = Load {
            arch: "gfx1201",
            tensors: &tensors,
            max_seq: walked.max_seq,
            spec: KernelSpecRoute::NativeMtp,
            flags: &flags,
            axes: &axes,
        };
        let refused =
            |load: &Load<'_>| matches!(plan(load), Err(RegistryError::UnsupportedRoute(_)));
        let name = layout.name;
        assert!(
            plan(&base).is_ok(),
            "{name}: the walked ceiling itself is admitted"
        );
        assert!(
            refused(&Load {
                max_seq: walked.max_seq + 1,
                ..base
            }),
            "{name}"
        );
        for spec in [
            KernelSpecRoute::DFlash,
            KernelSpecRoute::DDTree,
            KernelSpecRoute::DSpark,
        ] {
            assert!(refused(&Load { spec, ..base }), "{name} {spec:?}");
        }
        // A policy axis outside the walks, a missing axis, an unknown axis.
        let mut q8 = axes.clone();
        q8.iter_mut().find(|(axis, _)| *axis == "qsa").unwrap().1 = "q8".into();
        assert!(refused(&Load { axes: &q8, ..base }), "{name}");
        assert!(
            refused(&Load {
                axes: &axes[1..],
                ..base
            }),
            "{name}"
        );
        let mut extra = axes.clone();
        extra.push(("unwalked_axis", "on".into()));
        assert!(
            refused(&Load {
                axes: &extra,
                ..base
            }),
            "{name}"
        );
        // Any non-default feature flag, including kernel.pm_decode.
        let mut changed = flags.clone();
        changed.hipcc_extra_flags.push_str(" -DFN_ROUTE_TEST=1");
        assert!(
            refused(&Load {
                flags: &changed,
                ..base
            }),
            "{name}"
        );
        let mut pm = flags.clone();
        pm.pm_decode = !pm.pm_decode;
        assert!(refused(&Load { flags: &pm, ..base }), "{name}");
        // Archs without a walked plan have no closed route.
        for arch in ["gfx1100", "gfx906"] {
            let arch_flags = builtin_flags(arch);
            assert!(
                refused(&Load {
                    arch,
                    flags: &arch_flags,
                    ..base
                }),
                "{name} {arch}"
            );
        }
    }
}

/// A header that is neither walked layout (a re-quantized artifact, a partial
/// ngram shard family, an extra tensor) has no walked route, whichever walked
/// layout it was derived from.
#[test]
fn unknown_header_layout_has_no_walked_route() {
    let flags = builtin_flags("gfx1201");
    let canonical = flash_next_header();
    let shard_0 = canonical
        .iter()
        .find(|(name, _, _)| *name == ngram_shard_name(0))
        .unwrap()
        .clone();
    let last_shard = ngram_shard_name(NGRAM_SHARDS - 1);
    let walked_layouts: Vec<String> = walked_plans("gfx1201")
        .iter()
        .map(|plan| plan.layout.to_owned())
        .collect();
    for layout in layouts() {
        let axes = walked_axes(walked("gfx1201", &layout.sha));
        let mut unknown = Vec::new();
        let mut dropped = layout.header.clone();
        dropped.pop();
        unknown.push(("last tensor dropped", dropped));
        let mut extra = layout.header.clone();
        extra.push(("extra.weight".to_owned(), DType::Q8_0, vec![1]));
        unknown.push(("extra tensor", extra));
        let mut partial = layout.header.clone();
        if layout.name == "split" {
            partial.push(shard_0.clone());
        } else {
            partial.retain(|(name, _, _)| *name != last_shard);
        }
        unknown.push(("partial ngram shard family", partial));
        for (what, header) in unknown {
            let tensors = metas(&header);
            let sha = route_layout_sha256(&tensors);
            assert!(
                !walked_layouts.contains(&sha),
                "{} {what} must not hash to a walked layout",
                layout.name
            );
            for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
                let load = Load {
                    arch: "gfx1201",
                    tensors: &tensors,
                    max_seq: 16_384,
                    spec,
                    flags: &flags,
                    axes: &axes,
                };
                match plan(&load) {
                    Err(RegistryError::UnsupportedRoute(why)) => {
                        assert!(
                            why.contains("has no walked route"),
                            "{} {what} {spec:?}: {why}",
                            layout.name
                        )
                    }
                    other => panic!(
                        "{} {what} {spec:?}: expected UnsupportedRoute, got {:?}",
                        layout.name,
                        other.map(|p| p.entries.len())
                    ),
                }
            }
        }
    }
}
