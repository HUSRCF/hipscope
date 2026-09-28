//! gfx11 builder target: counter waits, LDS barrier drain, VOPD pairing, the
//! V2C-equivalent SET kernel, and byte identity of the gfx1201 products.
use hipfire_isa::{Arch, Builder, KernelSpec, KernargLayout, RegPlan};
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::kernels::{iu4_gemm, iu4_v2c, fp8_gemm};
use hipfire_isa::lds::Transition;
use hipfire_isa::reg::Live;
use hipfire_isa::vopd::{Operand, VopdF32, VopdOp};
use std::io::Write;
use std::process::{Command, Stdio};

const LLVM: &str = "/opt/rocm/core-10.0/lib/llvm/bin";

fn probe(arch: Arch) -> Builder {
    let mut plan = RegPlan::new(16, 8).unwrap();
    for (name, n) in [("a", 0), ("b", 1), ("addr", 2), ("c", 3), ("d", 4), ("e", 5)] { plan.v::<1>(name, n, Live::Whole).unwrap(); }
    plan.s::<2>("base", 0, Live::Whole).unwrap();
    Builder::new(KernelSpec {
        kernel_id: "probe".into(), variant: "default".into(), arch, symbol: "probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 64, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, plan)
}
fn v(n: u8) -> hipfire_isa::reg::RegRef { hipfire_isa::V::<1>(n).reg() }

#[test]
fn gfx11_lds_barrier_drains_pending_stores_first() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds.add("S", 0, 256).unwrap();
    b.ds_store(slot, Instruction::new("ds_store_b32 v2, v0", vec![], vec![v(2), v(0)]).memory(MemoryClass::DsStore)).unwrap();
    b.barrier(&[Transition::Ready(slot)]).unwrap();
    let text: Vec<&str> = b.program.instructions.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(&text[text.len() - 2..], ["s_waitcnt lgkmcnt(0)", "s_barrier"]);
    assert!(b.ledger.is_empty());
}

#[test]
fn gfx11_combines_vm_and_lgkm_waits_into_one_s_waitcnt() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds.add("S", 0, 256).unwrap();
    b.ds_store(slot, Instruction::new("ds_store_b32 v2, v0", vec![], vec![v(2), v(0)]).memory(MemoryClass::DsStore)).unwrap();
    b.barrier(&[Transition::Ready(slot)]).unwrap();
    b.push(Instruction::new("global_load_b32 v1, v2, s[0:1]", vec![v(1)], vec![v(2), hipfire_isa::S::<2>(0).reg()]).memory(MemoryClass::VmemLoad)).unwrap();
    b.push(Instruction::new("global_load_b32 v5, v2, s[0:1] offset:4", vec![v(5)], vec![v(2), hipfire_isa::S::<2>(0).reg()]).memory(MemoryClass::VmemLoad)).unwrap();
    b.ds_load(slot, Instruction::new("ds_load_b32 v3, v2", vec![v(3)], vec![v(2)]).memory(MemoryClass::DsLoad)).unwrap();
    b.ds_load(slot, Instruction::new("ds_load_b32 v4, v2 offset:4", vec![v(4)], vec![v(2)]).memory(MemoryClass::DsLoad)).unwrap();
    // Needs the older VMEM load and the older DS load: vmcnt(1) lgkmcnt(1).
    b.push(Instruction::new("v_add_f32_e32 v0, v1, v3", vec![v(0)], vec![v(1), v(3)])).unwrap();
    let waits: Vec<&str> = b.program.instructions.iter().map(|i| i.text.as_str()).filter(|t| t.starts_with("s_waitcnt")).collect();
    assert_eq!(waits.last(), Some(&"s_waitcnt vmcnt(1) lgkmcnt(1)"));
    assert_eq!(b.waits.iter().filter(|w| w.insn == "s_waitcnt vmcnt(1) lgkmcnt(1)").count(), 2);
}

#[test]
fn gfx11_vopd_halves_may_not_read_each_others_destination() {
    let x = VopdOp { op: VopdF32::Mul, dst: 8, src0: Operand::V(1), src1: 2 };
    let reads_x = VopdOp { op: VopdF32::Add, dst: 9, src0: Operand::Lit(0xcb40_0000), src1: 8 };
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1100, x, reads_x).is_err());
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1151, x, reads_x).is_err());
    // gfx12 pairing rules are unchanged.
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1201, x, reads_x).is_ok());
    let independent = VopdOp { op: VopdF32::Add, dst: 9, src0: Operand::Lit(0xcb40_0000), src1: 9 };
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1100, x, independent).is_ok());
}

fn v2c() -> hipfire_isa::Emitted { iu4_v2c::emit(iu4_v2c::Spec { arch: Arch::Gfx1100 }).unwrap() }

#[test]
fn v2c_set_is_deterministic_within_the_occupancy_ceiling() {
    let (a, b) = (v2c(), v2c());
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert!(a.shape.next_free_vgpr <= iu4_v2c::VGPR_CEILING);
    assert_eq!(a.proof.loop_fixpoints.len(), 1);
    assert!(iu4_v2c::emit(iu4_v2c::Spec { arch: Arch::Gfx1201 }).is_err());
}

#[test]
fn v2c_steady_state_trip_matches_the_v2c_algorithm() {
    // One trip = two K128 epochs; per epoch 64 K16 WMMAs, 2 x (8 DPP + 32
    // mul/add + 16 fmac packets), 8 A rebias XORs, 4 + 4 staging loads,
    // 40 fragment loads, 4 LDS stores and one barrier.
    let census = iu4_v2c::hot_loop_census(&v2c().s_text);
    for (name, n) in [("v_wmma_i32_16x16x16_iu4", 128), ("vopd_packets", 192), ("v_dual_mul_f32", 128), ("v_dual_fmac_f32", 64),
        ("v_mov_b32_dpp", 32), ("v_cvt_f32_f16_e64", 2), ("v_xor_b32_e32", 16), ("global_load_b64", 16), ("global_load_b32", 8),
        ("global_load_u16", 2), ("ds_load_2addr_b64", 80), ("ds_store_2addr_b64", 8), ("s_barrier", 2), ("valu_slots", 242)] {
        assert_eq!(census.get(name).copied().unwrap_or(0), n, "{name}");
    }
    for forbidden in ["buffer_gl0_inv", "s_nop", "v_nop", "scratch_load_b32"] { assert!(!census.contains_key(forbidden), "{forbidden}"); }
}

fn assemble(text: &str, arch: &str) {
    let mut child = Command::new(format!("{LLVM}/llvm-mc")).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj", "-o", "/dev/null"])
        .stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success() && out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn v2c_set_assembles_for_gfx1100_with_zero_diagnostics() { assemble(&v2c().s_text, "gfx1100") }

/// `.text` of the device ELF inside a clang offload bundle (or a bare ELF).
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let at = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[at..];
    let u16_ = |o: usize| u16::from_le_bytes(e[o..o + 2].try_into().unwrap()) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(40), u16_(58), u16_(60), u16_(62));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum).find_map(|i| {
        let name = &e[names + u32_(sh(i))..];
        name.starts_with(b".text\0").then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
    }).unwrap()
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("hipfire-isa-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (dir.join(format!("{stem}.s")), dir.join(format!("{stem}.o")), dir.join(format!("{stem}.co")));
    std::fs::write(&s, text).unwrap();
    assert!(Command::new(format!("{LLVM}/llvm-mc")).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj"]).arg(&s).arg("-o").arg(&o).status().unwrap().success());
    assert!(Command::new(format!("{LLVM}/ld.lld")).arg("-shared").arg(&o).arg("-o").arg(&co).status().unwrap().success());
    std::fs::read(co).unwrap()
}

/// The gfx11 target must not move a byte of the embedded gfx1201 products:
/// fresh emission of `_b1`, `_b1s` and the fp8 module links to the exact
/// `.text` of the committed bundles.
#[test]
fn committed_gfx1201_bundles_equal_fresh_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let token = iu4_gemm::emit_module(iu4_gemm::Fold::K128, iu4_gemm::Tile::T128x128x8, iu4_gemm::Cacc::One, iu4_gemm::ALayout::Token, Arch::Gfx1201).unwrap().1;
    let slab = iu4_gemm::emit_module(iu4_gemm::Fold::K128, iu4_gemm::Tile::T128x128x8, iu4_gemm::Cacc::One, iu4_gemm::ALayout::Slab, Arch::Gfx1201).unwrap().1;
    let mut fp8 = Vec::new();
    for act_scale in [fp8_gemm::ActScale::Row, fp8_gemm::ActScale::K128] {
        for epi in [fp8_gemm::Epi::Set, fp8_gemm::Epi::Add, fp8_gemm::Epi::GateUpSilu, fp8_gemm::Epi::Qkv, fp8_gemm::Epi::Qkvza] {
            fp8.push(fp8_gemm::emit(fp8_gemm::Spec { arch: Arch::Gfx1201, act_scale, epi }).unwrap());
        }
    }
    let fp8 = fp8_gemm::module(&fp8).unwrap().0;
    for (text, bundle) in [(token, "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1"), (slab, "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s"), (fp8, "gemm_mq4g256v2_wmma_fp8_gfx12_b1")] {
        let committed = std::fs::read(format!("{root}/{bundle}.hxaco")).unwrap();
        assert!(text_section(&link(&text, "gfx1201", bundle)) == text_section(&committed), "{bundle}: fresh gfx1201 emission differs from the committed bundle");
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, profile, toolchain::{assemble_link_bundle, Toolchain}};

    #[test]
    fn v2c_set_passes_gfx11_certification_checks() {
        let e = v2c();
        ledger_replay::replay_waits(&e.s_text, Arch::Gfx1100).unwrap();
        let dir = std::env::temp_dir().join(format!("hipfire-isa-v2c-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("v2c.s");
        std::fs::write(&s, &e.s_text).unwrap();
        let build = assemble_link_bundle(&Toolchain::default(), &s, &dir.join("v2c.hsaco"), "gfx1100").unwrap();
        let symbol = iu4_v2c::Spec { arch: Arch::Gfx1100 }.symbol();
        assert_eq!(pm_check::lds_bounds(&e.s_text, &symbol, 8, iu4_v2c::LDS_BYTES).unwrap(), iu4_v2c::LDS_BYTES);
        // A launch allocation smaller than the highest access is rejected.
        assert!(pm_check::lds_bounds(&e.s_text, &symbol, 8, iu4_v2c::LDS_BYTES - 8).is_err());
        let m7 = pm_check::m7(&build.elf, "gfx1100", &symbol).unwrap();
        assert_eq!(m7["lift"], "byte-exact");
        assert_eq!(m7["obligations"], serde_json::json!({}));
    }

    #[test]
    fn gfx11_profile_rewrite_is_verified_and_hazard_neutral() {
        let e = v2c();
        let points = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/profile/v2c_set.points.json")).unwrap();
        let cfg: profile::Config = serde_json::from_str(&points).unwrap();
        let (text, map) = profile::instrument(&e.s_text, &cfg, Arch::Gfx1100).unwrap();
        profile::verify(&e.s_text, &text, &map).unwrap();
        assert_eq!(map.cycle_bits, 20);
        assert!(map.vgpr_after <= iu4_v2c::VGPR_CEILING);
        let original = ledger_replay::replay_hazards(&profile::kernel_body(&e.s_text, &map.symbol).unwrap(), Arch::Gfx1100).unwrap();
        let profiled = ledger_replay::replay_hazards(&profile::kernel_body(&text, &map.profiled_symbol).unwrap(), Arch::Gfx1100).unwrap();
        assert!(original.is_empty() && profiled.is_empty(), "{profiled:?}");
        assemble(&text, "gfx1100");
        // A store wait the records would count is rejected.
        assert!(profile::instrument(&e.s_text.replace("\ts_endpgm", "\ts_waitcnt_vscnt null, 0x1\n\ts_endpgm"), &cfg, Arch::Gfx1100).is_err());
    }
}
