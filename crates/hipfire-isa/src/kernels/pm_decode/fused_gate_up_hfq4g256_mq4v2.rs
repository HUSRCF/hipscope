//! Exact gfx1201 gate/up projection arithmetic, frozen to clang24 MQ4v2.
use crate::{Arch, Builder, Emitted, KernelSpec, KernargLayout, RegPlan};
use crate::insn::MemoryClass;
use crate::kernels::common::{mem, op, s, smem, sr, v, vr};
use crate::reg::Live;

const SYMBOL: &str = "fused_gate_up_mq4g256v2";
const G0_SYMBOL: &str = "fused_gate_up_mq4g256v2_g0";

fn builder(symbol: &str) -> Result<Builder, String> {
    let spec = KernelSpec {
        kernel_id: "fused_gate_up_hfq4g256_mq4v2".into(),
        variant: "gfx1201".into(), arch: Arch::Gfx1201,
        symbol: symbol.into(),
        kernargs: KernargLayout::new(52)
            .pointer("A_gate", 0).pointer("A_up", 8).pointer("x", 16)
            .pointer("y_gate", 24).pointer("y_up", 32)
            .hidden("gate_m", 40, 4, "by_value")
            .hidden("up_m", 44, 4, "by_value")
            .hidden("K", 48, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0,
        wave32: true, cu_mode: false,
    };
    let mut regs = RegPlan::new(32, 32)?;
    for (name, base) in [("lane", 0), ("scratch", 4), ("x_lo", 8),
                         ("x_hi", 12), ("levels_lo", 16), ("levels_hi", 20),
                         ("accumulators", 24), ("addresses", 28)] {
        regs.v::<4>(name, base, Live::Whole)?;
    }
    for (name, base) in [("abi", 0), ("pointers0", 4), ("pointers1", 8),
                         ("resource", 12), ("shape", 16), ("loop", 20),
                         ("masks", 24), ("temps", 28)] {
        regs.s::<4>(name, base, Live::Whole)?;
    }
    Ok(Builder::new(spec, regs))
}

/// One group's eight-term DAG. clang24 starts with term 1, folds term 0
/// into it, then terms 2..7; the stream accumulator is added only last.
fn group_arithmetic(b: &mut Builder, accumulator: u8) -> Result<(), String> {
    for nibble in 0..8u8 {
        let dst = 16 + nibble;
        op(b, format!("v_bfe_u32 v{dst}, v4, {}, 4", nibble * 4),
           &[v(dst)], &[v(4)])?;
        op(b, format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
           &[v(dst)], &[v(dst)])?;
        op(b, format!("v_fma_mix_f32 v{dst}, v5, v{dst}, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
           &[v(dst)], &[v(5), v(dst)])?;
    }
    op(b, "v_mul_f32_e32 v6, v9, v17", &[v(6)], &[v(9), v(17)])?;
    for nibble in [0u8, 2, 3, 4, 5, 6, 7] {
        op(b, format!("v_fmac_f32_e32 v6, v{}, v{}", 8 + nibble, 16 + nibble),
           &[v(6)], &[v(6), v(8 + nibble), v(16 + nibble)])?;
    }
    op(b, format!("v_add_f32_e32 v{accumulator}, v{accumulator}, v6"),
       &[v(accumulator)], &[v(accumulator), v(6)])
}

/// G0 only: one 136-byte group and 256 input floats, producing all 32
/// lane partials in y_gate. This is deliberately not a projection twin.
fn build_region() -> Result<Emitted, String> {
    let mut b = builder(G0_SYMBOL)?;
    smem(&mut b, 4, 2, 0, 0)?;
    smem(&mut b, 8, 4, 0, 16)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_lshlrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?;
    // Header is independent of the input lane: load both half-wave headers.
    op(&mut b, "v_mov_b32_e32 v3, 0", &[v(3)], &[])?;
    mem(&mut b, "global_load_b64 v[6:7], v3, s[4:5]",
        &[vr(6, 2)], &[v(3), sr(4, 2)], MemoryClass::VmemLoad)?;
    op(&mut b, "v_cmp_lt_u32_e64 s24, v0, 16", &[s(24)], &[v(0)])?;
    op(&mut b, "v_cndmask_b32_e64 v5, v7, v6, s24",
        &[v(5)], &[v(7), v(6), s(24)])?;
    mem(&mut b, "global_load_b32 v4, v1, s[4:5] offset:8",
        &[v(4)], &[v(1), sr(4, 2)], MemoryClass::VmemLoad)?;
    mem(&mut b, "global_load_b128 v[8:11], v2, s[8:9]",
        &[vr(8, 4)], &[v(2), sr(8, 2)], MemoryClass::VmemLoad)?;
    mem(&mut b, "global_load_b128 v[12:15], v2, s[8:9] offset:16",
        &[vr(12, 4)], &[v(2), sr(8, 2)], MemoryClass::VmemLoad)?;
    op(&mut b, "v_mov_b32_e32 v24, 0", &[v(24)], &[])?;
    group_arithmetic(&mut b, 24)?;
    mem(&mut b, "global_store_b32 v1, v24, s[10:11]",
        &[], &[v(1), v(24), sr(10, 2)], MemoryClass::VmemStore)?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Err(format!("{SYMBOL}: G0 real-tensor region gate has not passed"))
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn g0_region_m7() {
        let emitted = build_region().unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/pm-decode/gfx1201");
        std::fs::create_dir_all(&root).unwrap();
        let stem = root.join("fused_gate_up_hfq4g256_mq4v2.g0");
        std::fs::write(stem.with_extension("g0.s"), &emitted.s_text).unwrap();
        std::fs::write(stem.with_extension("g0.proof.json"),
            serde_json::to_vec_pretty(&emitted.proof).unwrap()).unwrap();
        let shape = serde_json::json!({
            "symbol": G0_SYMBOL,
            "counts": {"v_fma_mix_f32": 8, "v_fmac_f32_e32": 7},
            "forbidden": ["scratch_*", "s_waitcnt", "v_wmma_*"],
            "vgpr_max": 32, "sgpr_max": 32,
            "require_wave32": true, "require_zero_spills": true,
            "require_zero_private": true, "launch_dynamic_lds_bytes": 0
        });
        std::fs::write(stem.with_extension("g0.shape.json"),
            serde_json::to_vec_pretty(&shape).unwrap()).unwrap();
        let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
        let path = stem.with_extension("g0.co");
        std::fs::write(&path, elf).unwrap();
        let m7 = crate::pm_check::m7(&path, "gfx1201", G0_SYMBOL).unwrap();
        println!("gate_up G0 M7: {m7}");
        assert_eq!(m7["lift"], "byte-exact");
        assert_eq!(m7["obligations"], serde_json::json!({}));
    }
}
