//! QKVZA's checked four-stream G0 arithmetic region, not an admitted twin.
use crate::{Arch, Builder, Emitted, KernelSpec, KernargLayout, RegPlan};
use crate::insn::{Instruction, MemoryClass};
use crate::reg::Live;
use crate::kernels::common::{op, mem, smem, srd_tail, v, vr, sr};

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Err("QKVZA admission requires the real-H2 G0 region byte oracle; no certified twin is available".into())
}

/// Four contiguous groups, one activation quad, and 32 reduced lane outputs.
/// The symbol and ABI deliberately differ from the production projection.
fn build_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(32, 16)?;
    regs.v::<1>("lane", 0, Live::Whole)?;
    regs.v::<1>("weight_offset", 1, Live::Whole)?;
    regs.v::<1>("header_offset", 2, Live::Whole)?;
    regs.v::<1>("activation_offset", 3, Live::Whole)?;
    regs.v::<4>("streams", 4, Live::Whole)?;
    regs.v::<4>("activation_lo", 8, Live::Whole)?;
    regs.v::<4>("activation_hi", 12, Live::Whole)?;
    regs.v::<2>("headers", 16, Live::Whole)?;
    regs.v::<1>("packed", 18, Live::Whole)?;
    regs.v::<1>("level", 19, Live::Whole)?;
    regs.v::<1>("weight", 20, Live::Whole)?;
    regs.v::<1>("dot", 21, Live::Whole)?;
    regs.v::<1>("pair", 22, Live::Whole)?;
    regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<4>("weights", 4, Live::Whole)?;
    regs.s::<2>("activation", 8, Live::Whole)?;
    regs.s::<2>("output", 10, Live::Whole)?;
    let mut b = Builder::new(KernelSpec {
        kernel_id: "pm_decode_qkvza_g0".into(), variant: "quad_region".into(),
        arch: Arch::Gfx1201, symbol: "pm_decode_qkvza_g0".into(),
        kernargs: KernargLayout::new(24).pointer("weights", 0)
            .pointer("activation", 8).pointer("lane_output", 16),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    smem(&mut b, 4, 2, 0, 0)?;
    smem(&mut b, 8, 2, 0, 8)?;
    smem(&mut b, 10, 2, 0, 16)?;
    srd_tail(&mut b, 4, None)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_mov_b32_e32 v2, 0", &[v(2)], &[])?;
    op(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[v(3)], &[v(0)])?;
    for stream in 0..4u8 {
        let acc = 4 + stream;
        op(&mut b, format!("v_mov_b32_e32 v{acc}, 0"), &[v(acc)], &[])?;
        let group = u32::from(stream) * 136;
        mem(&mut b, format!("buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:{group}"),
            &[vr(16, 2)], &[v(2), sr(4, 4)], MemoryClass::VmemLoad)?;
        mem(&mut b, format!("buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:{}", group + 8),
            &[v(18)], &[v(1), sr(4, 4)], MemoryClass::VmemLoad)?;
        for (dst, offset) in [(8, u32::from(stream) * 1024), (12, u32::from(stream) * 1024 + 16)] {
            mem(&mut b, format!("global_load_b128 v[{dst}:{}], v3, s[8:9] offset:{offset}", dst + 3),
                &[vr(dst, 4)], &[v(3), sr(8, 2)], MemoryClass::VmemLoad)?;
        }
        op(&mut b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
        op(&mut b, "v_cndmask_b32_e64 v16, v17, v16, vcc_lo", &[v(16)], &[v(17), v(16)])?;
        // LLVM24 starts with element 1, then fuses element 0, then 2..7.
        for element in [1u8, 0, 2, 3, 4, 5, 6, 7] {
            op(&mut b, format!("v_bfe_u32 v19, v18, {}, 4", element * 4), &[v(19)], &[v(18)])?;
            op(&mut b, "v_cvt_f32_ubyte0_e32 v19, v19", &[v(19)], &[v(19)])?;
            op(&mut b, "v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                &[v(20)], &[v(16), v(19)])?;
            let x = 8 + element;
            if element == 1 {
                op(&mut b, format!("v_mul_f32_e32 v21, v20, v{x}"), &[v(21)], &[v(20), v(x)])?;
            } else {
                op(&mut b, format!("v_fmac_f32_e32 v21, v20, v{x}"), &[v(21)], &[v(21), v(20), v(x)])?;
            }
        }
        op(&mut b, format!("v_add_f32_e32 v{acc}, v{acc}, v21"), &[v(acc)], &[v(acc), v(21)])?;
    }
    op(&mut b, "v_add_f32_e32 v4, v4, v5", &[v(4)], &[v(4), v(5)])?;
    op(&mut b, "v_add_f32_e32 v22, v6, v7", &[v(22)], &[v(6), v(7)])?;
    op(&mut b, "v_add_f32_e32 v4, v4, v22", &[v(4)], &[v(4), v(22)])?;
    // clang24's shfl-down16 sets lane bit 4; out-of-range lanes read themselves.
    b.ds_crosslane(Instruction::new(
        "ds_swizzle_b32 v21, v4 offset:0x20f",
        vec![v(21)], vec![v(4)],
    ).memory(MemoryClass::DsLoad))?;
    b.wait(crate::ledger::Counter::Ds, 0)?;
    op(&mut b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    for offset in [8, 4, 2, 1] {
        op(&mut b, format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset), &[], &[v(0)])?;
        op(&mut b, format!("v_cndmask_b32_e64 v22, 0, {offset}, vcc_lo"), &[v(22)], &[])?;
        op(&mut b, "v_add_lshl_u32 v22, v22, v0, 2", &[v(22)], &[v(22), v(0)])?;
        b.ds_crosslane(Instruction::new(
            "ds_bpermute_b32 v21, v22, v4",
            vec![v(21)], vec![v(22), v(4)],
        ).memory(MemoryClass::DsLoad))?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        op(&mut b, "v_add_f32_e32 v4, v4, v21", &[v(4)], &[v(4), v(21)])?;
    }
    mem(&mut b, "global_store_b32 v1, v4, s[10:11]", &[], &[v(1), v(4), sr(10, 2)], MemoryClass::VmemStore)?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    #[test]
    fn g0_quad_m7() -> Result<(), String> {
        let emitted = super::build_g0()?;
        let base = std::path::Path::new("/home/kaden/ClaudeCode/warpfront/wt-pmdt-plan/kernels/pm-decode/gfx1201/fused_qkvza_hfq4g256_mq4v2.g0");
        std::fs::write(base.with_extension("g0.s"), &emitted.s_text).map_err(|e| e.to_string())?;
        let bytes = crate::native::assemble(&emitted.s_text, crate::Arch::Gfx1201)?;
        let object = base.with_extension("g0.co");
        std::fs::write(&object, bytes).map_err(|e| e.to_string())?;
        let evidence = crate::pm_check::m7(&object, "gfx1201", "pm_decode_qkvza_g0")?;
        println!("QKVZA G0 M7: {evidence}");
        Ok(())
    }
}
