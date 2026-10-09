// SPDX-License-Identifier: Apache-2.0
//! QKV's gate-first, single-group diagnostic. This export is deliberately
//! distinct from the production twin: it writes 32 lane partials to `y_q`.
//! Its input is the first 136-byte Q group and the first 256 floats of `x`.
//! It is not admitted by the native decode selector.
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use crate::insn::MemoryClass;
use crate::reg::{Kind, Live};
use crate::kernels::common::{mem, op, smem, sr, v, vr};

const MODULE: &str = "fused_qkv_hfq4g256_mq4v2";
const G0_SYMBOL: &str = "fused_qkv_mq4g256v2_g0_group";

fn kernargs() -> KernargLayout {
    let mut args = KernargLayout::new(72);
    for (name, offset) in [
        ("A_q", 0), ("A_k", 8), ("A_v", 16), ("x", 24),
        ("y_q", 32), ("y_k", 40), ("y_v", 48),
    ] {
        args = args.pointer(name, offset);
    }
    for (i, name) in ["q_m", "k_m", "v_m", "K"].iter().enumerate() {
        args = args.hidden(name, 56 + i as u32 * 4, 4, "by_value");
    }
    args
}

/// Exact clang24 lane dot: product 1, FMA 0, then FMAs 2..7.
/// Dequantization consumes the packed low-half scale/high-half zero point
/// directly, using the selected body's mixed-precision FMA operand modes.
fn group_dot(b: &mut Builder) -> Result<(), String> {
    for i in 0..8u8 {
        let dst = 16 + i;
        op(b, format!("v_bfe_u32 v{dst}, v6, {}, 4", i * 4),
            &[v(dst)], &[v(6)])?;
        op(b, format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
            &[v(dst)], &[v(dst)])?;
        op(b, format!("v_fma_mix_f32 v{dst}, v5, v{dst}, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
            &[v(dst)], &[v(5), v(dst)])?;
    }
    op(b, "v_mul_f32_e32 v24, v17, v9", &[v(24)], &[v(17), v(9)])?;
    for i in [0u8, 2, 3, 4, 5, 6, 7] {
        op(b, format!("v_fmac_f32_e32 v24, v{}, v{}", 16 + i, 8 + i),
            &[v(24)], &[v(24), v(16 + i), v(8 + i)])?;
    }
    // The incumbent adds the completed dot to its zero-seeded stream,
    // independently of the seven FMAs (important for signed zero).
    op(b, "v_mov_b32_e32 v25, 0", &[v(25)], &[])?;
    op(b, "v_add_f32_e32 v24, v25, v24", &[v(24)], &[v(25), v(24)])
}

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Err("QKV production admission requires the real-H2 G0 oracle and checked crosslane reduction".into())
}

/// Diagnostic only: never returned from the production entry point.
pub fn build_g0() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(32, 16)?;
    regs.add_range("workitem", Kind::V, 0, 1, Live::Whole)?;
    for base in [1, 2, 3, 4, 5, 6, 7, 24, 25] {
        regs.add_range(&format!("v{base}"), Kind::V, base, 1, Live::Whole)?;
    }
    for base in [8, 16] {
        regs.add_range(&format!("v{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)?;
    regs.add_range("wg_x", Kind::S, 2, 1, Live::Whole)?;
    for base in [4, 6, 8] {
        regs.add_range(&format!("s{base}"), Kind::S, base, 2, Live::Whole)?;
    }
    let mut b = Builder::new(KernelSpec {
        kernel_id: MODULE.into(), variant: "g0-single-group-lane-dot".into(),
        arch: Arch::Gfx1201, symbol: G0_SYMBOL.into(), kernargs: kernargs(),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0,
        wave32: true, cu_mode: false,
    }, regs);
    smem(&mut b, 4, 2, 0, 0)?;
    smem(&mut b, 6, 2, 0, 24)?;
    smem(&mut b, 8, 2, 0, 32)?;
    op(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
    op(&mut b, "v_add_nc_u32_e32 v2, 8, v1", &[v(2)], &[v(1)])?;
    op(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[v(3)], &[v(0)])?;
    op(&mut b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[v(0)])?;
    op(&mut b, "v_cndmask_b32_e64 v4, 4, 0, vcc_lo", &[v(4)], &[])?;
    mem(&mut b, "global_load_b32 v5, v4, s[4:5]", &[v(5)],
        &[v(4), sr(4, 2)], MemoryClass::VmemLoad)?;
    mem(&mut b, "global_load_b32 v6, v2, s[4:5]", &[v(6)],
        &[v(2), sr(4, 2)], MemoryClass::VmemLoad)?;
    mem(&mut b, "global_load_b128 v[8:11], v3, s[6:7]", &[vr(8, 4)],
        &[v(3), sr(6, 2)], MemoryClass::VmemLoad)?;
    mem(&mut b, "global_load_b128 v[12:15], v3, s[6:7] offset:16", &[vr(12, 4)],
        &[v(3), sr(6, 2)], MemoryClass::VmemLoad)?;
    group_dot(&mut b)?;
    mem(&mut b, "global_store_b32 v1, v24, s[8:9]", &[],
        &[v(1), v(24), sr(8, 2)], MemoryClass::VmemStore)?;
    b.wait_all()?;
    op(&mut b, "s_endpgm", &[], &[])?;
    Ok(vec![b.finish()?])
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn g0_emit_and_m7() {
        let emitted = build_g0().expect("checked G0 emission");
        let e = &emitted[0];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/pm-decode/gfx1201");
        std::fs::create_dir_all(&root).unwrap();
        let path = |suffix: &str| root.join(format!("{MODULE}.g0.{suffix}"));
        std::fs::write(path("s"), &e.s_text).unwrap();
        std::fs::write(path("proof.json"), serde_json::to_vec_pretty(&e.proof).unwrap()).unwrap();
        let contract = crate::toolchain::IsaShapeContract {
            symbol: G0_SYMBOL.into(), vgpr_max: Some(32), sgpr_max: Some(16),
            require_wave32: true, require_zero_spills: true,
            require_zero_private: true, launch_dynamic_lds_bytes: Some(0),
            forbidden: vec!["scratch_*".into()],
            ..Default::default()
        };
        std::fs::write(path("shape.json"), serde_json::to_vec_pretty(&contract).unwrap()).unwrap();
        let elf = crate::native::assemble(&e.s_text, Arch::Gfx1201).expect("native code object");
        std::fs::write(path("co"), elf).unwrap();
        let m7 = crate::pm_check::m7(&path("co"), "gfx1201", G0_SYMBOL)
            .expect("G0 requires M7 obligations {}");
        println!("{m7}");
        std::fs::write(path("m7.json"), serde_json::to_vec_pretty(&m7).unwrap()).unwrap();
    }
}
