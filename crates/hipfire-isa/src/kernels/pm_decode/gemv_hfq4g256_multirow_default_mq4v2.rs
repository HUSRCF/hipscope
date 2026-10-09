//! Exact arithmetic admission region for the gfx1201 multirow-r2 twin.
//! The production selector fails closed until G0 and the complete twin pass.

use crate::{
    insn::{Instruction, MemoryClass},
    reg::{Live, V},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

const MODULE: &str = "gemv_hfq4g256_multirow_default_mq4v2";
const G0_SYMBOL: &str = "gemv_mq4g256v2_multirow_r2_dog_g0";

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Err(format!(
        "K[{MODULE}]: admission pending: real-H2 DOG region oracle and complete multirow-r2 M7/output identity"
    ))
}

/// One group of each of two independent rows, before stream accumulation.
///
/// Test-only launch contract: one wave32; A contains exactly two 136-byte
/// groups, x contains 256 f32s, y contains 64 lane partials (row-major).
/// M=2 and K=256 occupy the frozen by-value slots but are not shape selectors.
/// This symbol is never returned by the production entry point.
///
/// The selected clang24 tail DAG starts row0 with x1*w1, then fmac x0*w0;
/// row1 starts x0*w0, then fmac x1*w1. Keep that distinction: even the
/// first two terms must not be swapped under the byte-identity contract.
pub fn build_dog_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(20, 10)?;
    let tid = regs.v::<1>("tid", 0, Live::Whole)?;
    let addr = regs.v::<1>("address", 1, Live::Whole)?;
    let packed0 = regs.v::<1>("packed0", 2, Live::Whole)?;
    let packed1 = regs.v::<1>("packed1", 3, Live::Whole)?;
    let header0 = regs.v::<1>("header0", 4, Live::Whole)?;
    let header1 = regs.v::<1>("header1", 5, Live::Whole)?;
    let weight = regs.v::<1>("weight", 6, Live::Whole)?;
    let dot0 = regs.v::<1>("dot0", 7, Live::Whole)?;
    let dot1 = regs.v::<1>("dot1", 8, Live::Whole)?;
    let x_lo = regs.v::<4>("x_lo", 12, Live::Whole)?;
    let x_hi = regs.v::<4>("x_hi", 16, Live::Whole)?;
    let args = regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<1>("workgroup_x", 2, Live::Whole)?;
    let a_x = regs.s::<4>("a_x", 4, Live::Whole)?;
    let y = regs.s::<2>("y", 8, Live::Whole)?;
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: MODULE.into(),
            variant: "dog-g0-not-production".into(),
            arch: Arch::Gfx1201,
            symbol: G0_SYMBOL.into(),
            kernargs: KernargLayout::new(32)
                .pointer("A", 0)
                .pointer("x", 8)
                .pointer("y", 16)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value"),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    b.push(Instruction::new(
        "s_load_b128 s[4:7], s[0:1], 0x0",
        vec![a_x.reg()], vec![args.reg()],
    ).memory(MemoryClass::SmemLoad))?;
    b.push(Instruction::new(
        "s_load_b64 s[8:9], s[0:1], 0x10",
        vec![y.reg()], vec![args.reg()],
    ).memory(MemoryClass::SmemLoad))?;
    // Each subgroup of 16 lanes selects its own packed half scale/zero.
    b.push(Instruction::new("v_lshrrev_b32_e32 v1, 4, v0",
        vec![addr.reg()], vec![tid.reg()]))?;
    b.push(Instruction::new("v_lshlrev_b32_e32 v1, 2, v1",
        vec![addr.reg()], vec![addr.reg()]))?;
    for (header, offset) in [(header0, 0), (header1, 136)] {
        b.push(Instruction::new(
            if offset == 0 { format!("global_load_b32 {}, v1, s[4:5]", header.reg()) }
            else { format!("global_load_b32 {}, v1, s[4:5] offset:{offset}", header.reg()) },
            vec![header.reg()], vec![addr.reg(), crate::S::<2>(4).reg()],
        ).memory(MemoryClass::VmemLoad))?;
    }
    b.push(Instruction::new("v_lshlrev_b32_e32 v1, 2, v0",
        vec![addr.reg()], vec![tid.reg()]))?;
    for (packed, offset) in [(packed0, 8), (packed1, 144)] {
        b.push(Instruction::new(
            format!("global_load_b32 {}, v1, s[4:5] offset:{offset}", packed.reg()),
            vec![packed.reg()], vec![addr.reg(), crate::S::<2>(4).reg()],
        ).memory(MemoryClass::VmemLoad))?;
    }
    b.push(Instruction::new("v_lshlrev_b32_e32 v1, 5, v0",
        vec![addr.reg()], vec![tid.reg()]))?;
    for (x, offset) in [(x_lo, 0), (x_hi, 16)] {
        b.push(Instruction::new(
            if offset == 0 { format!("global_load_b128 {}, v1, s[6:7]", x.reg()) }
            else { format!("global_load_b128 {}, v1, s[6:7] offset:{offset}", x.reg()) },
            vec![x.reg()], vec![addr.reg(), crate::S::<2>(6).reg()],
        ).memory(MemoryClass::VmemLoad))?;
    }
    for (packed, header, dot, order) in [
        (packed0, header0, dot0, [1, 0, 2, 3, 4, 5, 6, 7]),
        (packed1, header1, dot1, [0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        for (term, nibble) in order.into_iter().enumerate() {
            b.push(Instruction::new(
                format!("v_bfe_u32 v6, {}, {}, 4", packed.reg(), nibble * 4),
                vec![weight.reg()], vec![packed.reg()],
            ))?;
            b.push(Instruction::new("v_cvt_f32_ubyte0_e32 v6, v6",
                vec![weight.reg()], vec![weight.reg()]))?;
            b.push(Instruction::new(
                format!("v_fma_mix_f32 v6, {h}, v6, {h} op_sel:[0,0,1] op_sel_hi:[1,0,1]", h = header.reg()),
                vec![weight.reg()], vec![header.reg(), weight.reg()],
            ))?;
            let x = V::<1>(12 + nibble);
            let mnemonic = if term == 0 { "v_mul_f32_e32" } else { "v_fmac_f32_e32" };
            let mut uses = vec![x.reg(), weight.reg()];
            if term != 0 { uses.push(dot.reg()); }
            b.push(Instruction::new(format!("{mnemonic} {}, {}, v6", dot.reg(), x.reg()),
                vec![dot.reg()], uses))?;
        }
    }
    b.push(Instruction::new("v_lshlrev_b32_e32 v1, 2, v0",
        vec![addr.reg()], vec![tid.reg()]))?;
    for (dot, offset) in [(dot0, 0), (dot1, 128)] {
        b.push(Instruction::new(
            if offset == 0 { format!("global_store_b32 v1, {}, s[8:9]", dot.reg()) }
            else { format!("global_store_b32 v1, {}, s[8:9] offset:{offset}", dot.reg()) },
            vec![], vec![addr.reg(), dot.reg(), y.reg()],
        ).memory(MemoryClass::VmemStore))?;
    }
    b.wait_all()?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;

    #[test]
    fn dog_region_native_m7() {
        let emitted = build_dog_g0().expect("checked DOG region");
        let object = crate::native::assemble(&emitted.s_text, Arch::Gfx1201)
            .expect("native region code object");
        let out = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/pm-decode/gfx1201/"
        ));
        let prefix = out.join(format!("{MODULE}.g0"));
        std::fs::write(prefix.with_extension("g0.s"), &emitted.s_text).unwrap();
        std::fs::write(prefix.with_extension("g0.proof.json"),
            serde_json::to_vec_pretty(&emitted.proof).unwrap()).unwrap();
        let contract = serde_json::json!({
            "symbol": G0_SYMBOL,
            "counts": {
                "v_fma_mix_f32": 16,
                "v_mul_f32_e32": 2,
                "v_fmac_f32_e32": 14
            },
            "forbidden": ["scratch_load_b32", "scratch_store_b32"],
            "vgpr_max": 20,
            "sgpr_max": 10,
            "require_wave32": true,
            "require_zero_spills": true,
            "require_zero_private": true,
            "launch_dynamic_lds_bytes": 0,
            "group_segment_fixed_bytes": 0
        });
        std::fs::write(prefix.with_extension("g0.shape.json"),
            serde_json::to_vec_pretty(&contract).unwrap()).unwrap();
        let co = prefix.with_extension("g0.co");
        std::fs::write(&co, object).unwrap();
        let m7 = crate::pm_check::m7(&co, "gfx1201", G0_SYMBOL).expect("M7{}");
        println!("{m7}");
        assert_eq!(m7["obligations"], serde_json::json!({}));
        std::fs::write(prefix.with_extension("g0.m7.json"),
            serde_json::to_vec_pretty(&m7).unwrap()).unwrap();
    }
}
