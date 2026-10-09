//! Exact residual DOG arithmetic admission probe; not a selectable decode twin.
//!
//! Each of 32 input records is 48 bytes: packed half scale/zero for row 0,
//! packed nibbles for row 0, packed half scale/zero for row 1, packed nibbles
//! for row 1, then eight f32 activations. Output is two f32 values per lane.
//! Headers must already select the lane's half-group. This isolates the
//! clang24 second-stream order before admitting the complete generic kernel.
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Live, S, V};

const G0_SYMBOL: &str = "pm_residual_dog_g0";

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    Err("residual twin not admitted: run the separately named DOG G0 region and real-H2 oracle before full emission".into())
}

fn build_dog_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(32, 8)?;
    regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<2>("input", 4, Live::Whole)?;
    regs.s::<2>("output", 6, Live::Whole)?;
    regs.v::<1>("lane", 0, Live::Whole)?;
    regs.v::<1>("input_offset", 1, Live::Whole)?;
    regs.v::<1>("output_offset", 2, Live::Whole)?;
    for (name, base) in [("headers",4),("x_lo",8),("x_hi",12),("weights_lo",16),("weights_hi",20)] {
        regs.v::<4>(name, base, Live::Whole)?;
    }
    regs.v::<2>("results",24, Live::Whole)?;
    let mut b = Builder::new(KernelSpec {
        kernel_id: "gemv_hfq4g256_residual_mq4v2".into(), variant: "dog_g0_only".into(),
        arch: Arch::Gfx1201, symbol: G0_SYMBOL.into(),
        kernargs: KernargLayout::new(16).pointer("records",0).pointer("out",8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    b.push(Instruction::new("s_load_b128 s[4:7], s[0:1], 0", vec![S::<2>(4).reg(), S::<2>(6).reg()], vec![S::<2>(0).reg()]).memory(MemoryClass::SmemLoad))?;
    b.push(Instruction::new("v_mul_lo_u32 v1, 48, v0", vec![V::<1>(1).reg()],vec![V::<1>(0).reg()]))?;
    b.push(Instruction::new("v_lshlrev_b32_e32 v2, 3, v0",vec![V::<1>(2).reg()],vec![V::<1>(0).reg()]))?;
    for (dst,offset) in [(4,0),(8,16),(12,32)] {
        b.push(Instruction::new(format!("global_load_b128 v[{}:{}], v1, s[4:5] offset:{}",dst,dst+3,offset),vec![V::<4>(dst).reg()],vec![V::<1>(1).reg(),S::<2>(4).reg()]).memory(MemoryClass::VmemLoad))?;
    }
    // The row-0 chain starts x0 then x1. The row-1 chain starts x1 then x0
    // in the frozen clang24 DAG (0x1d98/0x1db4), not the source's apparent order.
    for row in 0..2u8 {
        let header=4+row*2;
        let packed=header+1;
        for nibble in 0..8u8 {
            let dst=16+nibble;
            b.push(Instruction::new(format!("v_bfe_u32 v{dst}, v{packed}, {}, 4",nibble*4),vec![V::<1>(dst).reg()],vec![V::<1>(packed).reg()]))?;
            b.push(Instruction::new(format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),vec![V::<1>(dst).reg()],vec![V::<1>(dst).reg()]))?;
            b.push(Instruction::new(format!("v_fma_mix_f32 v{dst}, v{header}, v{dst}, v{header} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),vec![V::<1>(dst).reg()],vec![V::<1>(header).reg(),V::<1>(dst).reg()]))?;
        }
        let result=24+row;
        let order=if row==0 {[0,1,2,3,4,5,6,7]} else {[1,0,2,3,4,5,6,7]};
        for (step,nibble) in order.into_iter().enumerate() {
            let x=8+nibble;
            let weight=16+nibble;
            let (mnemonic,uses)=if step==0 {("v_mul_f32_e32",vec![V::<1>(x).reg(),V::<1>(weight).reg()])} else {("v_fmac_f32_e32",vec![V::<1>(x).reg(),V::<1>(weight).reg(),V::<1>(result).reg()])};
            b.push(Instruction::new(format!("{mnemonic} v{result}, v{x}, v{weight}"),vec![V::<1>(result).reg()],uses))?;
        }
    }
    b.push(Instruction::new("global_store_b64 v2, v[24:25], s[6:7]",vec![],vec![V::<1>(2).reg(),V::<2>(24).reg(),S::<2>(6).reg()]).memory(MemoryClass::VmemStore))?;
    b.wait_all()?;
    b.control(Instruction::new("s_endpgm",vec![],vec![]))?;
    b.finish()
}

#[cfg(all(test, feature="toolchain"))]
mod tests {
    use super::*;
    #[test]
    fn residual_dog_g0_m7() {
        let emitted=build_dog_g0().expect("checked DOG region");
        let dir=std::env::temp_dir().join(format!("pm-residual-g0-{}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source=dir.join("gemv_hfq4g256_residual_mq4v2.g0.s");
        let elf=dir.join("gemv_hfq4g256_residual_mq4v2.g0.co");
        std::fs::write(&source,&emitted.s_text).unwrap();
        std::fs::write(&elf,crate::native::assemble(&emitted.s_text,Arch::Gfx1201).expect("native assemble")).unwrap();
        let report=crate::pm_check::m7(&elf,"gfx1201",G0_SYMBOL).expect("M7");
        println!("{}\n{}",source.display(),report);
        assert_eq!(report["obligations"],serde_json::json!({}),"{report}");
    }
}
