use crate::{Builder, kernels::{bf16::Bf16, common::{op, v}, iu4_gemm::region::{Binding, Region, emit_interleaved}}};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epilogue { Set, Add, SiLU, Bf16Rne }
impl Epilogue {
    /// SET copies the sum; ADD keeps the existing output as the left operand.
    pub fn apply(self, b: &mut Builder, sum: u8, output: u8) -> Result<(), String> {
        match self {
            Self::Set if sum == output => Ok(()),
            Self::Set => op(b, format!("v_mov_b32_e32 v{output}, v{sum}"), &[v(output)], &[v(sum)]),
            Self::Add => op(b, format!("v_add_f32_e32 v{output}, v{output}, v{sum}"), &[v(output)], &[v(output), v(sum)]),
            _ => Err("SiLU/BF16 need explicit temporary bindings".into()),
        }
    }
    /// Instantiate the imported hipcc region, preserving its per-value DAG.
    pub fn silu(b: &mut Builder, bindings: &[Binding]) -> Result<(), String> {
        let region = if b.spec.arch.gfx12() { Region::silu()? } else { Region::silu_gfx1100()? };
        emit_interleaved(b, &region, bindings)
    }
    /// Dense V2B's interleaved lowering of the same hipcc SiLU contract.
    pub fn silu_dense(b: &mut Builder, gate: u8, up: u8, tmp: u8, mask: u8, n: u8) -> Result<(), String> {
        crate::kernels::iu4_v2b::silu_mul(b, gate, up, tmp, mask, n)
    }
    pub fn bf16_rne(b: &mut Builder, src: u8, dst: u8, scratch: [u8; 4]) -> Result<(), String> {
        Bf16::hip_bfloat16(b, src, dst, scratch)
    }
}
