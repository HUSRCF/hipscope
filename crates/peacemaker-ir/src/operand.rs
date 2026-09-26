use std::fmt;
use crate::{cfg::BlockId, reg::RegRef, wait::WaitImm};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Half { Hi, Lo }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Special { Vcc, VccLo, VccHi, Exec, ExecLo, ExecHi, Scc, M0, Null, Ttmp(u8), FlatScratch, SrcSharedBase, SrcSharedLimit, SrcPrivateBase, SrcPrivateLimit, Pc }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InlineConst { Integer(i8), FloatBits(u32), InvTwoPi }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImmField { Sopp(i16), Sopk(i16), DsOffset0(u8), DsOffset1(u8), VmemOffset(i32), SmemOffset(i32), Unsigned(u32) }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HwReg { pub id: u8, pub offset: u8, pub size: u8 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Msg { pub id: u8, pub op: u8 }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Operand { Reg(RegRef), Special(Special), Inline(InlineConst), Literal(u32), Imm(ImmField), Label(BlockId), Hwreg(HwReg), SendMsg(Msg), Half(RegRef, Half) }
impl Operand {
    pub fn validate(&self) -> Result<(), String> {
        match self { Self::Reg(r) | Self::Half(r, _) => r.validate(),
            Self::Imm(ImmField::VmemOffset(n)) if !(-8_388_608..=8_388_607).contains(n) => Err("VMEM offset exceeds signed 24-bit field".into()),
            Self::Imm(ImmField::SmemOffset(n)) if !(-1_048_576..=1_048_575).contains(n) => Err("SMEM offset exceeds signed 21-bit field".into()),
            Self::Inline(InlineConst::Integer(n)) if !(-16..=64).contains(n) => Err("inline integer outside ISA range".into()),
            _ => Ok(()) }
    }
}
impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(r) => write!(f, "{r}"),
            Self::Special(s) => f.write_str(match s { Special::Vcc => "vcc", Special::VccLo => "vcc_lo", Special::VccHi => "vcc_hi", Special::Exec => "exec", Special::ExecLo => "exec_lo", Special::ExecHi => "exec_hi", Special::Scc => "scc", Special::M0 => "m0", Special::Null => "null", Special::Ttmp(_) => return write!(f, "ttmp{}", if let Special::Ttmp(n) = s { n } else { unreachable!() }), Special::FlatScratch => "flat_scratch", Special::SrcSharedBase => "src_shared_base", Special::SrcSharedLimit => "src_shared_limit", Special::SrcPrivateBase => "src_private_base", Special::SrcPrivateLimit => "src_private_limit", Special::Pc => "pc" }),
            Self::Inline(InlineConst::Integer(n)) => write!(f, "{n}"),
            Self::Inline(InlineConst::FloatBits(bits)) => write!(f, "{}", f32::from_bits(*bits)),
            Self::Inline(InlineConst::InvTwoPi) => f.write_str("0.15915494"),
            Self::Literal(n) => write!(f, "{n:#x}"),
            Self::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => write!(f, "{n}"),
            Self::Imm(ImmField::DsOffset0(n)) => write!(f, "offset0:{n}"),
            Self::Imm(ImmField::DsOffset1(n)) => write!(f, "offset1:{n}"),
            Self::Imm(ImmField::VmemOffset(n) | ImmField::SmemOffset(n)) => write!(f, "{n}"),
            Self::Imm(ImmField::Unsigned(n)) => write!(f, "{n:#x}"),
            Self::Label(id) => write!(f, ".LBB{}", id.0),
            Self::Hwreg(r) => write!(f, "hwreg({}, {}, {})", r.id, r.offset, r.size),
            Self::SendMsg(m) => write!(f, "sendmsg({}, {})", m.id, m.op),
            Self::Half(r, part) => write!(f, "{r}.{}", match part { Half::Hi => "h", Half::Lo => "l" }),
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Omod { #[default] None, Mul2, Mul4, Div2 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CachePolicy { pub th: u8, pub scope: u8, pub nv: bool, pub glc: bool, pub slc: bool, pub dlc: bool }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dpp { pub ctrl: u16, pub row_mask: u8, pub bank_mask: u8, pub bound_ctrl: bool }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Sdwa { pub dst_sel: u8, pub src0_sel: u8, pub src1_sel: u8 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DsMods { pub gds: bool }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DelayAluHint { pub instid0: u8, pub instskip: u8, pub instid1: u8 }
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Modifiers {
    pub neg: u8, pub abs: u8, pub op_sel: u8, pub op_sel_hi: u8,
    pub neg_lo: u8, pub neg_hi: u8, pub clamp: bool, pub omod: Omod,
    pub cpol: CachePolicy, pub dpp: Option<Dpp>, pub sdwa: Option<Sdwa>,
    pub ds: DsMods, pub wait: Option<WaitImm>, pub delay: Option<DelayAluHint>, pub clause: Option<u8>,
}
impl Modifiers {
    pub fn text_suffix(&self) -> String {
        let mut out = String::new();
        if self.neg_lo != 0 { out.push_str(&format!(" neg_lo:[{},{},{}]", self.neg_lo & 1, (self.neg_lo >> 1) & 1, (self.neg_lo >> 2) & 1)); }
        if self.neg_hi != 0 { out.push_str(&format!(" neg_hi:[{},{},{}]", self.neg_hi & 1, (self.neg_hi >> 1) & 1, (self.neg_hi >> 2) & 1)); }
        if self.op_sel != 0 { out.push_str(&format!(" op_sel:[{},{},{}]", self.op_sel & 1, (self.op_sel >> 1) & 1, (self.op_sel >> 2) & 1)); }
        if self.op_sel_hi != 0 { out.push_str(&format!(" op_sel_hi:[{},{},{}]", self.op_sel_hi & 1, (self.op_sel_hi >> 1) & 1, (self.op_sel_hi >> 2) & 1)); }
        if self.clamp { out.push_str(" clamp"); }
        out
    }
}
