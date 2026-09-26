//! Table-gated gfx1201 decoder and encoder. No provenance bytes participate in encoding.
use smallvec::SmallVec;
use crate::{inst::{Arch, Form, FormFields, Inst, NamedField}, isa::{self, FieldClass, OpRow}, operand::{CacheScope, DelayAluHint, Half, ImmField, InlineConst, Modifiers, Msg, Omod, Operand, Special, VmemToken}, provenance::Provenance, reg::{Kind, RegRef}, wait::{Counter, WaitImm}};
use super::forms::{self, Field};

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("unrecognized machine instruction at byte offset {offset}: {reason}")]
    Rejected { offset: usize, reason: String },
}
fn reject(reason: impl Into<String>) -> DecodeError { DecodeError::Rejected { offset: 0, reason: reason.into() } }
fn value(field: Option<Field>, words: &[u32]) -> u32 { field.map_or(0, |f| f.value(words)) }
fn assign(field: Field, words: &mut [u32], v: u32) -> Result<(), DecodeError> {
    field.set(words, v).map_err(|name| reject(format!("{name} does not fit its encoding field")))
}
fn select(code: u32, width: u8, scalar_dest: bool, literal: Option<u32>) -> Result<Operand, DecodeError> {
    let reg = |kind, base, len| Operand::Reg(RegRef { kind, base, len });
    if code <= 105 { return Ok(reg(Kind::S, code as u16, width)); }
    if code >= 256 && code <= 511 { return Ok(reg(Kind::V, (code - 256) as u16, width)); }
    if scalar_dest && code <= 127 {
        return Ok(match code { 106 => Operand::Special(Special::VccLo), 107 => Operand::Special(Special::VccHi),
            124 => Operand::Special(Special::Null), 126 => Operand::Special(Special::ExecLo), 127 => Operand::Special(Special::ExecHi),
            _ => return Err(reject(format!("unknown scalar destination {code:#x}"))) });
    }
    Ok(match code {
        106 => Operand::Special(Special::VccLo), 107 => Operand::Special(Special::VccHi),
        108 => Operand::Special(Special::Ttmp(0)), 109 => Operand::Special(Special::Ttmp(1)),
        110..=123 => Operand::Reg(RegRef { kind: Kind::Ttmp, base: (code - 108) as u16, len: width }),
        124 => Operand::Special(Special::M0), 125 => Operand::Special(Special::Null),
        126 => Operand::Special(Special::ExecLo), 127 => Operand::Special(Special::ExecHi),
        128..=192 => Operand::Inline(InlineConst::Integer((code - 128) as i8)),
        193..=208 => Operand::Inline(InlineConst::Integer((192 - code as i32) as i8)),
        240..=247 => Operand::Inline(InlineConst::FloatBits(match code {
            240 => 0x3f000000, 241 => 0xbf000000, 242 => 0x3f800000, 243 => 0xbf800000,
            244 => 0x40000000, 245 => 0xc0000000, 246 => 0x40800000, _ => 0xc0800000,
        })),
        248 => Operand::Inline(InlineConst::InvTwoPi),
        255 => Operand::Literal(literal.ok_or_else(|| reject("literal selector without literal word"))?),
        _ => return Err(reject(format!("unknown source selector {code:#x}"))),
    })
}
fn selector(operand: &Operand, width: u8, scalar_dest: bool) -> Result<u32, DecodeError> {
    match operand {
        Operand::Reg(r) | Operand::Half(r, _) => {
            if r.len != width { return Err(reject(format!("register width mismatch: {} instead of {width}", r.len))); }
            Ok(match r.kind { Kind::V => u32::from(r.base) + 256, Kind::S => u32::from(r.base), Kind::Ttmp => u32::from(r.base) + 108 })
        }
        Operand::Special(s) => Ok(match s {
            Special::VccLo | Special::Vcc => 106, Special::VccHi => 107,
            Special::Ttmp(n) => u32::from(*n) + 108, Special::M0 => 124,
            Special::Null if scalar_dest => 124, Special::Null => 125,
            Special::ExecLo | Special::Exec => 126, Special::ExecHi => 127,
            _ => return Err(reject(format!("unencodable special register {s:?}"))),
        }),
        Operand::Inline(InlineConst::Integer(n)) if (0..=64).contains(n) => Ok((*n as u32) + 128),
        Operand::Inline(InlineConst::Integer(n)) if (-16..=-1).contains(n) => Ok((192 - i32::from(*n)) as u32),
        Operand::Inline(InlineConst::FloatBits(bits)) => [0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x40000000,0xc0000000,0x40800000,0xc0800000]
            .iter().position(|x| x == bits).map(|i| i as u32 + 240).ok_or_else(|| reject("unencodable inline float")),
        Operand::Inline(InlineConst::InvTwoPi) => Ok(248), Operand::Literal(_) => Ok(255),
        _ => Err(reject(format!("operand is not a register/source selector: {operand:?}"))),
    }
}
fn reg(kind: Kind, base: u32, bits: u16) -> Operand {
    Operand::Reg(RegRef { kind, base: base as u16, len: (bits / 32).max(1) as u8 })
}
fn operand(name: &str, bits: u16, code: u32, literal: Option<u32>, row: &OpRow, words: &[u32]) -> Result<Operand, DecodeError> {
    let form = row.form;
    let width = (bits / 32).max(1) as u8;
    if name == "SIMM16" {
        if row.name == "s_sendmsg" && code == 3 { return Ok(Operand::SendMsg(Msg { id: 3, op: 0 })); }
        return Ok(Operand::Imm(if form == Form::Sopk { ImmField::Sopk(code as i16) } else { ImmField::Sopp(code as i16) }));
    }
    if name == "SOFFSET" && form == Form::Smem && code == 124 {
        return Ok(Operand::Imm(ImmField::SmemOffset(field_value("IOFFSET", row, words) as i32)));
    }
    if name == "SBASE" { return Ok(reg(Kind::S, code * 2, bits)); }
    if name == "RSRC" { return Ok(reg(Kind::S, code, bits)); }
    if name == "SADDR" && code == 124 { return Ok(Operand::Vmem(VmemToken::Off)); }
    let scalar_dest = name == "SDST" || name == "SDATA"
        || name == "VDST" && (row.name.starts_with("v_cmp") || row.name.starts_with("v_s_")
            || row.name == "v_readlane_b32" || row.name == "v_readfirstlane_b32");
    if scalar_dest { return select(code, width, true, literal); }
    if name == "VDST" || name == "VDATA" || name == "VSRC" || name == "VADDR" || name == "ADDR"
        || name.starts_with("DATA") || name.starts_with("VDST") || name.starts_with("VSRC") {
        let r = RegRef { kind: Kind::V, base: code as u16, len: width };
        if bits == 16 && name == "VDST" { return Ok(Operand::Half(r, if field_value("OPSEL",row,words) & 0x8 != 0 { Half::Hi } else { Half::Lo })); }
        if form == Form::Vmem(crate::inst::VmemForm::Buffer) && name == "VADDR"
            && field_value("IDXEN",row,words) ^ field_value("OFFEN",row,words) == 1 {
            return Ok(Operand::Reg(RegRef { len: 1, ..r }));
        }
        return Ok(Operand::Reg(r));
    }
    let src=select(code, width, false, literal)?;
    if bits == 16 {
        if let Operand::Reg(r) = src { return Ok(Operand::Half(r, if field_value("OPSEL",row,words) & 1 != 0 { Half::Hi } else { Half::Lo })); }
    }
    Ok(src)
}
fn encoded_operand(name: &str, bits: u16, op: &Operand, form: Form) -> Result<u32, DecodeError> {
    match (name, op) {
        ("SIMM16", Operand::SendMsg(Msg { id:3,op:0 })) => Ok(3),
        ("SIMM16", Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n))) => Ok((*n as u16).into()),
        ("SOFFSET", Operand::Imm(ImmField::SmemOffset(_))) if form == Form::Smem => Ok(124),
        ("SBASE", Operand::Reg(r)) if r.kind == Kind::S && r.base % 2 == 0 => Ok(u32::from(r.base / 2)),
        ("SADDR", Operand::Vmem(VmemToken::Off)) => Ok(124),
        ("RSRC", Operand::Reg(r)) if r.kind == Kind::S => Ok(u32::from(r.base)),
        (_, Operand::Reg(r)) if name.starts_with('V') || name.starts_with("DATA") || name == "ADDR" => {
            if r.kind != Kind::V && !(name == "VDST" && (form == Form::Vop1 || form == Form::Vop3)) { return Err(reject("vector operand has wrong register bank")); }
            if r.len != (bits / 32).max(1) as u8 && !(name=="VADDR" && form==Form::Vmem(crate::inst::VmemForm::Buffer) && r.len==1) { return Err(reject(format!("{name} register width mismatch"))); }
            Ok(u32::from(r.base))
        }
        ("VDST", Operand::Special(s)) if form == Form::Vop3 => selector(&Operand::Special(*s), (bits/32).max(1) as u8, true),
        ("VDST" | "SRC0", Operand::Half(r, _)) => Ok(if name=="VDST" { u32::from(r.base) } else { u32::from(r.base) + 256 }),
        (_, _) => selector(op, (bits / 32).max(1) as u8, name == "SDST"),
    }
}
fn grammars(row: &OpRow) -> impl Iterator<Item = (&str, u16)> {
    row.grammar.split(',').filter_map(|part| {
        let (name, bits) = part.split_once(':')?;
        Some((name, bits.parse().ok()?))
    })
}
fn width(row: &OpRow, words: &[u32]) -> Result<usize, DecodeError> {
    let base = match row.form { Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Sopk | Form::Sopp | Form::Vop1 | Form::Vop2 | Form::Vopc => 1,
        Form::Vmem(_) => 3, _ => 2 };
    if words.len() < base { return Err(reject(format!("truncated {} instruction (need {base} words)", row.name))); }
    let literal = match row.form {
        Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Vop1 | Form::Vop2 | Form::Vopc | Form::Vop3 | Form::Vop3p | Form::Vopd => {
            grammars(row).any(|(name, _)| {
                let source = name.starts_with("SRC") || name.starts_with("SSRC");
                source && forms::field(row.form, name).is_some_and(|f| value(Some(f), words) == 255)
            }) || row.form == Form::Vopd && forms::field(row.form,"SRCY0").is_some_and(|f| value(Some(f),words)==255)
        }
        _ => false,
    };
    let size = base + usize::from(literal);
    if size > 3 || words.len() < size { return Err(reject(format!("truncated or overlong {} instruction (need {size} words)", row.name))); }
    Ok(size)
}
fn field_value(name: &str, row: &OpRow, words: &[u32]) -> u32 {
    if name == "SDST" && row.form == Form::Vop3 { return (words[0] >> 8) & 127; }
    value(forms::field(row.form,name), words)
}
fn encode_field(name: &str, row: &OpRow, words: &mut [u32], value: u32) -> Result<(), DecodeError> {
    if name == "SDST" && row.form == Form::Vop3 { return assign(Field::new("SDST",8,7),words,value); }
    assign(forms::field(row.form,name).ok_or_else(|| reject(format!("unknown bitfield {name} in {}",row.name)))?,words,value)
}
fn special_mods(row: &OpRow, words: &[u32], mods: &mut Modifiers) {
    if row.form == Form::Vop3 {
        let carry = row.grammar.contains("SDST:");
        if !carry { mods.abs = field_value("ABS",row,words) as u8; mods.op_sel = field_value("OPSEL",row,words) as u8; }
        mods.neg = field_value("NEG",row,words) as u8;
        mods.clamp = field_value("CLAMP",row,words) != 0;
        mods.omod = match field_value("OMOD",row,words) { 1 => Omod::Mul2, 2 => Omod::Mul4, 3 => Omod::Div2, _ => Omod::None };
    } else if row.form == Form::Vop3p {
        mods.neg_lo = field_value("NEG",row,words) as u8;
        mods.neg_hi = field_value("NEG_HI",row,words) as u8;
        mods.op_sel = field_value("OPSEL",row,words) as u8;
        mods.op_sel_hi = (field_value("OPSEL_HI_LO",row,words) | field_value("OPSEL_HI_2",row,words) << 2) as u8;
        mods.clamp = field_value("CLAMP",row,words) != 0;
    }
    if row.form == Form::Smem || matches!(row.form,Form::Vmem(_)) {
        mods.cpol.th = field_value("TH",row,words) as u8;
        mods.cpol.scope = field_value("SCOPE",row,words) as u8;
        mods.cpol.nv = field_value("NV",row,words) != 0;
    }
    if row.form == Form::Sopp && row.name.starts_with("s_wait_") && row.name != "s_wait_alu" {
        let raw = words[0] as u16;
        let mut wait = WaitImm::default();
        if row.name.ends_with("loadcnt_dscnt") || row.name.ends_with("storecnt_dscnt") {
            let counter = if row.name.ends_with("loadcnt_dscnt") { Counter::Load } else { Counter::Store };
            wait.per_counter[counter as usize] = Some((raw >> 8) as u8 & 63);
            wait.per_counter[Counter::Ds as usize] = Some(raw as u8 & 63);
        }
        else { let counter = if row.name.ends_with("loadcnt") { Counter::Load } else if row.name.ends_with("dscnt") { Counter::Ds } else if row.name.ends_with("storecnt") { Counter::Store } else if row.name.ends_with("kmcnt") { Counter::Km } else if row.name.ends_with("samplecnt") { Counter::Sample } else if row.name.ends_with("bvhcnt") { Counter::Bvh } else { Counter::Exp };
            wait.per_counter[counter as usize] = Some(raw as u8); }
        mods.wait = Some(wait);
    }
    if row.name == "s_clause" { mods.clause = Some(words[0] as u8); }
    if row.name == "s_delay_alu" {
        let imm=words[0] as u16;
        mods.delay=Some(DelayAluHint { instid0: (imm&15) as u8, instskip: ((imm>>4)&7) as u8,
            instid1: ((imm>>7)&0x1ff) as u8 });
    }
}
fn fields_from(row: &OpRow, words: &[u32], consumed: &mut [u32;3]) -> Result<FormFields, DecodeError> {
    let mut ignored = SmallVec::new(); let mut honored = SmallVec::new();
    if row.form == Form::Vop3 && !row.grammar.contains("SRC2:") && !row.benign_src2.is_empty() {
        let src2 = field_value("SRC2",row,words);
        if src2 == 255 { return Err(reject("dangerous-fill src2_unused=0xff (literal selector)")); }
        consumed[1] |= forms::field(row.form,"SRC2").expect("VOP3 SRC2 layout").mask();
        if row.fields.len() == 1 { return Ok(FormFields::Vop3b { src2_unused: src2 as u16 }); }
        ignored.push(NamedField { name: "src2_unused", value: src2 });
    }
    for rule in &row.fields {
        if rule.name == "src2_unused" { continue; }
        if rule.name == "src1_unused" {
            let src1=field_value("SRC1",row,words);
            if !rule.allowed.contains(&src1) { return Err(reject(format!("unknown don't-care src1_unused={src1:#x}"))); }
            consumed[1] |= forms::field(row.form,"SRC1").expect("VOP3 SRC1").mask();
            ignored.push(NamedField { name:rule.name,value:src1 });
            continue;
        }
        let (wi, raw) = match rule.name {
            "w0_extra" | "wait_unused" | "vbuffer_tfe" | "vbuffer_nv" | "global_nv" => (0,words[0] & rule.mask),
            "w1_extra" | "vsrc_unused" | "vbuffer_format" | "vbuffer_offen" | "vbuffer_idxen"
            | "vbuffer_scope" | "vbuffer_th" | "global_sve" | "global_scope" | "global_th" => (1,words[1] & rule.mask),
            "neg" => (1,(words[1] >> 29) & rule.mask),
            "abs" => (0,(words[0] >> 8) & rule.mask),
            "omod" => (1,(words[1] >> 27) & rule.mask),
            "op_sel" => (0,(words[0] >> 11) & rule.mask),
            "op_sel_hi" => (1,((words[1] >> 27)&3 | (((words[0]>>14)&1)<<2)) & rule.mask),
            other => return Err(reject(format!("unsupported table field {other}"))),
        };
        if matches!(rule.name, "w0_extra"|"w1_extra"|"wait_unused"|"vsrc_unused") || rule.name.starts_with("vbuffer_") || rule.name.starts_with("global_") { consumed[wi] |= rule.mask; }
        if rule.class == FieldClass::Ignored && !rule.allowed.contains(&raw) { return Err(reject(format!("unknown don't-care {}={raw:#x}",rule.name))); }
        let item = NamedField { name: rule.name, value: raw };
        if rule.class == FieldClass::Honored { honored.push(item); } else { ignored.push(item); }
    }
    if ignored.is_empty() && honored.is_empty() { Ok(FormFields::None) } else { Ok(FormFields::Bits { ignored, honored }) }
}
fn row_for(words: &[u32]) -> Result<&'static OpRow, DecodeError> {
    let w0 = *words.first().ok_or_else(|| reject("empty instruction stream"))?;
    isa::gfx12().iter().find(|r| {
        let Some((mask, prefix)) = forms::prefix(r.form) else { return false };
        if w0 & mask != prefix || Some(r.op.id as u32) != forms::opcode(r.form).map(|f| f.value(words)) { return false; }
        r.form != Form::Vopd || words.len() > 1 && isa::gfx12().iter().any(|y| y.form == Form::Vopd && y.op.id == ((words[0] >> 17) & 31) as u16)
    }).ok_or_else(|| reject(format!("undefined gfx1201 opcode/form for {w0:#010x}")))
}
/// Decode one table-declared instruction; caller owns its stream offset.
pub fn decode(words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    let row = row_for(words)?;
    let n = width(row,words)?;
    let words = &words[..n];
    let literal = if n > match row.form { Form::Sop1|Form::Sop2|Form::Sopc|Form::Sopk|Form::Sopp|Form::Vop1|Form::Vop2|Form::Vopc => 1, Form::Vmem(_) => 3, _ => 2 } { Some(words[n-1]) } else { None };
    let mut consumed = [0;3];
    let (mask,_prefix)=forms::prefix(row.form).expect("known form"); consumed[0] |= mask;
    consumed[0] |= forms::opcode(row.form).expect("form opcode").mask();
    let mut operands = SmallVec::new();
    let mut mods = Modifiers::default();
    for (name,_) in grammars(row) {
        if let Some(f)=forms::field(row.form,name) { consumed[usize::from(f.bit / 32)] |= f.mask(); }
    }
    if row.form == Form::Vop3 && row.grammar.contains("SDST:") { consumed[0] |= 0x7f00; }
    if row.form == Form::Vopc && row.name.starts_with("v_cmp_") {
        operands.push(Operand::Special(Special::VccLo));
    }
    for (name,bits) in grammars(row) {
        let v = field_value(name,row,words);
        if forms::field(row.form,name).is_none() && name != "SDST" { return Err(reject(format!("no {} bitfield for {name}",row.name))); }
        if name == "SRC2" && v==255 && literal.is_none() { return Err(reject("literal selector without literal word")); }
        let parsed=operand(name,bits,v,literal,row,words)?;
        operands.push(parsed);
    }
    if row.form == Form::Smem && matches!(operands.last(),Some(Operand::Imm(ImmField::SmemOffset(_)))) {
        consumed[1] |= forms::field(row.form,"IOFFSET").expect("SMEM IOFFSET").mask();
    }
    if row.form == Form::Ds {
        let low=field_value("OFFSET0",row,words) as u8;
        let high=field_value("OFFSET1",row,words) as u8;
        if row.name.contains("2addr") {
            if low!=0 { operands.push(Operand::Imm(ImmField::DsOffset0(low))); }
            if high!=0 { operands.push(Operand::Imm(ImmField::DsOffset1(high))); }
        } else if low!=0 || high!=0 {
            operands.push(Operand::Imm(ImmField::DsOffset(u16::from_le_bytes([low,high]))));
        }
        consumed[0] |= 0xffff;
    }
    if matches!(row.form,Form::Vmem(_)) {
        let off=field_value("IOFFSET",row,words);
        if off!=0 { operands.push(Operand::Imm(ImmField::VmemOffset(((off<<8) as i32)>>8))); }
        if row.form == Form::Vmem(crate::inst::VmemForm::Buffer) && field_value("OFFEN",row,words)==1 {
            operands.push(Operand::Vmem(VmemToken::Offen));
        }
        if row.name=="global_inv" && field_value("SCOPE",row,words)==1 {
            operands.push(Operand::Scope(CacheScope::Se));
        }
        consumed[2] = u32::MAX;
    }
    if row.form == Form::Vopd {
        let y_id = ((words[0] >> 17) & 31) as u16;
        let y = isa::gfx12().iter().find(|r| r.form == Form::Vopd && r.op.id == y_id).ok_or_else(|| reject("unknown VOPD Y opcode"))?;
        consumed[0] |= Field::new("OPY",17,5).mask();
        let count = operands.len() as u8;
        for (name,bits) in grammars(y) {
            let mapped=match name { "VDSTX"=>"VDSTY", "SRCX0"=>"SRCY0", "VSRCX1"=>"VSRCY1", _=>name };
            operands.push(operand(mapped,bits,field_value(mapped,y,words),literal,y,words)?);
            let f=forms::field(row.form,mapped).expect("VOPD Y field");
            consumed[usize::from(f.bit/32)] |= f.mask();
        }
        for i in 0..2 {
            if words[i] & !consumed[i] != 0 {
                return Err(reject(format!("unknown VOPD bits in word {i}: {:#x}", words[i] & !consumed[i])));
            }
        }
        let fields = FormFields::Vopd { y_op: y.op, x_operands: count };
        let inst=Inst::from_parts(Arch::Gfx1201,row.op,row.form,fields,operands,mods,literal,Provenance::default()).map_err(|e| reject(e.to_string()))?;
        return Ok((inst,n));
    }
    special_mods(row,words,&mut mods);
    if matches!(row.form,Form::Vop3|Form::Vop3p) {
        for name in if row.form==Form::Vop3 { &["ABS","OPSEL","CLAMP","OMOD","NEG"][..] }
            else { &["NEG_HI","OPSEL","OPSEL_HI_LO","OPSEL_HI_2","CLAMP","NEG"][..] } {
            let f=forms::field(row.form,name).expect("modifier layout");
            consumed[usize::from(f.bit/32)] |= f.mask();
        }
    }
    if row.form==Form::Smem || matches!(row.form,Form::Vmem(_)) {
        for name in ["TH","SCOPE","NV"] {
            let f=forms::field(row.form,name).expect("memory modifier layout");
            consumed[usize::from(f.bit/32)] |= f.mask();
        }
    }
    if row.name=="global_inv" {
        if field_value("SADDR",row,words)!=124 { return Err(reject("global_inv SADDR must be off")); }
        consumed[0] |= forms::field(row.form,"SADDR").expect("global SADDR").mask();
    }
    let fields=fields_from(row,words,&mut consumed)?;
    for i in 0..n { if Some(i) == literal.map(|_|n-1) || matches!(row.form,Form::Vmem(_)) && i==2 { continue; }
        let unknown=words[i] & !consumed[i]; if unknown!=0 { return Err(reject(format!("unknown bits in word {i}: {unknown:#010x}"))); }
    }
    let inst=Inst::from_parts(Arch::Gfx1201,row.op,row.form,fields,operands,mods,literal,Provenance::default()).map_err(|e| reject(e.to_string()))?;
    Ok((inst,n))
}
/// Rebuild bytes from typed instruction fields. Never reads `inst.prov.bytes`.
pub fn encode(inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    inst.validate(Arch::Gfx1201).map_err(|e|reject(e.to_string()))?;
    let row=isa::lookup(Arch::Gfx1201,inst.op,inst.form).ok_or_else(||reject("unknown opcode/form"))?;
    let mut words=[0u32;3];
    let (_,prefix)=forms::prefix(row.form).ok_or_else(||reject("unsupported form"))?; words[0]=prefix;
    assign(forms::opcode(row.form).expect("form opcode"),&mut words,inst.op.id.into())?;
    let mut operands=inst.operands.iter();
    if row.form == Form::Vopc && row.name.starts_with("v_cmp_")
        && operands.next() != Some(&Operand::Special(Special::VccLo)) {
        return Err(reject("VOPC compare destination must be vcc_lo"));
    }
    for (name,bits) in grammars(row) {
        let op=operands.next().ok_or_else(|| reject(format!("missing {name} operand")))?;
        encode_field(name,row,&mut words,encoded_operand(name,bits,op,row.form)?)?;
        if name=="SOFFSET" && row.form==Form::Smem {
            if let Operand::Imm(ImmField::SmemOffset(off))=op {
                encode_field("IOFFSET",row,&mut words,(*off as u32)&0x00ff_ffff)?;
            }
        }
    }
    if row.form == Form::Vopd {
        if let FormFields::Vopd { y_op,x_operands } = &inst.fields {
            if usize::from(*x_operands) != grammars(row).count() { return Err(reject("VOPD half boundary mismatch")); }
            let y=isa::lookup(Arch::Gfx1201,*y_op,Form::Vopd).ok_or_else(||reject("unknown VOPD Y opcode"))?;
            assign(Field::new("OPY",17,5),&mut words,y.op.id.into())?;
            for (name,bits) in grammars(y) {
                let mapped=match name { "VDSTX"=>"VDSTY", "SRCX0"=>"SRCY0", "VSRCX1"=>"VSRCY1", _=>name };
                let op=operands.next().ok_or_else(||reject(format!("missing Y {mapped} operand")))?;
                encode_field(mapped,y,&mut words,encoded_operand(mapped,bits,op,y.form)?)?;
            }
        } else { return Err(reject("missing VOPD Y half")); }
    }
    if row.form==Form::Ds {
        for op in operands.by_ref() {
            match op {
                Operand::Imm(ImmField::DsOffset(n)) if !row.name.contains("2addr") => {
                    encode_field("OFFSET0",row,&mut words,u32::from(*n&255))?;
                    encode_field("OFFSET1",row,&mut words,u32::from(*n>>8))?;
                }
                Operand::Imm(ImmField::DsOffset0(n)) if row.name.contains("2addr") =>
                    encode_field("OFFSET0",row,&mut words,u32::from(*n))?,
                Operand::Imm(ImmField::DsOffset1(n)) if row.name.contains("2addr") =>
                    encode_field("OFFSET1",row,&mut words,u32::from(*n))?,
                _ => return Err(reject("invalid DS offset")),
            }
        }
    }
    if matches!(row.form,Form::Vmem(_)) {
        for op in operands.by_ref() {
            match op {
                Operand::Imm(ImmField::VmemOffset(n)) =>
                    encode_field("IOFFSET",row,&mut words,(*n as u32)&0x00ff_ffff)?,
                Operand::Vmem(VmemToken::Offen) if row.form==Form::Vmem(crate::inst::VmemForm::Buffer) =>
                    encode_field("OFFEN",row,&mut words,1)?,
                Operand::Scope(CacheScope::Se) if row.name=="global_inv" =>
                    encode_field("SCOPE",row,&mut words,1)?,
                _ => return Err(reject("invalid VMEM modifier operand")),
            }
        }
    }
    if operands.next().is_some() { return Err(reject("extra operands not declared by opcode grammar")); }
    if matches!(row.form, Form::Vop3|Form::Vop3p) {
        let m=&inst.mods;
        if row.form==Form::Vop3 {
            if !row.grammar.contains("SDST:") { encode_field("ABS",row,&mut words,m.abs.into())?; encode_field("OPSEL",row,&mut words,m.op_sel.into())?; }
            encode_field("OMOD",row,&mut words,match m.omod { Omod::None=>0,Omod::Mul2=>1,Omod::Mul4=>2,Omod::Div2=>3 })?;
        } else {
            encode_field("NEG_HI",row,&mut words,m.neg_hi.into())?;
            encode_field("OPSEL",row,&mut words,m.op_sel.into())?;
            encode_field("OPSEL_HI_LO",row,&mut words,(m.op_sel_hi&3).into())?;
            encode_field("OPSEL_HI_2",row,&mut words,(m.op_sel_hi>>2).into())?;
        }
        encode_field("NEG",row,&mut words,if row.form==Form::Vop3p { m.neg_lo } else { m.neg }.into())?;
        encode_field("CLAMP",row,&mut words,u32::from(m.clamp))?;
    }
    if row.form==Form::Smem || matches!(row.form, Form::Vmem(_)) {
        encode_field("TH",row,&mut words,inst.mods.cpol.th.into())?;
        encode_field("SCOPE",row,&mut words,inst.mods.cpol.scope.into())?;
        encode_field("NV",row,&mut words,u32::from(inst.mods.cpol.nv))?;
        if row.name=="global_inv" { encode_field("SADDR",row,&mut words,124)?; }
    }
    let fields = match &inst.fields {
        FormFields::Vop3b { src2_unused } => {
            encode_field("SRC2",row,&mut words,u32::from(*src2_unused))?;
            None
        }
        FormFields::Bits { ignored,honored } => Some((ignored,honored)),
        FormFields::None | FormFields::Vopd { .. } => None,
    };
    if let Some((ignored,honored)) = fields {
        for field in ignored.iter().chain(honored) {
            match field.name {
                "src2_unused" => encode_field("SRC2",row,&mut words,field.value)?,
                "src1_unused" => encode_field("SRC1",row,&mut words,field.value)?,
                "w0_extra" | "wait_unused" | "vbuffer_tfe" | "vbuffer_nv" | "global_nv" => words[0] |= field.value,
                "w1_extra" | "vsrc_unused" | "vbuffer_format" | "vbuffer_offen" | "vbuffer_idxen"
                | "vbuffer_scope" | "vbuffer_th" | "global_sve" | "global_scope" | "global_th" => words[1] |= field.value,
                "neg" | "abs" | "omod" | "op_sel" | "op_sel_hi" => { /* Encoded by the semantic modifier. */ },
                other => return Err(reject(format!("unhandled field {other}"))),
            }
        }
    }
    let mut n=match row.form { Form::Sop1|Form::Sop2|Form::Sopc|Form::Sopk|Form::Sopp|Form::Vop1|Form::Vop2|Form::Vopc=>1,
        Form::Vmem(_)=>3,_=>2 };
    if let Some(lit)=inst.literal { if n>=3 { return Err(reject("literal exceeds three-word instruction")); } words[n]=lit; n+=1; }
    let (decoded,used)=decode(&words[..n])?;
    if used!=n || decoded.op!=inst.op || decoded.form!=inst.form || decoded.operands!=inst.operands || decoded.fields!=inst.fields || decoded.mods!=inst.mods || decoded.literal!=inst.literal {
        return Err(reject(format!("inconsistent typed fields for {}",row.name)));
    }
    Ok(SmallVec::from_slice(&words[..n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_example_roundtrips() {
        let mut failures = Vec::new();
        for row in isa::gfx12() {
            let words: Vec<u32> = row.encoding.split_whitespace()
                .map(|w| u32::from_str_radix(w, 16).unwrap()).collect();
            match decode(&words).and_then(|(inst, len)| {
                if len != words.len() { return Err(reject(format!("width {len} != {}", words.len()))); }
                let result = encode(&inst)?;
                if &result[..] != words { return Err(reject(format!("words {result:08x?} != {words:08x?}"))); }
                Ok(())
            }) {
                Ok(()) => (),
                Err(error) => failures.push(format!("{}: {error}", row.name)),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn declared_dont_care_combinations_roundtrip() {
        for row in isa::gfx12() {
            let example: Vec<u32> = row.encoding.split_whitespace()
                .map(|w| u32::from_str_radix(w,16).unwrap()).collect();
            let mut cases = vec![example];
            for rule in row.fields.iter().filter(|rule| rule.class == FieldClass::Ignored) {
                let (index,shift,mask) = match rule.name {
                    "src1_unused" => (1,9,rule.mask << 9),
                    "src2_unused" => (1,18,rule.mask << 18),
                    "omod" => (1,27,rule.mask << 27),
                    "w0_extra" | "wait_unused" => (0,0,rule.mask),
                    "w1_extra" | "vsrc_unused" => (1,0,rule.mask),
                    other => panic!("unexpected ignored field {other} on {}",row.name),
                };
                cases = cases.into_iter().flat_map(|words| {
                    rule.allowed.iter().map(move |value| {
                        let mut variant = words.clone();
                        variant[index] = (variant[index] & !mask) | (*value << shift);
                        variant
                    })
                }).collect();
            }
            for words in cases {
                let (inst,len) = decode(&words).unwrap_or_else(|e| panic!("{} {words:08x?}: {e}",row.name));
                assert_eq!(len,words.len(),"{}",row.name);
                assert_eq!(&encode(&inst).unwrap()[..],words,"{}",row.name);
            }
        }
    }

    fn kt48_typed_insts() -> Vec<Inst> {
        const IMAGE: &[u8] = include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00; // .text file offset 0x2400 + (kernel VA 0x7f00 - .text VA 0x3400)
        const SIZE: usize = 10_604;
        let words: Vec<_> = IMAGE[START..START + SIZE].chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect();
        let mut result = Vec::with_capacity(1696);
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = decode(&words[index..]).unwrap_or_else(|e| panic!("KT48 byte {:#x}: {e}", index * 4));
            let output = encode(&inst).unwrap_or_else(|e| panic!("KT48 byte {:#x}: {e}", index * 4));
            assert_eq!(&output[..], &words[index..index + count], "KT48 byte {:#x}", index * 4);
            result.push(inst);
            index += count;
        }
        assert_eq!(index * 4, SIZE);
        assert_eq!(result.len(), 1696, "selected KT48 instruction census");
        result
    }

    #[test]
    fn kt48_selected_kernel_reencodes_exactly() {
        kt48_typed_insts();
    }

    #[test]
    fn kt48_inst_snapshot_is_typed() {
        let insts = kt48_typed_insts();
        let typed = insts.iter().map(|inst| format!("{inst:?}")).collect::<Vec<_>>().join("\n");
        insta::assert_snapshot!("kt48_typed_insts", typed);
    }

    #[test]
    fn dangerous_literal_fill_and_misaligned_smem_are_rejected() {
        // The first two words are the pinned v_add_co_u32 VOP3b example.
        let mut add = [0xd7006a01, 0x0202020e];
        add[1] = (add[1] & !(0x1ff << 18)) | (0xff << 18);
        assert!(decode(&add).unwrap_err().to_string().contains("dangerous-fill"));
        let aligned = [0xf4004100, 0xf8000030]; // s_load_b128 s[4:7], s[0:1], 0x30
        let misaligned = [(aligned[0] & !(0x7f << 6)) | (2 << 6), aligned[1]];
        assert!(decode(&misaligned).unwrap_err().to_string().contains("must be aligned"));
    }

    #[test]
    fn honored_modifiers_and_noncanonical_fill_reencode_without_provenance() {
        let add=[0xd5250001,0x02020702]; // integer VOP3 with no semantic SRC2
        let changed=[add[0] | (5<<8), (add[1] & !(0x7<<29) & !(0x1ff<<18)) | (3<<29)];
        let (inst,words)=decode(&changed).unwrap();
        assert_eq!(words,2);
        assert_eq!(inst.mods.abs,5);
        assert_eq!(inst.mods.neg,3);
        assert_eq!(&encode(&inst).unwrap()[..],changed);

        let zero_fill=[add[0], add[1] & !(0x1ff<<18)];
        let (mut lifted,_)=decode(&zero_fill).unwrap();
        lifted.prov.bytes=Some([add[0],add[1],0]);
        assert_eq!(&encode(&lifted).unwrap()[..],zero_fill);
        assert_ne!(&encode(&lifted).unwrap()[..],add);

        let dot=[0xcc164000,0x7c0e0501]; // dot op_sel/op_sel_hi are honored
        let dot_modified=[(dot[0] & !(0x7<<11)) | (2<<11),
            (dot[1] & !(3<<27)) | (1<<27)];
        let (inst,_)=decode(&dot_modified).unwrap();
        assert_eq!(inst.mods.op_sel,2);
        assert_eq!(inst.mods.op_sel_hi,5);
        assert_eq!(&encode(&inst).unwrap()[..],dot_modified);
    }

    #[test]
    fn packed_neg_lo_and_combined_wait_are_typed() {
        let packed=[0xcc4a4039,0x7a020347];
        let (inst,_) = decode(&packed).unwrap();
        assert_eq!(inst.mods.neg_lo,3);
        assert_eq!(inst.mods.neg,0);
        assert_eq!(&encode(&inst).unwrap()[..],packed);

        let wait=[0xbfc90102];
        let (inst,_) = decode(&wait).unwrap();
        let counters=&inst.mods.wait.as_ref().unwrap().per_counter;
        assert_eq!(counters[Counter::Store as usize],Some(1));
        assert_eq!(counters[Counter::Ds as usize],Some(2));
        assert_eq!(&encode(&inst).unwrap()[..],wait);
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(512))]
        #[test]
        fn encoded_insts_recover_declared_form(
            row_index in 0usize..isa::gfx12().len(),
            seed in proptest::prelude::any::<u8>(),
        ) {
            let row=&isa::gfx12()[row_index];
            let words:Vec<u32>=row.encoding.split_whitespace()
                .map(|w| u32::from_str_radix(w,16).unwrap()).collect();
            let (mut inst,_) = decode(&words).unwrap();
            if let Some(first) = inst.operands.iter_mut()
                .find(|op| matches!(op,Operand::Reg(RegRef { kind: Kind::V,.. }))) {
                if let Operand::Reg(r)=first {
                    let bank_limit=if inst.form==Form::Vopd {128} else {256};
                    r.base=u16::from(seed) % (bank_limit+1-u16::from(r.len));
                }
                inst=Inst::from_parts(Arch::Gfx1201,inst.op,inst.form,inst.fields.clone(),
                    inst.operands.clone(),inst.mods.clone(),inst.literal,Provenance::default()).unwrap();
            }
            if let Some(literal)=&mut inst.literal {
                let replacement=*literal ^ (u32::from(seed)*0x0101_0101);
                *literal=replacement;
                for op in &mut inst.operands {
                    if let Operand::Literal(value)=op { *value=replacement; }
                }
            }
            if let FormFields::Vop3b { src2_unused }=&mut inst.fields {
                *src2_unused=if seed&1==0 {0} else {128};
            }
            let encoded=encode(&inst).unwrap();
            let (decoded,count)=decode(&encoded).unwrap();
            proptest::prop_assert_eq!(count,encoded.len());
            proptest::prop_assert_eq!(decoded,inst);
        }
    }
}
