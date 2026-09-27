//! gfx1201 opcode/encoding rows. All source text is compiled into the IR crate.
use std::sync::LazyLock;
use crate::inst::{Arch, Family, Form, Opcode, VmemForm};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldClass { Ignored, Honored }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldRule { pub name: &'static str, pub class: FieldClass, pub mask: u32, pub allowed: Vec<u32> }
#[derive(Clone, Debug)]
pub struct OpRow {
    pub op: Opcode, pub form: Form, pub name: &'static str,
    pub grammar: &'static str, pub defs: &'static str, pub uses: &'static str,
    pub implicit: &'static str, pub counter: &'static str,
    pub sample: &'static str, pub encoding: &'static str,
    pub benign_src2: Vec<u16>, pub fields: Vec<FieldRule>,
}
#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("line {line}: malformed opcode row: {reason}")]
    Malformed { line: usize, reason: String },
    #[error("line {line}: duplicate opcode/form")]
    Duplicate { line: usize },
}
fn form(value: &str) -> Option<Form> {
    Some(match value {
        "sop1" => Form::Sop1, "sop2" => Form::Sop2, "sopc" => Form::Sopc, "sopk" => Form::Sopk,
        "sopp" => Form::Sopp, "smem" => Form::Smem, "vop1" => Form::Vop1,
        "vop1_dpp" => Form::Vop1Dpp, "vop2" => Form::Vop2, "vop2_dpp" => Form::Vop2Dpp, "vopc" => Form::Vopc, "vop3" => Form::Vop3,
        "vop3p" => Form::Vop3p, "vopd" => Form::Vopd, "vinterp" => Form::Vinterp,
        "ds" => Form::Ds, "global" => Form::Vmem(VmemForm::Global),
        "scratch" => Form::Vmem(VmemForm::Scratch), "flat" => Form::Vmem(VmemForm::Flat),
        "buffer" => Form::Vmem(VmemForm::Buffer), "image" => Form::Vmem(VmemForm::Image),
        "export" => Form::Export, _ => return None,
    })
}
fn family(form: Form) -> Family {
    match form {
        Form::Sop1 => Family::Sop1, Form::Sop2 => Family::Sop2, Form::Sopc => Family::Sopc,
        Form::Sopk => Family::Sopk, Form::Sopp => Family::Sopp, Form::Smem => Family::Smem,
        Form::Vop1 | Form::Vop1Dpp => Family::Vop1, Form::Vop2 | Form::Vop2Dpp => Family::Vop2, Form::Vopc => Family::Vopc,
        Form::Vop3 => Family::Vop3, Form::Vop3p => Family::Vop3p,
        Form::Vopd => Family::Vopd, Form::Vinterp => Family::Vinterp, Form::Ds => Family::Ds,
        Form::Vmem(_) => Family::Vmem, Form::Export => Family::Export,
    }
}
/// Tab-separated: `name form opcode grammar defs uses implicit counter sample words field-rules`.
pub fn parse_gfx12() -> Result<Vec<OpRow>, TableError> {
    let mut rows = Vec::new();
    for (index, line) in include_str!("../isa/gfx12.tbl").lines().enumerate() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let fields: Vec<_> = line.split('\t').collect();
        let fail = |reason: &str| TableError::Malformed { line: index + 1, reason: reason.into() };
        if fields.len() != 11 { return Err(fail("expected 11 tab-delimited columns")); }
        let kind = form(fields[1]).ok_or_else(|| fail("unknown instruction form"))?;
        let id = u16::from_str_radix(fields[2].trim_start_matches("0x"), 16).map_err(|_| fail("bad opcode id"))?;
        let mut rule_fields = Vec::new();
        let mut benign_src2 = Vec::new();
        for field in fields[10].split(',').filter(|s| !s.is_empty() && *s != "-") {
            let parts: Vec<_> = field.split(':').collect();
            if parts.len() < 3 { return Err(fail("malformed field rule")); }
            let class = match parts[1] { "ignored" => FieldClass::Ignored, "honored" => FieldClass::Honored, _ => return Err(fail("unknown field class")) };
            let mask = u32::from_str_radix(parts[2].trim_start_matches("0x"), 16).map_err(|_| fail("bad field mask"))?;
            let allowed = parts.get(3).unwrap_or(&"").split('/').filter(|s| !s.is_empty())
                .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|_| fail("bad benign field value")))
                .collect::<Result<Vec<_>, _>>()?;
            if parts[0] == "src2_unused" { benign_src2 = allowed.iter().map(|&v| v as u16).collect(); }
            rule_fields.push(FieldRule { name: parts[0], class, mask, allowed });
        }
        let row = OpRow { op: Opcode { family: family(kind), id }, form: kind, name: fields[0],
            grammar: fields[3], defs: fields[4], uses: fields[5], implicit: fields[6], counter: fields[7],
            sample: fields[8], encoding: fields[9], benign_src2, fields: rule_fields };
        if rows.iter().any(|r: &OpRow| r.op == row.op && r.form == row.form) { return Err(TableError::Duplicate { line: index + 1 }); }
        rows.push(row);
    }
    Ok(rows)
}
pub fn gfx12() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| parse_gfx12().expect("shipped gfx12 table must parse"));
    &TABLE
}
pub fn lookup(arch: Arch, opcode: Opcode, form: Form) -> Option<&'static OpRow> {
    if arch != Arch::Gfx1201 { return None; }
    gfx12().iter().find(|row| row.op == opcode && row.form == form)
}
pub fn lookup_opcode(arch: Arch, opcode: Opcode) -> Option<&'static OpRow> {
    if arch != Arch::Gfx1201 { return None; }
    gfx12().iter().find(|row| row.op == opcode)
}
