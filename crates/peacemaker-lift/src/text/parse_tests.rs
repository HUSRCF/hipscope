//! Parser gate tests: every canonical KT48 line parses back to the
//! instruction the codec decoded (semantics plus re-encoding), and the `.s`
//! source structure carries labels/directives for `lift_text`.

use super::{label_block_ids, parse_line, parse_source, SourceKind};
use crate::text::{print::canonical, support};
use peacemaker_ir::{
    codec::gfx12,
    inst::{Arch, Inst},
};

const KT48_VA: u32 = 0x7f00;
const KT48_SIZE: usize = 10_604;

fn decode_kt48() -> Vec<(u32, Vec<u32>, Inst)> {
    let (stream, va) = support::kt48_stream();
    assert_eq!(va, KT48_VA);
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < stream.len() {
        let (inst, n) = gfx12::decode(&stream[index..])
            .unwrap_or_else(|e| panic!("KT48 word {index}: {e}"));
        out.push((
            va + (index as u32) * 4,
            stream[index..index + n].to_vec(),
            inst,
        ));
        index += n;
    }
    assert_eq!(index * 4, KT48_SIZE);
    out
}

fn kt48_texts() -> Vec<(u32, Vec<u32>, String)> {
    support::pinned_objdump_fixture()
        .into_iter()
        .filter(|l| l.addr >= KT48_VA && l.addr < KT48_VA + KT48_SIZE as u32)
        .map(|l| (l.addr, l.words, l.text))
        .collect()
}

/// The parser agrees with the codec on all 1,696 instructions: same op,
/// form, operands, modifiers and literal — and the parsed instruction
/// re-encodes to the original words, proving the canonical don't-care
/// defaults. Any semantic disagreement is a test failure.
#[test]
fn parser_matches_codec_on_kt48() {
    let decoded = decode_kt48();
    let texts = kt48_texts();
    assert_eq!(decoded.len(), 1696);
    assert_eq!(texts.len(), 1696);
    let mut failures = Vec::new();
    for ((addr, words, inst), (taddr, _, text)) in
        decoded.iter().zip(texts.iter())
    {
        assert_eq!(addr, taddr);
        // Sanity: the test parses the canonical printing, which T3 proves
        // equals the pinned text.
        let printed =
            canonical(inst, Arch::Gfx1201).expect("canonical prints");
        if &printed != text {
            failures.push(format!("{addr:#x}: T3 drift for {text}"));
            continue;
        }
        let parsed = match parse_line(text, Arch::Gfx1201) {
            Ok(inst) => inst,
            Err(e) => {
                failures.push(format!("{addr:#x}: parse {text:?}: {e}"));
                continue;
            }
        };
        // Modifiers compare with textually invisible bits cleared (the
        // same lossiness the T9 harness defines: e.g. WMMA op_sel_hi).
        if parsed.op != inst.op
            || parsed.form != inst.form
            || parsed.operands != inst.operands
            || support::masked_mods(&parsed) != support::masked_mods(inst)
            || parsed.literal != inst.literal
        {
            failures.push(format!(
                "{addr:#x}: semantic drift on {text}\n  codec: {:?} {:?} {:?} {:?}\n  parse: {:?} {:?} {:?} {:?}",
                inst.op,
                inst.form,
                inst.operands,
                inst.mods,
                parsed.op,
                parsed.form,
                parsed.operands,
                parsed.mods
            ));
            continue;
        }
        // Re-encoding matches modulo the same unspellable bits.
        let mask = support::lossy_mask(inst);
        match gfx12::encode(&parsed) {
            Ok(enc)
                if enc.len() == words.len()
                    && enc.iter().zip(words.iter()).enumerate().all(
                        |(i, (a, b))| {
                            (a ^ b) & !mask.get(i).copied().unwrap_or(0) == 0
                        },
                    ) => {}
            Ok(enc) => failures.push(format!(
                "{addr:#x}: re-encode drift on {text}: {enc:08X?} != {words:08X?}"
            )),
            Err(e) => failures.push(format!(
                "{addr:#x}: re-encode {text:?}: {e}"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} parser mismatches:\n{}",
        failures.len(),
        failures.into_iter().take(10).collect::<Vec<_>>().join("\n")
    );
}

/// `.s` structure: labels map to instruction ordinals, directives pass
/// through, every instruction line parses.
#[test]
fn source_parser_handles_labels_and_directives() {
    let src = "\
.text
.globl my_kernel
my_kernel:
    s_nop 0
.LBB0:
    s_branch 3
    // a comment
    s_endpgm
";
    let file = parse_source(src).expect("source parses");
    assert_eq!(
        file.labels,
        vec![
            ("my_kernel".to_owned(), 0),
            (".LBB0".to_owned(), 1),
        ]
    );
    assert_eq!(file.inst_len(), 3);
    let ids = label_block_ids(&file);
    assert_eq!(ids.len(), 2);
    // Every instruction line parses to a typed instruction.
    for text in file.insts() {
        parse_line(text, Arch::Gfx1201)
            .unwrap_or_else(|e| panic!("parse {text:?}: {e}"));
    }
    // Comment and directive lines are classified, not parsed as code.
    assert!(
        file.lines.iter().any(|l| matches!(
            l.kind,
            SourceKind::Directive(_)
        )),
        "directives retained"
    );
    // Duplicate labels fail closed.
    assert!(parse_source("a:\n    s_nop 0\na:\n    s_nop 0\n").is_err());
    // A KT48-sized body parses end to end through the source path.
    let body = kt48_texts()
        .into_iter()
        .map(|(_, _, t)| t)
        .collect::<Vec<_>>()
        .join("\n");
    let file =
        parse_source(&format!(".text\nmy_kernel:\n{body}\n")).expect("body");
    assert_eq!(file.inst_len(), 1696);
    assert_eq!(file.labels, vec![("my_kernel".to_owned(), 0)]);
}

/// Garbage is rejected, never normalised or panicked on.
#[test]
fn parse_rejects_garbage() {
    for bad in [
        "",
        "not_an_instruction",
        "s_nop",
        "s_nop 0, 1",
        "v_fma_f32 v177, -v175, v176",
        "v_fma_f32 v177, -v175, v176, 1.0, v0",
        "s_branch x",
        "s_delay_alu instid0(BOGUS)",
        "s_wait_alu depctr_nope(0)",
        "s_wait_alu depctr_sa_sdst(2)",
        "v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, -1, v3",
        "s_clause 0x100",
        "v_cndmask_b32_e32 v15, -1, v10",
        "s_load_b32 s3, s[0:1]",
    ] {
        assert!(
            parse_line(bad, Arch::Gfx1201).is_err(),
            "must reject {bad:?}"
        );
    }
    assert!(parse_line("s_nop 0", Arch::Gfx1201).is_ok());
}

/// C7 lift_text regressions on hipcc's own `.s`: quoted `//` inside
/// `.ident`, the `.amdgpu_metadata` YAML region, and the hidden-EXEC
/// cmpx shape (two visible operands).
#[test]
fn source_parser_handles_hipcc_s_constructs() {
    let src = "\
\t.ident\t\"AMD clang version 20.1.0 (https://github.com/rocm/llvm-project.git)\"
\t.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
# a yaml comment
  - .args:
...
\t.end_amdgpu_metadata
my_kernel:
\ts_nop 0
";
    let file = parse_source(src).expect("hipcc constructs parse");
    let directives: Vec<&str> = file
        .lines
        .iter()
        .filter_map(|l| match &l.kind {
            SourceKind::Directive(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        directives.iter().any(|d| d.starts_with(".ident")
            && d.contains("https://github.com/rocm/llvm-project.git")),
        "quoted // must not start a comment: {directives:?}"
    );
    for marker in [
        ".amdgpu_metadata",
        "amdhsa.version: [ 1, 2 ]",
        "# a yaml comment",
        "- .args:",
        "...",
        ".end_amdgpu_metadata",
    ] {
        assert!(
            directives.iter().any(|d| d.trim() == marker),
            "metadata region kept verbatim, missing {marker:?}: {directives:?}"
        );
    }
    assert_eq!(file.labels, vec![("my_kernel".to_owned(), 0)]);
    assert_eq!(file.inst_len(), 1);
}

/// `v_cmpx*` VOP3 canonical text has two visible operands; the hidden
/// exec VDST is fixed 0x7e in the encoding and absent from the type.
#[test]
fn cmpx_parses_two_visible_operands() {
    let inst = parse_line("v_cmpx_gt_i32_e64 s7, v4", Arch::Gfx1201)
        .expect("cmpx parses");
    assert_eq!(inst.operands.len(), 2);
    let enc = gfx12::encode(&inst).expect("cmpx encodes");
    assert_eq!(enc.len(), 2);
    let (back, n) = gfx12::decode(&enc).expect("cmpx re-decodes");
    assert_eq!(n, 2);
    assert_eq!(back.operands, inst.operands);
    assert_eq!(back.mods, inst.mods);
}

/// C7 lift_text census: hipcc's own `.s` omits redundant `op_sel`, uses
/// negative VMEM offsets, and relies on assembler defaults. parse_line
/// types exactly what llvm-mc encodes: op_sel implied by `.h`/`.l`,
/// assembler defaults for unspelled bits.
#[test]
fn hipcc_spellings_match_codec() {
    // op_sel implied by halves (redundant suffix omitted by hipcc).
    for (text, op_sel) in [
        ("v_mov_b16_e64 v139.l, v5.h", 1),
        ("v_cvt_f32_f16_e64 v12, v198.h", 1),
        ("v_cvt_pk_fp8_f32 v141.h, v140, v139", 8),
    ] {
        let inst =
            parse_line(text, Arch::Gfx1201).expect("hipcc spelling parses");
        assert_eq!(inst.mods.op_sel, op_sel, "{text}");
        // Re-prints with the redundant suffix objdump shows.
        let canon = canonical(&inst, Arch::Gfx1201).expect("prints");
        assert!(
            canon.contains("op_sel:["),
            "canonical restores suffix: {canon}"
        );
        gfx12::encode(&inst).expect("encodes");
    }
    // Negative VMEM offsets.
    let inst = parse_line(
        "global_load_b64 v[173:174], v[2:3], off offset:-48",
        Arch::Gfx1201,
    )
    .expect("negative vmem offset parses");
    assert!(matches!(
        inst.operands.last(),
        Some(peacemaker_ir::operand::Operand::Imm(
            peacemaker_ir::operand::ImmField::VmemOffset(-48)
        ))
    ));
    gfx12::encode(&inst).expect("encodes");
    // Omitted WMMA op_sel_hi takes the assembler default.
    let inst = parse_line(
        "v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]",
        Arch::Gfx1201,
    )
    .expect("wmma parses");
    assert_eq!(inst.mods.op_sel_hi, 7);
    gfx12::encode(&inst).expect("encodes");
}
