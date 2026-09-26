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
