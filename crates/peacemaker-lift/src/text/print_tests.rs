//! T3/T9 gate tests for the C3b second opinion. Tool-gated: with a pinned
//! toolchain (`PEACEMAKER_ROCM`, else `/opt/rocm/core-10.0`) the live tools
//! must agree; without one the committed objdump fixture is the reference.
//! A set-but-unusable `PEACEMAKER_ROCM` fails, never skips.

use super::{canonical, kernel_lines};
use crate::text::support::{self, Parity, classify, lossy_mask};
use peacemaker_ir::{
codec::gfx12,
inst::{Arch, Form, FormFields, Inst},
isa, operand::Operand,
};
use smallvec::SmallVec;

const KT48_VA: u32 = 0x7f00;
const KT48_SIZE: usize = 10_604;

struct Kt48 {
    addrs: Vec<u32>,
    words: Vec<Vec<u32>>,
    insts: Vec<Inst>,
}

fn decode_kt48() -> Kt48 {
    let (stream, va) = support::kt48_stream();
    assert_eq!(va, KT48_VA);
    let mut addrs = Vec::new();
    let mut words = Vec::new();
    let mut insts = Vec::new();
    let mut index = 0usize;
    while index < stream.len() {
        let (inst, n) = gfx12::decode(&stream[index..])
            .unwrap_or_else(|e| panic!("KT48 word {index}: {e}"));
        addrs.push(va + (index as u32) * 4);
        words.push(stream[index..index + n].to_vec());
        insts.push(inst);
        index += n;
    }
    assert_eq!(index * 4, KT48_SIZE, "selected kernel size");
    assert_eq!(insts.len(), 1696, "selected kernel census");
    Kt48 { addrs, words, insts }
}

fn objdump_lines() -> Vec<support::ObjdumpLine> {
    let lines = match support::toolchain()
        .expect("toolchain discovery must not fail")
    {
        Some(tc) => {
            let output = std::process::Command::new(&tc.objdump)
                .args([
                    "--mcpu=gfx1201",
                    "-d",
                    "--disassemble-symbols=attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
                ])
                .arg(
                    concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/tests/fixtures/kt48/hipcc.co"
                    ),
                )
                .output()
                .expect("run pinned llvm-objdump");
            assert!(
                output.status.success(),
                "pinned llvm-objdump failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let live = support::parse_objdump_stdout(
                &String::from_utf8_lossy(&output.stdout),
            );
            // The committed fixture must equal the live tool output.
            let pinned = support::pinned_objdump_fixture();
            assert_eq!(
                live.len(),
                pinned.len(),
                "pinned objdump fixture is stale (line count)"
            );
            for (l, p) in live.iter().zip(pinned.iter()) {
                assert_eq!(l.addr, p.addr, "stale fixture addr");
                assert_eq!(l.text, p.text, "stale fixture text");
                assert_eq!(l.words, p.words, "stale fixture words");
            }
            live
        }
        None => support::pinned_objdump_fixture(),
    };
    lines
        .into_iter()
        .filter(|l| l.addr >= KT48_VA && l.addr < KT48_VA + KT48_SIZE as u32)
        .collect()
}

/// T3: boundaries, mnemonics and canonical operand text of all 1,696
/// instructions equal the pinned objdump's; 57 distinct branch targets
/// equal its `<sym+0x…>` annotations.
#[test]
fn t3_decoder_agrees_with_pinned_objdump() {
    let kt = decode_kt48();
    let lines = objdump_lines();
    assert_eq!(lines.len(), 1696, "objdump lines in symbol range");
    // Boundaries: objdump addresses must advance exactly as the codec's
    // widths say.
    for (i, line) in lines.iter().enumerate() {
        assert_eq!(
            line.addr, kt.addrs[i],
            "boundary mismatch at instruction {i}"
        );
        assert_eq!(
            line.words.len(),
            kt.words[i].len(),
            "width mismatch at {:#x}",
            line.addr
        );
        for (a, b) in line.words.iter().zip(kt.words[i].iter()) {
            assert_eq!(*a, *b, "word mismatch at {:#x}", line.addr);
        }
    }
    // Canonical text equality; semantic disagreement is a failure.
    let mut diffs = Vec::new();
    let mut printed = Vec::new();
    for (i, inst) in kt.insts.iter().enumerate() {
        let text = canonical(inst, Arch::Gfx1201)
            .unwrap_or_else(|e| panic!("print {:#x}: {e}", kt.addrs[i]));
        printed.push(text.clone());
        if text != lines[i].text {
            diffs.push(format!(
                "{:#x}:\n  ours: {text}\n  ref : {}",
                kt.addrs[i], lines[i].text
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "{} canonical text mismatches:\n{}",
        diffs.len(),
        diffs.into_iter().take(10).collect::<Vec<_>>().join("\n")
    );
    // Branch targets from encodings (signed simm16, dwords from next
    // PC) equal the annotated `<sym+0x…>` targets; decoding from the
    // unsigned text operand instead gives 10 wrong targets, so this
    // also pins the print/parse direction of the pitfall.
    const BRANCH_SET: [&str; 7] = [
        "s_branch",
        "s_cbranch_scc0",
        "s_cbranch_scc1",
        "s_cbranch_vccz",
        "s_cbranch_vccnz",
        "s_cbranch_execz",
        "s_cbranch_execnz",
    ];
    let mut targets = std::collections::BTreeSet::new();
    let mut branches = 0usize;
    for (i, inst) in kt.insts.iter().enumerate() {
        let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
        if !BRANCH_SET.contains(&row.name) {
            continue;
        }
        branches += 1;
        let simm = match inst.operands.first() {
            Some(Operand::Imm(
                peacemaker_ir::operand::ImmField::Sopp(n),
            )) => *n as i32,
            other => panic!("branch without simm16 at {:#x}: {other:?}", kt.addrs[i]),
        };
        let target = kt.addrs[i] as i32
            + (kt.words[i].len() as i32) * 4
            + simm * 4;
        let ann = &lines[i].ann;
        let hex = ann
            .strip_prefix("<attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201+0x")
            .and_then(|s| s.strip_suffix('>'))
            .unwrap_or_else(|| {
                panic!("branch at {:#x} lacks annotation: {ann:?}", kt.addrs[i])
            });
        let expected = u32::from_str_radix(hex, 16).unwrap();
        assert_eq!(
            (target - KT48_VA as i32) as u32,
            expected,
            "branch target mismatch at {:#x}",
            kt.addrs[i]
        );
        targets.insert(expected);
        // The printed unsigned operand re-parses to the same simm.
        let printed_operand = printed[i]
            .split_whitespace()
            .nth(1)
            .unwrap()
            .to_owned();
        let reparsed = printed_operand.parse::<u32>().unwrap() as u16 as i16;
        assert_eq!(reparsed, simm as i16, "unsigned branch pitfall");
    }
    assert_eq!(branches, 67, "KT48 branch census");
    assert_eq!(targets.len(), 57, "distinct branch targets");
}

/// T9: assembler parity for every KT48 instruction (all 1,696 have
/// lossless syntax, so the reported set is empty); a synthetic VOP3b
/// with `src2_unused = 0x00` lands in the reported set, differing only
/// in that field, without failing. This also discharges T3's
/// `mc(Inst::text(i)) == encode(i)` sentence.
#[test]
fn t9_assembler_parity_where_syntax_is_lossless() {
    let kt = decode_kt48();
    let lines = kernel_lines(&kt.insts, Arch::Gfx1201)
        .expect("all KT48 instructions print");
    // Synthetic VOP3b with the non-canonical benign fill first, so the
    // reported-set logic is exercised even without tools.
    let vop3b_index =
        kt.insts.iter().position(|inst| {
            matches!(inst.fields, FormFields::Vop3b { .. })
        }).expect("KT48 holds VOP3b instructions");
    let mut synth = kt.insts[vop3b_index].clone();
    synth.fields = FormFields::Vop3b { src2_unused: 0x00 };
    let synth = Inst::from_parts(
        Arch::Gfx1201,
        synth.op,
        synth.form,
        synth.fields.clone(),
        SmallVec::from_vec(synth.operands.to_vec()),
        synth.mods.clone(),
        synth.literal,
        peacemaker_ir::provenance::Provenance::default(),
    )
    .expect("synthetic VOP3b validates");
    let synth_text =
        canonical(&synth, Arch::Gfx1201).expect("synthetic prints");
    assert_eq!(
        synth_text, lines[vop3b_index],
        "don't-care fill is invisible in canonical text"
    );
    let enc_synth = gfx12::encode(&synth).expect("synthetic encodes");
    let enc_canon =
        gfx12::encode(&kt.insts[vop3b_index]).expect("canonical encodes");
    assert_ne!(enc_synth.as_slice(), enc_canon.as_slice());
    let mask = lossy_mask(&synth);
    assert!(
        mask.iter().any(|m| *m != 0),
        "synthetic has unspellable bits"
    );
    let only_masked = enc_synth.len() == enc_canon.len()
        && enc_synth
            .iter()
            .zip(enc_canon.iter())
            .enumerate()
            .all(|(i, (a, b))| {
                (a ^ b) & !mask.get(i).copied().unwrap_or(0) == 0
            });
    assert!(only_masked, "synthetic differs only in src2");
    match support::toolchain().expect("toolchain discovery") {
        Some(tc) => {
            let mc_words = support::mc_batch(&tc.mc, &lines)
                .expect("pinned llvm-mc assembles KT48 text");
            let mut reported = Vec::new();
            let mut mismatched = Vec::new();
            for (i, (inst, mc)) in
                kt.insts.iter().zip(mc_words.iter()).enumerate()
            {
                match classify(inst, mc).unwrap_or(Parity::Mismatch) {
                    Parity::Exact => {}
                    Parity::Reported => reported.push(i),
                    Parity::Mismatch => mismatched.push(format!(
                        "{:#x}: {} mc={mc:08X?} enc={:08X?}",
                        kt.addrs[i],
                        lines[i],
                        gfx12::encode(inst)
                            .map(|e| e.to_vec())
                            .unwrap_or_default()
                    )),
                }
            }
            assert!(
                mismatched.is_empty(),
                "{} assembler mismatches:\n{}",
                mismatched.len(),
                mismatched.into_iter().take(10).collect::<Vec<_>>().join("\n")
            );
            assert!(
                reported.is_empty(),
                "KT48 non-canonical list is not empty: {reported:?}"
            );
            // The synthetic lands reported through the same path: mc
            // fills src2 with 0x80 while the instruction holds 0x00.
            let mc_synth = support::mc_batch(&tc.mc, &[synth_text])
                .expect("mc assembles synthetic")[0]
                .clone();
            assert_eq!(
                classify(&synth, &mc_synth).unwrap(),
                Parity::Reported,
                "synthetic VOP3b must land in the reported set"
            );
        }
        None => {
            if std::env::var("PEACEMAKER_ROCM").is_ok() {
                panic!("PEACEMAKER_ROCM is set but unusable");
            }
            // No assembler: the reported-set logic still runs against
            // the canonical encoding (which is exactly what llvm-mc
            // fills for don't-care fields).
            eprintln!("no pinned toolchain: mc parity via canonical fill");
            assert_eq!(
                classify(&synth, enc_canon.as_slice()).unwrap(),
                Parity::Reported,
                "synthetic VOP3b must land in the reported set"
            );
        }
    }
}
