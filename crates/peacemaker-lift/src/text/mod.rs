//! C3b: canonical objdump text (<->) typed instructions.
//!
//! The typed codec (`peacemaker-ir`) owns bytes; this module owns the
//! independent second opinion: the pinned `llvm-objdump` spelling of every
//! instruction ([`print`]) and the parser that reads it back ([`parse`]).
//! Semantic disagreement with the codec is a test failure, never a
//! normalisation. `llvm-mc` byte parity is required only where the syntax is
//! lossless; instructions carrying don't-care state the syntax cannot spell
//! land in a *reported* set instead of failing.
pub mod parse;
pub mod print;

pub use parse::{SourceFile, SourceKind, SourceLine, parse_line, parse_source};
pub use print::{PrintError, canonical, kernel_lines};

#[cfg(test)]
pub(crate) mod support {
    use std::path::PathBuf;

    /// Pinned second-opinion toolchain (`PEACEMAKER_ROCM`, else the default
    /// install). `Some` means the live tools must agree; `None` means the
    /// committed objdump fixture is the reference. A set-but-unusable
    /// `PEACEMAKER_ROCM` is a hard error, never a silent skip.
    pub(crate) struct Toolchain {
        pub(crate) objdump: PathBuf,
        pub(crate) mc: PathBuf,
    }

    pub(crate) fn toolchain() -> Result<Option<Toolchain>, String> {
        const BIN: &str = "lib/llvm/bin";
        if let Ok(root) = std::env::var("PEACEMAKER_ROCM") {
            let root = PathBuf::from(root);
            let objdump = root.join(BIN).join("llvm-objdump");
            let mc = root.join(BIN).join("llvm-mc");
            if !objdump.is_file() || !mc.is_file() {
                return Err(format!(
                    "PEACEMAKER_ROCM={} does not hold llvm-objdump/llvm-mc",
                    root.display()
                ));
            }
            return Ok(Some(Toolchain { objdump, mc }));
        }
        let root = PathBuf::from("/opt/rocm/core-10.0");
        let objdump = root.join(BIN).join("llvm-objdump");
        let mc = root.join(BIN).join("llvm-mc");
        if objdump.is_file() && mc.is_file() {
            return Ok(Some(Toolchain { objdump, mc }));
        }
        Ok(None)
    }

    /// Selected KT48 kernel stream as little-endian words plus base VA.
    /// Same constants as the C3 codec test (fixture pinned by provenance.json).
    pub(crate) fn kt48_stream() -> (Vec<u32>, u32) {
        const IMAGE: &[u8] =
            include_bytes!("../../tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        const VA: u32 = 0x7f00;
        let words = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (words, VA)
    }

    /// One parsed objdump disassembly line.
    pub(crate) struct ObjdumpLine {
        pub(crate) addr: u32,
        pub(crate) words: Vec<u32>,
        pub(crate) text: String,
        pub(crate) ann: String,
    }

    /// Parse `llvm-objdump -d` stdout. Objump comment words are big-endian;
    /// the `//` comment separator may directly follow long lines.
    pub(crate) fn parse_objdump_stdout(s: &str) -> Vec<ObjdumpLine> {
        let mut out = Vec::new();
        for line in s.lines() {
            let body = line.strip_prefix('\t').unwrap_or("");
            if body.is_empty() {
                continue;
            }
            let Some(mark) = body.find("//") else { continue };
            let text = body[..mark].trim_end().to_owned();
            if text.is_empty() {
                continue;
            }
            let comment = body[mark + 2..].trim();
            let mut parts = comment.split_whitespace();
            let addr = parts.next().and_then(|a| {
                u32::from_str_radix(a.trim_end_matches(':'), 16).ok()
            });
            let Some(addr) = addr else { continue };
            let mut words = Vec::new();
            let mut ann = String::new();
            for tok in parts {
                if let Some(a) = tok.strip_prefix('<') {
                    ann = format!("<{a}");
                    for rest in parts {
                        ann.push(' ');
                        ann.push_str(rest);
                    }
                    break;
                }
                if tok.len() == 8
                    && let Ok(w) = u32::from_str_radix(tok, 16)
                {
                    words.push(w);
                }
            }
            out.push(ObjdumpLine { addr, words, text, ann });
        }
        out
    }

    /// Committed pinned objdump of the selected symbol (fallback when no
    /// toolchain is present).
    pub(crate) fn pinned_objdump_fixture() -> Vec<ObjdumpLine> {
        const FIXTURE: &str =
            include_str!("../../tests/fixtures/kt48/hipcc.objdump.txt");
        parse_objdump_stdout(FIXTURE)
    }

    /// Run the pinned `llvm-mc` over bare canonical lines, returning the
    /// emitted little-endian words per input line in order. Any assembler
    /// error is returned with its line number; the caller decides whether an
    /// error is a failure or (for crafted probes) the expected outcome.
    pub(crate) fn mc_batch(
        mc: &std::path::Path,
        lines: &[String],
    ) -> Result<Vec<Vec<u32>>, String> {
        let path =
            std::env::temp_dir().join(format!("pm-c3b-mc-{}.s", std::process::id()));
        std::fs::write(&path, lines.join("\n") + "\n")
            .map_err(|e| format!("mc input write: {e}"))?;
        let output = std::process::Command::new(mc)
            .args(["-arch=amdgcn", "-mcpu=gfx1201", "-show-encoding"])
            .arg(&path)
            .output()
            .map_err(|e| format!("spawn llvm-mc: {e}"))?;
        let _ = std::fs::remove_file(&path);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut encodings: Vec<Vec<u32>> = Vec::new();
        for line in stdout.lines() {
            let Some(start) = line.find("; encoding: [") else { continue };
            let bytes = line[start + "; encoding: [".len()..]
                .trim_end_matches(']')
                .split(',')
                .map(|b| {
                    u8::from_str_radix(b.trim().trim_start_matches("0x"), 16)
                        .map_err(|_| format!("bad mc encoding byte in {line:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if bytes.len() % 4 != 0 {
                return Err(format!("mc encoding not word-aligned in {line:?}"));
            }
            encodings.push(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            );
        }
        if !output.status.success() {
            return Err(format!(
                "llvm-mc failed ({} of {} lines assembled):\n{stderr}",
                encodings.len(),
                lines.len()
            ));
        }
        if encodings.len() != lines.len() {
            return Err(format!(
                "llvm-mc emitted {} encodings for {} lines",
                encodings.len(),
                lines.len()
            ));
        }
        Ok(encodings)
    }
}
