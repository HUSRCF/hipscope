//! Typed RDNA machine-code representation and gfx12 instruction tables.
pub mod cfg;
pub mod codec;
pub mod descriptor;
pub mod edit;
pub mod effects;
pub mod inst;
pub mod isa;
pub mod lds;
pub mod metadata;
pub mod operand;
pub mod passes;
pub mod provenance;
pub mod reg;
pub mod state;
pub mod wait;

pub use cfg::{Block, BlockId, Body, InstId};
pub use inst::{Arch, Form, FormFields, Inst, Kernel, Opcode, Program, Target};

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::{HashMap, HashSet}, process::Command};

    #[test]
    fn opcode_examples_match_pinned_llvm_mc_for_every_declared_form() {
        let mc = "/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc";
        assert!(std::path::Path::new(mc).exists(), "pinned llvm-mc required for the table gate");
        for row in isa::gfx12() {
            let mut output = Command::new(mc).args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-show-encoding"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn().expect("launch pinned llvm-mc");
            use std::io::Write;
            output.stdin.take().expect("stdin").write_all(format!("{}\n", row.sample).as_bytes()).expect("write instruction");
            let result = output.wait_with_output().expect("llvm-mc completion");
            assert!(result.status.success(), "{} {:?}: {}", row.name, row.form, String::from_utf8_lossy(&result.stderr));
            let stdout = String::from_utf8(result.stdout).expect("llvm-mc UTF-8 output");
            let encoded = stdout.split("encoding: [").nth(1).unwrap_or_else(|| panic!("no encoding for {}", row.sample))
                .split(']').next().expect("closing bracket");
            let bytes: Vec<_> = encoded.split(',').map(|word| u8::from_str_radix(word.trim().trim_start_matches("0x"), 16).unwrap()).collect();
            let expected: Vec<_> = row.encoding.split_whitespace()
                .flat_map(|word| u32::from_str_radix(word, 16).unwrap().to_le_bytes()).collect();
            assert_eq!(bytes, expected, "{} {:?}", row.sample, row.form);
        }
    }

    #[test]
    fn runtime_dependency_graph_does_not_include_peacemaker() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
        let output = Command::new("cargo").args(["metadata", "--format-version", "1", "--manifest-path"])
            .arg(manifest).output().expect("cargo metadata");
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let meta: serde_json::Value = serde_json::from_slice(&output.stdout).expect("cargo metadata JSON");
        let packages: HashMap<&str, &str> = meta["packages"].as_array().unwrap().iter()
            .map(|p| (p["id"].as_str().unwrap(), p["name"].as_str().unwrap())).collect();
        let graph: HashMap<&str, Vec<&str>> = meta["resolve"]["nodes"].as_array().unwrap().iter()
            .map(|node| (node["id"].as_str().unwrap(), node["deps"].as_array().unwrap().iter()
                .map(|dep| dep["pkg"].as_str().unwrap()).collect())).collect();
        for root in ["hipfire-daemon", "rdna-compute"] {
            let id = *packages.iter().find(|(_, &name)| name == root).expect("runtime crate present").0;
            let mut stack = vec![id];
            let mut visited = HashSet::new();
            while let Some(pkg) = stack.pop() {
                if !visited.insert(pkg) { continue; }
                assert!(!packages[pkg].starts_with("peacemaker-"), "{root} reaches {}", packages[pkg]);
                stack.extend(graph[pkg].iter().copied());
            }
        }
    }

    #[test]
    fn dangerous_literal_selector_and_unaligned_smem_are_rejected() {
        use reg::{Kind, RegRef};
        use operand::Operand;
        let row = isa::gfx12().iter().find(|r| r.name == "v_add_co_u32").unwrap();
        let mut inst = Inst { op: row.op, form: row.form, fields: FormFields::Vop3b { src2_unused: 0xff },
            operands: Default::default(), mods: Default::default(), literal: None, effects: Default::default(), prov: Default::default() };
        assert!(matches!(inst.validate(Arch::Gfx1201), Err(inst::ValidateError::DangerousFill { field: "src2_unused", value: 0xff })));
        inst.fields = FormFields::Vop3b { src2_unused: 0x80 };
        inst.validate(Arch::Gfx1201).unwrap();
        let row = isa::gfx12().iter().find(|r| r.name == "s_load_b128").unwrap();
        inst.op = row.op; inst.form = row.form; inst.fields = FormFields::None;
        inst.operands.push(Operand::Reg(RegRef { kind: Kind::S, base: 5, len: 4 }));
        assert!(matches!(inst.validate(Arch::Gfx1201), Err(inst::ValidateError::MisalignedSmemSdata { .. })));
    }
}
