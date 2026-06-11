//! SP4: selective re-quant overlay builder. See
//! docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md §3.
//!
//! Two pure-ish units shared by the overlay (and, later, bake) flow:
//!   * `quantize_to_format` — tier-name → existing `quantize_*` encoder dispatch.
//!   * `reap_override_for`   — arch-aware tensor-name → override-tier resolver.

use crate::{
    gen_fwht_signs, quantize_hfq4g256, quantize_hfq6g256, quantize_mq2g256_lloyd,
    quantize_mq3g256_lloyd, quantize_mq4g256, quantize_mq4g256_lloyd, quantize_mq6g256,
    quantize_q8f16, HfqTensor, QuantType,
};
use hipfire_reap::plan::{QuantOverride, ReapPlan, Role};

/// Quantize one tensor's f32 data to the named tier, returning the HFQ tensor.
/// Covers the self-calibrating tiers usable in an overlay without an imatrix.
/// `shape` is the row-major tensor shape (e.g. `[rows, cols]`).
///
/// The FWHT-rotated tiers (mq*/lloyd) use the canonical sign seeds 42 / 1042 —
/// the same seeds the monolithic quantize loop uses (`main.rs` `gen_fwht_signs(42|1042, 256)`),
/// so the bytes produced here are byte-identical to `--format <tier>`.
pub fn quantize_to_format(
    name: &str,
    fmt: &str,
    f32_data: &[f32],
    shape: &[usize],
) -> Result<HfqTensor, String> {
    let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
    // Canonical FWHT sign tables (only built for the rotated tiers).
    let signs = || (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
    let (qt, gs, data) = match fmt {
        "q8" | "q8f16" => (QuantType::Q8F16, 0u32, quantize_q8f16(f32_data)),
        "hfq4" | "hfq4g256" => (QuantType::HFQ4G256, 256, quantize_hfq4g256(f32_data)),
        "hfq6" | "hfq6g256" => (QuantType::HFQ6G256, 256, quantize_hfq6g256(f32_data)),
        "mq4" | "mq4g256" => {
            let (s1, s2) = signs();
            (QuantType::MQ4G256, 256, quantize_mq4g256(f32_data, &s1, &s2))
        }
        "mq6" | "mq6g256" => {
            let (s1, s2) = signs();
            (QuantType::MQ6G256, 256, quantize_mq6g256(f32_data, &s1, &s2))
        }
        "mq2lloyd" | "mq2g256lloyd" => {
            let (s1, s2) = signs();
            (QuantType::MQ2G256Lloyd, 256, quantize_mq2g256_lloyd(f32_data, &s1, &s2))
        }
        "mq3lloyd" | "mq3g256lloyd" => {
            let (s1, s2) = signs();
            (QuantType::MQ3G256Lloyd, 256, quantize_mq3g256_lloyd(f32_data, &s1, &s2))
        }
        "mq4lloyd" | "mq4g256lloyd" => {
            let (s1, s2) = signs();
            (QuantType::MQ4G256Lloyd, 256, quantize_mq4g256_lloyd(f32_data, &s1, &s2))
        }
        other => return Err(format!("reap: unsupported overlay tier '{other}' for {name}")),
    };
    Ok(HfqTensor {
        name: name.to_string(),
        quant_type: qt,
        shape: shape_u32,
        group_size: gs,
        data,
        spilled_len: 0,
    })
}

/// Detected arch family for tensor-name matching (the quantizer already knows
/// the arch_id; pass the matching variant in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReapArch {
    Deepseek4,
    Qwen35,
    Lfm2Moe,
    Minimax,
}

/// Resolve a tensor name to its override tier under `plan`, or None.
/// Matches by (layer, role, [expert]) using the arch's tensor naming.
pub fn reap_override_for<'a>(name: &str, arch: ReapArch, plan: &'a ReapPlan) -> Option<&'a str> {
    for ov in &plan.quant_overrides {
        if tensor_matches(name, arch, ov) {
            return Some(ov.tier.as_str());
        }
    }
    None
}

fn tensor_matches(name: &str, arch: ReapArch, ov: &QuantOverride) -> bool {
    // Layer gate: the name must reference `ov.layer`. All four arches embed the
    // layer index as `.layers.{L}.` or `layers.{L}.`.
    let layer_tok = format!("layers.{}.", ov.layer);
    if !name.contains(&layer_tok) {
        return false;
    }
    match ov.role {
        Role::RoutedExperts => {
            let (seg, w_ok): (&str, fn(&str) -> bool) = match arch {
                ReapArch::Deepseek4 => (
                    ".ffn.experts.",
                    |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight"),
                ),
                ReapArch::Qwen35 => (
                    ".mlp.experts.",
                    |n| n.ends_with(".gate_up_proj.weight") || n.ends_with(".down_proj.weight"),
                ),
                ReapArch::Lfm2Moe => (
                    ".feed_forward.experts.",
                    |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight"),
                ),
                ReapArch::Minimax => (
                    ".block_sparse_moe.experts.",
                    |n| n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight"),
                ),
            };
            if !name.contains(seg) || !w_ok(name) {
                return false;
            }
            if ov.experts.is_empty() {
                return true; // whole role at this layer
            }
            // expert index: the token right after `seg`. ds4 layout (verified
            // against main.rs:5843 — `layers.L.ffn.experts.E.{w1,w2,w3}.weight`)
            // makes the first dotted token after `seg` the expert index.
            let after = &name[name.find(seg).unwrap() + seg.len()..];
            let eidx: u32 = match after.split('.').next().and_then(|s| s.parse().ok()) {
                Some(e) => e,
                None => return false,
            };
            ov.experts.contains(&eidx)
        }
        Role::Attention => {
            name.contains(".self_attn.") || name.contains(".attn.") || name.contains(".attention.")
        }
        Role::Router => {
            name.contains(".gate.weight") || name.contains(".router") || name.contains(".gate.tid2eid")
        }
        Role::SharedExpert => name.contains(".shared_expert") || name.contains(".shared_experts"),
        Role::LmHead => name.contains("lm_head") || name.contains("output.weight"),
        Role::Embed => name.contains("embed_tokens") || name.contains("tok_embeddings"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_underlying_encoder_byte_for_byte() {
        // 512 f32 (two 256-groups) of varied values.
        let f32: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let direct = quantize_hfq4g256(&f32);
        let t = quantize_to_format("x", "hfq4g256", &f32, &[2, 256]).unwrap();
        assert_eq!(t.data, direct, "overlay encode must equal direct encode");
        assert_eq!(t.shape, vec![2u32, 256]);
        assert_eq!(t.quant_type, QuantType::HFQ4G256);
    }

    #[test]
    fn mq4_matches_with_canonical_signs() {
        let f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let direct = quantize_mq4g256(&f32, &s1, &s2);
        let t = quantize_to_format("x", "mq4", &f32, &[1, 256]).unwrap();
        assert_eq!(t.data, direct);
    }

    #[test]
    fn rejects_unknown_tier() {
        let err = quantize_to_format("x", "bogus", &[0.0; 256], &[1, 256]).unwrap_err();
        assert!(err.contains("unsupported overlay tier 'bogus'"), "got: {err}");
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    fn plan_with(json: &str) -> ReapPlan {
        // write to a tempdir & load_unchecked
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("reap_plan.json"), json).unwrap();
        ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn ds4_specific_experts() {
        let p = plan_with(
            r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7],"tier":"mq3lloyd"}]}"#,
        );
        assert_eq!(
            reap_override_for("layers.20.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p),
            Some("mq3lloyd")
        );
        assert_eq!(
            reap_override_for("layers.20.ffn.experts.8.w1.weight", ReapArch::Deepseek4, &p),
            None
        ); // wrong expert
        assert_eq!(
            reap_override_for("layers.21.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p),
            None
        ); // wrong layer
    }

    #[test]
    fn qwen35_whole_role() {
        let p = plan_with(
            r#"{"original_experts":128,"num_layers":48,
            "quant_overrides":[{"layer":5,"role":"routed_experts","tier":"hfq6"}]}"#,
        );
        assert_eq!(
            reap_override_for("model.layers.5.mlp.experts.99.gate_up_proj.weight", ReapArch::Qwen35, &p),
            Some("hfq6")
        );
        assert_eq!(
            reap_override_for("model.layers.5.mlp.experts.99.down_proj.weight", ReapArch::Qwen35, &p),
            Some("hfq6")
        );
        assert_eq!(
            reap_override_for("model.layers.5.self_attn.q_proj.weight", ReapArch::Qwen35, &p),
            None
        );
    }

    #[test]
    fn attention_role() {
        let p = plan_with(
            r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":41,"role":"attention","tier":"q8"}]}"#,
        );
        assert_eq!(
            reap_override_for("model.layers.41.self_attn.q_proj.weight", ReapArch::Qwen35, &p),
            Some("q8")
        );
        assert_eq!(
            reap_override_for("model.layers.40.self_attn.q_proj.weight", ReapArch::Qwen35, &p),
            None
        );
    }
}
