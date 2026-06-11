//! SP4: selective re-quant overlay builder. See
//! docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md §3.
//!
//! `quantize_to_format` — tier-name → existing `quantize_*` encoder dispatch.
//! (The arch-aware `reap_override_for` resolver is added in the next task.)

use crate::{
    gen_fwht_signs, quantize_hfq4g256, quantize_hfq6g256, quantize_mq2g256_lloyd,
    quantize_mq3g256_lloyd, quantize_mq4g256, quantize_mq4g256_lloyd, quantize_mq6g256,
    quantize_q8f16, HfqTensor, QuantType,
};

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
