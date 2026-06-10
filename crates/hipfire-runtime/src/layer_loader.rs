// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative weight-name resolution for arch crates.
//!
//! Each arch crate declares static `WeightSpec` arrays describing candidate
//! tensor name patterns. `resolve_candidate` finds the first candidate that
//! exists in the source, substituting `{prefix}` and `{layer}` as needed.
//!
//! Loading, uploading, and augmentation are the caller's responsibility —
//! this module only handles name resolution.

use crate::model_source::ModelSource;

/// A descriptor for one weight tensor: its logical name within a layer and the
/// ordered list of candidate on-disk paths to try.
///
/// Patterns may contain `{prefix}` (e.g. `"model.language_model"`) and `{layer}`
/// (e.g. `"3"`), which are substituted before lookup.
pub struct WeightSpec {
    /// Short logical name used as a map key or for error messages.
    pub logical: &'static str,
    /// Candidate path templates tried in order; first match wins.
    pub candidates: &'static [&'static str],
    /// If true, `resolve_required` returns an error when all candidates miss.
    pub required: bool,
}

/// A collection of weight specs describing one layer variant
/// (e.g. full-attention layer, DeltaNet layer, MoE layer).
pub struct LayerSpec {
    pub weights: &'static [WeightSpec],
}

/// Substitute `{prefix}` and `{layer}` into a candidate template.
fn apply_template(tmpl: &str, prefix: &str, layer: usize) -> String {
    tmpl.replace("{prefix}", prefix)
        .replace("{layer}", &layer.to_string())
}

/// Return the first candidate for `spec` that exists in `source`, or `None`.
pub fn resolve_candidate(
    source: &dyn ModelSource,
    spec: &WeightSpec,
    prefix: &str,
    layer: usize,
) -> Option<String> {
    spec.candidates
        .iter()
        .map(|tmpl| apply_template(tmpl, prefix, layer))
        .find(|name| source.tensor_info(name).is_some())
}

/// Like `resolve_candidate` but returns `Err` if the spec is required and
/// no candidate is found.
pub fn resolve_required(
    source: &dyn ModelSource,
    spec: &WeightSpec,
    prefix: &str,
    layer: usize,
) -> Result<String, String> {
    resolve_candidate(source, spec, prefix, layer).ok_or_else(|| {
        format!(
            "required weight '{}' not found in source (prefix={prefix:?}, layer={layer}, \
             tried {} candidates)",
            spec.logical,
            spec.candidates.len()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_source::{QuantConfig, TensorInfo};
    use std::collections::HashMap;
    use std::path::Path;

    struct MockSource(HashMap<String, TensorInfo>);

    fn make_source(names: &[&str]) -> MockSource {
        MockSource(names.iter().map(|n| (n.to_string(), TensorInfo {
            name: n.to_string(),
            dtype: "F16".into(),
            shape: vec![1, 1],
            quant_type: 0xFF,
            data_offset: 0,
            data_size: 2,
        })).collect())
    }

    impl ModelSource for MockSource {
        fn metadata_json(&self) -> &str { "{}" }
        fn arch_id(&self) -> u32 { 0 }
        fn quant_config(&self) -> Option<&QuantConfig> { None }
        fn tensor_data(&self, _: &str) -> Option<(&TensorInfo, &[u8])> { None }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> { self.0.get(name) }
        fn tensor_names(&self) -> Vec<&str> { self.0.keys().map(|s| s.as_str()).collect() }
        fn path(&self) -> &Path { Path::new("/tmp/mock") }
    }

    static SPEC: WeightSpec = WeightSpec {
        logical: "wq",
        candidates: &[
            "{prefix}.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.q_proj.weight",
        ],
        required: true,
    };

    #[test]
    fn resolve_picks_first_match() {
        let src = make_source(&[
            "model.language_model.layers.3.self_attn.q_proj.weight",
        ]);
        let result = resolve_candidate(
            &src,
            &SPEC,
            "model.language_model",
            3,
        );
        assert_eq!(
            result.unwrap(),
            "model.language_model.layers.3.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn resolve_falls_back_to_second_candidate() {
        let src = make_source(&[
            "model.layers.3.self_attn.q_proj.weight",
        ]);
        let result = resolve_candidate(&src, &SPEC, "model.language_model", 3);
        assert_eq!(result.unwrap(), "model.layers.3.self_attn.q_proj.weight");
    }

    #[test]
    fn resolve_returns_none_when_not_found() {
        let src = make_source(&[]);
        let result = resolve_candidate(&src, &SPEC, "model", 0);
        assert!(result.is_none());
    }

    #[test]
    fn resolve_errors_when_required_and_not_found() {
        let src = make_source(&[]);
        let spec = WeightSpec { logical: "wq", candidates: &[], required: true };
        let result = resolve_required(&src, &spec, "model", 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("wq"));
    }
}
