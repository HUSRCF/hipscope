use std::path::{Path, PathBuf};

/// One selective-requant edit (applied in SP2+; parsed & validated now).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantOverride {
    pub layer: usize,
    pub role: Role,
    /// Only meaningful for `Role::RoutedExperts`; empty ⇒ whole role at this layer.
    pub experts: Vec<u32>,
    pub tier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    RoutedExperts,
    SharedExpert,
    Attention,
    Router,
    LmHead,
    Embed,
}

impl Role {
    pub fn parse(s: &str) -> Result<Role, String> {
        Ok(match s {
            "routed_experts" => Role::RoutedExperts,
            "shared_expert" => Role::SharedExpert,
            "attention" => Role::Attention,
            "router" => Role::Router,
            "lm_head" => Role::LmHead,
            "embed" => Role::Embed,
            other => return Err(format!("reap: unknown role '{other}'")),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ReapPlan {
    pub model_arch: Option<String>,
    pub num_layers: usize,
    pub original_experts: usize,
    /// `keep[l][slot]` = original expert index in compact slot `slot`.
    /// `None` ⇒ no pruning (keep all `original_experts`).
    pub keep: Option<Vec<Vec<u32>>>,
    pub quant_overrides: Vec<QuantOverride>,
    pub dir: PathBuf,
}

impl ReapPlan {
    /// Returns 0 if keep is Some with an empty outer vec (cannot arise from load()).
    pub fn kept_per_layer(&self) -> usize {
        match &self.keep {
            Some(k) => k.first().map(|r| r.len()).unwrap_or(0),
            None => self.original_experts,
        }
    }

    /// Load `<dir>/reap_plan.json`, validating against the model's layer/expert
    /// counts (passed BEFORE any n_routed_experts override).
    pub fn load(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        let path = Path::new(dir).join("reap_plan.json");
        let txt = std::fs::read_to_string(&path)
            .map_err(|e| format!("reap: read {path:?}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&txt).map_err(|e| format!("reap: parse {path:?}: {e}"))?;

        let original_experts = v["original_experts"]
            .as_u64()
            .unwrap_or(orig_experts_expected as u64) as usize;
        if original_experts != orig_experts_expected {
            return Err(format!(
                "reap: original_experts {original_experts} != model n_routed_experts {orig_experts_expected}"
            ));
        }
        let num_layers = v["num_layers"].as_u64().unwrap_or(num_layers_expected as u64) as usize;
        if num_layers != num_layers_expected {
            return Err(format!(
                "reap: num_layers {num_layers} != model num_hidden_layers {num_layers_expected}"
            ));
        }

        let keep = match v["keep"]["per_layer"].as_array() {
            None => None,
            Some(arr) => {
                if arr.len() != num_layers_expected {
                    return Err(format!(
                        "reap: keep.per_layer has {} layers, model has {num_layers_expected}",
                        arr.len()
                    ));
                }
                let kept = arr.first().and_then(|r| r.as_array()).map(|r| r.len()).unwrap_or(0);
                let mut out = Vec::with_capacity(arr.len());
                for (l, row) in arr.iter().enumerate() {
                    let r = row
                        .as_array()
                        .ok_or_else(|| format!("reap: keep layer {l} not an array"))?;
                    if r.len() != kept {
                        return Err(format!(
                            "reap: keep layer {l} has {} entries, expected {kept}",
                            r.len()
                        ));
                    }
                    let mut v32 = Vec::with_capacity(kept);
                    for x in r {
                        let idx = x
                            .as_u64()
                            .ok_or_else(|| format!("reap: keep layer {l} non-integer index"))?
                            as u32;
                        if idx as usize >= original_experts {
                            return Err(format!(
                                "reap: keep layer {l} index {idx} >= original_experts {original_experts}"
                            ));
                        }
                        v32.push(idx);
                    }
                    out.push(v32);
                }
                Some(out)
            }
        };

        let mut quant_overrides = Vec::new();
        if let Some(arr) = v["quant_overrides"].as_array() {
            for (i, o) in arr.iter().enumerate() {
                let layer = o["layer"]
                    .as_u64()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing layer"))?
                    as usize;
                if layer >= num_layers_expected {
                    return Err(format!(
                        "reap: quant_override[{i}] layer {layer} >= num_layers {num_layers_expected}"
                    ));
                }
                let role = Role::parse(
                    o["role"].as_str().ok_or_else(|| format!("reap: quant_override[{i}] missing role"))?,
                )?;
                let experts: Vec<u32> = if let Some(a) = o["experts"].as_array() {
                    a.iter().enumerate().map(|(j, x)| {
                        let n = x.as_u64()
                            .ok_or_else(|| format!("reap: quant_override[{i}] experts[{j}] not an integer"))? as u32;
                        if n as usize >= original_experts {
                            return Err(format!("reap: quant_override[{i}] expert {n} >= original_experts {original_experts}"));
                        }
                        Ok(n)
                    }).collect::<Result<Vec<_>, String>>()?
                } else {
                    Vec::new()
                };
                if !experts.is_empty() && role != Role::RoutedExperts {
                    return Err(format!(
                        "reap: quant_override[{i}] lists experts but role is not routed_experts"
                    ));
                }
                let tier = o["tier"]
                    .as_str()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing tier"))?
                    .to_string();
                quant_overrides.push(QuantOverride { layer, role, experts, tier });
            }
        }

        Ok(ReapPlan {
            model_arch: v["model_arch"].as_str().map(|s| s.to_string()),
            num_layers: num_layers_expected,
            original_experts,
            keep,
            quant_overrides,
            dir: PathBuf::from(dir),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_plan(json: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(d.path().join("reap_plan.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        d
    }

    #[test]
    fn keep_all_when_keep_absent() {
        let d = write_plan(r#"{"original_experts":8,"num_layers":2}"#);
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 8).unwrap();
        assert!(p.keep.is_none());
        assert_eq!(p.kept_per_layer(), 8);
    }

    #[test]
    fn parses_keep_and_overrides() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":2,
                "keep":{"per_layer":[[0,2,3],[1,2,3]]},
                "quant_overrides":[{"layer":1,"role":"routed_experts","experts":[2],"tier":"mq3lloyd"}]}"#,
        );
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 4).unwrap();
        assert_eq!(p.kept_per_layer(), 3);
        assert_eq!(p.keep.as_ref().unwrap()[0], vec![0, 2, 3]);
        assert_eq!(p.quant_overrides.len(), 1);
        assert_eq!(p.quant_overrides[0].tier, "mq3lloyd");
    }

    #[test]
    fn rejects_out_of_range_index() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,"keep":{"per_layer":[[0,9]]}}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("index 9 >= original_experts 4"), "got: {err}");
    }

    #[test]
    fn rejects_experts_on_non_routed_role() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"attention","experts":[1],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("not routed_experts"), "got: {err}");
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let d = write_plan(r#"{"original_experts":4,"num_layers":3,"keep":{"per_layer":[[0,1]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 3, 4).unwrap_err();
        assert!(err.contains("keep.per_layer has 1 layers"), "got: {err}");
    }

    #[test]
    fn rejects_non_integer_override_expert() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"routed_experts","experts":[1,"bad"],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("not an integer"), "got: {err}");
    }

    #[test]
    fn rejects_out_of_range_override_expert() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"routed_experts","experts":[9],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains(">= original_experts 4"), "got: {err}");
    }
}
