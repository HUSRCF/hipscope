//! Experimental dense-FFN group pruner for Qwen3.6 MQ4 containers.
//!
//! This rewrites complete 256-channel FFN groups. MQ4's down-projection FWHT
//! is independent per 256 values, so group gather preserves the rotation
//! boundary. The weight-only ranking is only an admission probe; the output
//! still requires model-quality evaluation.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
const MQ4_QT: u8 = 13;
const GROUP: usize = 256;
const MQ4_GROUP_BYTES: usize = 136;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Projection {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Debug)]
enum Rewrite {
    Copy,
    OutputRows { layer: usize },
    InputGroups { layer: usize },
}

#[derive(Clone)]
struct OutputTensor {
    source: HfqTensorInfo,
    shape: Vec<u32>,
    data_size: usize,
    rewrite: Rewrite,
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    activation_energy: Option<PathBuf>,
    keep_groups: usize,
    dry_run: bool,
}

struct ActivationCalibration {
    energy: BTreeMap<usize, Vec<f64>>,
    model_bytes: u64,
    model_sha256: String,
    metadata_fingerprint: String,
    hidden_dim: usize,
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn sha256_file(path: &Path) -> String {
    let mut file = File::open(path).expect("open model for SHA-256");
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).expect("read model for SHA-256");
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn usage() -> ! {
    eprintln!(
        "usage: prune_dense_ffn_groups --input MODEL --keep-groups N \
         [--output MODEL] [--activation-energy FILE.json] [--dry-run]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut input = None;
    let mut output = None;
    let mut activation_energy = None;
    let mut keep_groups = None;
    let mut dry_run = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input = args.get(i).map(PathBuf::from);
            }
            "--output" => {
                i += 1;
                output = args.get(i).map(PathBuf::from);
            }
            "--activation-energy" => {
                i += 1;
                activation_energy = args.get(i).map(PathBuf::from);
            }
            "--keep-groups" => {
                i += 1;
                keep_groups = args.get(i).and_then(|v| v.parse().ok());
            }
            "--dry-run" => dry_run = true,
            _ => usage(),
        }
        i += 1;
    }
    let args = Args {
        input: input.unwrap_or_else(|| usage()),
        output,
        activation_energy,
        keep_groups: keep_groups.unwrap_or_else(|| usage()),
        dry_run,
    };
    if !args.dry_run && args.output.is_none() {
        usage();
    }
    args
}

fn projection(name: &str) -> Option<(usize, Projection)> {
    let (_, suffix) = name.split_once(".layers.")?;
    let (layer, rest) = suffix.split_once('.')?;
    let layer = layer.parse().ok()?;
    let kind = match rest {
        "mlp.gate_proj.weight" => Projection::Gate,
        "mlp.up_proj.weight" => Projection::Up,
        "mlp.down_proj.weight" => Projection::Down,
        _ => return None,
    };
    Some((layer, kind))
}

fn block_energy(block: &[u8]) -> f64 {
    debug_assert_eq!(block.len(), MQ4_GROUP_BYTES);
    let scale = f32::from_le_bytes(block[0..4].try_into().unwrap()) as f64;
    let zero = f32::from_le_bytes(block[4..8].try_into().unwrap()) as f64;
    let mut sum_q = 0.0f64;
    let mut sum_q2 = 0.0f64;
    for &packed in &block[8..] {
        let lo = (packed & 0x0f) as f64;
        let hi = (packed >> 4) as f64;
        sum_q += lo + hi;
        sum_q2 += lo * lo + hi * hi;
    }
    scale * scale * sum_q2 + 2.0 * scale * zero * sum_q + GROUP as f64 * zero * zero
}

fn output_group_energy(data: &[u8], m: usize, k: usize) -> Vec<f64> {
    let groups_per_row = k / GROUP;
    let row_bytes = groups_per_row * MQ4_GROUP_BYTES;
    let output_groups = m / GROUP;
    let mut result = vec![0.0f64; output_groups];
    for (group, energy) in result.iter_mut().enumerate() {
        for row in group * GROUP..(group + 1) * GROUP {
            let row_start = row * row_bytes;
            for k_group in 0..groups_per_row {
                let start = row_start + k_group * MQ4_GROUP_BYTES;
                *energy += block_energy(&data[start..start + MQ4_GROUP_BYTES]);
            }
        }
    }
    result
}

fn input_group_energy(data: &[u8], m: usize, k: usize) -> Vec<f64> {
    let groups_per_row = k / GROUP;
    let row_bytes = groups_per_row * MQ4_GROUP_BYTES;
    let mut result = vec![0.0f64; groups_per_row];
    for row in 0..m {
        let row_start = row * row_bytes;
        for (group, energy) in result.iter_mut().enumerate() {
            let start = row_start + group * MQ4_GROUP_BYTES;
            *energy += block_energy(&data[start..start + MQ4_GROUP_BYTES]);
        }
    }
    result
}

fn assert_canonical_mq4(tensor: &HfqTensorInfo, m: usize, k: usize, label: &str) {
    assert_eq!(
        tensor.quant_type, MQ4_QT,
        "{label}: expected MQ4G256 quant_type={MQ4_QT}"
    );
    assert_eq!(
        tensor.group_size, GROUP as u32,
        "{label}: expected group_size={GROUP}"
    );
    assert_eq!(m % GROUP, 0, "{label}: output width must be 256-aligned");
    assert_eq!(k % GROUP, 0, "{label}: input width must be 256-aligned");
    let expected = m
        .checked_mul(k / GROUP)
        .and_then(|groups| groups.checked_mul(MQ4_GROUP_BYTES))
        .expect("MQ4 tensor size overflow");
    assert_eq!(
        tensor.data_size, expected,
        "{label}: non-canonical MQ4G256 payload size"
    );
}

fn load_activation_energy(path: &Path) -> ActivationCalibration {
    let root: Value = serde_json::from_slice(&std::fs::read(path).expect("read activation energy"))
        .expect("parse activation energy JSON");
    assert_eq!(
        root.get("group_size").and_then(Value::as_u64),
        Some(GROUP as u64),
        "activation calibration group_size mismatch"
    );
    assert_eq!(
        root.get("metric").and_then(Value::as_str),
        Some("sum_square_silu_gate_mul_up"),
        "unsupported activation calibration metric"
    );
    let mut result = BTreeMap::new();
    for entry in root
        .get("layers")
        .and_then(Value::as_array)
        .expect("activation calibration layers array")
    {
        let layer = entry
            .get("layer")
            .and_then(Value::as_u64)
            .expect("activation calibration layer id") as usize;
        let energy: Vec<f64> = entry
            .get("energy")
            .and_then(Value::as_array)
            .expect("activation calibration energy array")
            .iter()
            .map(|v| {
                let x = v.as_f64().expect("activation energy number");
                assert!(x.is_finite() && x >= 0.0, "invalid activation energy");
                x
            })
            .collect();
        assert!(
            result.insert(layer, energy).is_none(),
            "duplicate layer {layer}"
        );
    }
    assert!(!result.is_empty(), "empty activation calibration");
    ActivationCalibration {
        energy: result,
        model_bytes: root
            .get("model_bytes")
            .and_then(Value::as_u64)
            .expect("activation calibration model_bytes"),
        model_sha256: root
            .get("model_sha256")
            .and_then(Value::as_str)
            .expect("activation calibration model_sha256")
            .to_string(),
        metadata_fingerprint: root
            .get("model_metadata_fingerprint")
            .and_then(Value::as_str)
            .expect("activation calibration model_metadata_fingerprint")
            .to_string(),
        hidden_dim: root
            .get("hidden_dim")
            .and_then(Value::as_u64)
            .expect("activation calibration hidden_dim") as usize,
    }
}

fn collect_selections(
    hfq: &HfqFile,
    keep_groups: usize,
    activation: Option<&BTreeMap<usize, Vec<f64>>>,
) -> BTreeMap<usize, Vec<usize>> {
    let mut layers: BTreeMap<usize, [Option<&HfqTensorInfo>; 3]> = BTreeMap::new();
    for tensor in hfq.tensors() {
        let Some((layer, kind)) = projection(&tensor.name) else {
            continue;
        };
        let slot = match kind {
            Projection::Gate => 0,
            Projection::Up => 1,
            Projection::Down => 2,
        };
        layers.entry(layer).or_insert([None, None, None])[slot] = Some(tensor);
    }

    let mut selections = BTreeMap::new();
    for (layer, tensors) in layers {
        let [Some(gate), Some(up), Some(down)] = tensors else {
            panic!("layer {layer}: incomplete dense FFN projection set");
        };
        let [m, k] = [gate.shape[0] as usize, gate.shape[1] as usize];
        assert_eq!(
            up.shape, gate.shape,
            "layer {layer}: gate/up shape mismatch"
        );
        assert_eq!(down.shape, vec![k as u32, m as u32]);
        assert_canonical_mq4(gate, m, k, &format!("layer {layer} gate"));
        assert_canonical_mq4(up, m, k, &format!("layer {layer} up"));
        assert_canonical_mq4(down, k, m, &format!("layer {layer} down"));
        let total_groups = m / GROUP;
        assert!(keep_groups > 0 && keep_groups <= total_groups);

        let gate_data = hfq.tensor_data(&gate.name).unwrap().1;
        let up_data = hfq.tensor_data(&up.name).unwrap().1;
        let down_data = hfq.tensor_data(&down.name).unwrap().1;
        let gate_energy = activation
            .is_none()
            .then(|| output_group_energy(gate_data, m, k));
        let up_energy = activation
            .is_none()
            .then(|| output_group_energy(up_data, m, k));
        let down_energy = input_group_energy(down_data, k, m);
        let activation_energy = activation.map(|all| {
            let energy = all
                .get(&layer)
                .unwrap_or_else(|| panic!("activation calibration missing layer {layer}"));
            assert_eq!(
                energy.len(),
                total_groups,
                "layer {layer}: activation group count mismatch"
            );
            energy
        });

        let mut ranked: Vec<(usize, f64)> = (0..total_groups)
            .map(|group| {
                let score = if let Some(energy) = activation_energy {
                    energy[group].max(f64::MIN_POSITIVE).ln()
                        + down_energy[group].max(f64::MIN_POSITIVE).ln()
                } else {
                    (gate_energy.as_ref().unwrap()[group]
                        .max(f64::MIN_POSITIVE)
                        .ln()
                        + up_energy.as_ref().unwrap()[group]
                            .max(f64::MIN_POSITIVE)
                            .ln()
                        + down_energy[group].max(f64::MIN_POSITIVE).ln())
                        / 3.0
                };
                (group, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let cutoff = ranked[keep_groups - 1].1;
        let best = ranked[0].1;
        let worst = ranked.last().unwrap().1;
        let mut selected: Vec<usize> = ranked[..keep_groups].iter().map(|x| x.0).collect();
        selected.sort_unstable();
        eprintln!(
            "layer={layer:02} keep={keep_groups}/{total_groups} score_exp(best/cut/worst)={:.4e}/{:.4e}/{:.4e} groups={selected:?}",
            best.exp(),
            cutoff.exp(),
            worst.exp(),
        );
        selections.insert(layer, selected);
    }
    assert!(!selections.is_empty(), "no dense FFN layers found");
    selections
}

fn output_tensors(hfq: &HfqFile, selections: &BTreeMap<usize, Vec<usize>>) -> Vec<OutputTensor> {
    hfq.tensors()
        .iter()
        .map(|tensor| {
            let mut out = OutputTensor {
                source: tensor.clone(),
                shape: tensor.shape.clone(),
                data_size: tensor.data_size,
                rewrite: Rewrite::Copy,
            };
            let Some((layer, kind)) = projection(&tensor.name) else {
                return out;
            };
            let keep = selections.get(&layer).unwrap();
            match kind {
                Projection::Gate | Projection::Up => {
                    let k = tensor.shape[1] as usize;
                    let row_bytes = (k / GROUP) * MQ4_GROUP_BYTES;
                    out.shape[0] = (keep.len() * GROUP) as u32;
                    out.data_size = keep.len() * GROUP * row_bytes;
                    out.rewrite = Rewrite::OutputRows { layer };
                }
                Projection::Down => {
                    let m = tensor.shape[0] as usize;
                    out.shape[1] = (keep.len() * GROUP) as u32;
                    out.data_size = m * keep.len() * MQ4_GROUP_BYTES;
                    out.rewrite = Rewrite::InputGroups { layer };
                }
            }
            out
        })
        .collect()
}

fn rewrite_metadata(metadata: &str, keep_groups: usize, ranking: &str) -> String {
    let mut root: Value = serde_json::from_str(metadata).expect("valid metadata JSON");
    let width = (keep_groups * GROUP) as u64;
    let text_config = root
        .pointer_mut("/config/text_config")
        .and_then(Value::as_object_mut)
        .expect("Qwen text_config metadata");
    text_config.insert("intermediate_size".to_string(), Value::from(width));
    root.as_object_mut().unwrap().insert(
        "hipfire_ffn_group_prune".to_string(),
        json!({
            "group_size": GROUP,
            "keep_groups": keep_groups,
            "intermediate_size": width,
            "ranking": ranking,
            "experimental": true,
        }),
    );
    serde_json::to_string(&root).unwrap()
}

fn build_index(tensors: &[OutputTensor]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
        let name = tensor.source.name.as_bytes();
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.push(tensor.source.quant_type);
        bytes.push(tensor.shape.len() as u8);
        for &dim in &tensor.shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.source.group_size.to_le_bytes());
        bytes.extend_from_slice(&(tensor.data_size as u64).to_le_bytes());
    }
    bytes
}

fn write_output(
    hfq: &HfqFile,
    path: &Path,
    metadata: &str,
    tensors: &[OutputTensor],
    selections: &BTreeMap<usize, Vec<usize>>,
) -> std::io::Result<()> {
    let index = build_index(tensors);
    let metadata = metadata.as_bytes();
    let header_size = 32u64;
    let metadata_offset = header_size;
    let index_offset = metadata_offset + metadata.len() as u64;
    let data_start = index_offset + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    out.write_all(HFQ_MAGIC)?;
    out.write_all(&HFQ_VERSION.to_le_bytes())?;
    out.write_all(&hfq.arch_id.to_le_bytes())?;
    out.write_all(&(tensors.len() as u32).to_le_bytes())?;
    out.write_all(&metadata_offset.to_le_bytes())?;
    out.write_all(&data_offset.to_le_bytes())?;
    out.write_all(metadata)?;
    out.write_all(&index)?;
    out.write_all(&vec![0u8; (data_offset - data_start) as usize])?;

    for (index, tensor) in tensors.iter().enumerate() {
        let data = hfq.tensor_data(&tensor.source.name).unwrap().1;
        match tensor.rewrite {
            Rewrite::Copy => out.write_all(data)?,
            Rewrite::OutputRows { layer } => {
                let keep = &selections[&layer];
                let k = tensor.source.shape[1] as usize;
                let row_bytes = (k / GROUP) * MQ4_GROUP_BYTES;
                for &group in keep {
                    let start = group * GROUP * row_bytes;
                    out.write_all(&data[start..start + GROUP * row_bytes])?;
                }
            }
            Rewrite::InputGroups { layer } => {
                let keep = &selections[&layer];
                let source_groups = tensor.source.shape[1] as usize / GROUP;
                let row_bytes = source_groups * MQ4_GROUP_BYTES;
                for row in 0..tensor.source.shape[0] as usize {
                    let row_start = row * row_bytes;
                    for &group in keep {
                        let start = row_start + group * MQ4_GROUP_BYTES;
                        out.write_all(&data[start..start + MQ4_GROUP_BYTES])?;
                    }
                }
            }
        }
        if index % 64 == 0 || index + 1 == tensors.len() {
            eprintln!(
                "write {}/{}: {}",
                index + 1,
                tensors.len(),
                tensor.source.name
            );
        }
    }
    out.flush()
}

fn main() {
    let args = parse_args();
    let hfq = HfqFile::open_at_offset(&args.input, 0).expect("open input HFQ");
    let activation = args
        .activation_energy
        .as_deref()
        .map(load_activation_energy);
    if let Some(calibration) = activation.as_ref() {
        assert_eq!(
            calibration.model_bytes,
            std::fs::metadata(&args.input)
                .expect("stat input HFQ")
                .len(),
            "activation calibration model size mismatch"
        );
        assert_eq!(
            calibration.model_sha256,
            sha256_file(&args.input),
            "activation calibration model SHA-256 mismatch"
        );
        assert_eq!(
            calibration.metadata_fingerprint,
            format!("fnv1a64:{:016x}", fnv1a64(hfq.metadata_json.as_bytes())),
            "activation calibration model metadata mismatch"
        );
        let model_hidden_dim = hfq
            .tensors()
            .iter()
            .find_map(|tensor| {
                projection(&tensor.name)
                    .filter(|(_, kind)| *kind == Projection::Gate)
                    .map(|_| tensor.shape[0] as usize)
            })
            .expect("model gate projection");
        assert_eq!(
            calibration.hidden_dim, model_hidden_dim,
            "activation calibration hidden_dim mismatch"
        );
    }
    let ranking = if activation.is_some() {
        "activation_swiglu_energy_times_down_weight_energy"
    } else {
        "weight_energy_geomean_gate_up_down"
    };
    let selections = collect_selections(
        &hfq,
        args.keep_groups,
        activation.as_ref().map(|calibration| &calibration.energy),
    );
    let tensors = output_tensors(&hfq, &selections);
    let input_bytes: usize = hfq.tensors().iter().map(|x| x.data_size).sum();
    let output_bytes: usize = tensors.iter().map(|x| x.data_size).sum();
    eprintln!(
        "layers={} input_tensor_gib={:.3} output_tensor_gib={:.3} saved_gib={:.3}",
        selections.len(),
        input_bytes as f64 / 2f64.powi(30),
        output_bytes as f64 / 2f64.powi(30),
        (input_bytes - output_bytes) as f64 / 2f64.powi(30),
    );
    if args.dry_run {
        return;
    }
    let metadata = rewrite_metadata(&hfq.metadata_json, args.keep_groups, ranking);
    write_output(
        &hfq,
        args.output.as_deref().unwrap(),
        &metadata,
        &tensors,
        &selections,
    )
    .expect("write pruned HFQ");
}
