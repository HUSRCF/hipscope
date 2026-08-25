// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare source-GGUF weights with the exact stored bytes of HFQ4 and MQ4.
//!
//! The MQ4 reconstruction is inverse-FWHT'd before comparison, so all three
//! arms are measured in the source weight domain.  With no explicit mappings,
//! the oracle checks the Qwen3.5 layer-0 DeltaNet projections most likely to
//! amplify quantization error.

use half::f16;
use hipfire_runtime::gguf::GgufFile;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use std::path::Path;

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;
const QT_HFQ4: u8 = 6;
const QT_MQ4_V2: u8 = 44;

const DEFAULT_MAPPINGS: &[(&str, &str)] = &[
    (
        "blk.0.attn_gate.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
    ),
    (
        "blk.0.attn_qkv.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
    ),
    (
        "blk.0.ssm_alpha.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ),
    (
        "blk.0.ssm_beta.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
    ),
    (
        "blk.0.ssm_out.weight",
        "model.layers.0.linear_attn.out_proj.weight",
    ),
];

#[derive(Debug, Clone, Copy)]
struct Metrics {
    n: usize,
    rmse: f64,
    nrmse: f64,
    mae: f64,
    max_abs: f32,
    cosine: f64,
    source_rms: f64,
}

fn metrics(reference: &[f32], candidate: &[f32]) -> Metrics {
    assert_eq!(reference.len(), candidate.len());
    let mut sum_sq_err = 0.0f64;
    let mut sum_abs_err = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut cand_sq = 0.0f64;
    for (&a, &b) in reference.iter().zip(candidate) {
        let af = a as f64;
        let bf = b as f64;
        let err = af - bf;
        sum_sq_err += err * err;
        sum_abs_err += err.abs();
        max_abs = max_abs.max((a - b).abs());
        dot += af * bf;
        ref_sq += af * af;
        cand_sq += bf * bf;
    }
    let n = reference.len();
    let rmse = (sum_sq_err / n as f64).sqrt();
    let source_rms = (ref_sq / n as f64).sqrt();
    Metrics {
        n,
        rmse,
        nrmse: rmse / source_rms.max(f64::MIN_POSITIVE),
        mae: sum_abs_err / n as f64,
        max_abs,
        cosine: dot / (ref_sq.sqrt() * cand_sq.sqrt()).max(f64::MIN_POSITIVE),
        source_rms,
    }
}

fn decode_hfq4(bytes: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(n % GROUP, 0);
    assert_eq!(bytes.len(), n / GROUP * GROUP_BYTES);
    let mut out = Vec::with_capacity(n);
    for block in bytes.chunks_exact(GROUP_BYTES) {
        let scale = f32::from_le_bytes(block[0..4].try_into().unwrap());
        let zero = f32::from_le_bytes(block[4..8].try_into().unwrap());
        for lane in 0..GROUP {
            let packed = block[8 + lane / 2];
            let q = if lane & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            out.push(scale * q as f32 + zero);
        }
    }
    out
}

fn inverse_fwht_group(group: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    for i in 0..GROUP {
        group[i] *= signs2[i];
    }
    let mut stride = 1;
    while stride < GROUP {
        for base in (0..GROUP).step_by(stride * 2) {
            for i in 0..stride {
                let a = group[base + i];
                let b = group[base + i + stride];
                group[base + i] = a + b;
                group[base + i + stride] = a - b;
            }
        }
        stride *= 2;
    }
    for i in 0..GROUP {
        group[i] *= 0.0625 * signs1[i];
    }
}

fn decode_mq4(bytes: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(n % GROUP, 0);
    assert_eq!(bytes.len(), n / GROUP * GROUP_BYTES);
    let mut out = Vec::with_capacity(n);
    for block in bytes.chunks_exact(GROUP_BYTES) {
        let scales = [
            f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32(),
            f16::from_bits(u16::from_le_bytes(block[4..6].try_into().unwrap())).to_f32(),
        ];
        let zeros = [
            f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32(),
            f16::from_bits(u16::from_le_bytes(block[6..8].try_into().unwrap())).to_f32(),
        ];
        for lane in 0..GROUP {
            let packed = block[8 + lane / 2];
            let q = if lane & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            let half = lane / 128;
            out.push(scales[half] * q as f32 + zeros[half]);
        }
    }
    let signs1 = llama::KvCache::gen_fwht_signs(42, GROUP);
    let signs2 = llama::KvCache::gen_fwht_signs(1042, GROUP);
    for group in out.chunks_exact_mut(GROUP) {
        inverse_fwht_group(group, &signs1, &signs2);
    }
    out
}

fn dequantize_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (b, block) in data.chunks_exact(176).enumerate() {
        let d = llama::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = llama::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = packed[i] & 63;
            mins[i] = packed[4 + i] & 63;
            scales[4 + i] = (packed[8 + i] & 15) | ((packed[i] >> 6) << 4);
            mins[4 + i] = (packed[8 + i] >> 4) | ((packed[4 + i] >> 6) << 4);
        }
        let qh = &block[16..48];
        let ql = &block[48..176];
        for group in 0..4 {
            for lane in 0..32 {
                let byte = ql[group * 32 + lane];
                let even = b * GROUP + group * 64 + lane;
                let odd = even + 32;
                if even < n {
                    let q = (byte & 15) | (((qh[lane] >> (2 * group)) & 1) << 4);
                    out[even] =
                        q as f32 * d * scales[group * 2] as f32 - dmin * mins[group * 2] as f32;
                }
                if odd < n {
                    let q = (byte >> 4) | (((qh[lane] >> (2 * group + 1)) & 1) << 4);
                    out[odd] = q as f32 * d * scales[group * 2 + 1] as f32
                        - dmin * mins[group * 2 + 1] as f32;
                }
            }
        }
    }
    out
}

fn source_values(gguf: &GgufFile, name: &str) -> (Vec<usize>, Vec<f32>) {
    let info = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("GGUF tensor not found: {name}"));
    let bytes = gguf.tensor_data(info);
    let values = match info.dtype {
        hipfire_runtime::gguf::GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        hipfire_runtime::gguf::GgmlType::Q4K => llama::dequantize_q4_k(bytes, info.numel()),
        hipfire_runtime::gguf::GgmlType::Q5K => dequantize_q5_k(bytes, info.numel()),
        hipfire_runtime::gguf::GgmlType::Q6K => llama::dequantize_q6_k(bytes, info.numel()),
        hipfire_runtime::gguf::GgmlType::Q8_0 => llama::dequantize_q8_0(bytes, info.numel()),
        other => panic!("unsupported source dtype {other:?} for {name}"),
    };
    (info.shape.clone(), values)
}

fn artifact_values(hfq: &HfqFile, name: &str, expected_qt: u8) -> (Vec<usize>, Vec<f32>) {
    let (info, bytes) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("artifact tensor not found: {name}"));
    assert_eq!(info.quant_type, expected_qt, "unexpected dtype for {name}");
    let n: usize = info.shape.iter().map(|&v| v as usize).product();
    let values = if expected_qt == QT_HFQ4 {
        decode_hfq4(&bytes, n)
    } else {
        decode_mq4(&bytes, n)
    };
    (info.shape.iter().map(|&v| v as usize).collect(), values)
}

fn print_metrics(label: &str, m: Metrics) {
    println!(
        "  {label:<4} n={} source_rms={:.8e} rmse={:.8e} nrmse={:.8e} mae={:.8e} max_abs={:.8e} cosine={:.10}",
        m.n, m.source_rms, m.rmse, m.nrmse, m.mae, m.max_abs, m.cosine
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!(
            "usage: gguf_hfq_mq4_value_oracle <source.gguf> <model.hf4> <model.mq4> [gguf-name=artifact-name ...]"
        );
        std::process::exit(2);
    }
    let gguf = GgufFile::open(Path::new(&args[0])).expect("open source GGUF");
    let hfq = HfqFile::open(Path::new(&args[1])).expect("open HFQ4 artifact");
    let mq4 = HfqFile::open(Path::new(&args[2])).expect("open MQ4 artifact");
    let owned;
    let mappings: Vec<(&str, &str)> = if args.len() == 3 {
        DEFAULT_MAPPINGS.to_vec()
    } else {
        owned = args[3..]
            .iter()
            .map(|arg| {
                arg.split_once('=')
                    .unwrap_or_else(|| panic!("mapping must be gguf-name=artifact-name: {arg}"))
            })
            .collect::<Vec<_>>();
        owned
    };

    for (source_name, artifact_name) in mappings {
        let (source_shape, source) = source_values(&gguf, source_name);
        let (hfq_shape, hfq_values) = artifact_values(&hfq, artifact_name, QT_HFQ4);
        let (mq4_shape, mq4_values) = artifact_values(&mq4, artifact_name, QT_MQ4_V2);
        assert_eq!(source.len(), hfq_values.len(), "HFQ element mismatch");
        assert_eq!(source.len(), mq4_values.len(), "MQ4 element mismatch");
        println!(
            "tensor source={source_name} artifact={artifact_name} source_shape={source_shape:?} artifact_shape={hfq_shape:?} mq4_shape={mq4_shape:?}"
        );
        print_metrics("HFQ4", metrics(&source, &hfq_values));
        print_metrics("MQ4", metrics(&source, &mq4_values));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_hfq4(values: &[u8], scale: f32, zero: f32) -> Vec<u8> {
        assert_eq!(values.len(), GROUP);
        let mut out = Vec::with_capacity(GROUP_BYTES);
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&zero.to_le_bytes());
        for pair in values.chunks_exact(2) {
            out.push(pair[0] | (pair[1] << 4));
        }
        out
    }

    #[test]
    fn hfq4_decode_uses_stored_scale_zero_and_nibbles() {
        let q: Vec<u8> = (0..GROUP).map(|i| (i % 16) as u8).collect();
        let decoded = decode_hfq4(&encode_hfq4(&q, 0.25, -1.0), GROUP);
        for i in 0..GROUP {
            assert_eq!(decoded[i], 0.25 * q[i] as f32 - 1.0);
        }
    }

    #[test]
    fn metrics_are_exact_for_identical_vectors() {
        let values = [1.0, -2.0, 3.5, 0.25];
        let got = metrics(&values, &values);
        assert_eq!(got.rmse, 0.0);
        assert_eq!(got.mae, 0.0);
        assert!((got.cosine - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn inverse_fwht_is_orthonormal() {
        let signs1 = llama::KvCache::gen_fwht_signs(42, GROUP);
        let signs2 = llama::KvCache::gen_fwht_signs(1042, GROUP);
        let original: Vec<f32> = (0..GROUP).map(|i| (i as f32 - 127.5) / 64.0).collect();
        let mut rotated = original.clone();
        for i in 0..GROUP {
            rotated[i] *= signs1[i];
        }
        let mut stride = 1;
        while stride < GROUP {
            for base in (0..GROUP).step_by(stride * 2) {
                for i in 0..stride {
                    let a = rotated[base + i];
                    let b = rotated[base + i + stride];
                    rotated[base + i] = a + b;
                    rotated[base + i + stride] = a - b;
                }
            }
            stride *= 2;
        }
        for i in 0..GROUP {
            rotated[i] *= 0.0625 * signs2[i];
        }
        inverse_fwht_group(&mut rotated, &signs1, &signs2);
        assert!(metrics(&original, &rotated).max_abs < 1.0e-5);
    }
}
