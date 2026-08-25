// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stored-byte HFQ4G256 oracle for a real HFQ artifact.
//!
//! Reads one tensor exactly as the runtime sees it, validates its container
//! shape/byte contract, computes `W*x` by decoding those stored bytes on CPU,
//! and compares that result with the production GPU GEMV.

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

const QT_HFQ4G256: u8 = 6;
const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: hfq4_artifact_gemv_oracle <artifact.hf4> <tensor-name>");
        std::process::exit(2);
    });
    let name = args.next().unwrap_or_else(|| {
        eprintln!("missing tensor name");
        std::process::exit(2);
    });

    let hfq = HfqFile::open(Path::new(&path)).expect("open HFQ artifact");
    let (info, bytes) = hfq
        .tensor_data_vec(&name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    assert_eq!(info.quant_type, QT_HFQ4G256, "expected HFQ4G256 qt=6");
    assert_eq!(info.shape.len(), 2, "oracle requires a 2D matrix");
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    assert_eq!(k % GROUP, 0, "K must be divisible by {GROUP}");
    let row_bytes = (k / GROUP) * GROUP_BYTES;
    let expected_bytes = m * row_bytes;
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "container blob disagrees with shape [M={m}, K={k}]"
    );
    assert_eq!(info.group_size, GROUP as u32, "unexpected group size");

    let x: Vec<f32> = (0..k)
        .map(|col| {
            let z = ((col as u64).wrapping_mul(0x9e37_79b9) >> 8) as u32;
            (z as f32 / u32::MAX as f32 - 0.5) * 0.25
        })
        .collect();
    let mut cpu = vec![0.0f32; m];
    for (row, out) in cpu.iter_mut().enumerate() {
        let row_off = row * row_bytes;
        let mut acc = 0.0f32;
        for group in 0..k / GROUP {
            let off = row_off + group * GROUP_BYTES;
            let scale = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let zero = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            assert!(
                scale.is_finite() && zero.is_finite(),
                "non-finite header row={row} group={group}"
            );
            for lane in 0..GROUP {
                let packed = bytes[off + 8 + lane / 2];
                let q = if lane & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                acc += (scale * q as f32 + zero) * x[group * GROUP + lane];
            }
        }
        *out = acc;
    }

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let d_w = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload W");
    let d_x = gpu.upload_f32(&x, &[k]).expect("upload x");
    let d_y = gpu
        .zeros(&[m], rdna_compute::DType::F32)
        .expect("allocate y");
    gpu.gemv_hfq4g256(&d_w, &d_x, &d_y, m, k)
        .expect("HFQ4 GPU GEMV");
    let actual = gpu.download_f32(&d_y).expect("download y");

    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut worst_row = 0usize;
    for row in 0..m {
        let abs = (actual[row] - cpu[row]).abs();
        let rel = abs / cpu[row].abs().max(1.0e-6);
        if abs > worst_abs {
            worst_abs = abs;
            worst_rel = rel;
            worst_row = row;
        }
    }
    println!(
        "HFQ4 artifact oracle tensor={name} shape=[{m},{k}] bytes={} worst_row={worst_row} cpu={:.9e} gpu={:.9e} max_abs={worst_abs:.9e} rel_at_worst={worst_rel:.9e}",
        bytes.len(), cpu[worst_row], actual[worst_row]
    );
    assert!(
        worst_abs <= 2.0e-3 || worst_rel <= 2.0e-4,
        "stored-byte CPU/GPU GEMV mismatch"
    );
}
