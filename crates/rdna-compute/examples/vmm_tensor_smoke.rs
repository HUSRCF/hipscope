// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Verify that a VMM-backed GpuTensor grows and bypasses the hipFree pool path.

use rdna_compute::{DType, Gpu};

const CHUNK_BYTES: usize = 2 << 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let device = gpu.device_id;
    let mut tensor =
        unsafe { gpu.alloc_vmm_tensor(&[CHUNK_BYTES * 2], DType::Raw, CHUNK_BYTES, &[device])? };
    assert_eq!(gpu.vmm_allocation_count(), 1);
    assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(CHUNK_BYTES));
    assert_eq!(tensor.buf.size(), CHUNK_BYTES);

    let first: Vec<u8> = (0..CHUNK_BYTES).map(|i| (i % 251) as u8).collect();
    gpu.hip.memcpy_htod(&tensor.buf, &first)?;

    let mapped = gpu.grow_vmm_tensor(&mut tensor, CHUNK_BYTES, &[device])?;
    assert_eq!(mapped, CHUNK_BYTES * 2);
    assert_eq!(tensor.buf.size(), CHUNK_BYTES * 2);
    let second: Vec<u8> = (0..CHUNK_BYTES)
        .map(|i| ((i * 17 + 3) % 251) as u8)
        .collect();
    gpu.hip
        .memcpy_htod_offset(&tensor.buf, CHUNK_BYTES, &second)?;

    let mut readback = vec![0u8; CHUNK_BYTES * 2];
    gpu.hip.memcpy_dtoh(&mut readback, &tensor.buf)?;
    assert_eq!(&readback[..CHUNK_BYTES], first.as_slice());
    assert_eq!(&readback[CHUNK_BYTES..], second.as_slice());

    let alias = tensor.shallow_clone();
    assert!(gpu.free_tensor(alias).is_err());
    assert_eq!(gpu.vmm_allocation_count(), 1);

    gpu.free_tensor(tensor)?;
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_tensor_smoke: PASS");
    Ok(())
}
