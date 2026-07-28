// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Smoke test for the explicit VMM arena ownership and growth path.

use hip_bridge::{HipRuntime, VmmArena};

const DEFAULT_CHUNK_BYTES: usize = 2 << 20;

fn bytes_from_env(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CHUNK_BYTES)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0);
    let access_device = std::env::var("HIPFIRE_VMM_ACCESS_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(device);
    let first_bytes = bytes_from_env("HIPFIRE_VMM_FIRST_BYTES");
    let second_bytes = bytes_from_env("HIPFIRE_VMM_SECOND_BYTES");

    let hip = HipRuntime::load()?;
    let mut arena = VmmArena::reserve(&hip, device, first_bytes + second_bytes)?;
    println!(
        "reserved={} granularity={} owner={}",
        arena.reserved_bytes(),
        arena.granularity(),
        arena.owner_device()
    );

    arena.map_next(&hip, first_bytes, &[access_device])?;
    let first = arena.buffer(first_bytes)?;
    let first_pattern: Vec<u8> = (0..first_bytes).map(|i| (i % 251) as u8).collect();
    hip.memcpy_htod(&first, &first_pattern)?;

    arena.map_next(&hip, second_bytes, &[access_device])?;
    let full = arena.buffer(first_bytes + second_bytes)?;
    let second_pattern: Vec<u8> = (0..second_bytes)
        .map(|i| ((i * 17 + 3) % 251) as u8)
        .collect();
    hip.memcpy_htod_offset(&full, first_bytes, &second_pattern)?;

    let mut readback = vec![0u8; first_bytes + second_bytes];
    hip.memcpy_dtoh(&mut readback, &full)?;
    assert_eq!(&readback[..first_bytes], first_pattern.as_slice());
    assert_eq!(&readback[first_bytes..], second_pattern.as_slice());
    if access_device != device {
        hip.set_device(access_device)?;
        let mut peer_readback = vec![0u8; first_bytes + second_bytes];
        hip.memcpy_dtoh(&mut peer_readback, &full)?;
        assert_eq!(peer_readback, readback);
        println!("peer device {access_device} full-prefix read VERIFIED");
    }
    println!(
        "mapped={} two-chunk round-trip VERIFIED",
        arena.mapped_bytes()
    );

    arena.release(&hip)?;
    println!("vmm_arena_smoke: PASS");
    Ok(())
}
