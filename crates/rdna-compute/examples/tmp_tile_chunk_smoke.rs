// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! THROWAWAY smoke: `gemm_f16_x_f16_wmma` (16-row tiles) and `gemm_hfq4g128`
//! (8-row tiles) past 65536 grid.y tiles. The big launch output (poisoned
//! first) must byte-match independent small launches of the head and tail rows.

use rdna_compute::{DType, Gpu};

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

fn f16_bits(v: f32) -> [u8; 2] {
    half_bits(v).to_le_bytes()
}

// Small-range exact conversion: values are multiples of 1/16 in [-2, 2].
fn half_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = ((b >> 13) & 0x3ff) as u16;
    sign | ((exp as u16) << 10) | man
}

fn check(
    gpu: &mut Gpu,
    name: &str,
    rows: usize,
    m: usize,
    x_row_bytes: usize,
    x: &[u8],
    run: &mut dyn FnMut(&mut Gpu, &rdna_compute::GpuTensor, &rdna_compute::GpuTensor, usize),
) -> bool {
    let xt = gpu.upload_raw(x, &[x.len()]).expect("x");
    let poison = vec![0xffu8; rows * m * 4];
    let y = gpu.upload_raw(&poison, &[poison.len()]).expect("y");
    run(gpu, &xt, &y, rows);
    gpu.hip.device_synchronize().expect("sync");
    let big = gpu.download_raw_bytes(&y).expect("dl");
    let mut ok = true;
    for (r0, n) in [(0usize, 40usize), (rows - 4008, 4008)] {
        let xs = &x[r0 * x_row_bytes..(r0 + n) * x_row_bytes];
        let xst = gpu.upload_raw(xs, &[xs.len()]).expect("xs");
        let ys = gpu.zeros(&[n * m], DType::F32).expect("ys");
        run(gpu, &xst, &ys, n);
        gpu.hip.device_synchronize().expect("sync");
        let small = gpu.download_raw_bytes(&ys).expect("dl");
        let same = small[..] == big[r0 * m * 4..(r0 + n) * m * 4];
        println!("{} {name} rows {r0}..{} of {rows} byte-match small launch", if same { "PASS" } else { "FAIL" }, r0 + n);
        ok &= same;
    }
    let unwritten = big.chunks_exact(4).filter(|w| w == &[0xff; 4]).count();
    println!("{} {name} poisoned words remaining = {unwritten}", if unwritten == 0 { "PASS" } else { "FAIL" });
    ok && unwritten == 0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);
    let mut seed = 0x5eed_u64;
    let mut ok = true;

    // Tile16 WMMA: 1048576 + 40 rows => 65539 grid.y tiles unchunked.
    {
        let (m, k, rows) = (16usize, 64usize, (1usize << 20) + 40);
        let a: Vec<u8> = (0..m * k)
            .flat_map(|_| f16_bits((lcg(&mut seed) % 65) as f32 / 16.0 - 2.0))
            .collect();
        let x: Vec<u8> = (0..rows * k)
            .flat_map(|_| f16_bits((lcg(&mut seed) % 65) as f32 / 16.0 - 2.0))
            .collect();
        let at = gpu.upload_raw(&a, &[a.len()]).expect("a");
        ok &= check(&mut gpu, "gemm_f16_x_f16_wmma", rows, m, k * 2, &x, &mut |g, xt, yt, n| {
            g.gemm_f16_x_f16_wmma(&at, xt, yt, m, k, n).expect("wmma")
        });
    }
    // Tile8 HFQ4-G128: 524288 + 40 rows => 65541 grid.y tiles unchunked.
    {
        let (m, k, rows) = (8usize, 128usize, (1usize << 19) + 40);
        let mut a = Vec::with_capacity(m * 72);
        for _ in 0..m {
            a.extend_from_slice(&0.125f32.to_le_bytes());
            a.extend_from_slice(&(-1.0f32).to_le_bytes());
            for _ in 0..64 {
                a.push(lcg(&mut seed) as u8);
            }
        }
        let x: Vec<u8> = (0..rows * k)
            .flat_map(|_| ((lcg(&mut seed) % 65) as f32 / 16.0 - 2.0).to_le_bytes())
            .collect();
        let at = gpu.upload_raw(&a, &[a.len()]).expect("a");
        ok &= check(&mut gpu, "gemm_hfq4g128", rows, m, k * 4, &x, &mut |g, xt, yt, n| {
            g.gemm_hfq4g128(&at, xt, yt, m, k, n).expect("hfq4g128")
        });
    }
    if !ok {
        std::process::exit(1);
    }
    println!("tmp_tile_chunk_smoke: all checks passed");
}
