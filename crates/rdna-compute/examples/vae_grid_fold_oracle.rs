// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle for the VAE launch-grid fold (`vae_im2col_f16_lds`,
//! `vae_transpose_f32_banded`, `vae_transpose_cast_f16`).
//!
//! The old kernels took a 2-D/3-D tile grid whose `gridDim.y` is
//! `ceil(rows/8)` / `ceil(m/32)`; raw HIP on gfx1201 rejects `gridDim.y >=
//! 65536`, i.e. a tall im2col band of >524,280 rows or a tiled transpose over
//! m > 2,097,120 pixels. The new kernels decode the same tiles from a folded
//! `blockIdx.x`.
//!
//! 1. **old-vs-new, small shapes** — the verbatim base-commit kernels
//!    (`vae_grid_fold_oracle/*_old2d.hip`, entry names suffixed `_old2d`) are
//!    launched with their original 2-D/3-D grids next to the current launchers;
//!    outputs must be byte-identical (partial tiles in every dimension, a
//!    strided/offset banded destination, c_in with 1 and 3 channel tiles).
//! 2. **new-vs-independent, past 65535 row tiles** — the old kernels cannot
//!    run there on gfx1201, so the oracle is a different path: the 1-D
//!    channel-fastest im2col (`map = "c"`) and a host transpose (+ the GPU
//!    `cast_f32_to_f16` for the f16 transpose).
//!
//! Needs `HIPFIRE_VAE_TRANSPOSE=tiled` so `vae_transpose_f32_banded` takes the
//! tiled kernel (the default is the 1-D naive scatter). Take
//! `scripts/gpu-lock.sh` first.
//!
//! Run: `HIPFIRE_VAE_TRANSPOSE=tiled cargo run --release --features lab
//!       --example vae_grid_fold_oracle -p rdna-compute`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const OLD_LAYOUT: &str = include_str!("vae_grid_fold_oracle/vae_layout_old2d.hip");
const OLD_IM2COL: &str = include_str!("vae_grid_fold_oracle/vae_im2col_old2d.hip");

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

fn bytes_equal(gpu: &Gpu, a: &GpuTensor, b: &GpuTensor) -> usize {
    gpu.hip.device_synchronize().expect("sync");
    let x = gpu.download_raw_bytes(a).expect("download a");
    let y = gpu.download_raw_bytes(b).expect("download b");
    assert_eq!(x.len(), y.len());
    x.iter().zip(&y).filter(|(p, q)| p != q).count()
}

fn report(failures: &mut usize, what: &str, diff: usize) {
    let ok = diff == 0;
    println!(
        "{what}: {diff} bytes differ  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    *failures += usize::from(!ok);
}

fn old_entries(gpu: &mut Gpu) {
    let layout = OLD_LAYOUT
        .replace(
            "void vae_transpose_f32_banded(",
            "void vae_transpose_f32_banded_old2d(",
        )
        .replace(
            "void vae_transpose_cast_f16(",
            "void vae_transpose_cast_f16_old2d(",
        );
    let im2col = OLD_IM2COL.replace(
        "void vae_im2col_f16_lds(",
        "void vae_im2col_f16_lds_old2d(",
    );
    for entry in [
        "vae_transpose_f32_banded_old2d",
        "vae_transpose_cast_f16_old2d",
    ] {
        gpu.ensure_kernel_public("vae_layout_old2d", &layout, entry)
            .expect("compile old layout");
    }
    gpu.ensure_kernel_public("vae_im2col_old2d", &im2col, "vae_im2col_f16_lds_old2d")
        .expect("compile old im2col");
}

/// Old (x, y) tile grid of the 32x32 transposes of a `[m][n]` source.
fn old_tile_grid(m: usize, n: usize) -> [u32; 3] {
    [((n + 31) / 32) as u32, ((m + 31) / 32) as u32, 1]
}

fn host_transpose(src: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            dst[j * m + i] = src[i * n + j];
        }
    }
    dst
}

fn main() {
    assert_eq!(
        std::env::var("HIPFIRE_VAE_TRANSPOSE").as_deref(),
        Ok("tiled"),
        "run with HIPFIRE_VAE_TRANSPOSE=tiled so vae_transpose_f32_banded launches the tiled kernel"
    );
    let mut gpu = Gpu::init().expect("GPU init failed");
    println!("arch={}", gpu.arch);
    old_entries(&mut gpu);
    let mut failures = 0usize;

    // ── 1a: banded tiled transpose, old 2-D grid vs folded ───────────────
    for (m, n, stride, off) in [
        (1037usize, 77usize, 1037usize + 13, 5usize),
        (64, 33, 64, 0),
        (33, 130, 40, 7),
    ] {
        let src = gpu
            .upload_f32(&fill(m * n, 0xa11ce ^ (m * n) as u64), &[m, n])
            .expect("upload src");
        let new_dst = gpu.zeros(&[n * stride], DType::F32).expect("zeros new");
        let old_dst = gpu.zeros(&[n * stride], DType::F32).expect("zeros old");
        gpu.vae_transpose_f32_banded(&src, &new_dst, m, n, stride, off)
            .expect("new banded transpose");
        let mut kb = KernargBlob::new();
        kb.push_ptr(src.buf.as_ptr() as *const c_void);
        kb.push_ptr(old_dst.buf.as_ptr() as *const c_void);
        kb.push_i32(m as i32);
        kb.push_i32(n as i32);
        kb.push_u64(stride as u64);
        kb.push_u64(off as u64);
        gpu.launch_kernel_blob(
            "vae_transpose_f32_banded_old2d",
            old_tile_grid(m, n),
            [32, 8, 1],
            0,
            kb.as_mut_slice(),
        )
        .expect("old banded transpose");
        let diff = bytes_equal(&gpu, &new_dst, &old_dst);
        report(
            &mut failures,
            &format!("banded tiled transpose old2d-vs-fold m={m} n={n} stride={stride} off={off}"),
            diff,
        );
        for t in [src, new_dst, old_dst] {
            gpu.free_tensor(t).expect("free");
        }
    }

    // ── 1b: transpose+cast f16, both orientations ────────────────────────
    for (m, n) in [(1037usize, 77usize), (45, 1301), (32, 32)] {
        let src = gpu
            .upload_f32(&fill(m * n, 0xca57 ^ (m * n) as u64), &[m, n])
            .expect("upload src");
        let new_dst = gpu.zeros(&[n, m], DType::F16).expect("zeros new");
        let old_dst = gpu.zeros(&[n, m], DType::F16).expect("zeros old");
        gpu.vae_transpose_cast_f16(&src, &new_dst, m, n)
            .expect("new transpose cast");
        let mut kb = KernargBlob::new();
        kb.push_ptr(src.buf.as_ptr() as *const c_void);
        kb.push_ptr(old_dst.buf.as_ptr() as *const c_void);
        kb.push_i32(m as i32);
        kb.push_i32(n as i32);
        gpu.launch_kernel_blob(
            "vae_transpose_cast_f16_old2d",
            old_tile_grid(m, n),
            [32, 8, 1],
            0,
            kb.as_mut_slice(),
        )
        .expect("old transpose cast");
        let diff = bytes_equal(&gpu, &new_dst, &old_dst);
        report(
            &mut failures,
            &format!("transpose_cast_f16 old2d-vs-fold m={m} n={n}"),
            diff,
        );
        for t in [src, new_dst, old_dst] {
            gpu.free_tensor(t).expect("free");
        }
    }

    // ── 1c: im2col lds, old 3-D grid vs folded (default tile c/8/16) ─────
    // (c_in, h, w, y0, rows): c_in 64 -> c_tile 64 (1 channel tile), 96 ->
    // c_tile 32 (3 channel tiles); partial tiles in x and rows.
    for (c_in, h, w, y0, rows) in [
        (64usize, 21usize, 37usize, 3usize, 13usize),
        (96, 18, 33, 0, 18),
        (16, 40, 5, 9, 31),
    ] {
        let k = c_in * 9;
        let x = gpu
            .upload_f32(&fill(c_in * h * w, 0x1a2c ^ (c_in * h * w) as u64), &[c_in * h * w])
            .expect("upload x");
        let new_out = gpu.zeros(&[rows * w, k], DType::F16).expect("zeros new");
        let old_out = gpu.zeros(&[rows * w, k], DType::F16).expect("zeros old");
        gpu.vae_im2col_f16_variant(&x, &new_out, c_in, h, w, y0, rows, "lds")
            .expect("new im2col");
        let c_tile = [64usize, 32, 16, 8, 4, 2]
            .into_iter()
            .find(|d| c_in % d == 0)
            .unwrap_or(1);
        let (th, tw) = (8usize, 16usize);
        let mut kb = KernargBlob::new();
        kb.push_ptr(x.buf.as_ptr() as *const c_void);
        kb.push_ptr(old_out.buf.as_ptr() as *const c_void);
        for v in [c_in, h, w, y0, rows, c_tile, th, tw] {
            kb.push_i32(v as i32);
        }
        gpu.launch_kernel_blob(
            "vae_im2col_f16_lds_old2d",
            [
                w.div_ceil(tw) as u32,
                rows.div_ceil(th) as u32,
                c_in.div_ceil(c_tile) as u32,
            ],
            [256, 1, 1],
            (c_tile * (th + 2) * (tw + 2) * 2) as u32,
            kb.as_mut_slice(),
        )
        .expect("old im2col");
        let diff = bytes_equal(&gpu, &new_out, &old_out);
        report(
            &mut failures,
            &format!("im2col lds old3d-vs-fold c_in={c_in} h={h} w={w} y0={y0} rows={rows}"),
            diff,
        );
        for t in [x, new_out, old_out] {
            gpu.free_tensor(t).expect("free");
        }
    }

    // ── 2a: tiled transpose past 65535 row tiles vs host transpose ───────
    // m = 2,097,153 -> 65,537 row tiles: the old grid.y.
    {
        let (m, n) = (2_097_153usize, 5usize);
        let host = fill(m * n, 0xb16);
        let src = gpu.upload_f32(&host, &[m, n]).expect("upload src");
        let dst = gpu.zeros(&[n * m], DType::F32).expect("zeros dst");
        gpu.vae_transpose_f32_banded(&src, &dst, m, n, m, 0)
            .expect("banded transpose m past 65535 tiles");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(&dst).expect("download dst");
        let want = host_transpose(&host, m, n);
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        report(
            &mut failures,
            &format!("banded tiled transpose m={m} n={n} ({} row tiles) vs host", m.div_ceil(32)),
            diff,
        );
        for t in [src, dst] {
            gpu.free_tensor(t).expect("free");
        }
    }

    // ── 2b: transpose+cast past 65535 tiles on EITHER axis ───────────────
    for (m, n) in [(2_097_153usize, 5usize), (5, 2_097_153)] {
        let host = fill(m * n, 0xca5 ^ m as u64);
        let src = gpu.upload_f32(&host, &[m, n]).expect("upload src");
        let dst = gpu.zeros(&[n, m], DType::F16).expect("zeros dst");
        gpu.vae_transpose_cast_f16(&src, &dst, m, n)
            .expect("transpose cast past 65535 tiles");
        // Oracle: host transpose, then the plain 1-D f32->f16 cast.
        let want32 = gpu
            .upload_f32(&host_transpose(&host, m, n), &[n, m])
            .expect("upload want");
        let want16 = gpu.zeros(&[n, m], DType::F16).expect("zeros want16");
        gpu.cast_f32_to_f16(&want32, &want16).expect("cast want");
        let diff = bytes_equal(&gpu, &dst, &want16);
        report(
            &mut failures,
            &format!("transpose_cast_f16 m={m} n={n} vs host transpose + cast_f32_to_f16"),
            diff,
        );
        for t in [src, dst, want32, want16] {
            gpu.free_tensor(t).expect("free");
        }
    }

    // ── 2c: im2col lds past 65535 row tiles vs the 1-D `c` gather ────────
    // rows = 524,296 -> ceil(rows/8) = 65,537 row tiles (old grid.y).
    {
        let (c_in, h, w) = (16usize, 524_296usize, 2usize);
        let k = c_in * 9;
        let x = gpu
            .upload_f32(&fill(c_in * h * w, 0x7a11), &[c_in * h * w])
            .expect("upload x");
        let lds = gpu.zeros(&[h * w, k], DType::F16).expect("zeros lds");
        let cfast = gpu.zeros(&[h * w, k], DType::F16).expect("zeros c");
        gpu.vae_im2col_f16_variant(&x, &lds, c_in, h, w, 0, h, "lds")
            .expect("lds im2col past 65535 row tiles");
        gpu.vae_im2col_f16_variant(&x, &cfast, c_in, h, w, 0, h, "c")
            .expect("c im2col");
        let diff = bytes_equal(&gpu, &lds, &cfast);
        report(
            &mut failures,
            &format!("im2col lds c_in={c_in} h={h} w={w} ({} row tiles) vs map c", h.div_ceil(8)),
            diff,
        );
        for t in [x, lds, cfast] {
            gpu.free_tensor(t).expect("free");
        }
    }

    if failures > 0 {
        eprintln!("{failures} FAILED");
        std::process::exit(1);
    }
    println!("all vae grid-fold oracles PASS");
}
