//! GPU smoke for the `norm.rs` token-row launchers that used to put the batch
//! row in `gridDim.y` (raw HIP rejects `gridDim.y >= 65536` on gfx1201):
//!
//!   * `rope_batched_f32`                      (ROPE_BATCHED)
//!   * `rope_partial_interleaved_f32_batched`  (run under both values of the
//!                                              `rope_interleaved_legacy` flag:
//!                                              halfsplit default and legacy
//!                                              interleaved kernel)
//!   * `rope_mrope_halfsplit_f32_batched`
//!   * `rope_interleaved_f32_batched`
//!   * `rope_partial_halved_f32_batched`
//!   * `repeat_interleave_qk_f32_batched`
//!   * `deinterleave_f32_batched`
//!   * `deinterleave_q_rmsnorm_f32_batched`
//!   * `gated_norm_f32_batched`
//!   * `deepseek4_silu_mul_clamp_f32_batched` (in place, out == gate)
//!
//! Each launcher now issues <= 65535-row launches over real row sub-views, with
//! the chunk row count as the kernel's batch argument. Every case compares the
//! launcher's whole output buffer BYTE-FOR-BYTE against an oracle that runs
//! the *unchanged* kernel sources (compiled here under an `old_` name,
//! independent of the launcher) with the original geometry: one launch with
//! `grid.y = rows` when rows <= 65535 (so small rows prove byte identity with
//! the pre-change launch), and <= 65535-row slices with shifted pointers past
//! that (so rows > 65535 prove every row was processed with the old math).
//! Row counts default to 37, 1000, 65535, 65536, 70000, 131077 (two chunk
//! boundaries); odd values catch off-by-one row offsets between chunks.
//!
//!     cargo run --release -p rdna-compute --features lab,deltanet \
//!         --example norm_rows_grid_smoke [-- --rows 37,1000,65536,70000,131077]
//!
//! Exit status 1 on any mismatch.

use hip_bridge::KernargBlob;
use std::sync::Arc;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const SLICE: usize = 65_535;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform f32 in [-scale, scale).
    fn f32(&mut self, scale: f32) -> f32 {
        let u = ((self.next() >> 40) as f32) / ((1u64 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }
}

fn randv(r: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| r.f32(scale)).collect()
}

/// i32 positions stored as f32 bits (the scratch convention: F32 dtype is
/// cosmetic for position buffers).
fn pos_bits(vals: &[i32]) -> Vec<f32> {
    vals.iter().map(|v| f32::from_bits(*v as u32)).collect()
}

fn shift(t: &GpuTensor, bytes: usize) -> *const c_void {
    t.buf.as_ptr().wrapping_byte_add(bytes) as *const c_void
}

fn launch(
    gpu: &Gpu,
    f: &str,
    grid: [u32; 3],
    block: u32,
    shared: u32,
    a: &mut KernargBlob,
) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], shared, a.as_mut_slice())?;
    Ok(())
}

/// `(first_row, rows)` slices of at most SLICE rows; one slice when it fits.
fn slices(rows: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(SLICE)
        .map(|o| (o, SLICE.min(rows - o)))
        .collect()
}

fn old_module(gpu: &mut Gpu, src: &str, entry: &str) -> Res<String> {
    let old = format!("old_{entry}");
    gpu.ensure_kernel_public(&old, &src.replace(entry, &old), &old)?;
    Ok(old)
}

fn diff(name: &str, rows: usize, got: &[f32], want: &[f32]) -> usize {
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let first = got
        .iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    println!(
        "  {name:<26} rows={rows:<7} elems={:<10} mismatched={bad}{}",
        got.len(),
        first.map(|i| format!(" first@{i}")).unwrap_or_default()
    );
    bad
}

fn free(gpu: &mut Gpu, ts: Vec<GpuTensor>) -> Res<()> {
    for t in ts {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

// ---------------------------------------------------------------- RoPE ----

fn case_rope_batched(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NHQ: usize = 3;
    const NHK: usize = 2;
    const HD: usize = 64;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/rope_batched.hip"),
        "rope_batched_f32",
    )?;
    let q = randv(r, rows * NHQ * HD, 2.0);
    let k = randv(r, rows * NHK * HD, 2.0);
    let pos: Vec<i32> = (0..rows).map(|i| ((i * 7919) % 100_003) as i32).collect();
    let fb = 10_000.0f32;
    let (nq, nk) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let (oq, ok) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let p = gpu.upload_f32(&pos_bits(&pos), &[rows])?;
    gpu.rope_batched_f32(&nq, &nk, &p, NHQ, NHK, HD, fb, rows)?;
    let half = (HD / 2) as u32;
    let block = 256u32.min(half);
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&oq, o * NHQ * HD * 4));
        b.push_ptr(shift(&ok, o * NHK * HD * 4));
        b.push_ptr(shift(&p, o * 4));
        b.push_i32(NHQ as i32);
        b.push_i32(NHK as i32);
        b.push_i32(HD as i32);
        b.push_f32(fb);
        b.push_i32(n as i32);
        launch(gpu, &old, [half.div_ceil(block), n as u32, 1], block, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("rope_batched q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("rope_batched k", rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    free(gpu, vec![nq, nk, oq, ok, p])?;
    Ok(bad)
}

fn case_rope_partial_batched(gpu: &mut Gpu, rows: usize, legacy: bool, r: &mut Rng) -> Res<usize> {
    const NHQ: usize = 4;
    const NHK: usize = 2;
    const HD: usize = 64;
    const NROT: usize = 32;
    let mut flags = (*gpu.flags).clone();
    flags.rope_interleaved_legacy = legacy;
    gpu.flags = Arc::new(flags);
    let (src, entry) = if legacy {
        (
            include_str!("../../../kernels/src/rope_partial_interleaved_batched.hip"),
            "rope_partial_interleaved_batched_f32",
        )
    } else {
        (
            include_str!("../../../kernels/src/rope_partial_halfsplit_batched.hip"),
            "rope_partial_halfsplit_batched_f32",
        )
    };
    let old = old_module(gpu, src, entry)?;
    let q = randv(r, rows * NHQ * HD, 2.0);
    let k = randv(r, rows * NHK * HD, 2.0);
    let pos: Vec<i32> = (0..rows).map(|i| ((i * 104_729) % 200_003) as i32).collect();
    let (fb, po) = (1_000_000.0f32, 5i32);
    let (nq, nk) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let (oq, ok) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let p = gpu.upload_f32(&pos_bits(&pos), &[rows])?;
    gpu.rope_partial_interleaved_f32_batched(&nq, &nk, &p, NHQ, NHK, HD, NROT, fb, rows, po)?;
    let n_pairs = (NROT / 2) as u32;
    let (grid_x, block, shared) = if legacy {
        let b = 32u32.min(n_pairs);
        (n_pairs.div_ceil(b), b, 0)
    } else {
        (1, 256, 2 * n_pairs * 4)
    };
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&oq, o * NHQ * HD * 4));
        b.push_ptr(shift(&ok, o * NHK * HD * 4));
        b.push_ptr(shift(&p, o * 4));
        b.push_i32(NHQ as i32);
        b.push_i32(NHK as i32);
        b.push_i32(HD as i32);
        b.push_i32(NROT as i32);
        b.push_f32(fb);
        b.push_i32(n as i32);
        b.push_i32(po);
        launch(gpu, &old, [grid_x, n as u32, 1], block, shared, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let tag = if legacy { "rope_partial(legacy)" } else { "rope_partial(halfsplit)" };
    let mut bad = diff(&format!("{tag} q"), rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff(&format!("{tag} k"), rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    free(gpu, vec![nq, nk, oq, ok, p])?;
    Ok(bad)
}

fn case_mrope_batched(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NHQ: usize = 4;
    const NHK: usize = 2;
    const HD: usize = 64;
    const NROT: usize = 32;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/rope_mrope_halfsplit_batched.hip"),
        "rope_mrope_halfsplit_batched_f32",
    )?;
    let q = randv(r, rows * NHQ * HD, 2.0);
    let k = randv(r, rows * NHK * HD, 2.0);
    let pos: Vec<i32> = (0..rows * 3)
        .map(|i| ((i * 6151 + i / 3) % 90_001) as i32)
        .collect();
    let (fb, po, section) = (10_000.0f32, 3i32, [4usize, 3, 3]);
    let (nq, nk) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let (oq, ok) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let p = gpu.upload_f32(&pos_bits(&pos), &[rows * 3])?;
    gpu.rope_mrope_halfsplit_f32_batched(
        &nq, &nk, &p.buf, NHQ, NHK, HD, NROT, fb, rows, po, section,
    )?;
    let half = (NROT / 2) as u32;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&oq, o * NHQ * HD * 4));
        b.push_ptr(shift(&ok, o * NHK * HD * 4));
        b.push_ptr(shift(&p, o * 3 * 4));
        b.push_i32(NHQ as i32);
        b.push_i32(NHK as i32);
        b.push_i32(HD as i32);
        b.push_i32(NROT as i32);
        b.push_f32(fb);
        b.push_i32(n as i32);
        b.push_i32(po);
        b.push_i32(section[1] as i32);
        b.push_i32(section[2] as i32);
        launch(gpu, &old, [half.div_ceil(64), n as u32, 1], 64, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("mrope q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("mrope k", rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    free(gpu, vec![nq, nk, oq, ok, p])?;
    Ok(bad)
}

fn case_rope_interleaved_batched(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NHQ: usize = 4;
    const NHK: usize = 2;
    const HD: usize = 64;
    const NROT: usize = 64;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/rope_partial_interleaved_batched.hip"),
        "rope_partial_interleaved_batched_f32",
    )?;
    let q = randv(r, rows * NHQ * HD, 2.0);
    let k = randv(r, rows * NHK * HD, 2.0);
    let pos: Vec<i32> = (0..rows).map(|i| ((i * 31) % 70_001) as i32).collect();
    let fb = 10_000.0f32;
    let (nq, nk) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let (oq, ok) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let p = gpu.upload_f32(&pos_bits(&pos), &[rows])?;
    gpu.rope_interleaved_f32_batched(&nq, &nk, &p, NHQ, NHK, HD, NROT, fb, rows)?;
    let n_pairs = (NROT / 2) as u32;
    let block = 32u32.min(n_pairs);
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&oq, o * NHQ * HD * 4));
        b.push_ptr(shift(&ok, o * NHK * HD * 4));
        b.push_ptr(shift(&p, o * 4));
        b.push_i32(NHQ as i32);
        b.push_i32(NHK as i32);
        b.push_i32(HD as i32);
        b.push_i32(NROT as i32);
        b.push_f32(fb);
        b.push_i32(n as i32);
        b.push_i32(0);
        launch(gpu, &old, [n_pairs.div_ceil(block), n as u32, 1], block, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("rope_interleaved q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("rope_interleaved k", rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    free(gpu, vec![nq, nk, oq, ok, p])?;
    Ok(bad)
}

fn case_rope_halved_batched(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NHQ: usize = 3;
    const NHK: usize = 1;
    const HD: usize = 64;
    const PAIRS: usize = 16;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/rope_partial_halved_batched.hip"),
        "rope_partial_halved_f32_batched",
    )?;
    let q = randv(r, rows * NHQ * HD, 2.0);
    let k = randv(r, rows * NHK * HD, 2.0);
    let pos: Vec<i32> = (0..rows).map(|i| ((i * 13) % 50_021) as i32).collect();
    let fb = 10_000.0f32;
    let (nq, nk) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let (oq, ok) = (gpu.upload_f32(&q, &[q.len()])?, gpu.upload_f32(&k, &[k.len()])?);
    let p = gpu.upload_f32(&pos_bits(&pos), &[rows])?;
    gpu.rope_partial_halved_f32_batched(&nq, &nk, &p, NHQ, NHK, HD, PAIRS, fb, rows)?;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&oq, o * NHQ * HD * 4));
        b.push_ptr(shift(&ok, o * NHK * HD * 4));
        b.push_ptr(shift(&p, o * 4));
        b.push_i32(NHQ as i32);
        b.push_i32(NHK as i32);
        b.push_i32(HD as i32);
        b.push_i32(PAIRS as i32);
        b.push_f32(fb);
        b.push_i32(n as i32);
        launch(gpu, &old, [(PAIRS as u32).div_ceil(256), n as u32, 1], 256, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("rope_halved q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("rope_halved k", rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    free(gpu, vec![nq, nk, oq, ok, p])?;
    Ok(bad)
}

// ------------------------------------------------- repeat / deinterleave ----

fn case_repeat_interleave(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NKH: usize = 2;
    const RATIO: usize = 3;
    const HD: usize = 128;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/repeat_interleave_qk_batched.hip"),
        "repeat_interleave_qk_f32_batched",
    )?;
    let qs = randv(r, rows * NKH * HD, 2.0);
    let ks = randv(r, rows * NKH * HD, 2.0);
    let nd = rows * NKH * RATIO * HD;
    let (nqs, nks) = (gpu.upload_f32(&qs, &[qs.len()])?, gpu.upload_f32(&ks, &[ks.len()])?);
    let (nqd, nkd) = (gpu.zeros(&[nd], DType::F32)?, gpu.zeros(&[nd], DType::F32)?);
    let (oqd, okd) = (gpu.zeros(&[nd], DType::F32)?, gpu.zeros(&[nd], DType::F32)?);
    gpu.repeat_interleave_qk_f32_batched(&nqs, &nks, &nqd, &nkd, NKH, RATIO, HD, rows)?;
    let total = (NKH * RATIO * HD) as u32;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&nqs, o * NKH * HD * 4));
        b.push_ptr(shift(&nks, o * NKH * HD * 4));
        b.push_ptr(shift(&oqd, o * NKH * RATIO * HD * 4));
        b.push_ptr(shift(&okd, o * NKH * RATIO * HD * 4));
        b.push_i32(NKH as i32);
        b.push_i32(RATIO as i32);
        b.push_i32(HD as i32);
        b.push_i32(n as i32);
        launch(gpu, &old, [total.div_ceil(256), n as u32, 1], 256, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("repeat_interleave q", rows, &gpu.download_f32(&nqd)?, &gpu.download_f32(&oqd)?);
    bad += diff("repeat_interleave k", rows, &gpu.download_f32(&nkd)?, &gpu.download_f32(&okd)?);
    free(gpu, vec![nqs, nks, nqd, nkd, oqd, okd])?;
    Ok(bad)
}

fn case_deinterleave(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NH: usize = 4;
    const HD: usize = 64;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/deinterleave_batched.hip"),
        "deinterleave_f32_batched",
    )?;
    let row = NH * HD;
    let inp = randv(r, rows * row * 2, 2.0);
    let x = gpu.upload_f32(&inp, &[inp.len()])?;
    let (nq, ng) = (gpu.zeros(&[rows * row], DType::F32)?, gpu.zeros(&[rows * row], DType::F32)?);
    let (oq, og) = (gpu.zeros(&[rows * row], DType::F32)?, gpu.zeros(&[rows * row], DType::F32)?);
    gpu.deinterleave_f32_batched(&x, &nq, &ng, NH, HD, rows)?;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&x, o * row * 2 * 4));
        b.push_ptr(shift(&oq, o * row * 4));
        b.push_ptr(shift(&og, o * row * 4));
        b.push_i32(NH as i32);
        b.push_i32(HD as i32);
        b.push_i32(n as i32);
        launch(gpu, &old, [(row as u32).div_ceil(256), n as u32, 1], 256, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("deinterleave q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("deinterleave gate", rows, &gpu.download_f32(&ng)?, &gpu.download_f32(&og)?);
    free(gpu, vec![x, nq, ng, oq, og])?;
    Ok(bad)
}

fn case_deinterleave_q_rmsnorm(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NH: usize = 4;
    const HD: usize = 128;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/deinterleave_q_rmsnorm_f32_batched.hip"),
        "deinterleave_q_rmsnorm_f32_batched",
    )?;
    let row = NH * HD;
    let inp = randv(r, rows * row * 2, 2.0);
    let w = randv(r, HD, 1.5);
    let eps = 1e-6f32;
    let x = gpu.upload_f32(&inp, &[inp.len()])?;
    let wt = gpu.upload_f32(&w, &[HD])?;
    let (nq, ng) = (gpu.zeros(&[rows * row], DType::F32)?, gpu.zeros(&[rows * row], DType::F32)?);
    let (oq, og) = (gpu.zeros(&[rows * row], DType::F32)?, gpu.zeros(&[rows * row], DType::F32)?);
    gpu.deinterleave_q_rmsnorm_f32_batched(&x, &nq, &ng, &wt, NH, HD, rows, eps)?;
    let block = 256u32.min(HD as u32);
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&x, o * row * 2 * 4));
        b.push_ptr(shift(&oq, o * row * 4));
        b.push_ptr(shift(&og, o * row * 4));
        b.push_ptr(wt.buf.as_ptr());
        b.push_i32(NH as i32);
        b.push_i32(HD as i32);
        b.push_i32(n as i32);
        b.push_f32(eps);
        launch(gpu, &old, [NH as u32, n as u32, 1], block, block * 4, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("deinterleave_qnorm q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("deinterleave_qnorm gate", rows, &gpu.download_f32(&ng)?, &gpu.download_f32(&og)?);
    free(gpu, vec![x, wt, nq, ng, oq, og])?;
    Ok(bad)
}

// ----------------------------------------------------------- gated norm ----

fn case_gated_norm(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const NH: usize = 4;
    const HD: usize = 128;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/gated_norm.hip"),
        "gated_norm_f32",
    )?;
    let row = NH * HD;
    let x = randv(r, rows * row, 2.0);
    let z = randv(r, rows * row, 3.0);
    let w = randv(r, HD, 1.5);
    let eps = 1e-6f32;
    let (tx, tz, tw) = (
        gpu.upload_f32(&x, &[x.len()])?,
        gpu.upload_f32(&z, &[z.len()])?,
        gpu.upload_f32(&w, &[HD])?,
    );
    let (no, oo) = (gpu.zeros(&[rows * row], DType::F32)?, gpu.zeros(&[rows * row], DType::F32)?);
    gpu.gated_norm_f32_batched(&tx, &tz, &tw, &no, NH, HD, eps, rows)?;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&tx, o * row * 4));
        b.push_ptr(shift(&tz, o * row * 4));
        b.push_ptr(tw.buf.as_ptr());
        b.push_ptr(shift(&oo, o * row * 4));
        b.push_i32(NH as i32);
        b.push_i32(HD as i32);
        b.push_f32(eps);
        launch(gpu, &old, [NH as u32, n as u32, 1], 32, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let bad = diff("gated_norm out", rows, &gpu.download_f32(&no)?, &gpu.download_f32(&oo)?);
    free(gpu, vec![tx, tz, tw, no, oo])?;
    Ok(bad)
}

// ------------------------------------------------------------ silu clamp ----

fn case_silu_clamp(gpu: &mut Gpu, rows: usize, r: &mut Rng) -> Res<usize> {
    const N: usize = 300;
    let old = old_module(
        gpu,
        include_str!("../../../kernels/src/deepseek4_silu_mul_clamp.hip"),
        "deepseek4_silu_mul_clamp_f32",
    )?;
    let gate = randv(r, rows * N, 14.0);
    let up = randv(r, rows * N, 14.0);
    let limit = 10.0f32;
    let (ng, nu) = (gpu.upload_f32(&gate, &[gate.len()])?, gpu.upload_f32(&up, &[up.len()])?);
    let (og, ou) = (gpu.upload_f32(&gate, &[gate.len()])?, gpu.upload_f32(&up, &[up.len()])?);
    // In place (out == gate), as the MoE pipelines call it.
    gpu.deepseek4_silu_mul_clamp_f32_batched(&ng, &nu, &ng, N, rows, limit)?;
    for (o, n) in slices(rows) {
        let mut b = KernargBlob::new();
        b.push_ptr(shift(&og, o * N * 4));
        b.push_ptr(shift(&ou, o * N * 4));
        b.push_ptr(shift(&og, o * N * 4));
        b.push_i32(N as i32);
        b.push_f32(limit);
        launch(gpu, &old, [(N as u32).div_ceil(256), n as u32, 1], 256, 0, &mut b)?;
    }
    gpu.hip.device_synchronize()?;
    let bad = diff("silu_mul_clamp (in place)", rows, &gpu.download_f32(&ng)?, &gpu.download_f32(&og)?);
    free(gpu, vec![ng, nu, og, ou])?;
    Ok(bad)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let rows_arg = args
        .iter()
        .position(|a| a == "--rows")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "37,1000,65535,65536,70000,131077".into());
    let mut gpu = Gpu::init()?;
    println!("device arch: {}", gpu.arch);
    let mut total = 0usize;
    for rows in rows_arg.split(',') {
        let rows: usize = rows.trim().parse()?;
        println!("rows = {rows}");
        let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ (rows as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
        total += case_rope_batched(&mut gpu, rows, &mut r)?;
        for legacy in [false, true] {
            total += case_rope_partial_batched(&mut gpu, rows, legacy, &mut r)?;
        }
        total += case_mrope_batched(&mut gpu, rows, &mut r)?;
        total += case_rope_interleaved_batched(&mut gpu, rows, &mut r)?;
        total += case_rope_halved_batched(&mut gpu, rows, &mut r)?;
        total += case_repeat_interleave(&mut gpu, rows, &mut r)?;
        total += case_deinterleave(&mut gpu, rows, &mut r)?;
        total += case_deinterleave_q_rmsnorm(&mut gpu, rows, &mut r)?;
        total += case_gated_norm(&mut gpu, rows, &mut r)?;
        total += case_silu_clamp(&mut gpu, rows, &mut r)?;
    }
    if total != 0 {
        eprintln!("FAIL: {total} mismatched values/bytes");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
