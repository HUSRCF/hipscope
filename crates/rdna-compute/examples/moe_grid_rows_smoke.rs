//! Temporary before/after smoke for the MoE k8 row-grid folds in `moe.rs`.
//!
//! Four objects moved their token / slot axis from grid.y to grid.x (limit
//! 2^31); gfx1201 caps grid.y/z at 65535 blocks:
//!
//!   * moe_unscatter_silu_clamp_k8           slot  y -> x
//!   * moe_down_combine_k8_batched (+vec4)   token y -> x
//!   * moe_down_combine_grouped_k8           token y -> x
//!
//! One binary proves before/after byte identity. The pristine base e371200c4
//! kernel sources live outside the tree in
//! `/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/` (fetched with
//! `git show e371200c4:kernels/src/<file>`), are compiled under unique
//! `mgrb_*` symbols and raw-launched with their ORIGINAL grid and ABI, in
//! slices of `CHUNK` (<= 40000) rows over shifted pointers so each old launch
//! stays under the 65535 limit. The new side is the production `Gpu` API or,
//! for vec4 (selector is rdna3-dGPU only), the CURRENT kernel source compiled
//! raw under `mgrn_*` with the NEW grid, run once over the whole row range.
//! Whole output buffers (0xA5 sentinel prefill for silu) must be
//! byte-identical.
//!
//!     cargo run --release -p rdna-compute --features lab \
//!         --example moe_grid_rows_smoke [-- --tokens 100,65600]
//!
//! Exit status 1 on any mismatch. Delete this example and its Cargo stanza
//! after the proof is recorded.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};

const CHUNK: usize = 40_000;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

macro_rules! src_pair {
    ($base:ident, $new:ident, $file:literal) => {
        const $base: &str = include_str!(concat!(
            "/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/",
            $file
        ));
        const $new: &str = include_str!(concat!("../../../kernels/src/", $file));
    };
}
src_pair!(B_SILU, _N_SILU, "moe_unscatter_silu_clamp_k8.hip");
src_pair!(B_COMB, _N_COMB, "moe_down_combine_k8_batched.hip");
src_pair!(B_VEC4, N_VEC4, "moe_down_combine_k8_batched_vec4.hip");
src_pair!(B_GRP, _N_GRP, "moe_down_combine_grouped_k8.hip");

const S_SILU: &str = "moe_unscatter_silu_clamp_k8";
const S_COMB: &str = "moe_down_combine_k8_batched";
const S_VEC4: &str = "moe_down_combine_k8_batched_vec4";
const S_GRP: &str = "moe_down_combine_grouped_k8";

/// Compile `src` with `sym` renamed to `{prefix}{sym}` and return that name.
fn compile(gpu: &mut Gpu, prefix: &str, src: &str, sym: &str) -> Res<String> {
    let renamed = format!("{prefix}{sym}");
    let body = src.replace(sym, &renamed);
    assert!(body.contains(&renamed), "rename failed for {sym}");
    gpu.ensure_kernel_public(&renamed, &body, &renamed)?;
    Ok(renamed)
}

fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, a: &mut KernargBlob) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, a.as_mut_slice())?;
    Ok(())
}

fn shift(p: *mut std::ffi::c_void, bytes: usize) -> *const std::ffi::c_void {
    (p as usize + bytes) as *const std::ffi::c_void
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        (((self.next() >> 11) as f64) / ((1u64 << 53) as f64) * 8.0 - 4.0) as f32
    }
}

fn fnv(b: &[u8]) -> u64 {
    b.iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ *x as u64).wrapping_mul(0x100_0000_01b3))
}

fn rand_f32(r: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| r.f32()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn sentinel(gpu: &mut Gpu, elems: usize) -> Res<GpuTensor> {
    let t = gpu.zeros(&[elems], DType::F32)?;
    gpu.hip.memset(&t.buf, 0xA5, elems * 4)?;
    Ok(t)
}

fn read(gpu: &mut Gpu, t: &GpuTensor, bytes: usize) -> Res<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf)?;
    Ok(out)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn report(name: &str, n: usize, grid_new: [u32; 3], new: &[u8], old: &[u8]) -> usize {
    let bad = new.iter().zip(old).filter(|(a, b)| a != b).count();
    let unwritten = new.iter().filter(|b| **b == 0xA5).count();
    println!(
        "{name} n={n} new_grid={grid_new:?} mismatched_bytes={bad} \
         sentinel_bytes={unwritten} fnv={:016x}",
        fnv(new)
    );
    bad
}

struct Names {
    old_silu: String,
    old_comb: String,
    old_vec4: String,
    new_vec4: String,
    old_grp: String,
}

fn run_combine(gpu: &mut Gpu, n: usize, m: usize, names: &Names) -> Res<usize> {
    const K: usize = 8;
    let mut r = Rng(0x1234_5678_9ABC_DEF1 ^ (n as u64) << 7 | 1);
    let slots = n * K;
    let eo = rand_f32(&mut r, slots * m);
    let w = rand_f32(&mut r, slots);
    let x0 = rand_f32(&mut r, n * m);
    let mut a = 37usize;
    while gcd(a, slots) != 1 {
        a += 2;
    }
    let inv: Vec<i32> = (0..slots).map(|f| ((f * a) % slots) as i32).collect();
    let eo_t = gpu.upload_f32(&eo, &[eo.len()])?;
    let w_t = gpu.upload_f32(&w, &[w.len()])?;
    let inv_b = i32_bytes(&inv);
    let inv_t = gpu.upload_raw(&inv_b, &[inv_b.len()])?;
    let xbytes = n * m * 4;
    let mut bad = 0;

    for (label, grouped) in [("combine_k8_batched", false), ("combine_grouped_k8", true)] {
        let x_new = gpu.upload_f32(&x0, &[x0.len()])?;
        let x_old = gpu.upload_f32(&x0, &[x0.len()])?;
        // The grouped path reuses `eo` as Y_down_grouped [slots, m].
        if grouped {
            gpu.moe_down_combine_grouped_k8(&eo_t, &inv_t, &w_t, &x_new, m, K, n)?;
        } else {
            gpu.moe_down_combine_k8_batched(&eo_t, &w_t, &x_new, m, K, n)?;
        }
        let func = if grouped { &names.old_grp } else { &names.old_comb };
        let grid_x = m.div_ceil(256) as u32;
        let mut off = 0;
        while off < n {
            let c = CHUNK.min(n - off);
            let mut b = KernargBlob::new();
            if grouped {
                b.push_ptr(eo_t.buf.as_ptr()); // Y: slot ids are absolute
                b.push_ptr(shift(inv_t.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(w_t.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(x_old.buf.as_ptr(), off * m * 4));
            } else {
                b.push_ptr(shift(eo_t.buf.as_ptr(), off * K * m * 4));
                b.push_ptr(shift(w_t.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(x_old.buf.as_ptr(), off * m * 4));
            }
            b.push_i32(m as i32);
            b.push_i32(K as i32);
            launch(gpu, func, [grid_x, c as u32, 1], 256, &mut b)?;
            off += c;
        }
        let got = read(gpu, &x_new, xbytes)?;
        let want = read(gpu, &x_old, xbytes)?;
        bad += report(label, n, [n as u32, grid_x, 1], &got, &want);
        gpu.free_tensor(x_new)?;
        gpu.free_tensor(x_old)?;
    }

    // vec4: raw new source with the new grid vs raw old source, old grid.
    {
        let x_new = gpu.upload_f32(&x0, &[x0.len()])?;
        let x_old = gpu.upload_f32(&x0, &[x0.len()])?;
        let gx = m.div_ceil(256 * 4) as u32;
        let mut b = KernargBlob::new();
        b.push_ptr(eo_t.buf.as_ptr());
        b.push_ptr(w_t.buf.as_ptr());
        b.push_ptr(x_new.buf.as_ptr());
        b.push_i32(m as i32);
        b.push_i32(K as i32);
        launch(gpu, &names.new_vec4, [n as u32, gx, 1], 256, &mut b)?;
        let mut off = 0;
        while off < n {
            let c = CHUNK.min(n - off);
            let mut b = KernargBlob::new();
            b.push_ptr(shift(eo_t.buf.as_ptr(), off * K * m * 4));
            b.push_ptr(shift(w_t.buf.as_ptr(), off * K * 4));
            b.push_ptr(shift(x_old.buf.as_ptr(), off * m * 4));
            b.push_i32(m as i32);
            b.push_i32(K as i32);
            launch(gpu, &names.old_vec4, [gx, c as u32, 1], 256, &mut b)?;
            off += c;
        }
        let got = read(gpu, &x_new, xbytes)?;
        let want = read(gpu, &x_old, xbytes)?;
        bad += report("combine_k8_batched_vec4(raw)", n, [n as u32, gx, 1], &got, &want);
        gpu.free_tensor(x_new)?;
        gpu.free_tensor(x_old)?;
    }
    gpu.free_tensor(eo_t)?;
    gpu.free_tensor(w_t)?;
    gpu.free_tensor(inv_t)?;
    Ok(bad)
}

fn run_silu(gpu: &mut Gpu, tokens: usize, names: &Names) -> Res<usize> {
    const MI: usize = 520;
    const K: usize = 8;
    const LIMIT: f32 = 7.0;
    let mut r = Rng(0xDEAD_BEEF_0BAD_F00D ^ (tokens as u64) << 5 | 1);
    let slots = tokens * K;
    let mut a = 37usize;
    while gcd(a, slots) != 1 {
        a += 2;
    }
    let mut sorted = Vec::new();
    for i in 0..slots {
        sorted.push(((i * a) % slots) as i32);
        if i % 1000 == 999 {
            sorted.push(-1);
        }
    }
    let grouped = sorted.len();
    let y = rand_f32(&mut r, grouped * 2 * MI);
    let y_t = gpu.upload_f32(&y, &[y.len()])?;
    let sb = i32_bytes(&sorted);
    let s_t = gpu.upload_raw(&sb, &[sb.len()])?;
    let out_elems = slots * MI;
    let new = sentinel(gpu, out_elems)?;
    let old = sentinel(gpu, out_elems)?;
    gpu.moe_unscatter_silu_clamp_k8(&y_t, &s_t, &new, MI, K, grouped, LIMIT)?;
    let gx = MI.div_ceil(256) as u32;
    let mut off = 0;
    while off < grouped {
        let c = CHUNK.min(grouped - off);
        let mut b = KernargBlob::new();
        b.push_ptr(shift(y_t.buf.as_ptr(), off * 2 * MI * 4));
        b.push_ptr(shift(s_t.buf.as_ptr(), off * 4));
        b.push_ptr(old.buf.as_ptr()); // flat ids are absolute
        b.push_i32(MI as i32);
        b.push_i32(K as i32);
        b.push_i32(c as i32);
        b.push_f32(LIMIT);
        launch(gpu, &names.old_silu, [gx, c as u32, 1], 256, &mut b)?;
        off += c;
    }
    let got = read(gpu, &new, out_elems * 4)?;
    let want = read(gpu, &old, out_elems * 4)?;
    let bad = report("silu_clamp_k8", grouped, [grouped as u32, gx, 1], &got, &want);
    gpu.free_tensor(new)?;
    gpu.free_tensor(old)?;
    gpu.free_tensor(y_t)?;
    gpu.free_tensor(s_t)?;
    Ok(bad)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let tokens: Vec<usize> = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![100, 65_600]);
    let mut gpu = Gpu::init()?;
    if !matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") {
        return Err(format!("gfx1151/gfx1201-only, device is {}", gpu.arch).into());
    }
    let names = Names {
        old_silu: compile(&mut gpu, "mgrb_", B_SILU, S_SILU)?,
        old_comb: compile(&mut gpu, "mgrb_", B_COMB, S_COMB)?,
        old_vec4: compile(&mut gpu, "mgrb_", B_VEC4, S_VEC4)?,
        new_vec4: compile(&mut gpu, "mgrn_", N_VEC4, S_VEC4)?,
        old_grp: compile(&mut gpu, "mgrb_", B_GRP, S_GRP)?,
    };
    let mut bad = 0;
    for &n in &tokens {
        // M: two scalar column tiles at 65600 tokens (537 MB), wide at 100.
        let m = if n > 10_000 { 260 } else { 1032 };
        bad += run_combine(&mut gpu, n, m, &names)?;
        // Silu consumes top-8 slots: 8200 tokens -> 65600 slots + 65 padding
        // = 65665 grouped rows for the large case.
        let st = if n > 8_200 { 8_200 } else { n };
        bad += run_silu(&mut gpu, st, &names)?;
    }
    if bad != 0 {
        eprintln!("FAIL: {bad} mismatched bytes");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
