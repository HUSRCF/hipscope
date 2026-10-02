//! Temporary smoke for the batch-row launchers in `attention.rs` that now split a
//! batch into <= 65520-row launches (`ROW_LAUNCH_CHUNK`): the legacy
//! `HIPFIRE_PREFILL_MAX_BATCH` override is unbounded, and gfx1201 rejects
//! `gridDim.y/z >= 65536`.
//!
//! Every case runs the public API over `n` rows and compares the WHOLE output
//! byte-for-byte against an oracle built from the existing kernels:
//!   * KV writes: the unmodified kernel raw-launched with its original
//!     `[X, rows, 1]` grid and ABI — one launch while n <= 65535 (the exact old
//!     launch), otherwise <= 40000-row slices over shifted pointers.
//!   * Flash prefill (q8 batched masked, q8 rows masked): the same API on
//!     <= 40000-row views (below the cap, so the pre-change code path).
//!
//! Cases: q8/fp8/bf16/f16 batched writes; K+V writes asym2/3/4 and fwht2/3/4
//! (both K kernel and the Q8 V write); q8 flash batched-masked and rows-masked.
//!
//!     cargo run --release -p rdna-compute --features lab \
//!         --example kv_row_chunk_smoke \
//!         [-- --rows 37,4096,65535,65536,65537,70000,131072 --flash-rows 37,65536,70000 --kv-heads 4]
//!
//! Exit status 1 on any mismatch or unwritten output.

use hip_bridge::KernargBlob;
use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};
use std::ffi::c_void;

const HD: usize = 256;
const CHUNK: usize = 40_000;
const NH: usize = 16;
const CTX: usize = 128;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (((self.next() >> 40) as f32) / ((1u64 << 24) as f32)) * 2.0 - 1.0
    }
}

fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload(gpu: &mut Gpu, data: &[u8]) -> Res<GpuTensor> {
    assert!(data.len() % 4 == 0);
    let t = gpu.alloc_tensor(&[data.len() / 4], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, data)?;
    Ok(t)
}

fn filled(gpu: &mut Gpu, bytes: usize, v: u8) -> Res<GpuTensor> {
    upload(gpu, &vec![v; bytes])
}

fn download(gpu: &mut Gpu, t: &GpuTensor, bytes: usize) -> Res<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf)?;
    Ok(out)
}

fn shift(p: *mut c_void, bytes: usize) -> *const c_void {
    (p as usize + bytes) as *const c_void
}

/// Old-launch row slices: one launch up to 65535 rows, else <= CHUNK rows.
fn slices(n: usize) -> Vec<(usize, usize)> {
    let step = if n <= 65_535 { n.max(1) } else { CHUNK };
    (0..n.max(1)).step_by(step).map(|o| (o, step.min(n - o))).collect()
}

fn raw(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, shared: u32, mut a: KernargBlob) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], shared, a.as_mut_slice())?;
    Ok(())
}

fn q8_blocks(n_blocks: usize, r: &mut Rng) -> Vec<u8> {
    let mut out = vec![0u8; n_blocks * 34];
    for blk in out.chunks_exact_mut(34) {
        let half = 0x1800u16 + (r.next() % 0x0800) as u16;
        blk[..2].copy_from_slice(&half.to_le_bytes());
        for b in &mut blk[2..] {
            *b = (r.next() >> 24) as u8;
        }
    }
    out
}

fn report(name: &str, n: usize, nkv: usize, got: &[Vec<u8>], want: &[Vec<u8>], sentinel: u8) -> bool {
    let mut ok = true;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (idx, (g, w)) in got.iter().zip(want).enumerate() {
        if let Some(i) = g.iter().zip(w).position(|(a, b)| a != b) {
            eprintln!("  MISMATCH {name} n={n} buffer {idx} at byte {i}/{}", g.len());
            ok = false;
        }
        // The last row must have been written (not left at the sentinel).
        let tail = &g[g.len() - g.len().min(64)..];
        if sentinel != 0 && tail.iter().all(|&b| b == sentinel) {
            eprintln!("  UNWRITTEN tail {name} n={n} buffer {idx}");
            ok = false;
        }
        for &b in g {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    println!(
        "{:<28} rows={:<7} kv_heads={} fnv1a={:016x} {}",
        name,
        n,
        nkv,
        h,
        if ok { "ok" } else { "FAIL" }
    );
    ok
}

#[derive(Clone, Copy)]
enum Plain {
    Q8,
    Fp8,
    Bf16,
    F16,
}

fn run_plain(gpu: &mut Gpu, which: Plain, n: usize, nkv: usize) -> Res<bool> {
    let kv = nkv * HD;
    let mut r = Rng(0x1234_5678_9abc_def1 ^ n as u64);
    let src_f: Vec<f32> = (0..n * kv).map(|_| r.unit() * 4.0).collect();
    let src = upload(gpu, &bytes_of(&src_f))?;
    drop(src_f);
    let pos_b: Vec<u8> = (0..n as i32).flat_map(|p| p.to_le_bytes()).collect();
    let pos = upload(gpu, &pos_b)?;
    let (name, dst_bytes) = match which {
        Plain::Q8 => ("kv_write_q8_0_batched", n * nkv * (HD / 32) * 34),
        Plain::Fp8 => ("kv_write_fp8_e4m3_batched", n * nkv * (HD + 2)),
        Plain::Bf16 => ("kv_write_bf16_batched", n * kv * 2),
        Plain::F16 => ("kv_write_f16_batched", n * kv * 2),
    };
    let dst_bytes = (dst_bytes + 3) & !3;
    let got_t = filled(gpu, dst_bytes, 0xA5)?;
    match which {
        Plain::Q8 => gpu.kv_cache_write_q8_0_batched(&got_t, &src, &pos, nkv, HD, n)?,
        Plain::Fp8 => gpu.kv_cache_write_fp8_e4m3_batched(&got_t, &src, &pos, nkv, HD, n)?,
        Plain::Bf16 => gpu.kv_cache_write_bf16_batched(&got_t, &src, &pos, nkv, HD, n, None, None)?,
        Plain::F16 => gpu.kv_cache_write_f16_batched(&got_t, &src, &pos, nkv, HD, n, None, None)?,
    }
    let want_t = filled(gpu, dst_bytes, 0xA5)?;
    let (f, grid_x, block, has_slots) = match which {
        Plain::Q8 => ("kv_cache_write_q8_0_batched", (nkv * HD / 32) as u32, 32, true),
        Plain::Fp8 => ("kv_cache_write_fp8_e4m3_batched", nkv as u32, 32, false),
        Plain::Bf16 => ("kv_cache_write_bf16_batched", kv.div_ceil(64) as u32, 64, true),
        Plain::F16 => ("kv_cache_write_f16_batched", kv.div_ceil(64) as u32, 64, true),
    };
    for (off, len) in slices(n) {
        let mut a = KernargBlob::new();
        a.push_ptr(want_t.buf.as_ptr());
        a.push_ptr(shift(src.buf.as_ptr(), off * kv * 4));
        a.push_ptr(shift(pos.buf.as_ptr(), off * 4));
        a.push_i32(nkv as i32);
        a.push_i32(HD as i32);
        a.push_i32(len as i32);
        if has_slots {
            a.push_ptr(std::ptr::null());
            a.push_ptr(std::ptr::null());
        }
        raw(gpu, f, [grid_x, len as u32, 1], block, 0, a)?;
    }
    let got = download(gpu, &got_t, dst_bytes)?;
    let want = download(gpu, &want_t, dst_bytes)?;
    for t in [src, pos, got_t, want_t] {
        gpu.free_tensor(t)?;
    }
    Ok(report(name, n, nkv, &[got], &[want], 0xA5))
}

#[derive(Clone, Copy)]
enum KMode {
    Asym(u32),
    Fwht(u32),
}

fn run_k(gpu: &mut Gpu, mode: KMode, n: usize, nkv: usize, t1: &GpuTensor, t2: &GpuTensor) -> Res<bool> {
    let kv = nkv * HD;
    let (bits, kname, label) = match mode {
        KMode::Asym(b) => (b, format!("kv_cache_write_asym_k_givens{b}_batched"), format!("kv_write_asym{b}_batched")),
        KMode::Fwht(b) => (b, format!("kv_cache_write_asym_k_fwht{b}_batched"), format!("kv_write_fwht{b}_batched")),
    };
    let k_bph = 4 + HD * bits as usize / 8;
    let v_bph = HD / 32 * 34;
    let (kb, vb) = ((n * nkv * k_bph + 3) & !3, (n * nkv * v_bph + 3) & !3);
    let mut r = Rng(0x0fed_cba9_8765_4321 ^ n as u64 ^ bits as u64);
    let ks: Vec<f32> = (0..n * kv).map(|_| r.unit() * 4.0).collect();
    let vs: Vec<f32> = (0..n * kv).map(|_| r.unit() * 4.0).collect();
    let (k_src, v_src) = (upload(gpu, &bytes_of(&ks))?, upload(gpu, &bytes_of(&vs))?);
    drop((ks, vs));
    let pos_b: Vec<u8> = (0..n as i32).flat_map(|p| p.to_le_bytes()).collect();
    let pos = upload(gpu, &pos_b)?;
    let (gk, gv) = (filled(gpu, kb, 0xA5)?, filled(gpu, vb, 0xA5)?);
    match mode {
        KMode::Asym(2) => gpu.kv_cache_write_asym2_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n)?,
        KMode::Asym(3) => gpu.kv_cache_write_asym3_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n)?,
        KMode::Asym(_) => gpu.kv_cache_write_asym4_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n)?,
        KMode::Fwht(2) => gpu.kv_cache_write_fwht2_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n, 8)?,
        KMode::Fwht(3) => gpu.kv_cache_write_fwht3_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n, 8)?,
        KMode::Fwht(_) => gpu.kv_cache_write_fwht4_batched(&gk, &gv, &k_src, &v_src, &pos, t1, t2, nkv, HD, n, 8)?,
    }
    let (wk, wv) = (filled(gpu, kb, 0xA5)?, filled(gpu, vb, 0xA5)?);
    for (off, len) in slices(n) {
        let mut a = KernargBlob::new();
        a.push_ptr(wk.buf.as_ptr());
        a.push_ptr(shift(k_src.buf.as_ptr(), off * kv * 4));
        a.push_ptr(shift(pos.buf.as_ptr(), off * 4));
        a.push_ptr(t1.buf.as_ptr());
        a.push_ptr(t2.buf.as_ptr());
        a.push_i32(nkv as i32);
        a.push_i32(HD as i32);
        a.push_i32(len as i32);
        raw(gpu, &kname, [nkv as u32, len as u32, 1], 32, ((HD + 32) * 4) as u32, a)?;
        let mut v = KernargBlob::new();
        v.push_ptr(wv.buf.as_ptr());
        v.push_ptr(shift(v_src.buf.as_ptr(), off * kv * 4));
        v.push_ptr(shift(pos.buf.as_ptr(), off * 4));
        v.push_i32(nkv as i32);
        v.push_i32(HD as i32);
        v.push_i32(len as i32);
        v.push_ptr(std::ptr::null());
        v.push_ptr(std::ptr::null());
        raw(gpu, "kv_cache_write_q8_0_batched", [(nkv * HD / 32) as u32, len as u32, 1], 32, 0, v)?;
    }
    let got = [download(gpu, &gk, kb)?, download(gpu, &gv, vb)?];
    let want = [download(gpu, &wk, kb)?, download(gpu, &wv, vb)?];
    for t in [k_src, v_src, pos, gk, gv, wk, wv] {
        gpu.free_tensor(t)?;
    }
    Ok(report(&label, n, nkv, &got, &want, 0xA5))
}

/// q8 flash prefill rows over a tiny context so partials for `n` rows stay
/// bounded; oracle = the same API on <= CHUNK-row views (pre-change path).
fn run_flash(gpu: &mut Gpu, rows_variant: bool, n: usize, nkv: usize) -> Res<bool> {
    let q_dim = NH * HD;
    let mut r = Rng(0x7777_1111_3333_5555 ^ n as u64);
    let q_f: Vec<f32> = (0..n * q_dim).map(|_| r.unit()).collect();
    let q = upload(gpu, &bytes_of(&q_f))?;
    drop(q_f);
    let k = upload(gpu, &q8_blocks(CTX * nkv * HD / 32, &mut r))?;
    let v = upload(gpu, &q8_blocks(CTX * nkv * HD / 32, &mut r))?;
    // Rows attend causally up to a position inside the CTX window.
    let pos_b: Vec<u8> = (0..n as i32).flat_map(|p| (p % CTX as i32).to_le_bytes()).collect();
    let pos = upload(gpu, &pos_b)?;
    let tiles = {
        let a = rdna_compute::attention::q8_flash_tile_size(&gpu.arch, NH, nkv, HD, CTX);
        let b = gpu.attn_tile_size();
        CTX.div_ceil(a).max(CTX.div_ceil(b))
    };
    let part_f32 = n * NH * tiles * (2 + HD);
    let parts = gpu.alloc_tensor(&[part_f32], DType::F32)?;
    let out_bytes = n * q_dim * 4;
    let got_t = filled(gpu, out_bytes, 0xA5)?;
    let want_t = filled(gpu, out_bytes, 0xA5)?;
    let name = if rows_variant { "attn_q8_flash_rows_masked" } else { "attn_q8_flash_batched_masked" };

    let mut call = |gpu: &mut Gpu, out: &GpuTensor, qv: &GpuTensor, pv: &GpuTensor, len: usize| -> Res<bool> {
        if rows_variant {
            Ok(gpu.attention_flash_q8_0_rows_masked(qv, &k, &v, out, pv, NH, nkv, HD, CTX, len, &parts)?)
        } else {
            gpu.attention_flash_q8_0_batched_masked(qv, &k, &v, out, pv, NH, nkv, HD, CTX, CTX, len, &parts, None, 0, 0)?;
            Ok(true)
        }
    };
    if !call(gpu, &got_t, &q, &pos, n)? {
        println!("{name:<28} rows={n:<7} kv_heads={nkv} SKIP (route not admitted on {})", gpu.arch);
        return Ok(true);
    }
    let mut off = 0;
    while off < n {
        let len = CHUNK.min(n - off);
        let qv = q.sub_offset(off * q_dim, len * q_dim);
        let pv = pos.sub_offset(off, len);
        let ov = want_t.sub_offset(off * q_dim, len * q_dim);
        if !call(gpu, &ov, &qv, &pv, len)? {
            return Err("oracle route not admitted".into());
        }
        off += len;
    }
    let got = download(gpu, &got_t, out_bytes)?;
    let want = download(gpu, &want_t, out_bytes)?;
    for t in [q, k, v, pos, parts, got_t, want_t] {
        gpu.free_tensor(t)?;
    }
    Ok(report(name, n, nkv, &[got], &[want], 0xA5))
}

fn arg_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let rows = arg_after(&args, "--rows").unwrap_or_else(|| "37,4096,65535,65536,65537,70000,131072".into());
    let flash_rows = arg_after(&args, "--flash-rows").unwrap_or_else(|| "37,65536,70000".into());
    let nkv: usize = arg_after(&args, "--kv-heads").map_or(Ok(4), |s| s.parse())?;
    let mut gpu = Gpu::init()?;
    println!("arch={} head_dim={HD} chunk(oracle)={CHUNK}", gpu.arch);
    let s1 = gpu.upload_f32(&gen_fwht_signs(42, 256), &[256])?;
    let s2 = gpu.upload_f32(&gen_fwht_signs(1042, 256), &[256])?;
    let mut r = Rng(0xC0DE);
    let (cs, sn): (Vec<f32>, Vec<f32>) = (0..256)
        .map(|_| {
            let a = r.unit() * 3.14159;
            (a.cos(), a.sin())
        })
        .unzip();
    let c1 = gpu.upload_f32(&cs, &[256])?;
    let c2 = gpu.upload_f32(&sn, &[256])?;
    let mut fails = 0;
    for n in rows.split(',') {
        let n: usize = n.trim().parse()?;
        for w in [Plain::Q8, Plain::Fp8, Plain::Bf16, Plain::F16] {
            if matches!(w, Plain::Fp8) && gpu.arch != "gfx1201" {
                continue;
            }
            fails += !run_plain(&mut gpu, w, n, nkv)? as i32;
        }
        for m in [KMode::Asym(2), KMode::Asym(3), KMode::Asym(4)] {
            fails += !run_k(&mut gpu, m, n, nkv, &c1, &c2)? as i32;
        }
        for m in [KMode::Fwht(2), KMode::Fwht(3), KMode::Fwht(4)] {
            fails += !run_k(&mut gpu, m, n, nkv, &s1, &s2)? as i32;
        }
    }
    for n in flash_rows.split(',') {
        let n: usize = n.trim().parse()?;
        for rows_variant in [false, true] {
            fails += !run_flash(&mut gpu, rows_variant, n, nkv)? as i32;
        }
    }
    if fails != 0 {
        eprintln!("FAIL: {fails} case(s)");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
