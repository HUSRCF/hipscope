//! Temporary smoke for the adaptive-KV transcode launchers. The four kernels
//! (`transcode_v_q8_to_lloyd4`, `transcode_v_lloyd_down` 4->3 / 4->2 / 3->2,
//! `transcode_k_fwht4_to_fwht2`, `transcode_k_fwht4_to_fwht3`) fold
//! `(pos, kv_head)` into grid x; the old `[n_kv_heads, n_positions]` grid broke
//! at >= 65536 positions on gfx1201 (grid.y limit).
//!
//! One binary proves before/after identity. The pristine base e371200c4 kernel
//! sources live outside the tree in
//! `/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/` (fetched with
//! `git show e371200c4:kernels/src/kv_transcode_*.hip`), are compiled under
//! unique `kvt_base_*` symbols and raw-launched with their original
//! `[kv_heads, n, 1]` grid and ABI.
//!
//! For every variant (q8->lloyd4, lloyd 4->3, 4->2, 3->2, fwht4->fwht2,
//! fwht4->fwht3) and every position count it
//!   1. runs the production (folded-x) launcher once over all positions,
//!   2. runs the base kernel: one raw old launch for n <= 65535, otherwise
//!      in `CHUNK`-position slices over shifted pointers, and requires the
//!      whole dst buffers to be byte-identical,
//!   3. requires every (pos, head) record to have been written (dst starts as
//!      0xA5 sentinel),
//!   4. prints an FNV-1a digest of the output.
//!
//!     cargo run --release -p rdna-compute --features lab \
//!         --example kv_transcode_grid_smoke \
//!         [-- --positions 37,1000,65535,65536,65537,70000,262144 --kv-heads 4]
//!
//! Exit status 1 on any mismatch / unwritten record.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const HEAD_DIM: usize = 256;
const CHUNK: usize = 40_000;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
enum Variant {
    VQ8ToL4,
    VDown(i32, i32),
    KToFwht2,
    KToFwht3,
}

impl Variant {
    fn name(self) -> String {
        match self {
            Variant::VQ8ToL4 => "v_q8_to_lloyd4".into(),
            Variant::VDown(s, d) => format!("v_lloyd_down_{s}to{d}"),
            Variant::KToFwht2 => "k_fwht4_to_fwht2".into(),
            Variant::KToFwht3 => "k_fwht4_to_fwht3".into(),
        }
    }
    /// Source bytes per (pos, head).
    fn src_bph(self) -> usize {
        match self {
            Variant::VQ8ToL4 => (HEAD_DIM / 32) * 34,
            Variant::VDown(s, _) => 4 + HEAD_DIM * s as usize / 8,
            Variant::KToFwht2 | Variant::KToFwht3 => 4 + HEAD_DIM / 2,
        }
    }
    /// Destination bytes per (pos, head).
    fn dst_bph(self) -> usize {
        match self {
            Variant::VQ8ToL4 => 4 + HEAD_DIM / 2,
            Variant::VDown(_, d) => 4 + HEAD_DIM * d as usize / 8,
            Variant::KToFwht2 => 4 + HEAD_DIM * 2 / 8,
            Variant::KToFwht3 => 4 + HEAD_DIM * 3 / 8,
        }
    }
    fn launch(
        self,
        gpu: &mut Gpu,
        dst: &GpuTensor,
        src: &GpuTensor,
        s1: &GpuTensor,
        s2: &GpuTensor,
        nkv: usize,
        n: usize,
    ) -> Res<()> {
        match self {
            Variant::VQ8ToL4 => gpu.transcode_v_q8_to_lloyd4(dst, src, s1, s2, nkv, HEAD_DIM, n)?,
            Variant::VDown(s, d) => gpu.transcode_v_lloyd_down(dst, src, nkv, HEAD_DIM, n, s, d)?,
            Variant::KToFwht2 => gpu.transcode_k_fwht4_to_fwht2(dst, src, nkv, HEAD_DIM, n)?,
            Variant::KToFwht3 => {
                gpu.transcode_k_fwht4_to_fwht3(dst, src, s1, s2, nkv, HEAD_DIM, n)?
            }
        }
        Ok(())
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Deterministic source records for `n * nkv` (pos, head) slots.
fn build_src(v: Variant, n: usize, nkv: usize, seed: u64) -> Vec<u8> {
    let bph = v.src_bph();
    let mut r = Rng(seed | 1);
    let mut out = vec![0u8; n * nkv * bph];
    for rec in out.chunks_exact_mut(bph) {
        match v {
            Variant::VQ8ToL4 => {
                // 34-byte q8_0 blocks: f16 scale (finite, ~[2^-9, 2^-4]) + 32 i8.
                for blk in rec.chunks_exact_mut(34) {
                    let half = 0x1800u16 + (r.next() % 0x0800) as u16;
                    blk[..2].copy_from_slice(&half.to_le_bytes());
                    for b in &mut blk[2..] {
                        *b = (r.next() >> 24) as u8;
                    }
                }
            }
            _ => {
                // [f32 cnorm][packed indices]; random indices are all valid.
                let cn = 0.5f32 + ((r.next() >> 40) as f32) / ((1u64 << 24) as f32);
                rec[..4].copy_from_slice(&cn.to_le_bytes());
                for b in &mut rec[4..] {
                    *b = (r.next() >> 24) as u8;
                }
            }
        }
    }
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn alloc_bytes(gpu: &mut Gpu, bytes: usize) -> Res<GpuTensor> {
    assert!(bytes % 4 == 0);
    Ok(gpu.alloc_tensor(&[bytes / 4], DType::F32)?)
}

fn upload_bytes(gpu: &mut Gpu, data: &[u8]) -> Res<GpuTensor> {
    let t = alloc_bytes(gpu, data.len())?;
    gpu.hip.memcpy_htod(&t.buf, data)?;
    Ok(t)
}

const TURBO_COMMON_H: &str = include_str!("../../../kernels/src/turbo_common.h");
const GIVENS_COMMON_H: &str = include_str!("../../../kernels/src/givens_common.h");
const BASE_Q8_TO_L4: &str =
    include_str!("/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/kv_transcode_v_q8_to_lloyd4.hip");
const BASE_LLOYD_DOWN: &str =
    include_str!("/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/kv_transcode_v_lloyd_down.hip");
const BASE_K_TO_FWHT2: &str =
    include_str!("/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/kv_transcode_k_fwht4_to_fwht2.hip");
const BASE_K_TO_FWHT3: &str =
    include_str!("/home/kaden/qcal/release-0.4.1/grid-y/base-kernels/kv_transcode_k_fwht4_to_fwht3.hip");

fn old_func(v: Variant) -> &'static str {
    match v {
        Variant::VQ8ToL4 => "kvt_base_v_q8_to_lloyd4",
        Variant::VDown(..) => "kvt_base_v_lloyd_down",
        Variant::KToFwht2 => "kvt_base_k_fwht4_to_fwht2",
        Variant::KToFwht3 => "kvt_base_k_fwht4_to_fwht3",
    }
}

fn compile_old(gpu: &mut Gpu, v: Variant) -> Res<()> {
    let body = match v {
        Variant::VQ8ToL4 => BASE_Q8_TO_L4,
        Variant::VDown(..) => BASE_LLOYD_DOWN,
        Variant::KToFwht2 => BASE_K_TO_FWHT2,
        Variant::KToFwht3 => BASE_K_TO_FWHT3,
    };
    let stripped = body
        .replace("#include \"turbo_common.h\"", "")
        .replace("#include \"givens_common.h\"", "")
        .replace("void kv_transcode_", "void kvt_base_");
    assert!(stripped.contains(old_func(v)), "base kernel rename failed");
    let src = format!("{TURBO_COMMON_H}\n{GIVENS_COMMON_H}\n{stripped}");
    gpu.ensure_kernel_public(old_func(v), &src, old_func(v))?;
    Ok(())
}

/// Raw launch of the base kernel with its original ABI and [kv_heads, n, 1]
/// grid; `d_off`/`s_off` are byte offsets applied to the dst/src pointers.
fn launch_old(
    gpu: &mut Gpu,
    v: Variant,
    dst: &GpuTensor,
    src: &GpuTensor,
    s1: &GpuTensor,
    s2: &GpuTensor,
    nkv: usize,
    n: usize,
    d_off: usize,
    s_off: usize,
) -> Res<()> {
    let mut a = KernargBlob::new();
    a.push_ptr((dst.buf.as_ptr() as usize + d_off) as *const c_void);
    a.push_ptr((src.buf.as_ptr() as usize + s_off) as *const c_void);
    match v {
        Variant::VQ8ToL4 | Variant::KToFwht3 => {
            a.push_ptr(s1.buf.as_ptr());
            a.push_ptr(s2.buf.as_ptr());
            a.push_i32(nkv as i32);
            a.push_i32(HEAD_DIM as i32);
            a.push_i32(n as i32);
        }
        Variant::VDown(sb, db) => {
            a.push_i32(nkv as i32);
            a.push_i32(HEAD_DIM as i32);
            a.push_i32(n as i32);
            a.push_i32(sb);
            a.push_i32(db);
        }
        Variant::KToFwht2 => {
            a.push_i32(nkv as i32);
            a.push_i32(HEAD_DIM as i32);
            a.push_i32(n as i32);
        }
    }
    a.pad_to(16);
    gpu.launch_kernel_blob(
        old_func(v),
        [nkv as u32, n as u32, 1],
        [32, 1, 1],
        ((HEAD_DIM + 32) * 4) as u32,
        a.as_mut_slice(),
    )?;
    Ok(())
}

fn run(gpu: &mut Gpu, v: Variant, n: usize, nkv: usize, s1: &GpuTensor, s2: &GpuTensor) -> Res<bool> {
    let (sb, db) = (v.src_bph() * nkv, v.dst_bph() * nkv); // bytes per position
    let src_host = build_src(v, n, nkv, 0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let src = upload_bytes(gpu, &src_host)?;
    drop(src_host);
    let sentinel = vec![0xA5u8; n * db];
    let full = upload_bytes(gpu, &sentinel)?;
    v.launch(gpu, &full, &src, s1, s2, nkv, n)?;
    gpu.hip.device_synchronize()?;
    let mut got = vec![0u8; n * db];
    gpu.hip.memcpy_dtoh(&mut got, &full.buf)?;

    // (3) every record written.
    let bph = v.dst_bph();
    let unwritten = got
        .chunks_exact(bph)
        .enumerate()
        .filter(|(_, rec)| rec[..4] == [0xA5; 4])
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let mut ok = unwritten.is_empty();
    if !ok {
        eprintln!(
            "  UNWRITTEN records: {} (first pos {} head {})",
            unwritten.len(),
            unwritten[0] / nkv,
            unwritten[0] % nkv
        );
    }

    // (2) base-kernel oracle: the unmodified e371200c4 kernel, raw-launched
    // with its original [n_kv_heads, n, 1] grid (one launch while n <= 65535,
    // otherwise <= CHUNK-position slices over shifted pointers, each far below
    // the grid.y limit). Whole dst buffers must be byte-identical.
    {
        let oracle = upload_bytes(gpu, &sentinel)?;
        let step = if n <= 65_535 { n } else { CHUNK };
        let mut p0 = 0;
        while p0 < n {
            let len = step.min(n - p0);
            launch_old(gpu, v, &oracle, &src, s1, s2, nkv, len, p0 * db, p0 * sb)?;
            p0 += len;
        }
        gpu.hip.device_synchronize()?;
        let mut want = vec![0u8; n * db];
        gpu.hip.memcpy_dtoh(&mut want, &oracle.buf)?;
        gpu.free_tensor(oracle)?;
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            eprintln!("  MISMATCH vs base-kernel oracle at byte {i} (pos {})", i / db);
            ok = false;
        }
    }
    println!(
        "{:<22} n={:<7} kv_heads={} fnv1a={:016x} {}",
        v.name(),
        n,
        nkv,
        fnv1a(&got),
        if ok { "ok" } else { "FAIL" }
    );
    gpu.free_tensor(src)?;
    gpu.free_tensor(full)?;
    Ok(ok)
}

fn arg_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let positions = arg_after(&args, "--positions")
        .unwrap_or_else(|| "37,1000,65535,65536,65537,70000,262144".into());
    let nkv: usize = arg_after(&args, "--kv-heads").map_or(Ok(4), |s| s.parse())?;
    let mut gpu = Gpu::init()?;
    println!("arch={} head_dim={HEAD_DIM} chunk={CHUNK}", gpu.arch);

    // +-1 sign tables (256 wide: fwht4->fwht3 and q8->lloyd4 read all 256).
    let mut r = Rng(0xF00D);
    let signs = |r: &mut Rng| -> Vec<f32> {
        (0..256).map(|_| if r.next() & 0x100 != 0 { 1.0 } else { -1.0 }).collect()
    };
    let (a, b) = (signs(&mut r), signs(&mut r));
    let s1 = gpu.upload_f32(&a, &[256])?;
    let s2 = gpu.upload_f32(&b, &[256])?;
    // Pristine base e371200c4 kernels (grid [kv_heads, n, 1]) under unique
    // symbols; compiled exactly like the production givens4 path.
    for v in [Variant::VQ8ToL4, Variant::VDown(4, 3), Variant::KToFwht2, Variant::KToFwht3] {
        compile_old(&mut gpu, v)?;
    }

    let variants = [
        Variant::VQ8ToL4,
        Variant::VDown(4, 3),
        Variant::VDown(4, 2),
        Variant::VDown(3, 2),
        Variant::KToFwht2,
        Variant::KToFwht3,
    ];
    let mut fails = 0;
    for n in positions.split(',') {
        let n: usize = n.trim().parse()?;
        for v in variants {
            if !run(&mut gpu, v, n, nkv, &s1, &s2)? {
                fails += 1;
            }
        }
    }
    if fails != 0 {
        eprintln!("FAIL: {fails} variant/size combinations");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
