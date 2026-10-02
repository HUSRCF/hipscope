//! GPU smoke for the `gemv.rs` token/row-scaled grids that used to put the row
//! count on grid.y / grid.z (HIP bound: 65535 on gfx1201). Two mechanisms:
//!
//!   * FOLD (kernel change): the IU4/A8 sidecar producers and the silu/gated-norm
//!     rotate kernels now read `(row, group)` from `blockIdx.x` as
//!     `x = row * groups + group`. Oracle = the PRISTINE (e371200c4) kernel
//!     source, compiled from `--oracle-dir`, launched with the old
//!     `[groups, rows]` geometry in <= 65535-row slices with shifted pointers
//!     (rows <= 65535 is exactly the old single launch). Sidecar outputs are
//!     compared record-by-record through a layout-aware canonicaliser, so slab
//!     planes that depend on N still compare across different slice widths.
//!     Every shipped kernel variant is built here from the same define sets as
//!     `kernels.rs` (silu plain/awq/IU4/hin/bf16-hin/slab/tokfast/A8, gated-norm
//!     plain/awq x wave-group x xbf16 x slab/A8). The public launchers that
//!     map those kernels (`fused_silu_mul_rotate_mq_batched`,
//!     `..._awq_batched`, `..._awq_indexed_batched`, `..._i4_batched`,
//!     `gated_norm_rotate_mq_i4_gfx1{1,2}_batched`,
//!     `fused_silu_hin_rotate_mq_i8_gfx12_batched`,
//!     `gated_norm_rotate_mq_i8_gfx12_batched`) are run against the same
//!     oracle, which proves the launcher grid and the kernel decode agree.
//!   * CHUNK (launcher change, kernels untouched): `rotate_x_mq_128_v2`,
//!     `..._f16`, `silu_mul_bf16_rt_rotate_x_mq_128_v2_shared`, the three
//!     `gemv_*_residual_sigmoid_scaled_gpu_batched`, and the MoE batched
//!     gate_up / down launchers launch <= 65535 rows per piece with every
//!     per-row pointer advanced. Oracle = the unchanged raw kernel (rotate,
//!     sigmoid) or the same launcher called with <= 65535-row slices (MoE),
//!     which is the pre-change code path.
//!
//! Deterministic (xorshift), tiny K/M so memory stays bounded (<~2 GiB).
//! Not covered (no hardware path here): `fused_rmsnorm_rotate_mq_split_gfx942`
//! (gfx942 only; same pointer-advance loop), MoE paro/mixed need real
//! expert formats and run on random bytes (bitwise compare is still exact).
//!
//! Get the pristine kernels:
//!     mkdir -p /tmp/gf_oracle && for f in fused_silu_mul_mq_rotate.hip \
//!       fused_silu_mul_mq_rotate_awq.hip fused_silu_mul_mq_rotate_awq_indexed.hip \
//!       gated_norm_mq_rotate_quant.gfx12.hip; do \
//!       git -C /home/kaden/ClaudeCode/warpfront/wt-grid-y show e371200c4:kernels/src/$f > /tmp/gf_oracle/$f; done
//! Run (needs the `lab` feature only):
//!     cargo run --release -p rdna-compute --features lab --example gemv_batch_grid_smoke \
//!         -- --oracle-dir /tmp/gf_oracle [--rows 37,1000,65535,65536,70000,131075] [--ks 512,768] [--only silu]
//!
//! Exit status 1 on any mismatch.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const SLICE: usize = 65_535;

const I4_HDR: &str = include_str!("../../../kernels/src/block_i4_128_quant.hip");
const I8_HDR: &str = include_str!("../../../kernels/src/block_i8_128_quant.hip");
const I8_EMIT: &str = include_str!("../../../kernels/src/block_i8_128_emit.hip");
const SILU_NEW: &str = include_str!("../../../kernels/src/fused_silu_mul_mq_rotate.hip");
const SILU_AWQ_NEW: &str = include_str!("../../../kernels/src/fused_silu_mul_mq_rotate_awq.hip");
const GATED_NEW: &str = include_str!("../../../kernels/src/gated_norm_mq_rotate_quant.gfx12.hip");
const ROT128: &str = include_str!("../../../kernels/src/mq_rotate_x_128_v2.hip");
const SIG_Q4: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip");
const SIG_Q4G128: &str =
    include_str!("../../../kernels/src/gemv_hfq4g128_residual_sigmoid_scaled.hip");
const SIG_Q6: &str = include_str!("../../../kernels/src/gemv_hfq6g256_residual_sigmoid_scaled.hip");

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16_777_216.0
    }
}
fn rand_f32(r: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| lo + (hi - lo) * r.unit()).collect()
}
fn rand_bytes(r: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| (r.next() >> 24) as u8).collect()
}
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect()
}
fn up_f32(gpu: &mut Gpu, v: &[f32]) -> Res<GpuTensor> {
    Ok(gpu.upload_f32(v, &[v.len()])?)
}
fn up_u8(gpu: &Gpu, v: &[u8]) -> Res<GpuTensor> {
    Ok(gpu.upload_raw(v, &[v.len()])?)
}
fn zeros_f32(gpu: &mut Gpu, elems: usize, poison: i32) -> Res<GpuTensor> {
    let t = gpu.zeros(&[elems], DType::F32)?;
    gpu.hip.memset(&t.buf, poison, t.byte_size())?;
    Ok(t)
}
fn read_dev(gpu: &Gpu, ptr: *mut c_void, bytes: usize) -> Res<Vec<u8>> {
    let buf = unsafe { DeviceBuffer::from_raw(ptr, bytes) };
    let mut v = vec![0u8; bytes];
    gpu.hip.device_synchronize()?;
    gpu.hip.memcpy_dtoh(&mut v, &buf)?;
    Ok(v)
}
fn read_t(gpu: &Gpu, t: &GpuTensor) -> Res<Vec<u8>> {
    read_dev(gpu, t.buf.as_ptr(), t.byte_size())
}
fn shift(p: *mut c_void, bytes: usize) -> *mut c_void {
    (p as usize + bytes) as *mut c_void
}
fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, a: &mut KernargBlob) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, a.as_mut_slice())?;
    Ok(())
}
fn slices(total: usize) -> Vec<(usize, usize)> {
    (0..total.div_ceil(SLICE))
        .map(|i| (i * SLICE, (total - i * SLICE).min(SLICE)))
        .collect()
}
fn diff(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

// ───────────────────────────── FOLD matrix ─────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Side {
    None,
    I4,
    I4Slab,
    I8,
}
#[derive(Clone, Copy)]
enum Fam {
    Silu { awq: bool, hin: bool, bf16: bool, tokfast: bool },
    Gated { awq: bool, xbf16: bool, wave: bool },
}
#[derive(Clone)]
struct V {
    name: String,
    fam: Fam,
    side: Side,
    only_1201: bool,
}

fn variants() -> Vec<V> {
    let s = |name: &str, awq, hin, bf16, tokfast, side, only_1201| V {
        name: name.to_string(),
        fam: Fam::Silu { awq, hin, bf16, tokfast },
        side,
        only_1201,
    };
    let mut v = vec![
        s("silu", false, false, false, false, Side::None, false),
        s("silu_awq", true, false, false, false, Side::None, false),
        s("silu_i4", false, false, false, false, Side::I4, false),
        s("silu_awq_i4", true, false, false, false, Side::I4, false),
        s("silu_hin_i4", true, true, false, false, Side::I4, false),
        s("silu_hin_bf16_i4", true, true, true, false, Side::I4, false),
        s("silu_hin_slab", true, true, false, false, Side::I4Slab, true),
        s("silu_hin_slab_tokfast", true, true, false, true, Side::I4Slab, true),
        s("silu_hin_i8", true, true, false, false, Side::I8, true),
    ];
    for awq in [false, true] {
        for wave in [false, true] {
            for xbf16 in [false, true] {
                v.push(V {
                    name: format!(
                        "gated{}{}{}_i4",
                        if awq { "_awq" } else { "" },
                        if wave { "_wave" } else { "" },
                        if xbf16 { "_xbf16" } else { "" }
                    ),
                    fam: Fam::Gated { awq, xbf16, wave },
                    side: Side::I4,
                    only_1201: false,
                });
            }
        }
    }
    for xbf16 in [false, true] {
        v.push(V {
            name: format!("gated_awq_wave{}_slab", if xbf16 { "_xbf16" } else { "" }),
            fam: Fam::Gated { awq: true, xbf16, wave: true },
            side: Side::I4Slab,
            only_1201: true,
        });
    }
    for awq in [false, true] {
        v.push(V {
            name: format!("gated{}_wave_i8", if awq { "_awq" } else { "" }),
            fam: Fam::Gated { awq, xbf16: false, wave: true },
            side: Side::I8,
            only_1201: true,
        });
    }
    v
}

fn oracle_file(v: &V) -> (&'static str, &'static str) {
    match v.fam {
        Fam::Silu { awq, hin, .. } if awq || hin => {
            ("fused_silu_mul_mq_rotate_awq.hip", SILU_AWQ_NEW)
        }
        Fam::Silu { .. } => ("fused_silu_mul_mq_rotate.hip", SILU_NEW),
        Fam::Gated { .. } => ("gated_norm_mq_rotate_quant.gfx12.hip", GATED_NEW),
    }
}

/// Source text built exactly like the `kernels.rs` consts, with `body` as the
/// producer file (pristine for the oracle, worktree for the folded kernel).
fn build_src(v: &V, body: &str, entry: &str) -> String {
    let mut s = String::new();
    match v.side {
        Side::None => {}
        Side::I4 => s += "#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n",
        Side::I4Slab => {
            s += "#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n#define HIPFIRE_IU4_SLAB 1\n"
        }
        Side::I8 => s += "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n",
    }
    match v.side {
        Side::None => {}
        Side::I4 | Side::I4Slab => s += I4_HDR,
        Side::I8 => {
            s += I8_HDR;
            s += I8_EMIT;
        }
    }
    let macro_name = match v.fam {
        Fam::Silu { awq: _, hin, bf16, tokfast } => {
            if v.side != Side::None {
                s += "#define HIPFIRE_IU4_SIDECAR 1\n";
            }
            if hin {
                s += "#define HIPFIRE_SILU_HIN 1\n";
            }
            if bf16 {
                s += "#define HIPFIRE_SILU_H_BF16 1\n";
            }
            if tokfast {
                s += "#define HIPFIRE_GRID_TOKFAST 1\n";
            }
            "HIPFIRE_SILU_MQ_ROTATE_KERNEL"
        }
        Fam::Gated { awq, xbf16, wave } => {
            if xbf16 {
                s += "#define HIPFIRE_GATED_NORM_X_BF16 1\n";
            }
            if awq {
                s += "#define HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ 1\n";
            }
            if wave {
                s += "#define HIPFIRE_GATED_NORM_WAVE_GROUP 1\n";
            }
            "HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL"
        }
    };
    s += &format!("#define {macro_name} {entry}\n");
    s += body;
    s
}

fn rec_bytes(v: &V) -> usize {
    if v.side == Side::I8 { 136 } else { 72 }
}

/// One `block_i{4,8}_128` record in canonical `[d][s][qs..]` order.
fn canon(v: &V, buf: &[u8], n: usize, kb: usize, t: usize) -> Vec<u8> {
    let rec = rec_bytes(v);
    if v.side != Side::I4Slab {
        return buf[(kb * n + t) * rec..(kb * n + t) * rec + rec].to_vec();
    }
    let base = kb * n * 72;
    let mut r = Vec::with_capacity(72);
    r.extend_from_slice(&buf[base + t * 4..base + t * 4 + 4]);
    r.extend_from_slice(&buf[base + 4 * n + t * 4..base + 4 * n + t * 4 + 4]);
    r.extend_from_slice(&buf[base + 8 * n + t * 32..base + 8 * n + t * 32 + 32]);
    r.extend_from_slice(&buf[base + 40 * n + t * 32..base + 40 * n + t * 32 + 32]);
    r
}

struct In {
    a: GpuTensor,       // gate | h | x
    b: Option<GpuTensor>, // up | z
    w: Option<GpuTensor>, // gated norm weight [128]
    awq: Option<GpuTensor>,
    a_es: usize,
}

fn make_in(gpu: &mut Gpu, v: &V, n: usize, k: usize, seed: u64) -> Res<In> {
    let mut r = Rng(seed | 1);
    let big = rand_f32(&mut r, n * k, -4.0, 4.0);
    let awq = up_f32(gpu, &rand_f32(&mut r, k, 0.5, 1.5))?;
    Ok(match v.fam {
        Fam::Silu { awq: has_awq, hin, bf16, .. } => {
            let a = if hin && bf16 {
                up_u8(gpu, &bf16_bytes(&big))?
            } else {
                up_f32(gpu, &big)?
            };
            let b = if hin { None } else { Some(up_f32(gpu, &rand_f32(&mut r, n * k, -4.0, 4.0))?) };
            In {
                a,
                b,
                w: None,
                awq: (has_awq || hin).then_some(awq),
                a_es: if hin && bf16 { 2 } else { 4 },
            }
        }
        Fam::Gated { awq: has_awq, xbf16, .. } => {
            let a = if xbf16 { up_u8(gpu, &bf16_bytes(&big))? } else { up_f32(gpu, &big)? };
            let z = up_f32(gpu, &rand_f32(&mut r, n * k, -4.0, 4.0))?;
            let w = up_f32(gpu, &rand_f32(&mut r, 128, 0.5, 1.5))?;
            In { a, b: Some(z), w: Some(w), awq: has_awq.then_some(awq), a_es: if xbf16 { 2 } else { 4 } }
        }
    })
}

fn signs(gpu: &mut Gpu) -> Res<(*mut c_void, *mut c_void)> {
    gpu.ensure_mq_signs()?;
    Ok((
        gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr(),
        gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr(),
    ))
}

/// One raw launch of `entry` over rows `[off, off+nc)` of `inp`. `folded`
/// selects the new `[groups*nc]` grid (else the pristine `[groups, nc]`).
#[allow(clippy::too_many_arguments)]
fn raw_launch(
    gpu: &mut Gpu,
    v: &V,
    entry: &str,
    inp: &In,
    k: usize,
    off: usize,
    nc: usize,
    folded: bool,
    xrot: *mut c_void,
    side: *mut c_void,
    side_n: usize,
) -> Res<()> {
    let (s1, s2) = signs(gpu)?;
    let g = k / 256;
    let a_p = shift(inp.a.buf.as_ptr(), off * k * inp.a_es);
    let xr_p = shift(xrot, off * k * 4);
    let mut b = KernargBlob::new();
    let (grid, block) = match v.fam {
        Fam::Silu { awq, hin, tokfast, .. } => {
            b.push_ptr(a_p);
            if !hin {
                b.push_ptr(shift(inp.b.as_ref().unwrap().buf.as_ptr(), off * k * 4));
            }
            if awq || hin {
                b.push_ptr(inp.awq.as_ref().unwrap().buf.as_ptr());
            }
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_ptr(xr_p);
            if v.side != Side::None {
                b.push_ptr(side);
            }
            b.push_i32(k as i32);
            if v.side != Side::None {
                b.push_i32(side_n as i32);
            }
            let grid = if tokfast {
                [nc as u32, g as u32, 1]
            } else if folded {
                [(g * nc) as u32, 1, 1]
            } else {
                [g as u32, nc as u32, 1]
            };
            (grid, 32)
        }
        Fam::Gated { awq, wave, .. } => {
            b.push_ptr(a_p);
            b.push_ptr(shift(inp.b.as_ref().unwrap().buf.as_ptr(), off * k * 4));
            b.push_ptr(inp.w.as_ref().unwrap().buf.as_ptr());
            if awq {
                b.push_ptr(inp.awq.as_ref().unwrap().buf.as_ptr());
            }
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_ptr(xr_p);
            b.push_ptr(side);
            b.push_i32((k / 128) as i32);
            b.push_i32(128);
            b.push_f32(1e-6);
            b.push_i32(k as i32);
            b.push_i32(side_n as i32);
            let gx = if wave { g.div_ceil(2) } else { g };
            let grid = if folded { [(gx * nc) as u32, 1, 1] } else { [gx as u32, nc as u32, 1] };
            (grid, 64)
        }
    };
    launch(gpu, entry, grid, block, &mut b)
}

struct OldOut {
    xrot: Vec<u8>,
    side: Vec<(usize, usize, Vec<u8>)>, // (off, nc, chunk sidecar with N = nc)
}

fn run_old(gpu: &mut Gpu, v: &V, entry: &str, inp: &In, k: usize, n: usize) -> Res<OldOut> {
    let xrot = zeros_f32(gpu, n * k, 0x5A)?;
    let mut side = Vec::new();
    for (off, nc) in slices(n) {
        let sb = rec_bytes(v) * (k / 128) * nc;
        let sidebuf = zeros_f32(gpu, sb / 4, 0x7B)?;
        raw_launch(gpu, v, entry, inp, k, off, nc, false, xrot.buf.as_ptr(), sidebuf.buf.as_ptr(), nc)?;
        side.push((off, nc, read_t(gpu, &sidebuf)?));
        gpu.free_tensor(sidebuf)?;
    }
    let out = OldOut { xrot: read_t(gpu, &xrot)?, side };
    gpu.free_tensor(xrot)?;
    Ok(out)
}

fn compare(
    v: &V,
    label: &str,
    k: usize,
    n: usize,
    new_x: &[u8],
    new_side: Option<&[u8]>,
    old: &OldOut,
) -> usize {
    let xbad = diff(new_x, &old.xrot);
    let mut sbad = 0usize;
    if v.side != Side::None {
        if let Some(ns) = new_side {
            for (off, nc, ob) in &old.side {
                for kb in 0..k / 128 {
                    for t in 0..*nc {
                        if canon(v, ns, n, kb, off + t) != canon(v, ob, *nc, kb, t) {
                            sbad += 1;
                        }
                    }
                }
            }
        }
    }
    println!("[{label}] {} n={n} k={k} xrot_bad_bytes={xbad} sidecar_bad_records={sbad}", v.name);
    xbad + sbad
}

/// The public launcher for `v`, when one maps to this kernel variant.
fn run_api(
    gpu: &mut Gpu,
    v: &V,
    inp: &In,
    xrot: &GpuTensor,
    k: usize,
    n: usize,
) -> Res<Option<Option<Vec<u8>>>> {
    let a = &inp.a;
    let sb = rec_bytes(v) * (k / 128) * n;
    let is1201 = gpu.arch == "gfx1201";
    Ok(Some(match (v.fam, v.side) {
        (Fam::Silu { awq: false, hin: false, .. }, Side::None) => {
            gpu.fused_silu_mul_rotate_mq_batched(a, inp.b.as_ref().unwrap(), xrot, k, n)?;
            None
        }
        (Fam::Silu { awq: true, hin: false, .. }, Side::None) => {
            gpu.fused_silu_mul_rotate_mq_awq_batched(
                a, inp.b.as_ref().unwrap(), inp.awq.as_ref().unwrap(), xrot, k, n)?;
            None
        }
        (Fam::Silu { awq, hin: false, .. }, Side::I4) => {
            let b = inp.b.as_ref().unwrap();
            let aw = if awq { inp.awq.as_ref() } else { None };
            let res = gpu.reserve_int4_mmq(k, n)?;
            let p = res.ptr();
            gpu.hip.memset(&unsafe { DeviceBuffer::from_raw(p, sb) }, 0x7B, sb)?;
            if is1201 {
                gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(a, b, aw, Some(xrot), res, k, n)?;
            } else {
                gpu.fused_silu_mul_rotate_mq_i4_batched(a, b, aw, Some(xrot), res, k, n)?;
            }
            Some(read_dev(gpu, p, sb)?)
        }
        (Fam::Silu { hin: true, bf16: false, tokfast: false, .. }, Side::I8) => {
            let res = gpu.reserve_int8_mmq(k, n)?;
            let p = res.ptr();
            gpu.hip.memset(&unsafe { DeviceBuffer::from_raw(p, sb) }, 0x7B, sb)?;
            gpu.fused_silu_hin_rotate_mq_i8_gfx12_batched(a, inp.awq.as_ref().unwrap(), res, k, n)?;
            Some(read_dev(gpu, p, sb)?)
        }
        (Fam::Gated { awq: false, xbf16: false, wave: false }, Side::I4) => {
            let (z, w) = (inp.b.as_ref().unwrap(), inp.w.as_ref().unwrap());
            let res = gpu.reserve_int4_mmq(k, n)?;
            let p = res.ptr();
            gpu.hip.memset(&unsafe { DeviceBuffer::from_raw(p, sb) }, 0x7B, sb)?;
            if is1201 {
                gpu.gated_norm_rotate_mq_i4_gfx12_batched(
                    a, GdnScanOut::F32, z, w, None, Some(xrot), res, k / 128, 128, 1e-6, k, n)?;
            } else {
                gpu.gated_norm_rotate_mq_i4_gfx11_batched(
                    a, z, w, None, Some(xrot), res, k / 128, 128, 1e-6, k, n)?;
            }
            Some(read_dev(gpu, p, sb)?)
        }
        (Fam::Gated { awq, xbf16: false, wave: true }, Side::I8) => {
            let res = gpu.reserve_int8_mmq(k, n)?;
            let p = res.ptr();
            gpu.hip.memset(&unsafe { DeviceBuffer::from_raw(p, sb) }, 0x7B, sb)?;
            let aw = if awq { inp.awq.as_ref() } else { None };
            gpu.gated_norm_rotate_mq_i8_gfx12_batched(
                a, inp.b.as_ref().unwrap(), inp.w.as_ref().unwrap(), aw, Some(xrot), res,
                k / 128, 128, 1e-6, k, n)?;
            Some(read_dev(gpu, p, sb)?)
        }
        _ => return Ok(None),
    }))
}

fn run_fold(gpu: &mut Gpu, oracle_dir: &str, rows: &[usize], ks: &[usize], only: &str) -> Res<usize> {
    let mut bad = 0;
    for v in variants() {
        if !only.is_empty() && !v.name.contains(only) {
            continue;
        }
        if v.only_1201 && gpu.arch != "gfx1201" {
            println!("[skip] {} (gfx1201 only)", v.name);
            continue;
        }
        let (file, new_body) = oracle_file(&v);
        let old_body = std::fs::read_to_string(format!("{oracle_dir}/{file}"))?;
        let (old_entry, new_entry) = (format!("gf_old_{}", v.name), format!("gf_new_{}", v.name));
        gpu.ensure_kernel_public(&old_entry, &build_src(&v, &old_body, &old_entry), &old_entry)?;
        gpu.ensure_kernel_public(&new_entry, &build_src(&v, new_body, &new_entry), &new_entry)?;
        for &k in ks {
            if matches!(v.fam, Fam::Gated { .. }) && k % 256 != 0 {
                continue;
            }
            for &n in rows {
                let inp = make_in(gpu, &v, n, k, 0x9E37_79B9_7F4A_7C15 ^ (n as u64) ^ ((k as u64) << 32))?;
                let old = run_old(gpu, &v, &old_entry, &inp, k, n)?;
                // New kernel, raw launch with the folded grid (kernel decode).
                let xrot = zeros_f32(gpu, n * k, 0x5A)?;
                let sb = rec_bytes(&v) * (k / 128) * n;
                let side = zeros_f32(gpu, sb / 4, 0x7B)?;
                raw_launch(gpu, &v, &new_entry, &inp, k, 0, n, true, xrot.buf.as_ptr(), side.buf.as_ptr(), n)?;
                let nx = read_t(gpu, &xrot)?;
                let ns = read_t(gpu, &side)?;
                bad += compare(&v, "raw-new", k, n, &nx, Some(&ns), &old);
                // Production launcher (launcher grid <-> kernel decode).
                let xrot2 = zeros_f32(gpu, n * k, 0x5A)?;
                match run_api(gpu, &v, &inp, &xrot2, k, n) {
                    Ok(Some(side_bytes)) => {
                        let nx2 = read_t(gpu, &xrot2)?;
                        // Sidecar-only kernels (hin) do not write x_rot; compare it only when written.
                        let wrote_x = !matches!(v.fam, Fam::Silu { hin: true, .. });
                        let nx2c = if wrote_x { nx2 } else { old.xrot.clone() };
                        bad += compare(&v, "api", k, n, &nx2c, side_bytes.as_deref(), &old);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        println!("[api] {} n={n} k={k} launcher error: {e}", v.name);
                        bad += 1;
                    }
                }
                for t in [xrot, side, xrot2] {
                    gpu.free_tensor(t)?;
                }
                gpu.free_tensor(inp.a)?;
                for t in [inp.b, inp.w, inp.awq].into_iter().flatten() {
                    gpu.free_tensor(t)?;
                }
            }
        }
    }
    Ok(bad)
}

/// `fused_silu_mul_rotate_mq_awq_indexed_batched`: row r reads topk[r], so the
/// oracle shifts the index pointer together with gate/up/x_rot.
fn run_indexed(gpu: &mut Gpu, oracle_dir: &str, rows: &[usize], ks: &[usize]) -> Res<usize> {
    let old_body = std::fs::read_to_string(format!("{oracle_dir}/fused_silu_mul_mq_rotate_awq_indexed.hip"))?;
    gpu.ensure_kernel_public("gf_old_idx", &old_body.replace("fused_silu_mul_mq_rotate_awq_indexed", "gf_old_idx"), "gf_old_idx")?;
    let mut bad = 0;
    const NEXP: usize = 4;
    for &k in ks {
        for &n in rows {
            let mut r = Rng(0xA5A5 ^ n as u64 ^ ((k as u64) << 20) | 1);
            let gate = up_f32(gpu, &rand_f32(&mut r, n * k, -4.0, 4.0))?;
            let up = up_f32(gpu, &rand_f32(&mut r, n * k, -4.0, 4.0))?;
            let mut awqs = Vec::new();
            let mut table = Vec::new();
            for _ in 0..NEXP {
                let t = up_f32(gpu, &rand_f32(&mut r, k, 0.5, 1.5))?;
                table.extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
                awqs.push(t);
            }
            let table_t = up_u8(gpu, &table)?;
            let idx: Vec<u8> = (0..n).map(|t| ((t * 3 + 1) % NEXP) as i32).flat_map(|x| x.to_le_bytes()).collect();
            let idx_t = up_u8(gpu, &idx)?;
            let (s1, s2) = signs(gpu)?;
            // Oracle.
            let want = zeros_f32(gpu, n * k, 0x5A)?;
            for (off, nc) in slices(n) {
                let mut b = KernargBlob::new();
                b.push_ptr(shift(gate.buf.as_ptr(), off * k * 4));
                b.push_ptr(shift(up.buf.as_ptr(), off * k * 4));
                b.push_ptr(table_t.buf.as_ptr());
                b.push_ptr(shift(idx_t.buf.as_ptr(), off * 4));
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(shift(want.buf.as_ptr(), off * k * 4));
                b.push_i32(k as i32);
                launch(gpu, "gf_old_idx", [(k / 256) as u32, nc as u32, 1], 32, &mut b)?;
            }
            let got = zeros_f32(gpu, n * k, 0x5A)?;
            gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(&gate, &up, &table_t, &idx_t, &got, k, n)?;
            let d = diff(&read_t(gpu, &got)?, &read_t(gpu, &want)?);
            println!("[api] silu_awq_indexed n={n} k={k} bad_bytes={d}");
            bad += d;
            for t in [gate, up, table_t, idx_t, want, got] {
                gpu.free_tensor(t)?;
            }
            for t in awqs {
                gpu.free_tensor(t)?;
            }
        }
    }
    Ok(bad)
}

// ───────────────────────────── CHUNK launchers ─────────────────────────────

fn run_rotate128(gpu: &mut Gpu, rows: &[usize]) -> Res<usize> {
    const K: usize = 640;
    gpu.ensure_kernel_public("gf_rot128", ROT128, "mq_rotate_x_128_v2")?;
    gpu.ensure_kernel_public("gf_rot128", ROT128, "mq_rotate_x_128_v2_f16")?;
    gpu.ensure_kernel_public("gf_rot128", ROT128, "mq_rotate_x_128_v2_silu_bf16")?;
    gpu.ensure_mq_signs_128()?;
    let s1 = gpu.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
    let s2 = gpu.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
    let mut bad = 0;
    for &n in rows {
        let mut r = Rng(0xC0DE ^ n as u64 | 1);
        let x = up_f32(gpu, &rand_f32(&mut r, n * K, -4.0, 4.0))?;
        let up = up_f32(gpu, &rand_f32(&mut r, n * K, -4.0, 4.0))?;
        // rotate_x_mq_128_v2
        let got = zeros_f32(gpu, n * K, 0x5A)?;
        gpu.rotate_x_mq_128_v2(&x, &got, K, n)?;
        let want = zeros_f32(gpu, n * K, 0x5A)?;
        for (off, nc) in slices(n) {
            let mut b = KernargBlob::new();
            b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
            b.push_ptr(shift(want.buf.as_ptr(), off * K * 4));
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_i32(K as i32);
            launch(gpu, "mq_rotate_x_128_v2", [(K / 128) as u32, nc as u32, 1], 32, &mut b)?;
        }
        let d = diff(&read_t(gpu, &got)?, &read_t(gpu, &want)?);
        println!("[chunk] rotate_x_mq_128_v2 n={n} bad_bytes={d}");
        bad += d;
        // _f16 (returns a view of the shared FP16 scratch: read it immediately)
        let h = gpu.rotate_x_mq_128_v2_f16(&x, K, n)?;
        let got16 = read_dev(gpu, h.buf.as_ptr(), n * K * 2)?;
        let want16 = gpu.zeros(&[n * K], DType::F16)?;
        for (off, nc) in slices(n) {
            let mut b = KernargBlob::new();
            b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
            b.push_ptr(shift(want16.buf.as_ptr(), off * K * 2));
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_i32(K as i32);
            launch(gpu, "mq_rotate_x_128_v2_f16", [(K / 128) as u32, nc as u32, 1], 32, &mut b)?;
        }
        let d = diff(&got16, &read_t(gpu, &want16)?);
        println!("[chunk] rotate_x_mq_128_v2_f16 n={n} bad_bytes={d}");
        bad += d;
        // silu_bf16 shared: rows = n (+1 shared tail). Small n: single old launch of n+1 rows;
        // large n: <=65535-row slices + one-row shared launch.
        let sn = K;
        let sg = up_f32(gpu, &rand_f32(&mut r, sn, -2.0, 2.0))?;
        let su = up_f32(gpu, &rand_f32(&mut r, sn, -2.0, 2.0))?;
        let sel_init = rand_f32(&mut r, 1, -1.0, 1.0);
        for shared in [false, true] {
            let sel_a = up_f32(gpu, &sel_init)?;
            let sel_b = up_f32(gpu, &sel_init)?;
            let so_a = zeros_f32(gpu, sn, 0x5A)?;
            let so_b = zeros_f32(gpu, sn, 0x5A)?;
            let got = zeros_f32(gpu, n * K, 0x5A)?;
            let want = zeros_f32(gpu, n * K, 0x5A)?;
            gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
                &x, &up, &got, K, n, shared.then_some((&sg, &su, &so_a, &sel_a)))?;
            let null = std::ptr::null_mut();
            let launch_rows = |gpu: &Gpu, off: usize, nc: usize, sh: bool, rows_y: usize| -> Res<()> {
                let mut b = KernargBlob::new();
                b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(up.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(want.buf.as_ptr(), off * K * 4));
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(K as i32);
                b.push_ptr(if sh { sg.buf.as_ptr() } else { null });
                b.push_ptr(if sh { su.buf.as_ptr() } else { null });
                b.push_ptr(if sh { so_b.buf.as_ptr() } else { null });
                b.push_ptr(if sh { sel_b.buf.as_ptr() } else { null });
                b.push_i32(if sh { sn as i32 } else { 0 });
                let _ = nc;
                launch(gpu, "mq_rotate_x_128_v2_silu_bf16", [(K / 128) as u32, rows_y as u32, 1], 32, &mut b)
            };
            if n + usize::from(shared) <= SLICE {
                launch_rows(gpu, 0, n, shared, n + usize::from(shared))?;
            } else {
                for (off, nc) in slices(n) {
                    launch_rows(gpu, off, nc, false, nc)?;
                }
                if shared {
                    launch_rows(gpu, 0, 0, true, 1)?;
                }
            }
            let mut d = diff(&read_t(gpu, &got)?, &read_t(gpu, &want)?);
            d += diff(&read_t(gpu, &so_a)?, &read_t(gpu, &so_b)?);
            d += diff(&read_t(gpu, &sel_a)?, &read_t(gpu, &sel_b)?);
            println!("[chunk] silu_mul_bf16_rt_rotate_x_mq_128_v2_shared shared={shared} n={n} bad_bytes={d}");
            bad += d;
            for t in [sel_a, sel_b, so_a, so_b, got, want] {
                gpu.free_tensor(t)?;
            }
        }
        for t in [x, up, got, want, sg, su, want16] {
            gpu.free_tensor(t)?;
        }
    }
    Ok(bad)
}

fn run_sigmoid(gpu: &mut Gpu, rows: &[usize]) -> Res<usize> {
    const M: usize = 8;
    const K: usize = 256;
    type F = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize) -> hip_bridge::HipResult<()>;
    let cases: [(&str, &str, &str, F); 3] = [
        ("hfq4g256", SIG_Q4, "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            Gpu::gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched),
        ("hfq4g128", SIG_Q4G128, "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
            Gpu::gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched),
        ("hfq6g256", SIG_Q6, "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
            Gpu::gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched),
    ];
    let mut bad = 0;
    for (name, src, func, f) in cases {
        gpu.ensure_kernel_public(&format!("gf_sig_{name}"), src, func)?;
        for &n in rows {
            let mut r = Rng(0x51 ^ n as u64 | 1);
            let a = up_u8(gpu, &rand_bytes(&mut r, M * 1024))?;
            let x = up_f32(gpu, &rand_f32(&mut r, n * K, -2.0, 2.0))?;
            let c = up_f32(gpu, &rand_f32(&mut r, n, -3.0, 3.0))?;
            let y0 = rand_f32(&mut r, n * M, -1.0, 1.0);
            let got = up_f32(gpu, &y0)?;
            let want = up_f32(gpu, &y0)?;
            f(gpu, &a, &x, &got, &c, M, K, n)?;
            for (off, nc) in slices(n) {
                let mut b = KernargBlob::new();
                b.push_ptr(a.buf.as_ptr());
                b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
                b.push_ptr(shift(want.buf.as_ptr(), off * M * 4));
                b.push_ptr(shift(c.buf.as_ptr(), off * 4));
                b.push_i32(M as i32);
                b.push_i32(K as i32);
                launch(gpu, func, [M as u32, nc as u32, 1], 32, &mut b)?;
            }
            let d = diff(&read_t(gpu, &got)?, &read_t(gpu, &want)?);
            println!("[chunk] {func} n={n} bad_bytes={d}");
            bad += d;
            for t in [a, x, c, got, want] {
                gpu.free_tensor(t)?;
            }
        }
    }
    Ok(bad)
}

struct Moe {
    ptrs: GpuTensor,
    tags: GpuTensor,
    _experts: Vec<GpuTensor>,
}

const NEXP: usize = 16;
const MM: usize = 64; // gate_up M (mi = 32) and down M
const KK: usize = 256;

fn moe_setup(gpu: &mut Gpu) -> Res<Moe> {
    let mut r = Rng(0xE0E0_0001);
    let mut experts = Vec::new();
    let mut table = Vec::new();
    for _ in 0..NEXP {
        // 64 KiB of random bytes covers 64 rows x k=256 for every packed format.
        let t = up_u8(gpu, &rand_bytes(&mut r, 64 * 1024))?;
        table.extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
        experts.push(t);
    }
    Ok(Moe { ptrs: up_u8(gpu, &table)?, tags: up_u8(gpu, &vec![0u8; NEXP])?, _experts: experts })
}

fn topk(gpu: &Gpu, n: usize, kt: usize) -> Res<GpuTensor> {
    let b: Vec<u8> = (0..n * kt)
        .map(|i| (((i / kt) * 5 + (i % kt) * 3) % NEXP) as i32)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    up_u8(gpu, &b)
}

type GateUp = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize, usize) -> hip_bridge::HipResult<()>;
type GateUpTags = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize, usize) -> hip_bridge::HipResult<()>;
type Down = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize, usize) -> hip_bridge::HipResult<()>;
type DownTags = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize, usize) -> hip_bridge::HipResult<()>;
type DownRes = fn(&mut Gpu, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor, usize, usize, usize, usize) -> hip_bridge::HipResult<()>;

fn run_moe(gpu: &mut Gpu, rows: &[usize]) -> Res<usize> {
    let moe = moe_setup(gpu)?;
    let mi = MM / 2;
    let gate_up: [(&str, GateUp); 7] = [
        ("hfq4", Gpu::gemv_hfq4g256_moe_gate_up_k8_indexed_batched),
        ("mq4v2", Gpu::gemv_mq4g256v2_moe_gate_up_k8_indexed_batched),
        ("mq6v2", Gpu::gemv_mq6g256v2_moe_gate_up_k8_indexed_batched),
        ("paro", Gpu::gemv_paro_q4g128_moe_gate_up_k8_indexed_batched),
        ("hfq6", Gpu::gemv_hfq6g256_moe_gate_up_k8_indexed_batched),
        ("hfq5", Gpu::gemv_hfq5g256_moe_gate_up_k8_indexed_batched),
        ("mfp4e8", Gpu::gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched),
    ];
    let down: [(&str, Down); 7] = [
        ("hfq4", Gpu::gemv_hfq4g256_moe_down_k8_indexed_batched_expanded),
        ("mq4v2", Gpu::gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded),
        ("mq6v2", Gpu::gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded),
        ("paro", Gpu::gemv_paro_q4g128_moe_down_k8_indexed_batched),
        ("hfq6", Gpu::gemv_hfq6g256_moe_down_k8_indexed_batched_expanded),
        ("hfq5", Gpu::gemv_hfq5g256_moe_down_k8_indexed_batched_expanded),
        ("mfp4e8", Gpu::gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded),
    ];
    let gate_up_mixed: GateUpTags = Gpu::gemv_mixed_moe_gate_up_k8_indexed_batched;
    let down_mixed: DownTags = Gpu::gemv_mixed_moe_down_k8_indexed_batched_expanded;
    let down_res: DownRes = Gpu::gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched;
    let mut bad = 0;
    let kt = 2usize;
    for &n in rows {
        let mut r = Rng(0xBEEF ^ n as u64 | 1);
        let idx = topk(gpu, n, kt)?;
        let x = up_f32(gpu, &rand_f32(&mut r, n * KK, -2.0, 2.0))?;
        let rot = up_f32(gpu, &rand_f32(&mut r, n * kt * KK, -2.0, 2.0))?;
        for (name, f) in gate_up {
            let yg = zeros_f32(gpu, n * kt * mi, 0x5A)?;
            let yu = zeros_f32(gpu, n * kt * mi, 0x5A)?;
            f(gpu, &moe.ptrs, &idx, &x, &yg, &yu, MM, KK, kt, n)?;
            let (wg, wu) = (zeros_f32(gpu, n * kt * mi, 0x5A)?, zeros_f32(gpu, n * kt * mi, 0x5A)?);
            for (off, nc) in slices(n) {
                f(gpu, &moe.ptrs, &idx.sub_offset(off * kt, nc * kt), &x.sub_offset(off * KK, nc * KK),
                  &wg.sub_offset(off * kt * mi, nc * kt * mi), &wu.sub_offset(off * kt * mi, nc * kt * mi),
                  MM, KK, kt, nc)?;
            }
            let d = diff(&read_t(gpu, &yg)?, &read_t(gpu, &wg)?) + diff(&read_t(gpu, &yu)?, &read_t(gpu, &wu)?);
            println!("[chunk] moe gate_up_{name} n={n} bad_bytes={d}");
            bad += d;
            for t in [yg, yu, wg, wu] { gpu.free_tensor(t)?; }
        }
        {
            let yg = zeros_f32(gpu, n * kt * mi, 0x5A)?;
            let yu = zeros_f32(gpu, n * kt * mi, 0x5A)?;
            gate_up_mixed(gpu, &moe.ptrs, &moe.tags, &idx, &x, &yg, &yu, MM, KK, kt, n)?;
            let (wg, wu) = (zeros_f32(gpu, n * kt * mi, 0x5A)?, zeros_f32(gpu, n * kt * mi, 0x5A)?);
            for (off, nc) in slices(n) {
                gate_up_mixed(gpu, &moe.ptrs, &moe.tags, &idx.sub_offset(off * kt, nc * kt), &x.sub_offset(off * KK, nc * KK),
                  &wg.sub_offset(off * kt * mi, nc * kt * mi), &wu.sub_offset(off * kt * mi, nc * kt * mi),
                  MM, KK, kt, nc)?;
            }
            let d = diff(&read_t(gpu, &yg)?, &read_t(gpu, &wg)?) + diff(&read_t(gpu, &yu)?, &read_t(gpu, &wu)?);
            println!("[chunk] moe gate_up_mixed n={n} bad_bytes={d}");
            bad += d;
            for t in [yg, yu, wg, wu] { gpu.free_tensor(t)?; }
        }
        for (name, f) in down {
            let out = zeros_f32(gpu, n * kt * MM, 0x5A)?;
            f(gpu, &moe.ptrs, &idx, &rot, &out, MM, KK, kt, n)?;
            let want = zeros_f32(gpu, n * kt * MM, 0x5A)?;
            for (off, nc) in slices(n) {
                f(gpu, &moe.ptrs, &idx.sub_offset(off * kt, nc * kt), &rot.sub_offset(off * kt * KK, nc * kt * KK),
                  &want.sub_offset(off * kt * MM, nc * kt * MM), MM, KK, kt, nc)?;
            }
            let d = diff(&read_t(gpu, &out)?, &read_t(gpu, &want)?);
            println!("[chunk] moe down_{name} n={n} bad_bytes={d}");
            bad += d;
            for t in [out, want] { gpu.free_tensor(t)?; }
        }
        {
            let out = zeros_f32(gpu, n * kt * MM, 0x5A)?;
            down_mixed(gpu, &moe.ptrs, &moe.tags, &idx, &rot, &out, MM, KK, kt, n)?;
            let want = zeros_f32(gpu, n * kt * MM, 0x5A)?;
            for (off, nc) in slices(n) {
                down_mixed(gpu, &moe.ptrs, &moe.tags, &idx.sub_offset(off * kt, nc * kt), &rot.sub_offset(off * kt * KK, nc * kt * KK),
                  &want.sub_offset(off * kt * MM, nc * kt * MM), MM, KK, kt, nc)?;
            }
            let d = diff(&read_t(gpu, &out)?, &read_t(gpu, &want)?);
            println!("[chunk] moe down_mixed n={n} bad_bytes={d}");
            bad += d;
            for t in [out, want] { gpu.free_tensor(t)?; }
        }
        {
            // atomicAdd across K_TOP blocks is order-dependent: one expert per token.
            let kt1 = 1usize;
            let idx1 = topk(gpu, n, kt1)?;
            let rot1 = up_f32(gpu, &rand_f32(&mut r, n * kt1 * KK, -2.0, 2.0))?;
            let w1 = up_f32(gpu, &rand_f32(&mut r, n * kt1, 0.0, 1.0))?;
            let y0 = rand_f32(&mut r, n * MM, -1.0, 1.0);
            let got = up_f32(gpu, &y0)?;
            let want = up_f32(gpu, &y0)?;
            down_res(gpu, &moe.ptrs, &idx1, &w1, &rot1, &got, MM, KK, kt1, n)?;
            for (off, nc) in slices(n) {
                down_res(gpu, &moe.ptrs, &idx1.sub_offset(off * kt1, nc * kt1), &w1.sub_offset(off * kt1, nc * kt1),
                  &rot1.sub_offset(off * kt1 * KK, nc * kt1 * KK), &want.sub_offset(off * MM, nc * MM),
                  MM, KK, kt1, nc)?;
            }
            let d = diff(&read_t(gpu, &got)?, &read_t(gpu, &want)?);
            println!("[chunk] moe down_residual_scaled_hfq4 n={n} bad_bytes={d}");
            bad += d;
            for t in [idx1, rot1, w1, got, want] { gpu.free_tensor(t)?; }
        }
        for t in [idx, x, rot] { gpu.free_tensor(t)?; }
    }
    Ok(bad)
}

fn list(args: &[String], flag: &str, default: &str) -> Vec<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or(default)
        .split(',')
        .map(|s| s.trim().parse().expect("number list"))
        .collect()
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let oracle_dir = args
        .iter()
        .position(|a| a == "--oracle-dir")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or("--oracle-dir <dir with pristine e371200c4 kernels> is required")?;
    let only = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let rows = list(&args, "--rows", "37,1000,65535,65536,70000,131075");
    let ks = list(&args, "--ks", "512,768");
    let mut gpu = Gpu::init()?;
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201") {
        return Err(format!("wave32 RDNA3/4 only, device is {}", gpu.arch).into());
    }
    println!("arch={} rows={rows:?} ks={ks:?}", gpu.arch);
    let mut bad = 0;
    if only.is_empty() || "silu".contains(only.as_str()) || only.starts_with("silu") || only.starts_with("gated") {
        bad += run_fold(&mut gpu, &oracle_dir, &rows, &ks, &only)?;
    }
    if only.is_empty() || only == "indexed" {
        bad += run_indexed(&mut gpu, &oracle_dir, &rows, &ks)?;
    }
    if only.is_empty() || only == "chunk" {
        bad += run_rotate128(&mut gpu, &rows)?;
        bad += run_sigmoid(&mut gpu, &rows)?;
        bad += run_moe(&mut gpu, &rows)?;
    }
    if bad != 0 {
        eprintln!("FAIL: {bad} mismatches");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
