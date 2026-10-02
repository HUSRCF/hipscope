//! gfx1201 smoke for the token/row axis of the scratch quantizers
//! (`quantize_q8_1_mmq_ds4[_x128]`, `quantize_int4_mmq_ds128`,
//! `quantize_int8_mmq_ds128`) and the AWQ batched rotate. Their launches fold the
//! row axis into grid x (`fold_rows_into_x`), so row counts >= 65536 must be
//! processed instead of being rejected by the device `gridDim.y` limit.
//!
//! Each production entry is driven through its public `Gpu` API and compared
//! byte-for-byte with the PRISTINE kernels from the base commit
//! (`examples/scratch_rows_oracle/*.hip`, `git show e371200c4:kernels/src/..`,
//! renamed at runtime so both generations live in one process). The pristine
//! kernels use grid y = rows, so the oracle runs them in `CHUNK`-row slices
//! (< 65536) over shifted pointers; the quantizers' `[K/128, N]` sidecar
//! layout is rebuilt on the host from the per-slice `[K/128, n]` outputs.
//!
//! Inputs are deterministic; every 5th row is all-zero (amax == 0 branch) and
//! row magnitudes span 2^-3 .. 2^3. K = 1280 is not a multiple of 1024, so the
//! quantizers' in-row tail block is exercised.
//!
//!     cargo run --release -p rdna-compute --features lab \
//!         --example scratch_rows_fold_smoke [-- --rows 1,37,65536,131073]
//!
//! Exit status 1 on any byte mismatch or on any production error on gfx1201.
//! Remove `examples/scratch_rows_oracle/` and this file after the gates.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const K: usize = 1280;
/// Largest slice the pristine (grid.y = rows) kernels are run with; < 65536.
const CHUNK: usize = 60_000;

const OLD_Q8: &str = include_str!("scratch_rows_oracle/old_q8_1_quant.hip");
const OLD_I4: &str = include_str!("scratch_rows_oracle/block_i4_128_quant.hip");
const OLD_I8: &str = include_str!("scratch_rows_oracle/block_i8_128_quant.hip");
const OLD_AWQ: &str = include_str!("scratch_rows_oracle/rotate_x_mq_awq.hip");

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (((self.next() >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
}

/// `rows * K` f32 in [-s, s], s = 2^((row % 7) - 3); every 5th row zero.
fn rows_data(rows: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed ^ (rows as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut v = Vec::with_capacity(rows * K);
    for row in 0..rows {
        let s = 2f32.powi((row % 7) as i32 - 3);
        for _ in 0..K {
            v.push(if row % 5 == 0 { 0.0 } else { (r.unit() * 2.0 - 1.0) * s });
        }
    }
    v
}

fn positive(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| lo + r.unit() * (hi - lo)).collect()
}

fn renamed(src: &str, from: &str, to: &str) -> String {
    assert!(src.contains(from), "oracle source lacks `{from}`");
    src.replace(from, to)
}

fn shift(base: *mut c_void, bytes: usize) -> *const c_void {
    (base as usize + bytes) as *const c_void
}

fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, b: KernargBlob) -> Res<()> {
    let mut b = b;
    b.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, b.as_mut_slice())?;
    Ok(())
}

fn compare(name: &str, rows: usize, got: &[u8], want: &[u8]) -> usize {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let first = got.iter().zip(want).position(|(a, b)| a != b);
    println!(
        "{name:<40} rows={rows:<7} bytes={:<11} mismatched={bad}{}",
        got.len(),
        first.map_or(String::new(), |i| format!(" first_at={i}"))
    );
    bad
}

/// Production error policy: fatal on gfx1201, informational skip elsewhere.
fn prod<T>(gpu: &Gpu, name: &str, r: Result<T, hip_bridge::HipError>) -> Res<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if gpu.arch != "gfx1201" => {
            println!("SKIP {name} on {}: {e}", gpu.arch);
            Ok(None)
        }
        Err(e) => Err(format!("{name}: {e}").into()),
    }
}

/// Pristine quantizer in <= CHUNK-row slices, sidecar rebuilt as `[K/128, rows]`.
fn oracle_quant(gpu: &mut Gpu, func: &str, x: &GpuTensor, rows: usize, bb: usize) -> Res<Vec<u8>> {
    let kb = K / 128;
    let mut want = vec![0u8; kb * rows * bb];
    let tmp = gpu.zeros(&[(kb * CHUNK.min(rows) * bb).div_ceil(4)], DType::F32)?;
    let mut off = 0;
    while off < rows {
        let n = CHUNK.min(rows - off);
        gpu.hip.memset(&tmp.buf, 0x7B, kb * n * bb)?;
        let mut b = KernargBlob::new();
        b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
        b.push_ptr(tmp.buf.as_ptr());
        b.push_i32(K as i32);
        b.push_i32(n as i32);
        launch(gpu, func, [K.div_ceil(1024) as u32, n as u32, 1], 256, b)?;
        gpu.hip.device_synchronize()?;
        let mut sl = vec![0u8; kb * n * bb];
        gpu.hip.memcpy_dtoh(&mut sl, &tmp.buf)?;
        for blk in 0..kb {
            for j in 0..n {
                let s = (blk * n + j) * bb;
                let d = (blk * rows + off + j) * bb;
                want[d..d + bb].copy_from_slice(&sl[s..s + bb]);
            }
        }
        off += n;
    }
    gpu.free_tensor(tmp)?;
    Ok(want)
}

fn scratch_bytes(gpu: &Gpu, which: &str, bytes: usize) -> Res<Vec<u8>> {
    let buf = match which {
        "q8" => gpu.scratch.q8_1_mmq_x_scratch.as_ref(),
        "i4" => gpu.scratch.int4_mmq_x_scratch.as_ref(),
        _ => gpu.scratch.int8_mmq_x_scratch.as_ref(),
    }
    .ok_or("scratch sidecar missing")?;
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, buf)?;
    Ok(v)
}

fn poison_scratch(gpu: &Gpu, which: &str, bytes: usize) -> Res<()> {
    let buf = match which {
        "q8" => gpu.scratch.q8_1_mmq_x_scratch.as_ref(),
        "i4" => gpu.scratch.int4_mmq_x_scratch.as_ref(),
        _ => gpu.scratch.int8_mmq_x_scratch.as_ref(),
    }
    .ok_or("scratch sidecar missing")?;
    gpu.hip.memset(buf, 0x7B, bytes)?;
    Ok(())
}

fn run_quantizers(gpu: &mut Gpu, rows: usize) -> Res<usize> {
    let x = gpu.upload_f32(&rows_data(rows, 11), &[rows * K])?;
    let mut bad = 0;
    // (name, scratch slot, pristine kernel, block bytes)
    let cases = [
        ("ensure_q8_1_mmq_x", "q8", "quantize_q8_1_mmq_ds4_old", 144usize),
        ("ensure_q8_1_mmq_x128", "q8", "quantize_q8_1_mmq_ds4_x128_old", 144),
        ("ensure_int4_mmq_x", "i4", "old_quantize_int4_mmq_ds128", 72),
        ("ensure_int8_mmq_x", "i8", "old_quantize_int8_mmq_ds128", 136),
    ];
    for (case, (name, which, func, bb)) in cases.into_iter().enumerate() {
        let bytes = (K / 128) * rows * bb;
        // First call allocates the sidecar; poison it; second call must
        // rewrite every block.
        let mut ok = true;
        for pass in 0..2 {
            let r = match case {
                0 => gpu.ensure_q8_1_mmq_x(&x, rows, K),
                1 => gpu.ensure_q8_1_mmq_x128(&x, rows, K),
                2 => gpu.ensure_int4_mmq_x(&x, rows, K),
                _ => gpu.ensure_int8_mmq_x(&x, rows, K),
            };
            if prod(gpu, name, r)?.is_none() {
                ok = false;
                break;
            }
            gpu.hip.device_synchronize()?;
            if pass == 0 {
                poison_scratch(gpu, which, bytes)?;
            }
        }
        if !ok {
            continue;
        }
        let got = scratch_bytes(gpu, which, bytes)?;
        let want = oracle_quant(gpu, func, &x, rows, bb)?;
        bad += compare(name, rows, &got, &want);
    }
    gpu.free_tensor(x)?;
    Ok(bad)
}

fn run_rotate_awq(gpu: &mut Gpu, rows: usize) -> Res<usize> {
    let name = "rotate_x_mq_awq_batched";
    let x = gpu.upload_f32(&rows_data(rows, 23), &[rows * K])?;
    let awq = gpu.upload_f32(&positive(K, 5, 0.5, 2.0), &[K])?;
    let out = gpu.zeros(&[rows * K], DType::F32)?;
    gpu.ensure_mq_signs()?;
    gpu.hip.memset(&out.buf, 0x7B, out.byte_size())?;
    let r = gpu.rotate_x_mq_awq_batched(&x, &awq, &out, K, rows);
    if prod(gpu, name, r)?.is_none() {
        return Ok(0);
    }
    gpu.hip.device_synchronize()?;
    let got = gpu.download_raw_bytes(&out)?;

    let refo = gpu.zeros(&[rows * K], DType::F32)?;
    gpu.hip.memset(&refo.buf, 0x7B, refo.byte_size())?;
    let s1 = gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
    let s2 = gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
    let mut off = 0;
    while off < rows {
        let n = CHUNK.min(rows - off);
        let mut b = KernargBlob::new();
        b.push_ptr(shift(x.buf.as_ptr(), off * K * 4));
        b.push_ptr(shift(refo.buf.as_ptr(), off * K * 4));
        b.push_ptr(awq.buf.as_ptr());
        b.push_ptr(s1);
        b.push_ptr(s2);
        b.push_i32(K as i32);
        launch(gpu, "rotate_x_mq_awq_old", [(K / 256) as u32, n as u32, 1], 32, b)?;
        off += n;
    }
    gpu.hip.device_synchronize()?;
    let want = gpu.download_raw_bytes(&refo)?;
    let bad = compare(name, rows, &got, &want);
    for t in [x, awq, out, refo] {
        gpu.free_tensor(t)?;
    }
    Ok(bad)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let rows_arg = args
        .iter()
        .position(|a| a == "--rows")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "1,37,1000,65535,65536,70000,131073".into());
    let mut gpu = Gpu::init()?;
    println!("device arch {}", gpu.arch);

    // Pristine kernels, renamed so they coexist with the folded production ones.
    let q8 = renamed(OLD_Q8, "void quantize_q8_1_mmq_ds4_x128(", "void quantize_q8_1_mmq_ds4_x128_old(");
    let q8 = renamed(&q8, "void quantize_q8_1_mmq_ds4(", "void quantize_q8_1_mmq_ds4_old(");
    let i4 = renamed(OLD_I4, "void quantize_int4_mmq_ds128(", "void old_quantize_int4_mmq_ds128(");
    let i8 = renamed(OLD_I8, "void quantize_int8_mmq_ds128(", "void old_quantize_int8_mmq_ds128(");
    let awq = renamed(OLD_AWQ, "void rotate_x_mq_awq(", "void rotate_x_mq_awq_old(");
    let jobs: [(&str, &String, &[&str]); 4] = [
        ("old_q8", &q8, &["quantize_q8_1_mmq_ds4_old", "quantize_q8_1_mmq_ds4_x128_old"]),
        ("old_i4", &i4, &["old_quantize_int4_mmq_ds128"]),
        ("old_i8", &i8, &["old_quantize_int8_mmq_ds128"]),
        ("old_awq", &awq, &["rotate_x_mq_awq_old"]),
    ];
    for (module, src, funcs) in jobs {
        for f in funcs {
            if let Err(e) = gpu.ensure_kernel_public(module, src, f) {
                if gpu.arch == "gfx1201" {
                    return Err(format!("oracle {f}: {e}").into());
                }
                println!("SKIP oracle {f} on {}: {e}", gpu.arch);
            }
        }
    }

    let mut total = 0;
    for rows in rows_arg.split(',') {
        let rows: usize = rows.trim().parse()?;
        total += run_quantizers(&mut gpu, rows)?;
        total += run_rotate_awq(&mut gpu, rows)?;
    }
    if total != 0 {
        eprintln!("FAIL: {total} mismatched bytes");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
