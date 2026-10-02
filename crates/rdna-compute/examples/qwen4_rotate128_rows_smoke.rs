//! gfx1151/gfx1201 smoke for `Gpu::qwen4_moe_rotate128_i4` row grids: the
//! launch folds grouped rows into grid x, so rows > 65535 must be processed.
//!
//! For each row count it runs the production kernel on pseudo-random BF16
//! rows with a permuted/padded `sorted` table and compares the whole
//! `[k/128, rows]` sidecar byte-for-byte with the independent oracle used by
//! `qwen4_moe_sym` (widen -> shipped `mq_rotate_x_128_v2` -> shared quantizer).
//! The oracle kernels still launch grid y = rows, so they run in chunks of
//! `CHUNK` rows with shifted pointers. Unwritten slots (padding) stay 0x7B in
//! both sidecars.
//!
//!     cargo run --release -p rdna-compute --features lab \
//!         --example qwen4_rotate128_rows_smoke [-- --rows 37,1000,70000,85808]
//!
//! Exit status 1 on any byte mismatch.

use hip_bridge::KernargBlob;
use rdna_compute::Gpu;
use std::ffi::c_void;

const K: usize = 640;
const BLK: usize = 72;
const CHUNK: usize = 60_000;
const QUANT: &str = include_str!("../../../kernels/src/block_i4_128_quant.hip");
const ROT128: &str = include_str!("../../../kernels/src/mq_rotate_x_128_v2.hip");
const ORACLE: &str = r#"
extern "C" __global__ __launch_bounds__(32) void rot128_smoke_ref_quant(
    const float* __restrict__ rot, const int* __restrict__ sorted,
    block_i4_128* __restrict__ Xq, int K, int P, int L) {
    const int group = blockIdx.x;
    const int p = blockIdx.y;
    const int slot = sorted[p];
    if (slot < 0) return;
    const int tid = threadIdx.x;
    float xv[4];
    for (int i = 0; i < 4; ++i) xv[i] = rot[(size_t)p * K + group * 128 + tid * 4 + i];
    quantize_block_i4_128_wave<true>(xv, Xq + (size_t)group * L + slot, tid);
}
extern "C" __global__ void rot128_smoke_ref_widen(const unsigned short* h, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((unsigned int)h[i] << 16);
}
"#;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, a: &mut KernargBlob) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, a.as_mut_slice())?;
    Ok(())
}

fn shift(base: *mut c_void, bytes: usize) -> *const c_void {
    (base as usize + bytes) as *const c_void
}

fn run_rows(gpu: &mut Gpu, rows: usize, seed: u64) -> Res<usize> {
    let l = rows; // total slots == rows: sorted is a permutation with padding
    let mut r = Rng(seed ^ (rows as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    // BF16 bits of finite values in about [-4, 4).
    let h: Vec<u8> = (0..rows * K)
        .flat_map(|_| {
            let u = ((r.next() >> 11) as f64) / ((1u64 << 53) as f64);
            let v = ((u - 0.5) * 8.0) as f32;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    // slot = p * a mod l is a bijection for gcd(a, l) == 1; every 7th row is padding.
    let mut a = 7919usize;
    while gcd(a, l) != 1 {
        a += 2;
    }
    let sorted: Vec<i32> = (0..rows)
        .map(|p| if p % 7 == 3 { -1 } else { ((p * a) % l) as i32 })
        .collect();
    let sorted_b: Vec<u8> = sorted.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ht = gpu.upload_raw(&h, &[h.len()])?;
    let st = gpu.upload_raw(&sorted_b, &[sorted_b.len()])?;
    let bytes = (K / 128) * l * BLK;

    // Production kernel; poison the sidecar after the first call (same size,
    // no growth) so unwritten slots read back as 0x7B.
    let _ = gpu.qwen4_moe_rotate128_i4(&ht, &st, K, rows, l)?;
    gpu.hip.device_synchronize()?;
    {
        let d = gpu.scratch.qwen4_moe_down_i4_scratch.as_ref().ok_or("no down sidecar")?;
        gpu.hip.memset(d, 0x7B, bytes)?;
    }
    let _ = gpu.qwen4_moe_rotate128_i4(&ht, &st, K, rows, l)?;
    gpu.hip.device_synchronize()?;
    let mut got = vec![0u8; bytes];
    {
        let d = gpu.scratch.qwen4_moe_down_i4_scratch.as_ref().ok_or("no down sidecar")?;
        gpu.hip.memcpy_dtoh(&mut got, d)?;
    }

    // Oracle, in CHUNK-row slices.
    let rot = gpu.zeros(&[rows * K], rdna_compute::DType::F32)?;
    let refq = gpu.zeros(&[bytes.div_ceil(4)], rdna_compute::DType::F32)?;
    gpu.hip.memset(&refq.buf, 0x7B, bytes)?;
    gpu.ensure_mq_signs_128()?;
    let s1 = gpu.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
    let s2 = gpu.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
    let mut off = 0;
    while off < rows {
        let n = CHUNK.min(rows - off);
        let mut b = KernargBlob::new();
        b.push_ptr(shift(ht.buf.as_ptr(), off * K * 2));
        b.push_ptr(shift(rot.buf.as_ptr(), off * K * 4));
        b.push_i32((n * K) as i32);
        launch(gpu, "rot128_smoke_ref_widen", [(n * K).div_ceil(256) as u32, 1, 1], 256, &mut b)?;
        let mut b = KernargBlob::new();
        b.push_ptr(shift(rot.buf.as_ptr(), off * K * 4));
        b.push_ptr(shift(rot.buf.as_ptr(), off * K * 4));
        b.push_ptr(s1);
        b.push_ptr(s2);
        b.push_i32(K as i32);
        launch(gpu, "mq_rotate_x_128_v2", [(K / 128) as u32, n as u32, 1], 32, &mut b)?;
        let mut b = KernargBlob::new();
        b.push_ptr(shift(rot.buf.as_ptr(), off * K * 4));
        b.push_ptr(shift(st.buf.as_ptr(), off * 4));
        b.push_ptr(refq.buf.as_ptr());
        b.push_i32(K as i32);
        b.push_i32(n as i32);
        b.push_i32(l as i32);
        launch(gpu, "rot128_smoke_ref_quant", [(K / 128) as u32, n as u32, 1], 32, &mut b)?;
        off += n;
    }
    gpu.hip.device_synchronize()?;
    let mut want = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut want, &refq.buf)?;
    gpu.free_tensor(rot)?;
    gpu.free_tensor(refq)?;
    gpu.free_tensor(ht)?;
    gpu.free_tensor(st)?;
    let written = sorted.iter().filter(|s| **s >= 0).count();
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    let untouched = want.iter().filter(|b| **b == 0x7B).count();
    println!(
        "rows={rows} grid=[{},1,1] valid_slots={written} mismatched_bytes={bad} \
         poison_bytes_in_ref={untouched}",
        (K / 128) * rows
    );
    Ok(bad)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let rows_arg = args
        .iter()
        .position(|a| a == "--rows")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "37,1000,65535,65536,70000,85808".into());
    let mut gpu = Gpu::init()?;
    if !matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") {
        return Err(format!("gfx1151/gfx1201-only, device is {}", gpu.arch).into());
    }
    let src = format!("#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n{QUANT}{ORACLE}");
    for f in ["rot128_smoke_ref_quant", "rot128_smoke_ref_widen"] {
        gpu.ensure_kernel_public("rot128_smoke_oracle", &src, f)?;
    }
    gpu.ensure_kernel_public("rot128_smoke_oracle_rot", ROT128, "mq_rotate_x_128_v2")?;
    let mut total = 0;
    for rows in rows_arg.split(',') {
        total += run_rows(&mut gpu, rows.trim().parse()?, 1)?;
    }
    if total != 0 {
        eprintln!("FAIL: {total} mismatched bytes");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
