//! GPU smoke for the Qwen3.5 DeltaNet batched row launchers that used to put
//! the batch row in grid.y (HIP limit 65535 blocks):
//!
//!   * `fused_sigmoid_alpha_gate_f32_batched`
//!   * `fused_qk_l2_norm_scale_f32_batched`
//!   * `fused_qk_l2_norm_scale_interleave_f32_batched`
//!   * `gemm_f32_batched`
//!
//! The first three now fold rows into grid.x; `gemm_f32_batched` launches in
//! <= 65535-row chunks with B/Y advanced by whole rows. Each case compares the
//! launcher's output byte-for-byte against an oracle that runs the *old*
//! grid.y kernels (embedded verbatim below, renamed) in <= 65535-row slices
//! with shifted pointers. For rows <= 65535 the oracle is exactly the old
//! single launch, so those rows prove small-row byte identity; for rows past
//! 65535 they prove every row was processed and matches. `gemm_f32_batched`
//! also checks the last rows against an f64 host reference.
//!
//!     cargo run --release -p rdna-compute --features lab,deltanet \
//!         --example qwen35_batch_rows_smoke [-- --rows 37,1000,65536,70000,85808]
//!
//! Exit status 1 on any mismatch.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const SLICE: usize = 65_535;

const OLD_KERNELS: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void old_sigmoid_alpha_gate(
    float* __restrict__ beta, float* __restrict__ alpha,
    const float* __restrict__ dt_bias, const float* __restrict__ a_log, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long batch_off = (long long)blockIdx.y * n;
    float* beta_b = beta + batch_off;
    float* alpha_b = alpha + batch_off;
    beta_b[i] = 1.0f / (1.0f + expf(-beta_b[i]));
    float biased = alpha_b[i] + dt_bias[i];
    float sp = (biased > 20.0f) ? biased
             : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
    alpha_b[i] = sp * (-expf(a_log[i]));
}
extern "C" __global__ void old_qk_l2(
    float* __restrict__ q, float* __restrict__ k,
    int n_heads, int head_dim, float q_scale, float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    float* q_head = q + batch_off + h * head_dim;
    float* k_head = k + batch_off + h * head_dim;
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32) q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1) q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    for (int i = tid; i < head_dim; i += 32) q_head[i] *= q_inv_norm;
    for (int i = tid; i < head_dim; i += 32) q_head[i] *= q_scale;
    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32) k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1) k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    for (int i = tid; i < head_dim; i += 32) k_head[i] *= k_inv_norm;
}
extern "C" __global__ void old_qk_l2_interleave(
    const float* __restrict__ q_src, const float* __restrict__ k_src,
    float* __restrict__ q_dst, float* __restrict__ k_dst,
    int n_key_heads, int ratio, int head_dim, float q_scale, float eps) {
    const int kh = blockIdx.x;
    if (kh >= n_key_heads) return;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const long long src_batch_off = (long long)b * n_key_heads * head_dim;
    const float* q_head = q_src + src_batch_off + kh * head_dim;
    const float* k_head = k_src + src_batch_off + kh * head_dim;
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32) q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1) q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32) k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1) k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    const int n_value_heads = n_key_heads * ratio;
    const long long dst_batch_off = (long long)b * n_value_heads * head_dim;
    for (int r = 0; r < ratio; r++) {
        const int vh = kh * ratio + r;
        float* q_out = q_dst + dst_batch_off + vh * head_dim;
        float* k_out = k_dst + dst_batch_off + vh * head_dim;
        for (int i = tid; i < head_dim; i += 32) {
            float qv = q_head[i] * q_inv_norm;
            qv *= q_scale;
            q_out[i] = qv;
            k_out[i] = k_head[i] * k_inv_norm;
        }
    }
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
    /// Uniform f32 in [-scale, scale).
    fn f32(&mut self, scale: f32) -> f32 {
        let u = ((self.next() >> 40) as f32) / ((1u64 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }
}

fn randv(r: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| r.f32(scale)).collect()
}

fn shift(t: &GpuTensor, f32_elems: usize) -> *mut c_void {
    t.buf.as_ptr().wrapping_byte_add(f32_elems * 4)
}

fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, a: &mut KernargBlob) -> Res<()> {
    a.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, a.as_mut_slice())?;
    Ok(())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn diff(label: &str, rows: usize, got: &[f32], want: &[f32]) -> usize {
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("rows={rows:>6} {label:<34} mismatched_f32={bad} (of {})", got.len());
    bad
}

fn sigmoid_alpha(gpu: &mut Gpu, rows: usize, seed: u64) -> Res<usize> {
    const N: usize = 48; // Qwen3.8 value heads
    let mut r = Rng(seed ^ 0x51);
    let beta = randv(&mut r, rows * N, 6.0);
    let alpha = randv(&mut r, rows * N, 6.0);
    let dt_bias = gpu.upload_f32(&randv(&mut r, N, 1.0), &[N])?;
    let a_log = gpu.upload_f32(&randv(&mut r, N, 1.0), &[N])?;
    let nb = gpu.upload_f32(&beta, &[rows * N])?;
    let na = gpu.upload_f32(&alpha, &[rows * N])?;
    let ob = gpu.upload_f32(&beta, &[rows * N])?;
    let oa = gpu.upload_f32(&alpha, &[rows * N])?;
    gpu.fused_sigmoid_alpha_gate_f32_batched(&nb, &na, &dt_bias, &a_log, N, rows)?;
    let mut off = 0;
    while off < rows {
        let take = SLICE.min(rows - off);
        let mut a = KernargBlob::new();
        a.push_ptr(shift(&ob, off * N));
        a.push_ptr(shift(&oa, off * N));
        a.push_ptr(dt_bias.buf.as_ptr());
        a.push_ptr(a_log.buf.as_ptr());
        a.push_i32(N as i32);
        launch(gpu, "old_sigmoid_alpha_gate", [N.div_ceil(256) as u32, take as u32, 1], 256, &mut a)?;
        off += take;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("sigmoid_alpha beta", rows, &gpu.download_f32(&nb)?, &gpu.download_f32(&ob)?);
    bad += diff("sigmoid_alpha alpha", rows, &gpu.download_f32(&na)?, &gpu.download_f32(&oa)?);
    for t in [dt_bias, a_log, nb, na, ob, oa] {
        gpu.free_tensor(t)?;
    }
    Ok(bad)
}

fn qk_l2(gpu: &mut Gpu, rows: usize, seed: u64) -> Res<usize> {
    const NKH: usize = 4;
    const HD: usize = 128;
    const RATIO: usize = 3;
    let (qs, eps) = (1.0 / (HD as f32).sqrt(), 1e-6f32);
    let mut r = Rng(seed ^ 0x52);
    let q = randv(&mut r, rows * NKH * HD, 3.0);
    let k = randv(&mut r, rows * NKH * HD, 3.0);
    let n = rows * NKH * HD;

    // In-place non-interleaved variant.
    let nq = gpu.upload_f32(&q, &[n])?;
    let nk = gpu.upload_f32(&k, &[n])?;
    let oq = gpu.upload_f32(&q, &[n])?;
    let ok = gpu.upload_f32(&k, &[n])?;
    gpu.fused_qk_l2_norm_scale_f32_batched(&nq, &nk, NKH, HD, qs, eps, rows)?;
    let mut off = 0;
    while off < rows {
        let take = SLICE.min(rows - off);
        let mut a = KernargBlob::new();
        a.push_ptr(shift(&oq, off * NKH * HD));
        a.push_ptr(shift(&ok, off * NKH * HD));
        a.push_i32(NKH as i32);
        a.push_i32(HD as i32);
        a.push_f32(qs);
        a.push_f32(eps);
        launch(gpu, "old_qk_l2", [NKH as u32, take as u32, 1], 32, &mut a)?;
        off += take;
    }
    gpu.hip.device_synchronize()?;
    let mut bad = diff("qk_l2 q", rows, &gpu.download_f32(&nq)?, &gpu.download_f32(&oq)?);
    bad += diff("qk_l2 k", rows, &gpu.download_f32(&nk)?, &gpu.download_f32(&ok)?);
    for t in [nq, nk, oq, ok] {
        gpu.free_tensor(t)?;
    }

    // Interleave variant (src untouched, dst widened by RATIO).
    let nd = rows * NKH * RATIO * HD;
    let sq = gpu.upload_f32(&q, &[n])?;
    let sk = gpu.upload_f32(&k, &[n])?;
    let nqd = gpu.zeros(&[nd], DType::F32)?;
    let nkd = gpu.zeros(&[nd], DType::F32)?;
    let oqd = gpu.zeros(&[nd], DType::F32)?;
    let okd = gpu.zeros(&[nd], DType::F32)?;
    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(&sq, &sk, &nqd, &nkd, NKH, RATIO, HD, qs, eps, rows)?;
    let mut off = 0;
    while off < rows {
        let take = SLICE.min(rows - off);
        let mut a = KernargBlob::new();
        a.push_ptr(shift(&sq, off * NKH * HD));
        a.push_ptr(shift(&sk, off * NKH * HD));
        a.push_ptr(shift(&oqd, off * NKH * RATIO * HD));
        a.push_ptr(shift(&okd, off * NKH * RATIO * HD));
        a.push_i32(NKH as i32);
        a.push_i32(RATIO as i32);
        a.push_i32(HD as i32);
        a.push_f32(qs);
        a.push_f32(eps);
        launch(gpu, "old_qk_l2_interleave", [NKH as u32, take as u32, 1], 32, &mut a)?;
        off += take;
    }
    gpu.hip.device_synchronize()?;
    bad += diff("qk_l2_interleave q_dst", rows, &gpu.download_f32(&nqd)?, &gpu.download_f32(&oqd)?);
    bad += diff("qk_l2_interleave k_dst", rows, &gpu.download_f32(&nkd)?, &gpu.download_f32(&okd)?);
    for t in [sq, sk, nqd, nkd, oqd, okd] {
        gpu.free_tensor(t)?;
    }
    Ok(bad)
}

/// `gemm_f32_batched`: new launcher vs the same kernel launched in 40000-row
/// slices (chunk boundaries differ from the launcher's 65535), plus an f64
/// host check on the last rows.
fn gemm_f32(gpu: &mut Gpu, rows: usize, seed: u64) -> Res<usize> {
    const M: usize = 64;
    const K: usize = 256;
    const OLD_SLICE: usize = 40_000;
    let mut r = Rng(seed ^ 0x53);
    let a = randv(&mut r, M * K, 1.0);
    let b = randv(&mut r, rows * K, 1.0);
    let ta = gpu.upload_f32(&a, &[M * K])?;
    let tb = gpu.upload_f32(&b, &[rows * K])?;
    let ty = gpu.zeros(&[rows * M], DType::F32)?;
    let oy = gpu.zeros(&[rows * M], DType::F32)?;
    gpu.gemm_f32_batched(&ta, &tb, &ty, M, K, rows)?;
    let mut off = 0;
    while off < rows {
        let take = OLD_SLICE.min(rows - off);
        let mut ka = KernargBlob::new();
        ka.push_ptr(ta.buf.as_ptr());
        ka.push_ptr(shift(&tb, off * K));
        ka.push_ptr(shift(&oy, off * M));
        ka.push_i32(M as i32);
        ka.push_i32(K as i32);
        ka.push_i32(take as i32);
        launch(gpu, "gemm_f32_batched", [M as u32, take as u32, 1], 32, &mut ka)?;
        off += take;
    }
    gpu.hip.device_synchronize()?;
    let got = gpu.download_f32(&ty)?;
    let want = gpu.download_f32(&oy)?;
    let mut bad = diff("gemm_f32_batched vs sliced", rows, &got, &want);
    let mut host_bad = 0;
    for row in rows.saturating_sub(64)..rows {
        for m in 0..M {
            let h: f64 = (0..K).map(|k| a[m * K + k] as f64 * b[row * K + k] as f64).sum();
            let g = got[row * M + m] as f64;
            host_bad += ((g - h).abs() > 1e-3 * (1.0 + h.abs())) as usize;
        }
    }
    println!("rows={rows:>6} {:<34} outside_tol={host_bad}", "gemm_f32_batched vs f64 host (tail)");
    bad += host_bad;
    for t in [ta, tb, ty, oy] {
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
        .unwrap_or_else(|| "37,1000,65535,65536,70000,85808".into());
    let mut gpu = Gpu::init()?;
    for f in ["old_sigmoid_alpha_gate", "old_qk_l2", "old_qk_l2_interleave"] {
        gpu.ensure_kernel_public("qwen35_batch_rows_old", OLD_KERNELS, f)?;
    }
    let mut total = 0;
    for rows in rows_arg.split(',') {
        let rows: usize = rows.trim().parse()?;
        total += sigmoid_alpha(&mut gpu, rows, 1)?;
        total += qk_l2(&mut gpu, rows, 1)?;
        total += gemm_f32(&mut gpu, rows, 1)?;
    }
    if total != 0 {
        eprintln!("FAIL: {total} mismatches");
        std::process::exit(1);
    }
    println!("PASS");
    Ok(())
}
