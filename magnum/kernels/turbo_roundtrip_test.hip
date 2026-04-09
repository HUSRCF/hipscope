// TurboQuant round-trip diagnostic: traces each pipeline stage in one kernel.
// Writes intermediate results (stage B, D) for CPU comparison.
// Does NOT use the actual write/attention kernels — isolates the math.

#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Full turbo4 round-trip in one kernel: normalize → FWHT → quant → dequant → inv FWHT
// 32 threads per block, head_dim=128 (4 elements per thread)
extern "C" __launch_bounds__(32, 1)
__global__ void turbo4_roundtrip_test(
    const float* __restrict__ input,     // [head_dim]
    float* __restrict__ output,          // [head_dim] final recovered
    float* __restrict__ after_norm,      // [head_dim] after unit normalization
    float* __restrict__ after_fwht,      // [head_dim] after forward FWHT
    float* __restrict__ after_dequant,   // [head_dim] after centroid lookup × cnorm
    float* __restrict__ debug_out,       // [8] cnorm, orig_norm, recon_norm, 0, 0, 0, 0, 0
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int dpt = head_dim / 32;  // 4
    const int d0 = tid * dpt;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    // Load input → shared memory
    for (int d = tid; d < head_dim; d += 32) x[d] = input[d];
    __syncthreads();

    // ── Stage A: L2 normalize ──
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid+s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    // Save normalized
    for (int d = tid; d < head_dim; d += 32) after_norm[d] = x[d];

    // ── Stage B: Forward FWHT ──
    float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
    fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

    after_fwht[d0] = fa; after_fwht[d0+1] = fb;
    after_fwht[d0+2] = fc; after_fwht[d0+3] = fd;

    // ── Stage C: 4-bit quantize ──
    float local_rsq = 0.0f;
    int idx0 = turbo_quantize_4bit(fa); local_rsq += TURBO_C4[idx0] * TURBO_C4[idx0];
    int idx1 = turbo_quantize_4bit(fb); local_rsq += TURBO_C4[idx1] * TURBO_C4[idx1];
    int idx2 = turbo_quantize_4bit(fc); local_rsq += TURBO_C4[idx2] * TURBO_C4[idx2];
    int idx3 = turbo_quantize_4bit(fd); local_rsq += TURBO_C4[idx3] * TURBO_C4[idx3];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid+s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    // ── Stage D: Dequant (centroid lookup × cnorm) ──
    float da = cnorm * TURBO_C4[idx0];
    float db = cnorm * TURBO_C4[idx1];
    float dc = cnorm * TURBO_C4[idx2];
    float dd = cnorm * TURBO_C4[idx3];

    after_dequant[d0] = da; after_dequant[d0+1] = db;
    after_dequant[d0+2] = dc; after_dequant[d0+3] = dd;

    // ── Stage E: Inverse FWHT ──
    fwht_shfl_inverse(da, db, dc, dd, signs1, signs2, tid);

    output[d0] = da; output[d0+1] = db; output[d0+2] = dc; output[d0+3] = dd;

    // Debug scalars
    if (tid == 0) {
        debug_out[0] = cnorm;
        debug_out[1] = orig_norm;
        debug_out[2] = recon_norm;
        debug_out[3] = (float)idx0;  // first element's quantized index
    }
}
