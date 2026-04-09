#include <hip/hip_runtime.h>

// conv1d + silu + Q/K/V split fused into one kernel.
//
// Replaces the sequence
//   conv1d_silu_f32(dn_conv_out, dn_qkv, weight, state, qkv_dim);
//   memcpy_dtod_at(q_out, 0, dn_conv_out, 0,           k_dim * 4);
//   memcpy_dtod_at(k_out, 0, dn_conv_out, k_dim*4,     k_dim * 4);
//   memcpy_dtod_at(v_out, 0, dn_conv_out, k_dim*2*4,   v_dim * 4);
//
// with a single launch. Each thread processes one channel: does the 1D
// conv + SiLU in registers (bit-identical to conv1d_silu.hip), then
// writes to q_out / k_out / v_out depending on channel index.
//
// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] in the `input`
// buffer, matching the wqkv projection output. Total n_channels =
// 2*k_dim + v_dim.

extern "C" __global__ void conv1d_silu_split_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    // Same math as conv1d_silu.hip — 4-tap causal depthwise conv + SiLU.
    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

    // SiLU
    float result = y / (1.0f + expf(-y));

    // Update ring buffer state
    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3]     = x;

    // Write to Q/K/V based on channel index.
    if (c < k_dim) {
        q_out[c] = result;
    } else if (c < 2 * k_dim) {
        k_out[c - k_dim] = result;
    } else {
        v_out[c - 2 * k_dim] = result;
    }
}
