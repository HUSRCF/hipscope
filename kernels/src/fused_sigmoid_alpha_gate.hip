#include <hip/hip_runtime.h>

// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops run in the
// DeltaNet attention preamble on independent buffers of size n_v_heads.
// Originally two back-to-back scalar launches; fused into one here.
//
// Each thread handles one head index, applying sigmoid to beta[i] and
// the alpha-gate transform to alpha[i]. Grid: ceil(n / 256) blocks.
//
// Bit-exact with the separate kernels since all operations are
// elementwise — no cross-thread dependencies or associativity changes.

extern "C" __global__ void fused_sigmoid_alpha_gate_f32(
    float* __restrict__ beta,         // [n] in-place sigmoid
    float* __restrict__ alpha,        // [n] in-place alpha_gate
    const float* __restrict__ dt_bias,// [n]
    const float* __restrict__ a_log,  // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Sigmoid on beta — matches sigmoid_f32.
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));

    // Alpha gate on alpha — matches alpha_gate_f32 (softplus(x+bias) then
    // multiply by -exp(a_log), with the same numerical guards).
    float biased = alpha[i] + dt_bias[i];
    float sp = (biased > 20.0f) ? biased
             : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
    alpha[i] = sp * (-expf(a_log[i]));
}
