#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

// One wave64 computes a 16-output-row x 16-routed-slot tile. MQ2-Lloyd
// weights are decoded directly into MFMA operands; no expanded weight shadow.
extern "C" __global__ __launch_bounds__(64)
void gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const half_t* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int x_row_div, int m_total
) {
    const int lane = threadIdx.x;
    const int n_lane = lane & 15;
    const int strip = lane >> 4;
    const int m_offset = blockIdx.x * 16;
    const int slot_offset = blockIdx.y * 16;
    const int expert_id = expert_tile_ids[blockIdx.y];
    if (expert_id < 0 || m_offset >= M || slot_offset >= m_total) return;

    const int weight_row = m_offset + n_lane;
    const int safe_row = weight_row < M ? weight_row : M - 1;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    const int groups = K / 256;
    const char* row_ptr = A + (long long)safe_row * groups * 72;
    const int slot = slot_offset + n_lane;
    const int flat = slot < m_total ? sorted_slot_index[slot] : -1;
    const int x_row = flat < 0 ? -1 : (x_row_div > 1 ? flat / x_row_div : flat);
    float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups; ++g) {
        const char* gp = row_ptr + g * 72;
        const __half* cb = reinterpret_cast<const __half*>(gp);
        const half_t c0 = (half_t)__half2float(cb[0]);
        const half_t c1 = (half_t)__half2float(cb[1]);
        const half_t c2 = (half_t)__half2float(cb[2]);
        const half_t c3 = (half_t)__half2float(cb[3]);
        #pragma unroll
        for (int ki = 0; ki < 256; ki += 16) {
            const int local_k = ki + strip * 4;
            const unsigned char packed = *(const unsigned char*)(gp + 8 + local_k / 4);
            half4 a;
#define CB(q) ((q) == 0 ? c0 : ((q) == 1 ? c1 : ((q) == 2 ? c2 : c3)))
            a[0] = CB( packed       & 3u);
            a[1] = CB((packed >> 2) & 3u);
            a[2] = CB((packed >> 4) & 3u);
            a[3] = CB((packed >> 6) & 3u);
#undef CB
            half4 b = {(half_t)0, (half_t)0, (half_t)0, (half_t)0};
            if (x_row >= 0) {
                const long long x_base = (long long)x_row * K + g * 256 + local_k;
                b[0] = X[x_base];
                b[1] = X[x_base + 1];
                b[2] = X[x_base + 2];
                b[3] = X[x_base + 3];
            }
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
        }
    }

    const int out_slot = slot_offset + n_lane;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int out_m = m_offset + strip * 4 + r;
        if (out_slot < m_total && out_m < M)
            Y[(long long)out_slot * M + out_m] = acc[r];
    }
}
