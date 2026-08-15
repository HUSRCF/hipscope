#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// One native wave64 computes two rows. Every lane contributes to both rows;
// the rows share x loads but never split the wave into logical sub-waves.
extern "C" __global__ __launch_bounds__(64)
void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_row2_gfx90a(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const int row0 = blockIdx.x * 2;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int krank = blockIdx.y;
    const int lane = threadIdx.x;
    const int expert_id = topk_indices[krank];
    if (expert_id < 0) {
        if (lane == 0) {
            const int mi = M >> 1;
            if (row0 < mi) y_gate[(long long)krank * mi + row0] = 0.0f;
            else y_up[(long long)krank * mi + row0 - mi] = 0.0f;
            if (row1 < M) {
                if (row1 < mi) y_gate[(long long)krank * mi + row1] = 0.0f;
                else y_up[(long long)krank * mi + row1 - mi] = 0.0f;
            }
        }
        return;
    }
    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups = K / 256;
    const long long row_bytes = (long long)groups * 72;
    const char* row_ptr0 = A + (long long)row0 * row_bytes;
    const char* row_ptr1 = A + (long long)(row1 < M ? row1 : row0) * row_bytes;
    __shared__ float codebooks[32];
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    const int quads = groups >> 2;

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp00 = row_ptr0 + g * 72;
        const char* gp10 = row_ptr1 + g * 72;
        if (lane < 32) {
            const int row_sel = lane >> 4;
            const int cb_lane = lane & 15;
            const char* gp = (row_sel ? gp10 : gp00) + (cb_lane >> 2) * 72;
            codebooks[lane] = __half2float(reinterpret_cast<const __half*>(gp)[cb_lane & 3]);
        }
        __syncwarp();
        const int base = g * 256 + lane * 4;
#define DOT_GROUP(gp, cb, xbase, acc) do {                                      \
        const unsigned char packed = *(const unsigned char*)((gp) + 8 + lane);  \
        (acc) += codebooks[(cb) + ( packed       & 3u)] * x[(xbase)]            \
               + codebooks[(cb) + ((packed >> 2) & 3u)] * x[(xbase) + 1]       \
               + codebooks[(cb) + ((packed >> 4) & 3u)] * x[(xbase) + 2]       \
               + codebooks[(cb) + ((packed >> 6) & 3u)] * x[(xbase) + 3];      \
    } while (0)
        DOT_GROUP(gp00,       0, base,       a00);
        DOT_GROUP(gp00 + 72,  4, base + 256, a01);
        DOT_GROUP(gp00 + 144, 8, base + 512, a02);
        DOT_GROUP(gp00 + 216, 12, base + 768, a03);
        DOT_GROUP(gp10,       16, base,       a10);
        DOT_GROUP(gp10 + 72,  20, base + 256, a11);
        DOT_GROUP(gp10 + 144, 24, base + 512, a12);
        DOT_GROUP(gp10 + 216, 28, base + 768, a13);
#undef DOT_GROUP
        __syncwarp();
    }

    for (int g = quads << 2; g < groups; ++g) {
        const char* gp0 = row_ptr0 + g * 72;
        const char* gp1 = row_ptr1 + g * 72;
        if (lane < 8) {
            const char* gp = lane < 4 ? gp0 : gp1;
            codebooks[lane] = __half2float(reinterpret_cast<const __half*>(gp)[lane & 3]);
        }
        __syncwarp();
        const unsigned char p0 = *(const unsigned char*)(gp0 + 8 + lane);
        const unsigned char p1 = *(const unsigned char*)(gp1 + 8 + lane);
        const int base = g * 256 + lane * 4;
#define DOT_TAIL(p, cb, acc) do {                                                \
        (acc) += codebooks[(cb) + ( (p)       & 3u)] * x[base]                  \
               + codebooks[(cb) + (((p) >> 2) & 3u)] * x[base + 1]             \
               + codebooks[(cb) + (((p) >> 4) & 3u)] * x[base + 2]             \
               + codebooks[(cb) + (((p) >> 6) & 3u)] * x[base + 3];            \
    } while (0)
        DOT_TAIL(p0, 0, a00);
        DOT_TAIL(p1, 4, a10);
#undef DOT_TAIL
        __syncwarp();
    }

    float sum0 = (a00 + a01) + (a02 + a03);
    float sum1 = (a10 + a11) + (a12 + a13);
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (lane == 0) {
        const int mi = M >> 1;
        if (row0 < mi) y_gate[(long long)krank * mi + row0] = sum0;
        else y_up[(long long)krank * mi + row0 - mi] = sum0;
        if (row1 < M) {
            if (row1 < mi) y_gate[(long long)krank * mi + row1] = sum1;
            else y_up[(long long)krank * mi + row1 - mi] = sum1;
        }
    }
}
