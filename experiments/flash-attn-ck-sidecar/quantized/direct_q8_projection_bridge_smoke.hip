// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kHeads = 24;

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};

static_assert(sizeof(block_q8_1_mmq) == 144);

void check_hip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ void fwht256(float& v0,
                                        float& v1,
                                        float& v2,
                                        float& v3,
                                        float& v4,
                                        float& v5,
                                        float& v6,
                                        float& v7,
                                        int lane)
{
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

#define HIPFIRE_HBFLY(v, pat, stride) do {                                      \
    const float peer = __int_as_float(                                           \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat)));                  \
    (v) = (lane & (stride)) ? peer - (v) : (v) + peer;                           \
} while(0)
#define HIPFIRE_HBFLY8(pat, stride)                                              \
    HIPFIRE_HBFLY(v0, pat, stride); HIPFIRE_HBFLY(v1, pat, stride);              \
    HIPFIRE_HBFLY(v2, pat, stride); HIPFIRE_HBFLY(v3, pat, stride);              \
    HIPFIRE_HBFLY(v4, pat, stride); HIPFIRE_HBFLY(v5, pat, stride);              \
    HIPFIRE_HBFLY(v6, pat, stride); HIPFIRE_HBFLY(v7, pat, stride)
    HIPFIRE_HBFLY8(0x041F, 1);
    HIPFIRE_HBFLY8(0x081F, 2);
    HIPFIRE_HBFLY8(0x101F, 4);
    HIPFIRE_HBFLY8(0x201F, 8);
    HIPFIRE_HBFLY8(0x401F, 16);
#undef HIPFIRE_HBFLY8
#undef HIPFIRE_HBFLY
}

__global__ void convert_f16_to_f32(const __half* input, float* output, int values)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if(i + 1 < values)
    {
        reinterpret_cast<float2*>(output)[i / 2] =
            __half22float2(reinterpret_cast<const __half2*>(input)[i / 2]);
    }
}

__global__ void sigmoid_mul(float* values, const float* gate, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count)
        values[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

__global__ __launch_bounds__(32, 16)
void mq_rotate(const float* input,
               float* output,
               const float* signs1,
               const float* signs2,
               int row_heads)
{
    const int row_head = blockIdx.x;
    const int lane = threadIdx.x;
    if(row_head >= row_heads || lane >= 32) return;
    const int d0 = lane * 8;
    const int base = row_head * kHeadDim + d0;
    float v0 = input[base] * signs1[d0];
    float v1 = input[base + 1] * signs1[d0 + 1];
    float v2 = input[base + 2] * signs1[d0 + 2];
    float v3 = input[base + 3] * signs1[d0 + 3];
    float v4 = input[base + 4] * signs1[d0 + 4];
    float v5 = input[base + 5] * signs1[d0 + 5];
    float v6 = input[base + 6] * signs1[d0 + 6];
    float v7 = input[base + 7] * signs1[d0 + 7];
    fwht256(v0, v1, v2, v3, v4, v5, v6, v7, lane);
    constexpr float scale = 0.0625f;
    output[base] = v0 * scale * signs2[d0];
    output[base + 1] = v1 * scale * signs2[d0 + 1];
    output[base + 2] = v2 * scale * signs2[d0 + 2];
    output[base + 3] = v3 * scale * signs2[d0 + 3];
    output[base + 4] = v4 * scale * signs2[d0 + 4];
    output[base + 5] = v5 * scale * signs2[d0 + 5];
    output[base + 6] = v6 * scale * signs2[d0 + 6];
    output[base + 7] = v7 * scale * signs2[d0 + 7];
}

__device__ __forceinline__ char quantize_value(float value, float inverse_scale)
{
    return static_cast<char>(lrintf(fmaxf(-127.0f, fminf(127.0f, value * inverse_scale))));
}

__global__ void quantize_q8_1_mmq_ds4(const float* input,
                                      block_q8_1_mmq* output,
                                      int k,
                                      int rows)
{
    const int row = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(row >= rows || i0 >= k) return;
    const float4 x = *reinterpret_cast<const float4*>(input + static_cast<long long>(row) * k + i0);
    float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
    float sum = x.x + x.y + x.z + x.w;
    for(int offset = 4; offset > 0; offset >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    const char4 q = make_char4(quantize_value(x.x, id),
                               quantize_value(x.y, id),
                               quantize_value(x.z, id),
                               quantize_value(x.w, id));
    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = output + static_cast<long long>(kblock) * rows + row;
    reinterpret_cast<char4*>(out->qs)[iqs / 4] = q;
    if((iqs & 31) == 0)
        out->ds4[iqs / 32] = make_half2(static_cast<__half>(d), static_cast<__half>(sum));
}

__global__ __launch_bounds__(32, 16)
void gated_mq_rotate_q8_1(const __half* attention,
                         const float* gate,
                         const float* signs1,
                         const float* signs2,
                         block_q8_1_mmq* output,
                         int rows,
                         int heads)
{
    const int row_head = blockIdx.x;
    const int lane = threadIdx.x;
    if(row_head >= rows * heads || lane >= 32) return;
    const int row = row_head / heads;
    const int head = row_head - row * heads;
    const int d0 = lane * 8;
    const int base = row_head * kHeadDim + d0;
    const __half2* half_input = reinterpret_cast<const __half2*>(attention + base);
    const float2 h0 = __half22float2(half_input[0]);
    const float2 h1 = __half22float2(half_input[1]);
    const float2 h2 = __half22float2(half_input[2]);
    const float2 h3 = __half22float2(half_input[3]);
    float v0 = h0.x * (1.0f / (1.0f + expf(-gate[base])));
    float v1 = h0.y * (1.0f / (1.0f + expf(-gate[base + 1])));
    float v2 = h1.x * (1.0f / (1.0f + expf(-gate[base + 2])));
    float v3 = h1.y * (1.0f / (1.0f + expf(-gate[base + 3])));
    float v4 = h2.x * (1.0f / (1.0f + expf(-gate[base + 4])));
    float v5 = h2.y * (1.0f / (1.0f + expf(-gate[base + 5])));
    float v6 = h3.x * (1.0f / (1.0f + expf(-gate[base + 6])));
    float v7 = h3.y * (1.0f / (1.0f + expf(-gate[base + 7])));
    v0 *= signs1[d0]; v1 *= signs1[d0 + 1];
    v2 *= signs1[d0 + 2]; v3 *= signs1[d0 + 3];
    v4 *= signs1[d0 + 4]; v5 *= signs1[d0 + 5];
    v6 *= signs1[d0 + 6]; v7 *= signs1[d0 + 7];
    fwht256(v0, v1, v2, v3, v4, v5, v6, v7, lane);
    constexpr float rotate_scale = 0.0625f;
    v0 *= rotate_scale * signs2[d0]; v1 *= rotate_scale * signs2[d0 + 1];
    v2 *= rotate_scale * signs2[d0 + 2]; v3 *= rotate_scale * signs2[d0 + 3];
    v4 *= rotate_scale * signs2[d0 + 4]; v5 *= rotate_scale * signs2[d0 + 5];
    v6 *= rotate_scale * signs2[d0 + 6]; v7 *= rotate_scale * signs2[d0 + 7];

    float amax0 = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax1 = fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7)));
    float sum0 = v0 + v1 + v2 + v3;
    float sum1 = v4 + v5 + v6 + v7;
    // Match quantize_q8_1_mmq_ds4's eight logical float4 lanes. Each
    // physical lane owns two adjacent float4s, so logical XOR 4/2 become
    // physical XOR 2/1 and the final logical XOR 1 combines sum0/sum1.
    amax0 = fmaxf(amax0, __shfl_xor(amax0, 2, 4));
    amax1 = fmaxf(amax1, __shfl_xor(amax1, 2, 4));
    sum0 += __shfl_xor(sum0, 2, 4);
    sum1 += __shfl_xor(sum1, 2, 4);
    amax0 = fmaxf(amax0, __shfl_xor(amax0, 1, 4));
    amax1 = fmaxf(amax1, __shfl_xor(amax1, 1, 4));
    sum0 += __shfl_xor(sum0, 1, 4);
    sum1 += __shfl_xor(sum1, 1, 4);
    const float amax = fmaxf(amax0, amax1);
    const float sum = sum0 + sum1;
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    const char4 q0 = make_char4(quantize_value(v0, id), quantize_value(v1, id),
                                quantize_value(v2, id), quantize_value(v3, id));
    const char4 q1 = make_char4(quantize_value(v4, id), quantize_value(v5, id),
                                quantize_value(v6, id), quantize_value(v7, id));
    const int half = lane >> 4;
    const int lane_in_half = lane & 15;
    block_q8_1_mmq* out = output + static_cast<long long>(head * 2 + half) * rows + row;
    reinterpret_cast<char4*>(out->qs)[lane_in_half * 2] = q0;
    reinterpret_cast<char4*>(out->qs)[lane_in_half * 2 + 1] = q1;
    if((lane & 3) == 0)
        out->ds4[lane_in_half >> 2] = make_half2(static_cast<__half>(d), static_cast<__half>(sum));
}

template <typename Launch>
float time_gpu(hipStream_t stream, Launch launch)
{
    for(int i = 0; i < 20; ++i) launch();
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "create start event");
    check_hip(hipEventCreate(&stop), "create stop event");
    check_hip(hipEventRecord(start, stream), "record start event");
    for(int i = 0; i < 200; ++i) launch();
    check_hip(hipEventRecord(stop, stream), "record stop event");
    check_hip(hipEventSynchronize(stop), "sync stop event");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, start, stop), "elapsed time");
    check_hip(hipEventDestroy(start), "destroy start event");
    check_hip(hipEventDestroy(stop), "destroy stop event");
    return elapsed / 200.0f;
}

} // namespace

int main(int argc, char** argv)
{
    const int rows = argc > 1 ? std::atoi(argv[1]) : 8192;
    if(rows <= 0) return 1;
    const int k = kHeads * kHeadDim;
    const int values = rows * k;
    const int q8_blocks = rows * (k / 128);
    std::mt19937 rng(20260808);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);
    std::vector<__half> attention(values);
    std::vector<float> gate(values);
    std::vector<float> signs1(kHeadDim);
    std::vector<float> signs2(kHeadDim);
    for(int i = 0; i < values; ++i)
    {
        attention[i] = __float2half(value_dist(rng));
        gate[i] = value_dist(rng);
    }
    for(int i = 0; i < kHeadDim; ++i)
    {
        signs1[i] = (rng() & 1) ? 1.0f : -1.0f;
        signs2[i] = (rng() & 1) ? 1.0f : -1.0f;
    }

    __half* d_attention = nullptr;
    float *d_gate = nullptr, *d_signs1 = nullptr, *d_signs2 = nullptr;
    float *d_f32 = nullptr, *d_rotated = nullptr;
    block_q8_1_mmq *d_baseline = nullptr, *d_fused = nullptr;
    check_hip(hipMalloc(&d_attention, values * sizeof(__half)), "malloc attention");
    check_hip(hipMalloc(&d_gate, values * sizeof(float)), "malloc gate");
    check_hip(hipMalloc(&d_signs1, kHeadDim * sizeof(float)), "malloc signs1");
    check_hip(hipMalloc(&d_signs2, kHeadDim * sizeof(float)), "malloc signs2");
    check_hip(hipMalloc(&d_f32, values * sizeof(float)), "malloc f32");
    check_hip(hipMalloc(&d_rotated, values * sizeof(float)), "malloc rotated");
    check_hip(hipMalloc(&d_baseline, q8_blocks * sizeof(block_q8_1_mmq)), "malloc baseline");
    check_hip(hipMalloc(&d_fused, q8_blocks * sizeof(block_q8_1_mmq)), "malloc fused");
    check_hip(hipMemcpy(d_attention, attention.data(), values * sizeof(__half), hipMemcpyHostToDevice), "copy attention");
    check_hip(hipMemcpy(d_gate, gate.data(), values * sizeof(float), hipMemcpyHostToDevice), "copy gate");
    check_hip(hipMemcpy(d_signs1, signs1.data(), kHeadDim * sizeof(float), hipMemcpyHostToDevice), "copy signs1");
    check_hip(hipMemcpy(d_signs2, signs2.data(), kHeadDim * sizeof(float), hipMemcpyHostToDevice), "copy signs2");
    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "create stream");

    auto baseline = [&]() {
        hipLaunchKernelGGL(convert_f16_to_f32, dim3((values / 2 + 255) / 256), dim3(256), 0, stream,
                           d_attention, d_f32, values);
        hipLaunchKernelGGL(sigmoid_mul, dim3((values + 255) / 256), dim3(256), 0, stream,
                           d_f32, d_gate, values);
        hipLaunchKernelGGL(mq_rotate, dim3(rows * kHeads), dim3(32), 0, stream,
                           d_f32, d_rotated, d_signs1, d_signs2, rows * kHeads);
        hipLaunchKernelGGL(quantize_q8_1_mmq_ds4, dim3((k + 1023) / 1024, rows), dim3(256), 0, stream,
                           d_rotated, d_baseline, k, rows);
    };
    auto fused = [&]() {
        hipLaunchKernelGGL(gated_mq_rotate_q8_1, dim3(rows * kHeads), dim3(32), 0, stream,
                           d_attention, d_gate, d_signs1, d_signs2, d_fused, rows, kHeads);
    };
    baseline();
    fused();
    check_hip(hipStreamSynchronize(stream), "sync correctness");
    std::vector<block_q8_1_mmq> baseline_host(q8_blocks);
    std::vector<block_q8_1_mmq> fused_host(q8_blocks);
    check_hip(hipMemcpy(baseline_host.data(), d_baseline, baseline_host.size() * sizeof(block_q8_1_mmq), hipMemcpyDeviceToHost), "copy baseline result");
    check_hip(hipMemcpy(fused_host.data(), d_fused, fused_host.size() * sizeof(block_q8_1_mmq), hipMemcpyDeviceToHost), "copy fused result");
    size_t q_mismatches = 0;
    float max_d_abs = 0.0f;
    float max_sum_abs = 0.0f;
    for(int block = 0; block < q8_blocks; ++block)
    {
        for(int i = 0; i < 128; ++i)
            q_mismatches += baseline_host[block].qs[i] != fused_host[block].qs[i];
        for(int i = 0; i < 4; ++i)
        {
            const float2 a = __half22float2(baseline_host[block].ds4[i]);
            const float2 b = __half22float2(fused_host[block].ds4[i]);
            max_d_abs = std::max(max_d_abs, std::abs(a.x - b.x));
            max_sum_abs = std::max(max_sum_abs, std::abs(a.y - b.y));
        }
    }
    const float baseline_ms = time_gpu(stream, baseline);
    const float fused_ms = time_gpu(stream, fused);
    std::printf("rows=%d baseline_ms=%.6f fused_ms=%.6f speedup=%.3fx q_mismatches=%zu max_d_abs=%.9g max_sum_abs=%.9g\n",
                rows, baseline_ms, fused_ms, baseline_ms / fused_ms, q_mismatches, max_d_abs, max_sum_abs);

    hipStreamDestroy(stream);
    hipFree(d_fused); hipFree(d_baseline); hipFree(d_rotated); hipFree(d_f32);
    hipFree(d_signs2); hipFree(d_signs1); hipFree(d_gate); hipFree(d_attention);
    return q_mismatches == 0 && max_d_abs == 0.0f && max_sum_abs == 0.0f ? 0 : 4;
}
