// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include "../../kernels/src/attention_dflash.hip"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Case
{
    int seqlen_q;
    int seqlen_k;
    int nhead_q;
    int nhead_k;
};

struct F16Buffers
{
    __half* q = nullptr;
    __half* k = nullptr;
    __half* v = nullptr;
    __half* out = nullptr;
};

void hip_check(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
        std::exit(2);
    }
}

void ck_check(int status, const char* operation, const char* error)
{
    if(status != 0)
    {
        std::fprintf(stderr, "%s failed with status %d: %s\n", operation, status, error);
        std::exit(2);
    }
}

__global__ void cast_f32_to_f16(const float* src, __half* dst, size_t count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < count)
    {
        dst[index] = __float2half(src[index]);
    }
}

__global__ void cast_f16_to_f32(const __half* src, float* dst, size_t count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < count)
    {
        dst[index] = __half2float(src[index]);
    }
}

unsigned next_power_of_two(unsigned value)
{
    if(value <= 1)
    {
        return 1;
    }
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

float deterministic_value(size_t index, unsigned seed)
{
    unsigned value = static_cast<unsigned>(index) ^ seed;
    value = value * 1664525u + 1013904223u;
    return (static_cast<float>((value >> 8) & 0xffffu) / 65535.0f - 0.5f) * 0.2f;
}

template <typename Launch>
float time_launches(Launch&& launch, int iterations)
{
    hipEvent_t start{};
    hipEvent_t stop{};
    hip_check(hipEventCreate(&start), "hipEventCreate(start)");
    hip_check(hipEventCreate(&stop), "hipEventCreate(stop)");
    hip_check(hipEventRecord(start, nullptr), "hipEventRecord(start)");
    for(int iteration = 0; iteration < iterations; ++iteration)
    {
        launch();
    }
    hip_check(hipEventRecord(stop, nullptr), "hipEventRecord(stop)");
    hip_check(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    hip_check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    hip_check(hipEventDestroy(start), "hipEventDestroy(start)");
    hip_check(hipEventDestroy(stop), "hipEventDestroy(stop)");
    return elapsed_ms / static_cast<float>(iterations);
}

template <typename Launch>
float time_wall_launches(Launch&& launch, int iterations)
{
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize(wall start)");
    const auto start = std::chrono::steady_clock::now();
    for(int iteration = 0; iteration < iterations; ++iteration)
    {
        launch();
    }
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize(wall stop)");
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<float, std::milli>(elapsed).count() /
           static_cast<float>(iterations);
}

void run_case(const Case& benchmark_case, int warmup, int trials, int iterations)
{
    constexpr int head_dim = 64;
    constexpr int batch = 1;
    const int b = benchmark_case.seqlen_q;
    const int l = benchmark_case.seqlen_k;
    const int nhead_q = benchmark_case.nhead_q;
    const int nhead_k = benchmark_case.nhead_k;

    const size_t q_count = static_cast<size_t>(b) * nhead_q * head_dim;
    const size_t kv_count = static_cast<size_t>(l) * nhead_k * head_dim;
    const size_t out_count = q_count;

    std::vector<float> q_host(q_count);
    std::vector<float> k_host(kv_count);
    std::vector<float> v_host(kv_count);
    std::vector<__half> q_half_host(q_count);
    std::vector<__half> k_half_host(kv_count);
    std::vector<__half> v_half_host(kv_count);
    for(size_t index = 0; index < q_count; ++index)
    {
        q_host[index] = deterministic_value(index, 0x1234u);
        q_half_host[index] = __float2half(q_host[index]);
    }
    for(size_t index = 0; index < kv_count; ++index)
    {
        k_host[index] = deterministic_value(index, 0x5678u);
        v_host[index] = deterministic_value(index, 0x9abcu);
        k_half_host[index] = __float2half(k_host[index]);
        v_half_host[index] = __float2half(v_host[index]);
    }

    float* q_f32 = nullptr;
    float* k_f32 = nullptr;
    float* v_f32 = nullptr;
    float* out_f32 = nullptr;
    float* qo_bridge_out_f32 = nullptr;
    float* full_f32_bridge_out_f32 = nullptr;
    F16Buffers direct_buffers;
    F16Buffers qo_bridge_buffers;
    F16Buffers full_f32_bridge_buffers;
    hip_check(hipMalloc(&q_f32, q_count * sizeof(float)), "hipMalloc(q_f32)");
    hip_check(hipMalloc(&k_f32, kv_count * sizeof(float)), "hipMalloc(k_f32)");
    hip_check(hipMalloc(&v_f32, kv_count * sizeof(float)), "hipMalloc(v_f32)");
    hip_check(hipMalloc(&out_f32, out_count * sizeof(float)), "hipMalloc(out_f32)");
    hip_check(
        hipMalloc(&qo_bridge_out_f32, out_count * sizeof(float)),
        "hipMalloc(qo_bridge_out_f32)");
    hip_check(
        hipMalloc(&full_f32_bridge_out_f32, out_count * sizeof(float)),
        "hipMalloc(full_f32_bridge_out_f32)");
    const auto allocate_f16_buffers = [&](F16Buffers& buffers, const char* label) {
        hip_check(hipMalloc(&buffers.q, q_count * sizeof(__half)), label);
        hip_check(hipMalloc(&buffers.k, kv_count * sizeof(__half)), label);
        hip_check(hipMalloc(&buffers.v, kv_count * sizeof(__half)), label);
        hip_check(hipMalloc(&buffers.out, out_count * sizeof(__half)), label);
    };
    allocate_f16_buffers(direct_buffers, "hipMalloc(direct F16 buffers)");
    allocate_f16_buffers(qo_bridge_buffers, "hipMalloc(Q/O bridge F16 buffers)");
    allocate_f16_buffers(full_f32_bridge_buffers, "hipMalloc(full-F32 bridge F16 buffers)");

    hip_check(
        hipMemcpy(q_f32, q_host.data(), q_count * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy(q_f32)");
    hip_check(
        hipMemcpy(k_f32, k_host.data(), kv_count * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy(k_f32)");
    hip_check(
        hipMemcpy(v_f32, v_host.data(), kv_count * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy(v_f32)");
    const auto initialize_f16_buffers = [&](F16Buffers& buffers, const char* label) {
        hip_check(
            hipMemcpy(
                buffers.q,
                q_half_host.data(),
                q_count * sizeof(__half),
                hipMemcpyHostToDevice),
            label);
        hip_check(
            hipMemcpy(
                buffers.k,
                k_half_host.data(),
                kv_count * sizeof(__half),
                hipMemcpyHostToDevice),
            label);
        hip_check(
            hipMemcpy(
                buffers.v,
                v_half_host.data(),
                kv_count * sizeof(__half),
                hipMemcpyHostToDevice),
            label);
    };
    initialize_f16_buffers(direct_buffers, "hipMemcpy(direct F16 buffers)");
    initialize_f16_buffers(qo_bridge_buffers, "hipMemcpy(Q/O bridge F16 buffers)");
    initialize_f16_buffers(
        full_f32_bridge_buffers, "hipMemcpy(full-F32 bridge F16 buffers)");

    const unsigned block_size =
        next_power_of_two(std::min(256, std::max(l, head_dim)));
    constexpr size_t lds_budget_f32 = 14336;
    const size_t fixed = block_size + head_dim;
    const size_t max_tile_room = lds_budget_f32 > fixed ? lds_budget_f32 - fixed : 1;
    const int tile_size = std::min<size_t>(l, std::max<size_t>(1, max_tile_room));
    const size_t shared_bytes =
        static_cast<size_t>(tile_size + block_size + head_dim) * sizeof(float);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const auto make_params = [&](const F16Buffers& buffers) {
        hipfire_flash_attn_ck_fwd_params params{};
        params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
        params.struct_size = sizeof(params);
        params.q = buffers.q;
        params.k = buffers.k;
        params.v = buffers.v;
        params.out = buffers.out;
        params.stream = nullptr;
        params.dtype = HIPFIRE_FLASH_ATTN_CK_F16;
        params.batch = batch;
        params.seqlen_q = b;
        params.seqlen_k = l;
        params.nhead_q = nhead_q;
        params.nhead_k = nhead_k;
        params.head_dim = head_dim;
        params.causal = 0;
        params.softmax_scale = scale;
        params.stride_q = nhead_q * head_dim;
        params.stride_k = nhead_k * head_dim;
        params.stride_v = nhead_k * head_dim;
        params.stride_out = nhead_q * head_dim;
        params.nhead_stride_q = head_dim;
        params.nhead_stride_k = head_dim;
        params.nhead_stride_v = head_dim;
        params.nhead_stride_out = head_dim;
        params.batch_stride_q = static_cast<int64_t>(b) * nhead_q * head_dim;
        params.batch_stride_k = static_cast<int64_t>(l) * nhead_k * head_dim;
        params.batch_stride_v = static_cast<int64_t>(l) * nhead_k * head_dim;
        params.batch_stride_out = static_cast<int64_t>(b) * nhead_q * head_dim;
        return params;
    };
    const auto direct_params = make_params(direct_buffers);
    const auto qo_bridge_params = make_params(qo_bridge_buffers);
    const auto full_f32_bridge_params = make_params(full_f32_bridge_buffers);

    char error[512]{};
    ck_check(
        hipfire_flash_attn_ck_fwd_supported(&direct_params, error, sizeof(error)),
        "hipfire_flash_attn_ck_fwd_supported",
        error);

    const auto launch_native = [&]() {
        hipLaunchKernelGGL(
            attention_dflash_f32,
            dim3(nhead_q, b, 1),
            dim3(block_size, 1, 1),
            shared_bytes,
            nullptr,
            q_f32,
            k_f32,
            v_f32,
            out_f32,
            b,
            l,
            nhead_q,
            nhead_k,
            head_dim,
            scale,
            tile_size);
    };
    const auto launch_ck = [&](const hipfire_flash_attn_ck_fwd_params& params) {
        error[0] = '\0';
        ck_check(
            hipfire_flash_attn_ck_fwd(&params, error, sizeof(error)),
            "hipfire_flash_attn_ck_fwd",
            error);
    };
    const auto launch_direct_ck = [&]() { launch_ck(direct_params); };
    const auto launch_qo_bridge = [&]() {
        constexpr unsigned threads = 256;
        hipLaunchKernelGGL(
            cast_f32_to_f16,
            dim3((q_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            q_f32,
            qo_bridge_buffers.q,
            q_count);
        launch_ck(qo_bridge_params);
        hipLaunchKernelGGL(
            cast_f16_to_f32,
            dim3((out_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            qo_bridge_buffers.out,
            qo_bridge_out_f32,
            out_count);
    };
    const auto launch_full_f32_bridge = [&]() {
        constexpr unsigned threads = 256;
        hipLaunchKernelGGL(
            cast_f32_to_f16,
            dim3((q_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            q_f32,
            full_f32_bridge_buffers.q,
            q_count);
        hipLaunchKernelGGL(
            cast_f32_to_f16,
            dim3((kv_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            k_f32,
            full_f32_bridge_buffers.k,
            kv_count);
        hipLaunchKernelGGL(
            cast_f32_to_f16,
            dim3((kv_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            v_f32,
            full_f32_bridge_buffers.v,
            kv_count);
        launch_ck(full_f32_bridge_params);
        hipLaunchKernelGGL(
            cast_f16_to_f32,
            dim3((out_count + threads - 1) / threads),
            dim3(threads),
            0,
            nullptr,
            full_f32_bridge_buffers.out,
            full_f32_bridge_out_f32,
            out_count);
    };

    for(int index = 0; index < warmup; ++index)
    {
        launch_native();
        launch_direct_ck();
        launch_qo_bridge();
        launch_full_f32_bridge();
    }
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize(warmup)");
    hip_check(hipGetLastError(), "warmup launch");

    std::vector<float> native_samples;
    std::vector<float> ck_samples;
    std::vector<float> qo_bridge_samples;
    std::vector<float> full_f32_bridge_samples;
    std::vector<float> native_wall_samples;
    std::vector<float> ck_wall_samples;
    std::vector<float> qo_bridge_wall_samples;
    std::vector<float> full_f32_bridge_wall_samples;
    native_samples.reserve(trials);
    ck_samples.reserve(trials);
    qo_bridge_samples.reserve(trials);
    full_f32_bridge_samples.reserve(trials);
    native_wall_samples.reserve(trials);
    ck_wall_samples.reserve(trials);
    qo_bridge_wall_samples.reserve(trials);
    full_f32_bridge_wall_samples.reserve(trials);
    for(int trial = 0; trial < trials; ++trial)
    {
        if(trial % 2 == 0)
        {
            native_samples.push_back(time_launches(launch_native, iterations));
            ck_samples.push_back(time_launches(launch_direct_ck, iterations));
            qo_bridge_samples.push_back(time_launches(launch_qo_bridge, iterations));
            full_f32_bridge_samples.push_back(
                time_launches(launch_full_f32_bridge, iterations));
            native_wall_samples.push_back(time_wall_launches(launch_native, iterations));
            ck_wall_samples.push_back(time_wall_launches(launch_direct_ck, iterations));
            qo_bridge_wall_samples.push_back(
                time_wall_launches(launch_qo_bridge, iterations));
            full_f32_bridge_wall_samples.push_back(
                time_wall_launches(launch_full_f32_bridge, iterations));
        }
        else
        {
            full_f32_bridge_wall_samples.push_back(
                time_wall_launches(launch_full_f32_bridge, iterations));
            qo_bridge_wall_samples.push_back(
                time_wall_launches(launch_qo_bridge, iterations));
            ck_wall_samples.push_back(time_wall_launches(launch_direct_ck, iterations));
            native_wall_samples.push_back(time_wall_launches(launch_native, iterations));
            full_f32_bridge_samples.push_back(
                time_launches(launch_full_f32_bridge, iterations));
            qo_bridge_samples.push_back(time_launches(launch_qo_bridge, iterations));
            ck_samples.push_back(time_launches(launch_direct_ck, iterations));
            native_samples.push_back(time_launches(launch_native, iterations));
        }
    }

    launch_native();
    launch_direct_ck();
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize(correctness)");
    std::vector<float> native_out(out_count);
    std::vector<__half> ck_out(out_count);
    hip_check(
        hipMemcpy(
            native_out.data(), out_f32, out_count * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy(native_out)");
    hip_check(
        hipMemcpy(
            ck_out.data(),
            direct_buffers.out,
            out_count * sizeof(__half),
            hipMemcpyDeviceToHost),
        "hipMemcpy(ck_out)");
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t index = 0; index < out_count; ++index)
    {
        const float difference =
            std::abs(native_out[index] - __half2float(ck_out[index]));
        max_abs = std::max(max_abs, difference);
        mean_abs += difference;
    }
    mean_abs /= static_cast<double>(out_count);

    const float native_ms = median(native_samples);
    const float ck_ms = median(ck_samples);
    const float qo_bridge_ms = median(qo_bridge_samples);
    const float full_f32_bridge_ms = median(full_f32_bridge_samples);
    const float native_wall_ms = median(native_wall_samples);
    const float ck_wall_ms = median(ck_wall_samples);
    const float qo_bridge_wall_ms = median(qo_bridge_wall_samples);
    const float full_f32_bridge_wall_ms = median(full_f32_bridge_wall_samples);
    std::printf(
        "%d,%d,%d,%d,%d,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,"
        "%.3f,%.3f,%.3f,"
        "%.3f,%.3f,%.3f,"
        "%.7g,%.7g\n",
        b,
        l,
        nhead_q,
        nhead_k,
        head_dim,
        native_ms,
        ck_ms,
        qo_bridge_ms,
        full_f32_bridge_ms,
        native_wall_ms,
        ck_wall_ms,
        qo_bridge_wall_ms,
        full_f32_bridge_wall_ms,
        native_ms / ck_ms,
        native_ms / qo_bridge_ms,
        native_ms / full_f32_bridge_ms,
        native_wall_ms / ck_wall_ms,
        native_wall_ms / qo_bridge_wall_ms,
        native_wall_ms / full_f32_bridge_wall_ms,
        max_abs,
        mean_abs);
    std::fflush(stdout);

    hip_check(hipFree(q_f32), "hipFree(q_f32)");
    hip_check(hipFree(k_f32), "hipFree(k_f32)");
    hip_check(hipFree(v_f32), "hipFree(v_f32)");
    hip_check(hipFree(out_f32), "hipFree(out_f32)");
    hip_check(hipFree(qo_bridge_out_f32), "hipFree(qo_bridge_out_f32)");
    hip_check(
        hipFree(full_f32_bridge_out_f32), "hipFree(full_f32_bridge_out_f32)");
    const auto free_f16_buffers = [&](F16Buffers& buffers, const char* label) {
        hip_check(hipFree(buffers.q), label);
        hip_check(hipFree(buffers.k), label);
        hip_check(hipFree(buffers.v), label);
        hip_check(hipFree(buffers.out), label);
    };
    free_f16_buffers(direct_buffers, "hipFree(direct F16 buffers)");
    free_f16_buffers(qo_bridge_buffers, "hipFree(Q/O bridge F16 buffers)");
    free_f16_buffers(full_f32_bridge_buffers, "hipFree(full-F32 bridge F16 buffers)");
}

int parse_positive(const char* value, const char* option)
{
    const int result = std::atoi(value);
    if(result <= 0)
    {
        std::fprintf(stderr, "%s must be positive\n", option);
        std::exit(2);
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    int warmup = 3;
    int trials = 9;
    int iterations = 5;
    for(int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if(index + 1 >= argc)
        {
            std::fprintf(stderr, "missing value for %s\n", option.c_str());
            return 2;
        }
        if(option == "--warmup")
        {
            warmup = parse_positive(argv[++index], "--warmup");
        }
        else if(option == "--trials")
        {
            trials = parse_positive(argv[++index], "--trials");
            if(trials % 2 == 0)
            {
                std::fprintf(stderr, "--trials must be odd for an exact median\n");
                return 2;
            }
        }
        else if(option == "--iterations")
        {
            iterations = parse_positive(argv[++index], "--iterations");
        }
        else
        {
            std::fprintf(stderr, "unknown option: %s\n", option.c_str());
            return 2;
        }
    }

    const std::vector<Case> cases = {
        {64, 512, 8, 8},
        {64, 1024, 8, 8},
        {64, 2048, 8, 8},
        {64, 4096, 8, 8},
        {96, 1024, 8, 8},
        {96, 2048, 8, 8},
        {128, 512, 8, 8},
        {128, 1024, 8, 8},
        {128, 2048, 8, 8},
        {192, 1024, 8, 8},
        {192, 2048, 8, 8},
        {256, 512, 8, 8},
        {256, 1024, 8, 8},
        {256, 2048, 8, 8},
        {512, 4096, 8, 8},
        {256, 2048, 8, 2},
    };
    std::puts(
        "seqlen_q,seqlen_k,nhead_q,nhead_k,head_dim,native_f32_ms,ck_f16_ms,"
        "ck_qo_bridge_ms,ck_full_f32_bridge_ms,"
        "native_f32_wall_ms,ck_f16_wall_ms,ck_qo_bridge_wall_ms,"
        "ck_full_f32_bridge_wall_ms,"
        "ck_f16_gpu_speedup_vs_native_f32,ck_qo_bridge_gpu_speedup,"
        "ck_full_f32_bridge_gpu_speedup,"
        "ck_f16_wall_speedup_vs_native_f32,ck_qo_bridge_wall_speedup,"
        "ck_full_f32_bridge_wall_speedup,max_abs,mean_abs");
    for(const Case& benchmark_case : cases)
    {
        run_case(benchmark_case, warmup, trials, iterations);
    }
    return 0;
}
