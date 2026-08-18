// SPDX-License-Identifier: Apache-2.0

#define HIPFIRE_PREDECODE_LIBRARY_ONLY 1
#include "quantized_kv_predecode_bench.hip"

#include "hipfire_flash_attn_ck_quantized.h"

#include <array>

namespace {

constexpr int kQueryHeads = 24;

float deterministic_value(size_t index, unsigned seed)
{
    unsigned value = static_cast<unsigned>(index) ^ seed;
    value = value * 1664525u + 1013904223u;
    return (static_cast<float>((value >> 8) & 0xffffu) / 65535.0f - 0.5f) * 0.2f;
}

template <typename Launch>
float time_once(hipStream_t stream, Launch&& launch)
{
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    launch();
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");
    return elapsed_ms;
}

void check_status(int status, const char* operation, const char* error)
{
    if(status != 0)
    {
        std::fprintf(stderr, "%s failed with status %d: %s\n", operation, status, error);
        std::exit(4);
    }
}

void run_staged_case(int seqlen_q, int seqlen_k)
{
    const int k_row_stride_bytes = kKvHeads * kAsym3HeadBytes;
    const int v_row_stride_bytes = kKvHeads * kQ8HeadBytes;
    const size_t q_values = static_cast<size_t>(seqlen_q) * kQueryHeads * kHeadDim;
    const size_t kv_values = static_cast<size_t>(seqlen_k) * kKvHeads * kHeadDim;
    const size_t k_bytes = static_cast<size_t>(seqlen_k) * k_row_stride_bytes;
    const size_t v_bytes = static_cast<size_t>(seqlen_k) * v_row_stride_bytes;

    std::vector<uint8_t> packed_k(k_bytes);
    std::vector<uint8_t> packed_v(v_bytes);
    std::vector<__half> ignored_k(kv_values);
    std::vector<__half> ignored_v(kv_values);
    make_packed(packed_k, packed_v, ignored_k, ignored_v, seqlen_k);
    std::vector<float> q(q_values);
    for(size_t index = 0; index < q_values; ++index)
    {
        q[index] = deterministic_value(index, 0x1234u);
    }
    std::array<float, kHeadDim / 2> cos_theta{};
    std::array<float, kHeadDim / 2> sin_theta{};
    for(size_t index = 0; index < cos_theta.size(); ++index)
    {
        const float angle = static_cast<float>(index + 1) * 0.0001f;
        cos_theta[index] = std::cos(angle);
        sin_theta[index] = std::sin(angle);
    }

    float* q_device = nullptr;
    uint8_t* packed_k_device = nullptr;
    uint8_t* packed_v_device = nullptr;
    float* packed_out_device = nullptr;
    void* packed_workspace = nullptr;
    float* cos_device = nullptr;
    float* sin_device = nullptr;
    void* staged_workspace = nullptr;
    float* staged_out_device = nullptr;
    check_hip(hipMalloc(&q_device, q_values * sizeof(float)), "hipMalloc(q)");
    check_hip(hipMalloc(&packed_k_device, k_bytes), "hipMalloc(packed K)");
    check_hip(hipMalloc(&packed_v_device, v_bytes), "hipMalloc(packed V)");
    check_hip(hipMalloc(&packed_out_device, q_values * sizeof(float)),
              "hipMalloc(packed output)");
    const size_t packed_workspace_bytes =
        hipfire_flash_attn_ck_quantized_prefill_workspace_bytes(
            seqlen_q, kQueryHeads, kHeadDim);
    check_hip(hipMalloc(&packed_workspace, packed_workspace_bytes),
              "hipMalloc(packed workspace)");
    check_hip(hipMalloc(&cos_device, cos_theta.size() * sizeof(float)),
              "hipMalloc(cos)");
    check_hip(hipMalloc(&sin_device, sin_theta.size() * sizeof(float)),
              "hipMalloc(sin)");
    const size_t staged_workspace_bytes =
        hipfire_flash_attn_ck_quantized_staged_workspace_bytes(
            seqlen_q, seqlen_k, kQueryHeads, kKvHeads, kHeadDim);
    check_hip(hipMalloc(&staged_workspace, staged_workspace_bytes),
              "hipMalloc(staged workspace)");
    check_hip(hipMalloc(&staged_out_device, q_values * sizeof(float)),
              "hipMalloc(staged output)");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
    check_hip(hipMemcpyAsync(q_device,
                            q.data(),
                            q_values * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpy(q)");
    check_hip(hipMemcpyAsync(packed_k_device,
                            packed_k.data(),
                            k_bytes,
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpy(packed K)");
    check_hip(hipMemcpyAsync(packed_v_device,
                            packed_v.data(),
                            v_bytes,
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpy(packed V)");
    check_hip(hipMemcpyAsync(cos_device,
                            cos_theta.data(),
                            cos_theta.size() * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpy(cos)");
    check_hip(hipMemcpyAsync(sin_device,
                            sin_theta.data(),
                            sin_theta.size() * sizeof(float),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpy(sin)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(inputs)");

    hipfire_flash_attn_ck_quantized_prefill_params packed_params{};
    packed_params.abi_version = HIPFIRE_FLASH_ATTN_CK_QUANTIZED_ABI_VERSION;
    packed_params.struct_size = sizeof(packed_params);
    packed_params.q = q_device;
    packed_params.packed_k = packed_k_device;
    packed_params.packed_v = packed_v_device;
    packed_params.out = packed_out_device;
    packed_params.workspace = packed_workspace;
    packed_params.workspace_bytes = packed_workspace_bytes;
    packed_params.cos_theta = cos_device;
    packed_params.sin_theta = sin_device;
    packed_params.stream = stream;
    packed_params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    packed_params.seqlen_q = seqlen_q;
    packed_params.seqlen_k = seqlen_k;
    packed_params.nhead_q = kQueryHeads;
    packed_params.nhead_k = kKvHeads;
    packed_params.head_dim = kHeadDim;
    packed_params.causal = 1;
    packed_params.k_row_stride_bytes = k_row_stride_bytes;
    packed_params.v_row_stride_bytes = v_row_stride_bytes;

    hipfire_flash_attn_ck_quantized_prefill_params staged_params = packed_params;
    staged_params.out = staged_out_device;
    staged_params.workspace = staged_workspace;
    staged_params.workspace_bytes = staged_workspace_bytes;

    char error[512]{};
    auto launch_packed = [&]() {
        check_status(hipfire_flash_attn_ck_quantized_prefill(
                         &packed_params, error, sizeof(error)),
                     "packed CK",
                     error);
    };
    auto launch_staged = [&]() {
        check_status(hipfire_flash_attn_ck_quantized_staged_prefill(
                         &staged_params, error, sizeof(error)),
                     "staged CK",
                     error);
    };

    for(int iteration = 0; iteration < 5; ++iteration)
    {
        launch_packed();
        launch_staged();
    }
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");

    constexpr int trials = 7;
    std::vector<float> packed_samples;
    std::vector<float> staged_samples;
    for(int trial = 0; trial < trials; ++trial)
    {
        if(trial % 2 == 0)
        {
            packed_samples.push_back(time_once(stream, launch_packed));
            staged_samples.push_back(time_once(stream, launch_staged));
        }
        else
        {
            staged_samples.push_back(time_once(stream, launch_staged));
            packed_samples.push_back(time_once(stream, launch_packed));
        }
    }

    launch_packed();
    launch_staged();
    std::vector<float> packed_output(q_values);
    std::vector<float> staged_output(q_values);
    check_hip(hipMemcpyAsync(packed_output.data(),
                            packed_out_device,
                            q_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpy(packed output)");
    check_hip(hipMemcpyAsync(staged_output.data(),
                            staged_out_device,
                            q_values * sizeof(float),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpy(staged output)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(outputs)");
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t index = 0; index < q_values; ++index)
    {
        const float error_value = std::abs(packed_output[index] - staged_output[index]);
        max_abs = std::max(max_abs, error_value);
        mean_abs += error_value;
    }
    mean_abs /= static_cast<double>(q_values);
    const float packed_ms = median(packed_samples);
    const float staged_ms = median(staged_samples);
    std::printf("seqlen_q=%d,seqlen_k=%d,packed_ck_ms=%.9g,staged_ck_ms=%.9g,"
                "staged_speedup=%.9g,max_abs=%.9g,mean_abs=%.9g,staged_scratch_bytes=%zu\n",
                seqlen_q,
                seqlen_k,
                packed_ms,
                staged_ms,
                packed_ms / staged_ms,
                max_abs,
                mean_abs,
                staged_workspace_bytes);
    if(max_abs > 0.03f)
    {
        std::fprintf(stderr, "staged CK correctness failed\n");
        std::exit(5);
    }

    check_hip(hipFree(q_device), "hipFree(q)");
    check_hip(hipFree(packed_k_device), "hipFree(packed K)");
    check_hip(hipFree(packed_v_device), "hipFree(packed V)");
    check_hip(hipFree(packed_out_device), "hipFree(packed output)");
    check_hip(hipFree(packed_workspace), "hipFree(packed workspace)");
    check_hip(hipFree(cos_device), "hipFree(cos)");
    check_hip(hipFree(sin_device), "hipFree(sin)");
    check_hip(hipFree(staged_workspace), "hipFree(staged workspace)");
    check_hip(hipFree(staged_out_device), "hipFree(staged output)");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
}

} // namespace

int main()
{
    for(const int seqlen_k : {2048, 4096, 6144, 8192})
    {
        run_staged_case(2048, seqlen_k);
    }
    return 0;
}
