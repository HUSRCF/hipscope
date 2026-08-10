// Full standalone screen for a high-memory rowwise-W8A8 prefill sidecar.
// Weight conversion is load-time work and is excluded from the hot-path
// timing; activation quantization, rocBLAS GEMM, and scale epilogue are timed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using rocblas_handle = void*;

extern "C" unsigned int rocblas_create_handle(rocblas_handle*);
extern "C" unsigned int rocblas_destroy_handle(rocblas_handle);
extern "C" unsigned int rocblas_set_stream(rocblas_handle, hipStream_t);
extern "C" unsigned int rocblas_gemm_ex(
    rocblas_handle, unsigned int, unsigned int, int, int, int,
    const void*, const void*, unsigned int, int, const void*, unsigned int,
    int, const void*, const void*, unsigned int, int, void*, unsigned int,
    int, unsigned int, unsigned int, int32_t, uint32_t);

namespace {

constexpr unsigned int kOpNone = 111;
constexpr unsigned int kOpTranspose = 112;
constexpr unsigned int kI8 = 160;
constexpr unsigned int kI32 = 162;
constexpr unsigned int kAlgoStandard = 160;
constexpr unsigned int kSuccess = 0;
constexpr int kThreads = 256;

__host__ __device__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__host__ __device__ float source_value(size_t index, uint32_t seed) {
    const uint32_t bits = hash32(static_cast<uint32_t>(index) ^ seed);
    return (static_cast<int>(bits & 0xffffu) - 32768) * (1.0f / 32768.0f);
}

__global__ void fill_source(float* dst, size_t count, uint32_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        dst[i] = source_value(i, seed);
    }
}

__global__ void quantize_rows_i8(
    const float* src, int8_t* dst, float* scales, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scratch[kThreads];
    float local_max = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(src[base + col]));
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = scratch[0] > 0.0f ? scratch[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int q = __float2int_rn(src[base + col] * inv_scale);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        dst[base + col] = static_cast<int8_t>(q);
    }
}

__global__ void scale_epilogue(
    const int32_t* input, const float* x_scale, const float* w_scale,
    float* output, int m, int n) {
    const size_t count = static_cast<size_t>(m) * n;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const int row = static_cast<int>(i / m);
        const int col = static_cast<int>(i - static_cast<size_t>(row) * m);
        output[i] = static_cast<float>(input[i]) * x_scale[row] * w_scale[col];
    }
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

void check_rocblas(unsigned int status, const char* what) {
    if (status != kSuccess) {
        std::fprintf(stderr, "%s: rocBLAS status %u\n", what, status);
        std::exit(3);
    }
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 17408;
    const int k = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int n = argc > 3 ? std::atoi(argv[3]) : 2048;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 5;
    const int trials = argc > 5 ? std::atoi(argv[5]) : 15;
    const uint32_t weight_seed = 0x13579bdu;
    const uint32_t activation_seed = 0x2468aceu;

    const size_t weight_count = static_cast<size_t>(m) * k;
    const size_t activation_count = static_cast<size_t>(n) * k;
    const size_t output_count = static_cast<size_t>(n) * m;
    float *weight_f32 = nullptr, *activation_f32 = nullptr, *weight_scale = nullptr,
          *activation_scale = nullptr, *output_f32 = nullptr;
    int8_t *weight_i8 = nullptr, *activation_i8 = nullptr;
    int32_t* output_i32 = nullptr;

    check_hip(hipMalloc(&weight_f32, weight_count * sizeof(float)), "alloc weight f32");
    check_hip(hipMalloc(&activation_f32, activation_count * sizeof(float)), "alloc activation f32");
    check_hip(hipMalloc(&weight_i8, weight_count), "alloc weight i8");
    check_hip(hipMalloc(&activation_i8, activation_count), "alloc activation i8");
    check_hip(hipMalloc(&weight_scale, static_cast<size_t>(m) * sizeof(float)), "alloc weight scale");
    check_hip(hipMalloc(&activation_scale, static_cast<size_t>(n) * sizeof(float)), "alloc activation scale");
    check_hip(hipMalloc(&output_i32, output_count * sizeof(int32_t)), "alloc output i32");
    check_hip(hipMalloc(&output_f32, output_count * sizeof(float)), "alloc output f32");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    const int fill_blocks = 1024;
    hipLaunchKernelGGL(fill_source, dim3(fill_blocks), dim3(kThreads), 0, stream,
                       weight_f32, weight_count, weight_seed);
    hipLaunchKernelGGL(fill_source, dim3(fill_blocks), dim3(kThreads), 0, stream,
                       activation_f32, activation_count, activation_seed);
    hipLaunchKernelGGL(quantize_rows_i8, dim3(m), dim3(kThreads), 0, stream,
                       weight_f32, weight_i8, weight_scale, m, k);

    rocblas_handle handle = nullptr;
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
    check_rocblas(rocblas_set_stream(handle, stream), "rocblas_set_stream");
    const int32_t alpha = 1;
    const int32_t beta = 0;
    auto gemm = [&]() {
        check_rocblas(
            rocblas_gemm_ex(
                handle, kOpTranspose, kOpNone, m, n, k, &alpha, weight_i8, kI8,
                k, activation_i8, kI8, k, &beta, output_i32, kI32, m,
                output_i32, kI32, m, kI32, kAlgoStandard, 0, 0),
            "rocblas_gemm_ex i8");
    };
    const int epilogue_blocks = std::min<int>(65535, (output_count + kThreads - 1) / kThreads);
    auto pipeline = [&]() {
        hipLaunchKernelGGL(quantize_rows_i8, dim3(n), dim3(kThreads), 0, stream,
                           activation_f32, activation_i8, activation_scale, n, k);
        gemm();
        hipLaunchKernelGGL(scale_epilogue, dim3(epilogue_blocks), dim3(kThreads), 0, stream,
                           output_i32, activation_scale, weight_scale, output_f32, m, n);
    };

    for (int i = 0; i < warmup; ++i) pipeline();
    check_hip(hipStreamSynchronize(stream), "warmup sync");

    hipEvent_t events[4] = {};
    for (auto& event : events) check_hip(hipEventCreate(&event), "event create");
    std::vector<float> quant_ms, gemm_ms, epilogue_ms, total_ms;
    for (int i = 0; i < trials; ++i) {
        check_hip(hipEventRecord(events[0], stream), "event 0");
        hipLaunchKernelGGL(quantize_rows_i8, dim3(n), dim3(kThreads), 0, stream,
                           activation_f32, activation_i8, activation_scale, n, k);
        check_hip(hipEventRecord(events[1], stream), "event 1");
        gemm();
        check_hip(hipEventRecord(events[2], stream), "event 2");
        hipLaunchKernelGGL(scale_epilogue, dim3(epilogue_blocks), dim3(kThreads), 0, stream,
                           output_i32, activation_scale, weight_scale, output_f32, m, n);
        check_hip(hipEventRecord(events[3], stream), "event 3");
        check_hip(hipEventSynchronize(events[3]), "event sync");
        float q = 0.0f, g = 0.0f, e = 0.0f, t = 0.0f;
        check_hip(hipEventElapsedTime(&q, events[0], events[1]), "quant elapsed");
        check_hip(hipEventElapsedTime(&g, events[1], events[2]), "gemm elapsed");
        check_hip(hipEventElapsedTime(&e, events[2], events[3]), "epilogue elapsed");
        check_hip(hipEventElapsedTime(&t, events[0], events[3]), "total elapsed");
        quant_ms.push_back(q);
        gemm_ms.push_back(g);
        epilogue_ms.push_back(e);
        total_ms.push_back(t);
    }

    const int samples = std::min(m, 32);
    std::vector<float> observed(samples);
    check_hip(hipMemcpy(observed.data(), output_f32, samples * sizeof(float),
                        hipMemcpyDeviceToHost), "copy output samples");
    double ref_norm2 = 0.0, err_norm2 = 0.0, dot = 0.0, obs_norm2 = 0.0;
    double max_abs = 0.0;
    for (int col = 0; col < samples; ++col) {
        double ref = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            ref += static_cast<double>(source_value(static_cast<size_t>(kk), activation_seed))
                * source_value(static_cast<size_t>(col) * k + kk, weight_seed);
        }
        const double err = static_cast<double>(observed[col]) - ref;
        max_abs = std::max(max_abs, std::abs(err));
        err_norm2 += err * err;
        ref_norm2 += ref * ref;
        obs_norm2 += static_cast<double>(observed[col]) * observed[col];
        dot += static_cast<double>(observed[col]) * ref;
    }

    std::printf("shape=M%d K%d N%d warmup=%d trials=%d\n", m, k, n, warmup, trials);
    std::printf("resident_weight_i8_bytes=%zu resident_weight_scale_bytes=%zu\n",
                weight_count, static_cast<size_t>(m) * sizeof(float));
    std::printf("activation_quant_median_ms=%.4f\n", median(quant_ms));
    std::printf("rocblas_i8_median_ms=%.4f\n", median(gemm_ms));
    std::printf("scale_epilogue_median_ms=%.4f\n", median(epilogue_ms));
    std::printf("pipeline_total_median_ms=%.4f\n", median(total_ms));
    std::printf("sample_max_abs=%.6g sample_relative_l2=%.6g sample_cosine=%.9f\n",
                max_abs, std::sqrt(err_norm2 / ref_norm2),
                dot / std::sqrt(ref_norm2 * obs_norm2));

    for (auto& event : events) check_hip(hipEventDestroy(event), "event destroy");
    rocblas_destroy_handle(handle);
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    check_hip(hipFree(output_f32), "free output f32");
    check_hip(hipFree(output_i32), "free output i32");
    check_hip(hipFree(activation_scale), "free activation scale");
    check_hip(hipFree(weight_scale), "free weight scale");
    check_hip(hipFree(activation_i8), "free activation i8");
    check_hip(hipFree(weight_i8), "free weight i8");
    check_hip(hipFree(activation_f32), "free activation f32");
    check_hip(hipFree(weight_f32), "free weight f32");
    return 0;
}
