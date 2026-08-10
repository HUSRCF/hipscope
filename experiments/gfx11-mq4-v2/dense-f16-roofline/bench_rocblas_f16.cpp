#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using rocblas_handle = void*;
extern "C" unsigned int rocblas_create_handle(rocblas_handle*);
extern "C" unsigned int rocblas_destroy_handle(rocblas_handle);
extern "C" unsigned int rocblas_gemm_ex(rocblas_handle,
                                         unsigned int,
                                         unsigned int,
                                         int,
                                         int,
                                         int,
                                         const void*,
                                         const void*,
                                         unsigned int,
                                         int,
                                         const void*,
                                         unsigned int,
                                         int,
                                         const void*,
                                         const void*,
                                         unsigned int,
                                         int,
                                         void*,
                                         unsigned int,
                                         int,
                                         unsigned int,
                                         unsigned int,
                                         int32_t,
                                         uint32_t);

namespace {
constexpr unsigned int kSuccess = 0;
constexpr unsigned int kNone    = 111;
constexpr unsigned int kF16     = 150;
constexpr unsigned int kF32     = 151;
constexpr unsigned int kAlgo    = 160;

void check_hip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

void check_rocblas(unsigned int status, const char* what)
{
    if(status != kSuccess)
    {
        std::fprintf(stderr, "%s: rocBLAS status %u\n", what, status);
        std::exit(3);
    }
}
} // namespace

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::fprintf(stderr, "usage: %s M N K\n", argv[0]);
        return 2;
    }
    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);

    __half *weight = nullptr, *activation = nullptr, *output = nullptr;
    check_hip(hipMalloc(&weight, sizeof(__half) * static_cast<size_t>(m) * k), "weight");
    check_hip(hipMalloc(&activation, sizeof(__half) * static_cast<size_t>(k) * n), "activation");
    check_hip(hipMalloc(&output, sizeof(__half) * static_cast<size_t>(m) * n), "output");
    check_hip(hipMemset(weight, 0, sizeof(__half) * static_cast<size_t>(m) * k), "zero weight");
    check_hip(hipMemset(activation, 0, sizeof(__half) * static_cast<size_t>(k) * n),
              "zero activation");

    rocblas_handle handle = nullptr;
    check_rocblas(rocblas_create_handle(&handle), "create handle");
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto launch       = [&] {
        check_rocblas(rocblas_gemm_ex(handle,
                                      kNone,
                                      kNone,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      weight,
                                      kF16,
                                      m,
                                      activation,
                                      kF16,
                                      k,
                                      &beta,
                                      output,
                                      kF16,
                                      m,
                                      output,
                                      kF16,
                                      m,
                                      kF32,
                                      kAlgo,
                                      0,
                                      0),
                      "gemm");
    };

    for(int i = 0; i < 1000; ++i)
        launch();
    check_hip(hipDeviceSynchronize(), "warmup");

    hipEvent_t start = nullptr, stop = nullptr;
    check_hip(hipEventCreate(&start), "start event");
    check_hip(hipEventCreate(&stop), "stop event");
    std::vector<float> samples;
    samples.reserve(51);
    for(int i = 0; i < 51; ++i)
    {
        check_hip(hipEventRecord(start), "record start");
        launch();
        check_hip(hipEventRecord(stop), "record stop");
        check_hip(hipEventSynchronize(stop), "sync stop");
        float ms = 0;
        check_hip(hipEventElapsedTime(&ms, start, stop), "elapsed");
        samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    const double ms = samples[samples.size() / 2];
    const double tflops = 2.0 * static_cast<double>(m) * n * k / 1.0e9 / ms;
    std::printf("m=%d n=%d k=%d median_ms=%.4f tflops=%.4f\n", m, n, k, ms, tflops);

    check_hip(hipEventDestroy(stop), "destroy stop");
    check_hip(hipEventDestroy(start), "destroy start");
    check_rocblas(rocblas_destroy_handle(handle), "destroy handle");
    check_hip(hipFree(output), "free output");
    check_hip(hipFree(activation), "free activation");
    check_hip(hipFree(weight), "free weight");
    return 0;
}
