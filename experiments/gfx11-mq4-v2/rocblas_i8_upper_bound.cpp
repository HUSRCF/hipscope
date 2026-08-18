// Standalone gfx11 upper-bound probe for an expanded rowwise-I8 execution
// format. This deliberately excludes quantization and scale epilogue cost.
#include <hip/hip_runtime.h>

#include <algorithm>
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

    const size_t weight_bytes = static_cast<size_t>(m) * k;
    const size_t activation_bytes = static_cast<size_t>(n) * k;
    const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(int32_t);

    int8_t* weight = nullptr;
    int8_t* activation = nullptr;
    int32_t* output = nullptr;
    check_hip(hipMalloc(&weight, weight_bytes), "hipMalloc weight");
    check_hip(hipMalloc(&activation, activation_bytes), "hipMalloc activation");
    check_hip(hipMalloc(&output, output_bytes), "hipMalloc output");
    check_hip(hipMemset(weight, 1, weight_bytes), "hipMemset weight");
    check_hip(hipMemset(activation, 1, activation_bytes), "hipMemset activation");
    check_hip(hipMemset(output, 0, output_bytes), "hipMemset output");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    rocblas_handle handle = nullptr;
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
    check_rocblas(rocblas_set_stream(handle, stream), "rocblas_set_stream");

    const int32_t alpha = 1;
    const int32_t beta = 0;
    auto launch = [&]() {
        // Row-major Y[N,M] = X[N,K] * W[M,K]^T is column-major
        // Y_col[M,N] = transpose(W_col[K,M]) * X_col[K,N].
        check_rocblas(
            rocblas_gemm_ex(
                handle, kOpTranspose, kOpNone, m, n, k, &alpha, weight, kI8,
                k, activation, kI8, k, &beta, output, kI32, m, output, kI32,
                m, kI32, kAlgoStandard, 0, 0),
            "rocblas_gemm_ex i8");
    };

    for (int i = 0; i < warmup; ++i) launch();
    check_hip(hipStreamSynchronize(stream), "warmup sync");

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate start");
    check_hip(hipEventCreate(&stop), "hipEventCreate stop");
    std::vector<float> samples;
    samples.reserve(trials);
    for (int i = 0; i < trials; ++i) {
        check_hip(hipEventRecord(start, stream), "event start");
        launch();
        check_hip(hipEventRecord(stop, stream), "event stop");
        check_hip(hipEventSynchronize(stop), "event sync");
        float ms = 0.0f;
        check_hip(hipEventElapsedTime(&ms, start, stop), "event elapsed");
        samples.push_back(ms);
    }

    const double med = median(samples);
    const double tops = 2.0 * static_cast<double>(m) * n * k / (med * 1.0e9);
    int32_t first_output = 0;
    check_hip(
        hipMemcpy(&first_output, output, sizeof(first_output), hipMemcpyDeviceToHost),
        "copy first output");
    std::printf("shape=M%d K%d N%d\n", m, k, n);
    std::printf("weight_bytes=%zu activation_bytes=%zu output_bytes=%zu\n",
                weight_bytes, activation_bytes, output_bytes);
    std::printf("rocblas_i8_raw_median_ms=%.4f\n", med);
    std::printf("effective_tops=%.3f\n", tops);
    std::printf("first_output=%d expected=%d\n", first_output, k);
    std::printf("samples_ms=");
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("%s%.4f", i ? "," : "", samples[i]);
    }
    std::printf("\n");

    check_hip(hipEventDestroy(stop), "hipEventDestroy stop");
    check_hip(hipEventDestroy(start), "hipEventDestroy start");
    rocblas_destroy_handle(handle);
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    check_hip(hipFree(output), "hipFree output");
    check_hip(hipFree(activation), "hipFree activation");
    check_hip(hipFree(weight), "hipFree weight");
    return 0;
}
