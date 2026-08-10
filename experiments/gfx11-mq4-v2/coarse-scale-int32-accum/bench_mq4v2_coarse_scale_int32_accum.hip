// SPDX-License-Identifier: MIT
// Standalone-only coarse-scale/int32-accumulation feasibility probe for gfx1100.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "mq4v2_coarse_scale_kernels.hip"

using namespace mq4v2_probe;

#define HIP_CHECK(expr)                                                           \
    do {                                                                          \
        const hipError_t err__ = (expr);                                          \
        if (err__ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,  \
                         hipGetErrorString(err__));                               \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

struct ShapeSpec {
    const char* name;
    int M;
    int K;
    int N;
    bool add_output;
};

struct KernelStats {
    int num_regs;
    int local_size_bytes;
    int shared_size_bytes;
    int max_threads_per_block;
};

struct ValidationStats {
    double max_abs;
    double relative_l2;
    double cosine;
};

struct TimingStats {
    std::vector<float> exact_ms;
    std::vector<float> coarse_ms;
    float exact_median;
    float coarse_median;
    float speedup;
    ValidationStats diff;
};

struct ProbeOptions {
    int warmup = 3;
    int pairs = 7;
    int dpm_warmup_ms = 5000;
    bool list_devices = false;
};

struct DeviceBuffers {
    ExactWeightGroup128* exact_w = nullptr;
    ExactActGroup128* exact_x = nullptr;
    CoarseWeightGroup512* coarse_w = nullptr;
    CoarseActGroup512* coarse_x = nullptr;
    float* exact_out = nullptr;
    float* coarse_out = nullptr;
    float* base_out = nullptr;
};

static uint8_t* weight_payload_bytes(ExactWeightGroup128& group) {
    return reinterpret_cast<uint8_t*>(group.payload_words);
}

static const uint8_t* weight_payload_bytes(const ExactWeightGroup128& group) {
    return reinterpret_cast<const uint8_t*>(group.payload_words);
}

static uint8_t* weight_payload_bytes(CoarseWeightGroup512& group) {
    return reinterpret_cast<uint8_t*>(group.payload_words);
}

static int8_t* act_q_bytes(ExactActGroup128& block) {
    return reinterpret_cast<int8_t*>(block.q_words);
}

static const int8_t* act_q_bytes(const ExactActGroup128& block) {
    return reinterpret_cast<const int8_t*>(block.q_words);
}

static int8_t* act_q_bytes(CoarseActGroup512& block) {
    return reinterpret_cast<int8_t*>(block.q_words);
}

static void set_packed_u4(uint8_t* payload, int index, uint8_t value) {
    const int byte_index = index >> 1;
    if (index & 1) {
        payload[byte_index] =
            static_cast<uint8_t>((payload[byte_index] & 0x0Fu) | (value << 4));
    } else {
        payload[byte_index] =
            static_cast<uint8_t>((payload[byte_index] & 0xF0u) | (value & 0x0Fu));
    }
}

static uint8_t get_packed_u4(const uint8_t* payload, int index) {
    const uint8_t packed = payload[index >> 1];
    return static_cast<uint8_t>((index & 1) ? (packed >> 4) : (packed & 0x0Fu));
}

static ShapeSpec make_small_control_shape() {
    return {"small_control_set", 16, 512, 16, false};
}

static std::vector<ShapeSpec> benchmark_shapes() {
    return {
        {"gate_up_full_set", 17408, 5120, 2048, false},
        {"down_full_add", 5120, 17408, 2048, true},
    };
}

static void list_devices() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::printf("visible_device_count=%d\n", count);
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        std::printf("device[%d]=%s arch=%s multiprocessors=%d\n",
                    i,
                    prop.name,
                    prop.gcnArchName,
                    prop.multiProcessorCount);
    }
}

static std::vector<float> make_base_output(const ShapeSpec& shape) {
    const size_t elements = static_cast<size_t>(shape.M) * shape.N;
    std::vector<float> out(elements, 0.0f);
    if (!shape.add_output) {
        return out;
    }
    for (size_t i = 0; i < elements; ++i) {
        const int numer = static_cast<int>(i % 17) - 8;
        out[i] = std::ldexp(static_cast<float>(numer), -5);
    }
    return out;
}

static void generate_quantized_inputs(
    const ShapeSpec& shape,
    std::vector<ExactWeightGroup128>& exact_w,
    std::vector<ExactActGroup128>& exact_x,
    std::vector<CoarseWeightGroup512>& coarse_w,
    std::vector<CoarseActGroup512>& coarse_x
) {
    if ((shape.K % kCoarseK) != 0 || (shape.K % kGroupK) != 0) {
        std::fprintf(stderr, "K must be divisible by %d and %d\n", kCoarseK, kGroupK);
        std::exit(2);
    }
    const int groups128 = shape.K / kGroupK;
    const int coarse_groups = shape.K / kCoarseK;

    exact_w.assign(static_cast<size_t>(shape.M) * groups128, {});
    exact_x.assign(static_cast<size_t>(groups128) * shape.N, {});
    coarse_w.assign(static_cast<size_t>(shape.M) * coarse_groups, {});
    coarse_x.assign(static_cast<size_t>(coarse_groups) * shape.N, {});

    for (int m = 0; m < shape.M; ++m) {
        for (int cg = 0; cg < coarse_groups; ++cg) {
            CoarseWeightGroup512& coarse = coarse_w[static_cast<size_t>(m) * coarse_groups + cg];
            coarse.scale = std::ldexp(static_cast<float>(1 + ((m + cg) & 3)), -5);
            coarse.zero = std::ldexp(static_cast<float>(((m + 3 * cg) % 17) - 8), -4);
            uint8_t* coarse_payload = weight_payload_bytes(coarse);
            std::memset(coarse_payload, 0, 256);
            for (int local_group = 0; local_group < kCoarseGroups; ++local_group) {
                ExactWeightGroup128& exact =
                    exact_w[static_cast<size_t>(m) * groups128 + cg * kCoarseGroups + local_group];
                exact.scale = coarse.scale;
                exact.zero = coarse.zero;
                uint8_t* exact_payload = weight_payload_bytes(exact);
                std::memset(exact_payload, 0, 64);
                for (int k = 0; k < kGroupK; ++k) {
                    const int absolute_k = local_group * kGroupK + k;
                    const uint8_t nibble =
                        static_cast<uint8_t>((m * 11 + cg * 7 + absolute_k * 3) & 0x0F);
                    set_packed_u4(coarse_payload, absolute_k, nibble);
                    set_packed_u4(exact_payload, k, nibble);
                }
            }
        }
    }

    for (int n = 0; n < shape.N; ++n) {
        for (int cg = 0; cg < coarse_groups; ++cg) {
            CoarseActGroup512& coarse = coarse_x[static_cast<size_t>(cg) * shape.N + n];
            coarse.scale = std::ldexp(static_cast<float>(1 + ((n + 2 * cg) & 3)), -7);
            int8_t* coarse_q = act_q_bytes(coarse);
            int total_sum_q = 0;
            for (int k = 0; k < kCoarseK; ++k) {
                const int value = ((n * 17 + cg * 13 + k * 5) % 63) - 31;
                coarse_q[k] = static_cast<int8_t>(value);
                total_sum_q += value;
            }
            coarse.sum = coarse.scale * static_cast<float>(total_sum_q);
            for (int local_group = 0; local_group < kCoarseGroups; ++local_group) {
                ExactActGroup128& exact =
                    exact_x[static_cast<size_t>(cg * kCoarseGroups + local_group) * shape.N + n];
                exact.scale = coarse.scale;
                int8_t* exact_q = act_q_bytes(exact);
                int group_sum_q = 0;
                for (int k = 0; k < kGroupK; ++k) {
                    const int8_t q = coarse_q[local_group * kGroupK + k];
                    exact_q[k] = q;
                    group_sum_q += q;
                }
                exact.sum = exact.scale * static_cast<float>(group_sum_q);
            }
        }
    }
}

static std::vector<float> cpu_exact_reference(
    const ShapeSpec& shape,
    const std::vector<ExactWeightGroup128>& weights,
    const std::vector<ExactActGroup128>& acts,
    const std::vector<float>& base
) {
    const int groups128 = shape.K / kGroupK;
    std::vector<float> out(static_cast<size_t>(shape.M) * shape.N, 0.0f);
    for (int n = 0; n < shape.N; ++n) {
        for (int m = 0; m < shape.M; ++m) {
            float acc = shape.add_output ? base[static_cast<size_t>(n) * shape.M + m] : 0.0f;
            for (int g = 0; g < groups128; ++g) {
                const ExactWeightGroup128& wg =
                    weights[static_cast<size_t>(m) * groups128 + g];
                const ExactActGroup128& xb =
                    acts[static_cast<size_t>(g) * shape.N + n];
                const uint8_t* payload = weight_payload_bytes(wg);
                const int8_t* q = act_q_bytes(xb);
                int dot = 0;
                for (int k = 0; k < kGroupK; ++k) {
                    dot += static_cast<int>(get_packed_u4(payload, k)) * static_cast<int>(q[k]);
                }
                acc += wg.scale * xb.scale * static_cast<float>(dot) + wg.zero * xb.sum;
            }
            out[static_cast<size_t>(n) * shape.M + m] = acc;
        }
    }
    return out;
}

static ValidationStats compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& candidate
) {
    if (reference.size() != candidate.size()) {
        std::fprintf(stderr, "size mismatch in compare_outputs\n");
        std::exit(3);
    }
    double max_abs = 0.0;
    double ref_norm2 = 0.0;
    double diff_norm2 = 0.0;
    double dot = 0.0;
    double cand_norm2 = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i];
        const double cand = candidate[i];
        const double diff = cand - ref;
        max_abs = std::max(max_abs, std::abs(diff));
        ref_norm2 += ref * ref;
        diff_norm2 += diff * diff;
        dot += ref * cand;
        cand_norm2 += cand * cand;
    }
    ValidationStats stats{};
    stats.max_abs = max_abs;
    stats.relative_l2 = ref_norm2 > 0.0 ? std::sqrt(diff_norm2 / ref_norm2) : 0.0;
    stats.cosine =
        (ref_norm2 > 0.0 && cand_norm2 > 0.0) ? dot / std::sqrt(ref_norm2 * cand_norm2) : 1.0;
    return stats;
}

template <typename Kernel>
static KernelStats query_kernel_stats(Kernel kernel) {
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)));
    KernelStats stats{};
    stats.num_regs = attr.numRegs;
    stats.local_size_bytes = attr.localSizeBytes;
    stats.shared_size_bytes = attr.sharedSizeBytes;
    stats.max_threads_per_block = attr.maxThreadsPerBlock;
    return stats;
}

static float median_of(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static void allocate_device_buffers(
    const ShapeSpec& shape,
    const std::vector<ExactWeightGroup128>& exact_w,
    const std::vector<ExactActGroup128>& exact_x,
    const std::vector<CoarseWeightGroup512>& coarse_w,
    const std::vector<CoarseActGroup512>& coarse_x,
    const std::vector<float>& base,
    DeviceBuffers& buffers
) {
    const size_t exact_w_bytes = exact_w.size() * sizeof(ExactWeightGroup128);
    const size_t exact_x_bytes = exact_x.size() * sizeof(ExactActGroup128);
    const size_t coarse_w_bytes = coarse_w.size() * sizeof(CoarseWeightGroup512);
    const size_t coarse_x_bytes = coarse_x.size() * sizeof(CoarseActGroup512);
    const size_t out_bytes = static_cast<size_t>(shape.M) * shape.N * sizeof(float);

    HIP_CHECK(hipMalloc(&buffers.exact_w, exact_w_bytes));
    HIP_CHECK(hipMalloc(&buffers.exact_x, exact_x_bytes));
    HIP_CHECK(hipMalloc(&buffers.coarse_w, coarse_w_bytes));
    HIP_CHECK(hipMalloc(&buffers.coarse_x, coarse_x_bytes));
    HIP_CHECK(hipMalloc(&buffers.exact_out, out_bytes));
    HIP_CHECK(hipMalloc(&buffers.coarse_out, out_bytes));
    if (shape.add_output) {
        HIP_CHECK(hipMalloc(&buffers.base_out, out_bytes));
    }

    HIP_CHECK(hipMemcpy(buffers.exact_w, exact_w.data(), exact_w_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers.exact_x, exact_x.data(), exact_x_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers.coarse_w, coarse_w.data(), coarse_w_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers.coarse_x, coarse_x.data(), coarse_x_bytes, hipMemcpyHostToDevice));
    if (shape.add_output) {
        HIP_CHECK(hipMemcpy(buffers.base_out, base.data(), out_bytes, hipMemcpyHostToDevice));
    }
}

static void free_device_buffers(DeviceBuffers& buffers) {
    if (buffers.base_out != nullptr) {
        HIP_CHECK(hipFree(buffers.base_out));
    }
    HIP_CHECK(hipFree(buffers.exact_out));
    HIP_CHECK(hipFree(buffers.coarse_out));
    HIP_CHECK(hipFree(buffers.exact_w));
    HIP_CHECK(hipFree(buffers.exact_x));
    HIP_CHECK(hipFree(buffers.coarse_w));
    HIP_CHECK(hipFree(buffers.coarse_x));
    buffers = {};
}

static void reset_output(const ShapeSpec& shape, float* output, const DeviceBuffers& buffers) {
    const size_t out_bytes = static_cast<size_t>(shape.M) * shape.N * sizeof(float);
    if (shape.add_output) {
        HIP_CHECK(hipMemcpy(output, buffers.base_out, out_bytes, hipMemcpyDeviceToDevice));
    }
}

static void launch_exact(const ShapeSpec& shape, const DeviceBuffers& buffers) {
    const dim3 grid(shape.M / kTileM, shape.N / kTileN);
    const dim3 block(32);
    if (shape.add_output) {
        hipLaunchKernelGGL(mq4v2_exact_group128_full_add,
                           grid,
                           block,
                           0,
                           0,
                           buffers.exact_w,
                           buffers.exact_x,
                           buffers.exact_out,
                           shape.M,
                           shape.K,
                           shape.N);
    } else {
        hipLaunchKernelGGL(mq4v2_exact_group128_full_set,
                           grid,
                           block,
                           0,
                           0,
                           buffers.exact_w,
                           buffers.exact_x,
                           buffers.exact_out,
                           shape.M,
                           shape.K,
                           shape.N);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_coarse(const ShapeSpec& shape, const DeviceBuffers& buffers) {
    const dim3 grid(shape.M / kTileM, shape.N / kTileN);
    const dim3 block(32);
    if (shape.add_output) {
        hipLaunchKernelGGL(mq4v2_coarse_group512_full_add,
                           grid,
                           block,
                           0,
                           0,
                           buffers.coarse_w,
                           buffers.coarse_x,
                           buffers.coarse_out,
                           shape.M,
                           shape.K,
                           shape.N);
    } else {
        hipLaunchKernelGGL(mq4v2_coarse_group512_full_set,
                           grid,
                           block,
                           0,
                           0,
                           buffers.coarse_w,
                           buffers.coarse_x,
                           buffers.coarse_out,
                           shape.M,
                           shape.K,
                           shape.N);
    }
    HIP_CHECK(hipGetLastError());
}

template <typename ResetFn, typename LaunchFn>
static float timed_launch(ResetFn&& reset, LaunchFn&& launch) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    reset();
    HIP_CHECK(hipEventRecord(start));
    launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
}

static void dpm_warmup(
    const ShapeSpec& shape,
    const ProbeOptions& options,
    const DeviceBuffers& buffers
) {
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(options.dpm_warmup_ms);
    while (std::chrono::steady_clock::now() < end) {
        reset_output(shape, buffers.exact_out, buffers);
        launch_exact(shape, buffers);
        reset_output(shape, buffers.coarse_out, buffers);
        launch_coarse(shape, buffers);
        HIP_CHECK(hipDeviceSynchronize());
    }
}

static TimingStats benchmark_shape(
    const ShapeSpec& shape,
    const ProbeOptions& options,
    const std::vector<ExactWeightGroup128>& exact_w,
    const std::vector<ExactActGroup128>& exact_x,
    const std::vector<CoarseWeightGroup512>& coarse_w,
    const std::vector<CoarseActGroup512>& coarse_x,
    const std::vector<float>& base
) {
    DeviceBuffers buffers{};
    allocate_device_buffers(shape, exact_w, exact_x, coarse_w, coarse_x, base, buffers);

    for (int i = 0; i < options.warmup; ++i) {
        reset_output(shape, buffers.exact_out, buffers);
        launch_exact(shape, buffers);
        reset_output(shape, buffers.coarse_out, buffers);
        launch_coarse(shape, buffers);
    }
    HIP_CHECK(hipDeviceSynchronize());
    dpm_warmup(shape, options, buffers);

    TimingStats stats{};
    stats.exact_ms.reserve(options.pairs);
    stats.coarse_ms.reserve(options.pairs);
    for (int pair = 0; pair < options.pairs; ++pair) {
        const bool exact_first = (pair & 1) == 0;
        if (exact_first) {
            stats.exact_ms.push_back(timed_launch(
                [&] { reset_output(shape, buffers.exact_out, buffers); },
                [&] { launch_exact(shape, buffers); }));
            stats.coarse_ms.push_back(timed_launch(
                [&] { reset_output(shape, buffers.coarse_out, buffers); },
                [&] { launch_coarse(shape, buffers); }));
        } else {
            stats.coarse_ms.push_back(timed_launch(
                [&] { reset_output(shape, buffers.coarse_out, buffers); },
                [&] { launch_coarse(shape, buffers); }));
            stats.exact_ms.push_back(timed_launch(
                [&] { reset_output(shape, buffers.exact_out, buffers); },
                [&] { launch_exact(shape, buffers); }));
        }
    }

    stats.exact_median = median_of(stats.exact_ms);
    stats.coarse_median = median_of(stats.coarse_ms);
    stats.speedup = stats.exact_median / stats.coarse_median;

    const size_t out_bytes = static_cast<size_t>(shape.M) * shape.N * sizeof(float);
    std::vector<float> exact_out(static_cast<size_t>(shape.M) * shape.N);
    std::vector<float> coarse_out(static_cast<size_t>(shape.M) * shape.N);
    reset_output(shape, buffers.exact_out, buffers);
    launch_exact(shape, buffers);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(exact_out.data(), buffers.exact_out, out_bytes, hipMemcpyDeviceToHost));
    reset_output(shape, buffers.coarse_out, buffers);
    launch_coarse(shape, buffers);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(coarse_out.data(), buffers.coarse_out, out_bytes, hipMemcpyDeviceToHost));
    stats.diff = compare_outputs(exact_out, coarse_out);

    free_device_buffers(buffers);
    return stats;
}

static void print_kernel_stats() {
    const KernelStats exact_set = query_kernel_stats(mq4v2_exact_group128_full_set);
    const KernelStats exact_add = query_kernel_stats(mq4v2_exact_group128_full_add);
    const KernelStats coarse_set = query_kernel_stats(mq4v2_coarse_group512_full_set);
    const KernelStats coarse_add = query_kernel_stats(mq4v2_coarse_group512_full_add);

    std::printf("kernel_stats exact_full_set regs=%d local_bytes=%d static_lds=%d max_threads=%d\n",
                exact_set.num_regs,
                exact_set.local_size_bytes,
                exact_set.shared_size_bytes,
                exact_set.max_threads_per_block);
    std::printf("kernel_stats exact_full_add regs=%d local_bytes=%d static_lds=%d max_threads=%d\n",
                exact_add.num_regs,
                exact_add.local_size_bytes,
                exact_add.shared_size_bytes,
                exact_add.max_threads_per_block);
    std::printf("kernel_stats coarse_full_set regs=%d local_bytes=%d static_lds=%d max_threads=%d\n",
                coarse_set.num_regs,
                coarse_set.local_size_bytes,
                coarse_set.shared_size_bytes,
                coarse_set.max_threads_per_block);
    std::printf("kernel_stats coarse_full_add regs=%d local_bytes=%d static_lds=%d max_threads=%d\n",
                coarse_add.num_regs,
                coarse_add.local_size_bytes,
                coarse_add.shared_size_bytes,
                coarse_add.max_threads_per_block);
}

static void print_shape_bytes(const ShapeSpec& shape) {
    const size_t exact_weight_bytes =
        static_cast<size_t>(shape.M) * (shape.K / kGroupK) * sizeof(ExactWeightGroup128);
    const size_t coarse_weight_bytes =
        static_cast<size_t>(shape.M) * (shape.K / kCoarseK) * sizeof(CoarseWeightGroup512);
    const size_t exact_act_bytes =
        static_cast<size_t>(shape.N) * (shape.K / kGroupK) * sizeof(ExactActGroup128);
    const size_t coarse_act_bytes =
        static_cast<size_t>(shape.N) * (shape.K / kCoarseK) * sizeof(CoarseActGroup512);
    std::printf(
        "resident_bytes shape=%s exact_weight=%zu coarse_weight=%zu exact_act=%zu coarse_act=%zu\n",
        shape.name,
        exact_weight_bytes,
        coarse_weight_bytes,
        exact_act_bytes,
        coarse_act_bytes);
}

static ProbeOptions parse_args(int argc, char** argv) {
    ProbeOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list-devices") {
            options.list_devices = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::atoi(argv[++i]);
        } else if (arg == "--pairs" && i + 1 < argc) {
            options.pairs = std::atoi(argv[++i]);
        } else if (arg == "--dpm-warmup-ms" && i + 1 < argc) {
            options.dpm_warmup_ms = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            std::exit(4);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const ProbeOptions options = parse_args(argc, argv);
    if (options.list_devices) {
        list_devices();
        return 0;
    }

    int device_count = 0;
    int current_device = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK(hipGetDevice(&current_device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, current_device));
    std::printf("device name=%s arch=%s visible_device_count=%d current_device=%d\n",
                prop.name,
                prop.gcnArchName,
                device_count,
                current_device);

    print_kernel_stats();

    const ShapeSpec small = make_small_control_shape();
    std::vector<ExactWeightGroup128> small_exact_w;
    std::vector<ExactActGroup128> small_exact_x;
    std::vector<CoarseWeightGroup512> small_coarse_w;
    std::vector<CoarseActGroup512> small_coarse_x;
    generate_quantized_inputs(
        small, small_exact_w, small_exact_x, small_coarse_w, small_coarse_x);
    const std::vector<float> small_base = make_base_output(small);
    const std::vector<float> cpu_ref =
        cpu_exact_reference(small, small_exact_w, small_exact_x, small_base);
    const TimingStats small_stats = benchmark_shape(
        small, options, small_exact_w, small_exact_x, small_coarse_w, small_coarse_x, small_base);
    std::vector<float> cpu_vs_exact = cpu_ref;
    std::vector<float> cpu_vs_coarse = cpu_ref;
    {
        DeviceBuffers buffers{};
        allocate_device_buffers(
            small, small_exact_w, small_exact_x, small_coarse_w, small_coarse_x, small_base, buffers);
        reset_output(small, buffers.exact_out, buffers);
        launch_exact(small, buffers);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(cpu_vs_exact.data(),
                            buffers.exact_out,
                            cpu_vs_exact.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        reset_output(small, buffers.coarse_out, buffers);
        launch_coarse(small, buffers);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(cpu_vs_coarse.data(),
                            buffers.coarse_out,
                            cpu_vs_coarse.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        free_device_buffers(buffers);
    }
    const ValidationStats cpu_exact_stats = compare_outputs(cpu_ref, cpu_vs_exact);
    const ValidationStats cpu_coarse_stats = compare_outputs(cpu_ref, cpu_vs_coarse);
    std::printf("validation small_case shape=%s exact_flushes_per_output=%d coarse_flushes_per_output=%d reduction=%.2fx\n",
                small.name,
                small.K / kGroupK,
                small.K / kCoarseK,
                static_cast<double>(kCoarseGroups));
    std::printf("validation cpu_vs_exact max_abs=%.9g relative_l2=%.9g cosine=%.9g\n",
                cpu_exact_stats.max_abs,
                cpu_exact_stats.relative_l2,
                cpu_exact_stats.cosine);
    std::printf("validation cpu_vs_coarse max_abs=%.9g relative_l2=%.9g cosine=%.9g\n",
                cpu_coarse_stats.max_abs,
                cpu_coarse_stats.relative_l2,
                cpu_coarse_stats.cosine);
    std::printf("validation exact_vs_coarse max_abs=%.9g relative_l2=%.9g cosine=%.9g\n",
                small_stats.diff.max_abs,
                small_stats.diff.relative_l2,
                small_stats.diff.cosine);

    for (const ShapeSpec& shape : benchmark_shapes()) {
        std::vector<ExactWeightGroup128> exact_w;
        std::vector<ExactActGroup128> exact_x;
        std::vector<CoarseWeightGroup512> coarse_w;
        std::vector<CoarseActGroup512> coarse_x;
        generate_quantized_inputs(shape, exact_w, exact_x, coarse_w, coarse_x);
        const std::vector<float> base = make_base_output(shape);
        print_shape_bytes(shape);
        const TimingStats stats =
            benchmark_shape(shape, options, exact_w, exact_x, coarse_w, coarse_x, base);
        std::printf("shape=%s M=%d K=%d N=%d add=%d\n",
                    shape.name,
                    shape.M,
                    shape.K,
                    shape.N,
                    shape.add_output ? 1 : 0);
        std::printf("exact_ms:");
        for (float ms : stats.exact_ms) {
            std::printf(" %.6f", ms);
        }
        std::printf("\n");
        std::printf("coarse_ms:");
        for (float ms : stats.coarse_ms) {
            std::printf(" %.6f", ms);
        }
        std::printf("\n");
        std::printf(
            "summary shape=%s exact_median_ms=%.6f coarse_median_ms=%.6f speedup=%.6fx max_abs=%.9g relative_l2=%.9g cosine=%.9g\n",
            shape.name,
            stats.exact_median,
            stats.coarse_median,
            stats.speedup,
            stats.diff.max_abs,
            stats.diff.relative_l2,
            stats.diff.cosine);
    }

    return 0;
}
