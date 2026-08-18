#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#ifdef CK_A16_BF16
#include "device_gemm_wmma_universal_bf16_i4_bf16_mk_nk_mn.hpp"
#else
#include "device_gemm_wmma_universal_f16_i4_f16_mk_nk_mn.hpp"
#endif
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cerr << "usage: " << argv[0] << " M N K\n";
        return 2;
    }

    const ck::index_t M = std::stoi(argv[1]);
    const ck::index_t N = std::stoi(argv[2]);
    const ck::index_t K = std::stoi(argv[3]);

    using namespace ck::tensor_operation::device::instance;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
#ifdef CK_A16_BF16
    using A16 = ck::bhalf_t;
    using Instances =
        device_gemm_wmma_universal_bf16_i4_bf16_mk_nk_mn_comp_instances<GemmDefault>;
#else
    using A16 = ck::half_t;
    using Instances =
        device_gemm_wmma_universal_f16_i4_f16_mk_nk_mn_comp_instances<GemmDefault>;
#endif

    ck::DeviceMem a(sizeof(A16) * static_cast<std::size_t>(M) * K);
    ck::DeviceMem b(sizeof(ck::pk_i4_t) * static_cast<std::size_t>(K) * N / 2);
    ck::DeviceMem c(sizeof(A16) * static_cast<std::size_t>(M) * N);

    float best_ms = std::numeric_limits<float>::infinity();
    std::string best_name;

    ck::static_for<0, std::tuple_size_v<Instances>, 1>{}([&](auto i) {
        auto gemm = std::get<i>(Instances{});
        auto argument = gemm.MakeArgument(static_cast<A16*>(a.GetDeviceBuffer()),
                                          static_cast<ck::pk_i4_t*>(b.GetDeviceBuffer()),
                                          static_cast<A16*>(c.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          K,
                                          K,
                                          N,
                                          1,
                                          PassThrough{},
                                          PassThrough{},
                                          PassThrough{});

        const auto name = gemm.GetTypeString();
        if(!gemm.IsSupportedArgument(argument))
        {
            std::cout << "instance=" << i << " supported=0 name=" << name << '\n';
            return;
        }

        auto invoker   = gemm.MakeInvoker();
        const float ms =
            invoker.Run(argument, StreamConfig{nullptr, true, 0, 10, 30, true, 0});
        const double tflops = 2.0 * static_cast<double>(M) * N * K / 1.0e9 / ms;
        std::cout << "instance=" << i << " supported=1 ms=" << ms
                  << " tflops=" << tflops << " name=" << name << '\n';
        if(ms < best_ms)
        {
            best_ms   = ms;
            best_name = name;
        }
    });

    std::cout << "best_ms=" << best_ms << " best_name=" << best_name << '\n';
    return 0;
}
