// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T>
__device__ __forceinline__ float load_as_f32(const T* p, long long i);

template <>
__device__ __forceinline__ float load_as_f32<float>(const float* p, long long i) {
    return p[i];
}

template <>
__device__ __forceinline__ float load_as_f32<__half>(const __half* p, long long i) {
    return __half2float(p[i]);
}

template <typename T>
__device__ __forceinline__ void accumulate_group_energy(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    float* __restrict__ energy,
    int K,
    int N,
    int layer
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups = K / 256;
    if (group >= groups || tid >= 256) return;

    const int channel = group * 256 + tid;
    float local = 0.0f;
    for (int row = 0; row < N; ++row) {
        const long long i = (long long)row * K + channel;
        const float g = load_as_f32<T>(gate, i);
        const float u = load_as_f32<T>(up, i);
        const float v = (g / (1.0f + expf(-g))) * u;
        local = fmaf(v, v, local);
    }

    __shared__ float reduce[256];
    reduce[tid] = local;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&energy[layer * groups + group], reduce[0]);
}

extern "C" __launch_bounds__(256, 1)
__global__ void ffn_group_energy_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ energy,
    int K,
    int N,
    int layer
) {
    accumulate_group_energy(gate, up, energy, K, N, layer);
}

extern "C" __launch_bounds__(256, 1)
__global__ void ffn_group_energy_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    float* __restrict__ energy,
    int K,
    int N,
    int layer
) {
    accumulate_group_energy(gate, up, energy, K, N, layer);
}
