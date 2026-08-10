// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ck_tile/core.hpp"

#include <cstdint>
#include <type_traits>

namespace hipfire::ck_attention {

#if defined(HIPFIRE_CK_ASYM3_CONSTANT_CODEBOOK) && defined(HIPFIRE_CK_ASYM3_LDS_CODEBOOK)
#error "select only one Asym3 codebook storage implementation"
#endif

#if defined(HIPFIRE_CK_ASYM3_CONSTANT_CODEBOOK)
inline __device__ __constant__ float kAsym3Centroids[8] = {
    -0.134860f,
    -0.083320f,
    -0.046469f,
    -0.015176f,
    0.015176f,
    0.046469f,
    0.083320f,
    0.134860f,
};
#endif

enum class PackedKvKind
{
    Asym3K,
    Q8V,
};

template <PackedKvKind Kind>
struct PackedKvBufferView
{
    using type = ck_tile::half_t;

    static constexpr ck_tile::index_t kHeadDim = 256;
    static constexpr ck_tile::index_t kAsym3ValuesPerPack = 8;
    static constexpr ck_tile::index_t kQ8ValuesPerBlock = 32;

    const uint8_t* p_data_ = nullptr;
    const float* asym3_codebook_ = nullptr;
    ck_tile::index_t logical_size_ = 0;
    ck_tile::index_t row_stride_bytes_ = 0;

    CK_TILE_HOST_DEVICE constexpr PackedKvBufferView() = default;

    CK_TILE_HOST_DEVICE constexpr PackedKvBufferView(const uint8_t* data,
                                                     ck_tile::index_t logical_size,
                                                     ck_tile::index_t row_stride_bytes,
                                                     const float* asym3_codebook = nullptr)
        : p_data_{data},
          asym3_codebook_{asym3_codebook},
          logical_size_{logical_size},
          row_stride_bytes_{row_stride_bytes}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr ck_tile::address_space_enum get_address_space()
    {
        return ck_tile::address_space_enum::global;
    }

    CK_TILE_DEVICE float asym3_centroid(int code) const
    {
#if defined(HIPFIRE_CK_ASYM3_LDS_CODEBOOK)
        return asym3_codebook_[code];
#elif defined(HIPFIRE_CK_ASYM3_CONSTANT_CODEBOOK)
        return kAsym3Centroids[code];
#else
        switch(code)
        {
        case 0: return -0.134860f;
        case 1: return -0.083320f;
        case 2: return -0.046469f;
        case 3: return -0.015176f;
        case 4: return 0.015176f;
        case 5: return 0.046469f;
        case 6: return 0.083320f;
        default: return 0.134860f;
        }
#endif
    }

    CK_TILE_DEVICE float decode(ck_tile::index_t logical_offset) const
    {
        const ck_tile::index_t row = logical_offset >> 8;
        const ck_tile::index_t dim = logical_offset & (kHeadDim - 1);
        const uint8_t* row_ptr = p_data_ + static_cast<size_t>(row) * row_stride_bytes_;

        if constexpr(Kind == PackedKvKind::Asym3K)
        {
            const float cnorm = *reinterpret_cast<const float*>(row_ptr);
            const ck_tile::index_t pack = dim >> 3;
            const ck_tile::index_t lane = dim & (kAsym3ValuesPerPack - 1);
            const uint8_t* bytes = row_ptr + 4 + pack * 3;
            const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                                  (static_cast<uint32_t>(bytes[1]) << 8) |
                                  (static_cast<uint32_t>(bytes[2]) << 16);
            return cnorm * asym3_centroid((word >> (lane * 3)) & 7u);
        }
        else
        {
            const ck_tile::index_t block = dim >> 5;
            const ck_tile::index_t lane = dim & (kQ8ValuesPerBlock - 1);
            const uint8_t* q8 = row_ptr + block * 34;
            const float scale = static_cast<float>(*reinterpret_cast<const _Float16*>(q8));
            const int value = static_cast<int>(*reinterpret_cast<const int8_t*>(q8 + 2 + lane));
            return scale * value;
        }
    }

    template <typename X, bool OobConditionalCheck = true>
    CK_TILE_DEVICE constexpr X get(ck_tile::index_t offset,
                                   ck_tile::index_t linear_offset,
                                   bool is_valid_element,
                                   ck_tile::bool_constant<OobConditionalCheck> = {}) const
    {
        static_assert(std::is_same_v<
                      typename ck_tile::vector_traits<ck_tile::remove_cvref_t<X>>::scalar_type,
                      ck_tile::half_t>);
        constexpr ck_tile::index_t count =
            ck_tile::vector_traits<ck_tile::remove_cvref_t<X>>::vector_size;
        X result{};

        const ck_tile::index_t first = offset + linear_offset;
        const ck_tile::index_t first_dim = first & (kHeadDim - 1);
        const bool full_vector_valid = is_valid_element && first >= 0 &&
                                       first + count <= logical_size_ &&
                                       first_dim + count <= kHeadDim;

        // CK requests naturally aligned 8-wide vectors for this view. Decode
        // one packed group at a time so Asym3 shares cnorm/word loads and Q8
        // shares its block scale across all eight returned values.
        if constexpr(count == 8)
        {
            if(full_vector_valid && (first_dim & 7) == 0)
            {
                const ck_tile::index_t row = first >> 8;
                const uint8_t* row_ptr =
                    p_data_ + static_cast<size_t>(row) * row_stride_bytes_;

                if constexpr(Kind == PackedKvKind::Asym3K)
                {
                    const float cnorm = *reinterpret_cast<const float*>(row_ptr);
                    const uint8_t* bytes = row_ptr + 4 + (first_dim >> 3) * 3;
                    const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                                          (static_cast<uint32_t>(bytes[1]) << 8) |
                                          (static_cast<uint32_t>(bytes[2]) << 16);
                    ck_tile::static_for<0, count, 1>{}([&](auto index) {
                        const int code = (word >> (index * 3)) & 7u;
                        result[index] = ck_tile::type_convert<ck_tile::half_t>(
                            cnorm * asym3_centroid(code));
                    });
                }
                else
                {
                    const uint8_t* q8 = row_ptr + (first_dim >> 5) * 34;
                    const float scale =
                        static_cast<float>(*reinterpret_cast<const _Float16*>(q8));
                    const int block_lane = first_dim & (kQ8ValuesPerBlock - 1);
                    ck_tile::static_for<0, count, 1>{}([&](auto index) {
                        const int value = static_cast<int>(
                            *reinterpret_cast<const int8_t*>(q8 + 2 + block_lane + index));
                        result[index] =
                            ck_tile::type_convert<ck_tile::half_t>(scale * value);
                    });
                }
                return result;
            }
        }

        ck_tile::static_for<0, count, 1>{}([&](auto index) {
            const ck_tile::index_t logical_offset = first + index;
            const bool valid = is_valid_element && logical_offset >= 0 &&
                               logical_offset < logical_size_;
            result[index] = valid ? ck_tile::type_convert<ck_tile::half_t>(decode(logical_offset))
                                  : ck_tile::half_t{0};
        });
        return result;
    }
};

template <PackedKvKind Kind>
CK_TILE_HOST_DEVICE constexpr auto make_packed_kv_tensor_view(const uint8_t* head_base,
                                                              ck_tile::index_t seqlen,
                                                              ck_tile::index_t row_stride_bytes,
                                                              const float* asym3_codebook = nullptr)
{
    auto descriptor = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(seqlen, ck_tile::number<PackedKvBufferView<Kind>::kHeadDim>{}),
        ck_tile::make_tuple(ck_tile::number<PackedKvBufferView<Kind>::kHeadDim>{}, 1),
        ck_tile::number<8>{},
        ck_tile::number<1>{});
    using BufferView = PackedKvBufferView<Kind>;
    return ck_tile::tensor_view<BufferView, ck_tile::remove_cvref_t<decltype(descriptor)>>{
        BufferView{
            head_base, seqlen * BufferView::kHeadDim, row_stride_bytes, asym3_codebook},
        descriptor};
}

} // namespace hipfire::ck_attention
