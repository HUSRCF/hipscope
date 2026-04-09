// MagnumQuant: Fused dequant + inverse butterfly rotation for G32 format
// Target: gfx1010 (RDNA1, RX 5700 XT)
//
// G32 format: quantization group = 32 elements, matching the rotation block.
// Per-block layout (24 bytes):
//   [0:4]   f32 scale
//   [4:8]   f32 zero
//   [8:24]  u8[16] packed 4-bit nibbles (2 per byte)
//
// Each wavefront (32 threads) processes exactly one block.
// Scale/zero are uniform across the wavefront → scalar loads.

#include <hip/hip_runtime.h>

#define BFLY_1  0x041F
#define BFLY_2  0x081F
#define BFLY_4  0x101F
#define BFLY_8  0x201F
#define BFLY_16 0x401F

#define BFLY_GIVENS(v, pattern, stride, c, s, lid) do { \
    float _p = __int_as_float(                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern))); \
    float _ss = ((lid) & (stride)) ? (s) : -(s);       \
    (v) = __fmaf_rn((c), (v), _ss * _p);               \
} while(0)

// Fused dequant (G32) + inverse butterfly rotation
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_dequant_g32(
    const char* __restrict__ data,           // [num_blocks * 24] G32 compressed
    float* __restrict__ output,              // [num_blocks * 32] decompressed
    const float* __restrict__ rot_params,    // [10] cos/sin per round
    const unsigned char* __restrict__ modes, // [num_blocks] per-block mode
    unsigned int num_blocks
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_blocks) return;

    // Block header: scale(4) + zero(4) + nibbles(16) = 24 bytes
    const char* bptr = data + (long long)wid * 24;
    float scale = __builtin_bit_cast(float, *(const unsigned int*)(bptr));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(bptr + 4));

    // Unpack nibble: each thread gets one of 32 elements
    const unsigned char* nibbles = (const unsigned char*)(bptr + 8);
    unsigned char packed = nibbles[lid >> 1];
    unsigned int nib = (lid & 1) ? (packed >> 4) : (packed & 0xFu);
    float val = __fmaf_rn(scale, (float)nib, zero);

    // Inverse butterfly rotation (mode-adaptive)
    unsigned int mode = (unsigned int)modes[wid];

    float c0 = rot_params[0], s0 = rot_params[1];
    float c1 = rot_params[2], s1 = rot_params[3];
    float c2 = rot_params[4], s2 = rot_params[5];
    float c3 = rot_params[6], s3 = rot_params[7];
    float c4 = rot_params[8], s4 = rot_params[9];

    if (mode >= 2) {
        BFLY_GIVENS(val, BFLY_16, 16, c4, -s4, lid);
        BFLY_GIVENS(val, BFLY_8,  8,  c3, -s3, lid);
    }
    if (mode >= 1) {
        BFLY_GIVENS(val, BFLY_4,  4,  c2, -s2, lid);
    }
    BFLY_GIVENS(val, BFLY_2,  2,  c1, -s1, lid);
    BFLY_GIVENS(val, BFLY_1,  1,  c0, -s0, lid);

    output[wid * 32 + lid] = val;
}

// Dequant-only baseline (G32, no rotation)
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_dequant_g32_norot(
    const char* __restrict__ data,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_blocks) return;

    const char* bptr = data + (long long)wid * 24;
    float scale = __builtin_bit_cast(float, *(const unsigned int*)(bptr));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(bptr + 4));

    const unsigned char* nibbles = (const unsigned char*)(bptr + 8);
    unsigned char packed = nibbles[lid >> 1];
    unsigned int nib = (lid & 1) ? (packed >> 4) : (packed & 0xFu);
    float val = __fmaf_rn(scale, (float)nib, zero);

    output[wid * 32 + lid] = val;
}
