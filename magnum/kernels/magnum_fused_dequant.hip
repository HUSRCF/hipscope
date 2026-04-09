// MagnumQuant Phase 3: Fused HFQ4 dequant + inverse butterfly rotation
// Target: gfx1010 (RDNA1, RX 5700 XT)
//
// Pipeline per wavefront (32 threads processing one 32-element sub-block):
//   1. Coalesced load of HFQ4-G256 group → LDS
//   2. SALU decodes metadata (scale, zero, mode)
//   3. VALU: nibble unpack → scale*nib+zero (dequant)
//   4. VALU: inverse butterfly rotation (mode-adaptive)
//   5. Store to output buffer
//
// Data format: standard HFQ4-G256 (136 bytes per 256 elements)
//   [0:4] f32 scale, [4:8] f32 zero, [8:136] packed 4-bit nibbles
// Mode bits in a separate buffer: 1 byte per 32-element sub-block

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

// ── Standalone dequant: reads HFQ4-G256, outputs float with inverse rotation ──
// Each wavefront processes one 32-element sub-block within a 256-element group.
// 8 wavefronts per group × num_groups total.
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_dequant_hfq4(
    const char* __restrict__ data,           // HFQ4-G256 compressed data
    float* __restrict__ output,              // [total_elements] decompressed
    const float* __restrict__ rot_params,    // [10] cos/sin per round
    const unsigned char* __restrict__ modes, // [total_blocks] 1 byte per 32-elem block
    int num_groups                           // number of 256-element groups
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    // 8 sub-blocks per group
    const int total_blocks = num_groups * 8;
    if ((int)wid >= total_blocks) return;

    const int group_idx = wid >> 3;         // wid / 8
    const int block_idx = wid & 7;          // wid % 8

    // ── Step 1+2: Load metadata (uniform → SGPR path) ──
    const char* gptr = data + (long long)group_idx * 136;
    float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));

    // ── Step 3: Load and dequant nibble ──
    // Each thread owns 1 element. 32 threads × 1 element = 32-element sub-block.
    // Sub-block starts at nibble byte offset: block_idx * 16
    // Within sub-block: thread lid owns element lid.
    // Two elements per byte: lid/2 selects byte, lid%2 selects high/low nibble.
    const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
    int byte_offset = block_idx * 16 + (lid >> 1);
    unsigned char packed = nibbles[byte_offset];
    unsigned int nib = (lid & 1) ? (packed >> 4) : (packed & 0xFu);
    float val = __fmaf_rn(scale, (float)nib, zero);

    // ── Step 4: Inverse butterfly rotation (mode-adaptive) ──
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

    // ── Step 5: Write output ──
    int out_idx = group_idx * 256 + block_idx * 32 + lid;
    output[out_idx] = val;
}

// ── Dequant-only baseline (no rotation, for comparison) ──
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_dequant_hfq4_norot(
    const char* __restrict__ data,
    float* __restrict__ output,
    int num_groups
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    const int total_blocks = num_groups * 8;
    if ((int)wid >= total_blocks) return;

    const int group_idx = wid >> 3;
    const int block_idx = wid & 7;

    const char* gptr = data + (long long)group_idx * 136;
    float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
    float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));

    const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
    int byte_offset = block_idx * 16 + (lid >> 1);
    unsigned char packed = nibbles[byte_offset];
    unsigned int nib = (lid & 1) ? (packed >> 4) : (packed & 0xFu);
    float val = __fmaf_rn(scale, (float)nib, zero);

    int out_idx = group_idx * 256 + block_idx * 32 + lid;
    output[out_idx] = val;
}
