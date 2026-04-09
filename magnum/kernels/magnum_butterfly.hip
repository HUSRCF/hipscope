// MagnumQuant: Butterfly Givens Rotation via ds_swizzle_b32
// Target: gfx1010 (RDNA1, RX 5700 XT)
//
// Core idea: log2(32) = 5 rounds of butterfly pair swaps executed entirely
// via ds_swizzle cross-lane operations. Each round applies a 2x2 Givens
// rotation between paired lanes. No LDS memory traffic, no shared memory.
//
// ds_swizzle_b32 BitMode encoding (bit15=0):
//   bits[14:10] = xor_mask, bits[9:5] = or_mask, bits[4:0] = and_mask
//   lane_j = ((lane_i & and) | or) ^ xor
//   For butterfly: and=0x1F, or=0x00, xor=stride

#include <hip/hip_runtime.h>

// Butterfly swap patterns for ds_swizzle_b32
#define BFLY_1  0x041F   // xor=1:  pairs (0,1),(2,3),...,(30,31)
#define BFLY_2  0x081F   // xor=2:  pairs (0,2),(1,3),...,(29,31)
#define BFLY_4  0x101F   // xor=4:  pairs (0,4),(1,5),...,(27,31)
#define BFLY_8  0x201F   // xor=8:  pairs (0,8),(1,9),...,(23,31)
#define BFLY_16 0x401F   // xor=16: pairs (0,16),(1,17),...,(15,31)

// Butterfly Givens rotation macro (pattern must be compile-time constant).
// Reads partner value via ds_swizzle (zero memory traffic), then applies:
//   lane (i & stride == 0): a' = c*a - s*b   (Givens row 1)
//   lane (i & stride != 0): b' = s*a + c*b   (Givens row 2)
// Unified: result = c*val + sign_s*partner, sign_s = (lane & stride) ? +s : -s
#define BFLY_GIVENS(v, pattern, stride, c, s, lid) do { \
    float _p = __int_as_float(                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern))); \
    float _ss = ((lid) & (stride)) ? (s) : -(s);       \
    (v) = __fmaf_rn((c), (v), _ss * _p);               \
} while(0)

// ==========================================================================
// Phase 1: Full 5-round butterfly rotation (per-round shared angles)
// ==========================================================================
// Rotation params layout: [cos0, sin0, cos1, sin1, ..., cos4, sin4]
// All 16 pairs in each round share the same (cos, sin) angle.
// Params are uniform across wavefront -> compiler uses SMEM (s_load_dword).
//
// VGPR budget estimate:
//   val(1) + partner(1) + sign_s(1) + lid(1) + addr regs(~2) = ~6
//   cos/sin from SGPRs (uniform load)
//   Target: <=12 VGPRs

extern "C" __launch_bounds__(256, 8)
__global__ void magnum_butterfly_rotate_f32(
    const float* __restrict__ input,    // [num_vectors * 32]
    float* __restrict__ output,         // [num_vectors * 32]
    const float* __restrict__ params,   // [10] cos/sin pairs per round
    unsigned int num_vectors
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_vectors) return;

    float v = input[wid * 32 + lid];

    // Load rotation params (uniform -> SGPR)
    float c0 = params[0], s0 = params[1];
    float c1 = params[2], s1 = params[3];
    float c2 = params[4], s2 = params[5];
    float c3 = params[6], s3 = params[7];
    float c4 = params[8], s4 = params[9];

    // 5 butterfly Givens rounds: strides 1, 2, 4, 8, 16
    BFLY_GIVENS(v, BFLY_1,  1,  c0, s0, lid);
    BFLY_GIVENS(v, BFLY_2,  2,  c1, s1, lid);
    BFLY_GIVENS(v, BFLY_4,  4,  c2, s2, lid);
    BFLY_GIVENS(v, BFLY_8,  8,  c3, s3, lid);
    BFLY_GIVENS(v, BFLY_16, 16, c4, s4, lid);

    output[wid * 32 + lid] = v;
}

// Inverse rotation: reverse round order, negated sin.
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_butterfly_rotate_inv_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ params,
    unsigned int num_vectors
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_vectors) return;

    float v = input[wid * 32 + lid];

    float c0 = params[0], s0 = params[1];
    float c1 = params[2], s1 = params[3];
    float c2 = params[4], s2 = params[5];
    float c3 = params[6], s3 = params[7];
    float c4 = params[8], s4 = params[9];

    // Reverse order, negated sin
    BFLY_GIVENS(v, BFLY_16, 16, c4, -s4, lid);
    BFLY_GIVENS(v, BFLY_8,  8,  c3, -s3, lid);
    BFLY_GIVENS(v, BFLY_4,  4,  c2, -s2, lid);
    BFLY_GIVENS(v, BFLY_2,  2,  c1, -s1, lid);
    BFLY_GIVENS(v, BFLY_1,  1,  c0, -s0, lid);

    output[wid * 32 + lid] = v;
}

// ==========================================================================
// Phase 2: Adaptive mode selection (2/3/5 rounds)
// ==========================================================================
// Mode bits in per-block metadata buffer (1 byte per block):
//   Mode 0: 2 rounds (strides 1, 2)     -- nearest-neighbor mixing
//   Mode 1: 3 rounds (strides 1, 2, 4)  -- moderate mixing
//   Mode 2: 5 rounds (strides 1..16)    -- full butterfly, near-global mixing
//
// SALU path: mode byte loaded uniformly -> s_cmp -> s_cbranch

extern "C" __launch_bounds__(256, 8)
__global__ void magnum_butterfly_adaptive(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ params,
    const unsigned char* __restrict__ modes,   // [num_vectors] per-block mode
    unsigned int num_vectors
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_vectors) return;

    float v = input[wid * 32 + lid];

    float c0 = params[0], s0 = params[1];
    float c1 = params[2], s1 = params[3];
    float c2 = params[4], s2 = params[5];
    float c3 = params[6], s3 = params[7];
    float c4 = params[8], s4 = params[9];

    // Mode uniform across wavefront -> scalar load + branch
    const unsigned int mode = (unsigned int)modes[wid];

    // Rounds 0-1: always execute (all modes >= 0)
    BFLY_GIVENS(v, BFLY_1,  1,  c0, s0, lid);
    BFLY_GIVENS(v, BFLY_2,  2,  c1, s1, lid);

    if (mode >= 1) {
        BFLY_GIVENS(v, BFLY_4,  4,  c2, s2, lid);
    }

    if (mode >= 2) {
        BFLY_GIVENS(v, BFLY_8,  8,  c3, s3, lid);
        BFLY_GIVENS(v, BFLY_16, 16, c4, s4, lid);
    }

    output[wid * 32 + lid] = v;
}

// Adaptive inverse
extern "C" __launch_bounds__(256, 8)
__global__ void magnum_butterfly_adaptive_inv(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ params,
    const unsigned char* __restrict__ modes,
    unsigned int num_vectors
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lid = tid & 31;

    if (wid >= num_vectors) return;

    float v = input[wid * 32 + lid];
    const unsigned int mode = (unsigned int)modes[wid];

    float c0 = params[0], s0 = params[1];
    float c1 = params[2], s1 = params[3];
    float c2 = params[4], s2 = params[5];
    float c3 = params[6], s3 = params[7];
    float c4 = params[8], s4 = params[9];

    if (mode >= 2) {
        BFLY_GIVENS(v, BFLY_16, 16, c4, -s4, lid);
        BFLY_GIVENS(v, BFLY_8,  8,  c3, -s3, lid);
    }
    if (mode >= 1) {
        BFLY_GIVENS(v, BFLY_4,  4,  c2, -s2, lid);
    }
    BFLY_GIVENS(v, BFLY_2,  2,  c1, -s1, lid);
    BFLY_GIVENS(v, BFLY_1,  1,  c0, -s0, lid);

    output[wid * 32 + lid] = v;
}
