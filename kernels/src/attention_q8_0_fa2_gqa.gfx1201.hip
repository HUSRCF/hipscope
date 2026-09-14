// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201-only GQA-fused FA2 prefill attention (research opt-in).
//
// One workgroup owns one KV head and eight token positions: 48 live rows
// (8 positions x 6 query heads, H=24/Hkv=4/D=256) packed into three 16-row
// WMMA tiles, plus one loader/helper wave. K and V are each dequantized
// exactly once per 64-key tile into two swizzled f16 LDS planes (32 KiB
// each), then reused by all six query heads. Scores, online-softmax state
// (m, l), and f32 O remain in VGPRs; dynamic LDS is exactly 65,536 B.
//
// Grid:  [ceil(batch/8), 4, 1] direct (x=batch tiles, y=KV head),
//        [ceil(batch/8), 4, n_splits] partial.
// Block: 128 threads (four wave32s; waves 0..2 compute, wave 3 helper).
//
// Row mapping (compute wave w=0..2, packed index t=0..15):
//   head_local = 2*w + (t >> 3);  q_local = t & 7;
//   h = 6*kv_h + head_local;      q_row = 8*blockIdx.x + q_local.
//
// Swapped operand order (no P transpose exists):
// gfx12 WMMA: lane (ml=lane&15, kg=lane>>4) supplies A[row=ml][k=kg*8..+8]
// and B[row=ml][k=kg*8..+8]; acc[j] = C[8*kg+j][ml] with
// C[i][n] = sum_k A[i][k]*B[n][k].
//   QK computes S^T = K*Q^T: A = K fragment (row ml = key 0..15),
//     B = Q fragment (row ml = packed query), acc[j] = S[query=ml][key].
//     Each lane owns one query row; the other key half sits in lane ml^16,
//     combined with one shfl_xor(16). m, l, alpha are per-lane scalars.
//   PV computes O^T = V^T*P^T: A = V^T fragment (row ml = dim 0..15 of the
//     dc tile), B = the packed P half8, Ofr[dc][j] = O[query][dim].
//     The P half8 from softmax feeds PV directly: zero shuffles.
//   O rescale/normalization fetch per-query alpha/l from the owner lane
//     (8 shuffles per subtile / 8 once in the epilogue).
//
// LDS swizzle (f16 pairs as dwords, S(c) = 8*(c>>2) + (c&3)):
//   K store/read: dword pair p for key k at K[k][(p + S(k&15)) & 127].
//   V store/read: key-pair p for dim d at V_T[d][(p + S(d&15)) & 31].
// Loads are four explicit uint32_t reads unpacked to half8_t; the circular
// row wrap is intentional.
//
// Q f16 rounding matches the incumbent (Q/K/V/P operands f16,
// accumulators/state/O f32). No slots, no windows, no FP8, no other arch.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;

__device__ __forceinline__ int fa2_swiz(int c)
{
    return 8 * (c >> 2) + (c & 3);
}

// Shared workgroup body. PARTIAL=false writes normalized O to `out`;
// PARTIAL=true writes [m, l, O_unnormalized[256]] records (stride 258) to
// `out`. `tile0/tile1` bound the KT64 tile range (direct: whole sequence;
// partial split: this split's disjoint range). `split` selects the record
// plane for PARTIAL (ignored otherwise).
template <bool PARTIAL>
__device__ __forceinline__ void fa2_gqa_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base,
    int tile0,
    int tile1,
    int split,
    int n_splits)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const bool compute = wave < 3;

    const int q_stride = 24 * 256;
    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + 8192;

    // ---- Transition 1: online state init (compute waves) ----
    // Q is NOT kept persistent (64 VGPR the tile loop cannot spare).
    // Instead each QK chunk reloads its B-fragment as two float4 loads
    // from L1-resident Q (the 16 KiB per-wave Q tile is read once per
    // tile into L1 and re-hit per subtile; L1 bandwidth makes the
    // re-read free). This is the authorized fallback for the register
    // gate: persistent Q left exactly 2 spills.
    typedef float __attribute__((ext_vector_type(4))) float4_t;
    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    if (compute) {
        h_ml = 6 * kv_h + (wave << 1) + (ml >> 3);
        qr_ml = q_base + (ml & 7);
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // ---- Per-WG causal bounds (positions[] is authoritative) ----
    // Lanes 0..7 read the tile's eight positions; wave 0 reduces; lane 0
    // publishes via two transient LDS mailbox dwords (overwritten by the
    // first V fill, so the 64 KiB plane budget is unaffected).
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (tid < 8) {
        const int qr = q_base + tid;
        if (qr < batch_size) {
            const int p = positions[qr];
            wmx = p;
            wmn = p;
        }
    }
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    if (tid == 0) {
        LDS[16382] = (uint32_t)wmx;
        LDS[16383] = (uint32_t)wmn;
    }
    __syncthreads();
    // Uniformize through lane 0: every lane loaded the same mailbox
    // dwords, but that uniformity is invisible through LDS.
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[16382]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[16383]);
    const int seq_len = gmax + 1;

    // ---- Transitions 2-4: KT64 tiles ----
    for (int tile = tile0; tile < tile1; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Transition 2: cooperative KT64 fill (all four waves) ----
        // K: 512 blocks (k*8+b), one block per slot. Codes are loaded on
        // demand per pair (no staging array) and dequantized straight
        // into the swizzled dword plane.
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 3;
            const int b = idx & 7;
            const int gk = ktile + k;
            const unsigned char* blk = (gk < seq_len)
                ? k_cache + (unsigned long long)gk * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t su = 0;
            if (blk != nullptr)
                su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            _Float16 s = ((fa2_dw_t){.u = (uint32_t)su}).h[0];
            const float sf = (float)s;
            const int rot = fa2_swiz(k & 15);
            const int base = k * 128;
            const int pb = b * 16;
#pragma unroll 1
            for (int i = 0; i < 16; ++i) {
                const float c0 =
                    (blk != nullptr) ? (float)(int8_t)blk[2 + 2 * i] : 0.0f;
                const float c1 =
                    (blk != nullptr) ? (float)(int8_t)blk[2 + 2 * i + 1] : 0.0f;
                fa2_dw_t dw;
                dw.h[0] = (_Float16)(sf * c0);
                dw.h[1] = (_Float16)(sf * c1);
                Kdw[base + ((pb + i + rot) & 127)] = dw.u;
            }
        }
        // V: 256 key-pairs (m*8+b); each slot reads the even and odd block
        // and packs (V[2m][d], V[2m+1][d]) per output dim. Codes are loaded
        // on demand per dim (no staging array).
        for (int t = 0; t < 2; ++t) {
            const int pair = tid + t * 128;
            const int m = pair >> 3;
            const int b = pair & 7;
            const int k0 = ktile + 2 * m;
            const unsigned char* blk0 = (k0 < seq_len)
                ? v_cache + (unsigned long long)k0 * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            const unsigned char* blk1 = (k0 + 1 < seq_len)
                ? v_cache + (unsigned long long)(k0 + 1) * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t s0u = 0;
            uint16_t s1u = 0;
            if (blk0 != nullptr)
                s0u = (uint16_t)blk0[0] | ((uint16_t)blk0[1] << 8);
            if (blk1 != nullptr)
                s1u = (uint16_t)blk1[0] | ((uint16_t)blk1[1] << 8);
            _Float16 s0 = ((fa2_dw_t){.u = (uint32_t)s0u}).h[0];
            _Float16 s1 = ((fa2_dw_t){.u = (uint32_t)s1u}).h[0];
            const float sf0 = (float)s0;
            const float sf1 = (float)s1;
#pragma unroll 1
            for (int i = 0; i < 32; ++i) {
                const int d = b * 32 + i;
                const float c0 =
                    (blk0 != nullptr) ? (float)(int8_t)blk0[2 + i] : 0.0f;
                const float c1 =
                    (blk1 != nullptr) ? (float)(int8_t)blk1[2 + i] : 0.0f;
                fa2_dw_t dw;
                dw.h[0] = (_Float16)(sf0 * c0);
                dw.h[1] = (_Float16)(sf1 * c1);
                Vdw[d * 32 + ((m + fa2_swiz(d & 15)) & 31)] = dw.u;
            }
        }
        __syncthreads();

        // ---- Transition 3: four KT16 QK/softmax/PV subtiles ----
        if (compute) {
#pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const int kt_sub = ktile + sub * 16;
                if (kt_sub > gmax)
                    continue;
                const bool full = (kt_sub + 15) <= gmin;
                // This lane's softmax query is ml; its position is loaded
                // once per subtile (masked path only).
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                // QK as S^T = K*Q^T: A = K fragment (row ml = key),
                // B = Q fragment reloaded per chunk (row ml = query).
                // acc[j] = S[query=ml][key=8*kg+j].
                float8_t sacc =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                {
                    const int key = sub * 16 + ml;
                    const int rot = fa2_swiz(key & 15);
                    const int base = key * 128;
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc) {
                        // Q B-fragment reloaded per chunk as two 16 B
                        // float4 loads from L1-resident Q (16 B aligned:
                        // qq_base is 4-B based and dc*16+kg*8 is a
                        // multiple of 8 floats).
                        half8_t qf;
                        {
                            const float* qb =
                                q + qq_base + (unsigned)(dc * 16 + kg * 8);
                            const float4_t lo = *(const float4_t*)qb;
                            const float4_t hi = *(const float4_t*)(qb + 4);
                            qf[0] = (_Float16)lo[0];
                            qf[1] = (_Float16)lo[1];
                            qf[2] = (_Float16)lo[2];
                            qf[3] = (_Float16)lo[3];
                            qf[4] = (_Float16)hi[0];
                            qf[5] = (_Float16)hi[1];
                            qf[6] = (_Float16)hi[2];
                            qf[7] = (_Float16)hi[3];
                        }
                        half8_t kf;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            fa2_dw_t dw;
                            dw.u = Kdw[base
                                + ((dc * 8 + kg * 4 + i + rot) & 127)];
                            kf[2 * i] = dw.h[0];
                            kf[2 * i + 1] = dw.h[1];
                        }
                        sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                            kf, qf, sacc);
                        // Bound load hoisting to this chunk: without it
                        // all 16 iterations' loads pile up in VGPRs.
                        asm volatile("" ::: "memory");
                    }
                }

                // Mask + row max. The mx chain serializes the unrolled
                // rows; the other key half is combined with one shuffle.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int key_g = kt_sub + 8 * kg + j;
                    const bool ok = qok_ml && (full || key_g <= pos_ml);
                    const float x = ok ? sacc[j] * scale_attn : -INFINITY;
                    // NOTE: max-accumulation happens in the second pass
                    // below; this pass only masks into sc.
                    sacc[j] = x;
                }
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    mx = fmaxf(mx, sacc[j]);
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float m_new = fmaxf(m_old, mx);
                const float alpha =
                    (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                m_old = m_new;

                // Probabilities pack straight into the PV B operand.
                // The sum chain likewise serializes rows.
                half8_t prow;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY)
                        ? 0.0f
                        : __expf(sacc[j] - m_new);
                    prow[j] = (_Float16)e;
                    sum += e;
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;

                // O rescale with per-query alpha fetched from the owner
                // lane (ml=q, kg=0), i.e. lane 8*kg+j for O-query 8*kg+j.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float a_j = __shfl(alpha, 8 * kg + j, 32);
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        Ofr[dc][j] *= a_j;
                }

                // PV as O^T = V^T*P^T: A = V^T fragment (row ml = dim of
                // the dc tile), B = prow. Ofr[dc][j] = O[query][dim].
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const int d = dc * 16 + ml;
                    const int rot = fa2_swiz(d & 15);
                    const int vbase = d * 32;
                    const int ps = sub * 8 + kg * 4;
                    half8_t vf;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        fa2_dw_t dw;
                        dw.u = Vdw[vbase + ((ps + i + rot) & 31)];
                        vf[2 * i] = dw.h[0];
                        vf[2 * i + 1] = dw.h[1];
                    }
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        vf, prow, Ofr[dc]);
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }
            }
        }
        // Tile turnover: no wave may overwrite LDS before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Transition 4: completion ----
    // Ofr[dc][j] at lane (ml,kg) = O[query=8*kg+j][dim=dc*16+ml].
    if (compute) {
        if (!PARTIAL) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int qf2 = 8 * kg + j;
                const int qr = q_base + (qf2 & 7);
                if (qr < batch_size) {
                    const int h = 6 * kv_h + (wave << 1) + (qf2 >> 3);
                    // Owner lane (ml=qf2, kg=0) holds this query's l.
                    const float lq = __shfl(l_val, qf2, 32);
                    const float inv = lq > 0.0f ? 1.0f / lq : 0.0f;
                    // 32-bit row base (whole O tensor < 10 MiB), hoisted
                    // out of the dc loop so the unrolled stores share it.
                    const unsigned bo =
                        (unsigned)qr * 6144u + (unsigned)h * 256u;
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc) {
                        out[bo + (unsigned)(dc * 16 + ml)] =
                            Ofr[dc][j] * inv;
                    }
                }
            }
        } else {
            // m/l owner per query is lane (ml=query, kg=0).
            if (kg == 0) {
                if (qok_ml) {
                    // 32-bit record base (partials < 64 MiB even at S=8).
                    const unsigned rb =
                        (((unsigned)qr_ml * 24u + (unsigned)h_ml)
                             * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                    float* rec = out + rb;
                    rec[0] = m_old;
                    rec[1] = l_val;
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int qf2 = 8 * kg + j;
                const int qr = q_base + (qf2 & 7);
                if (qr < batch_size) {
                    const int h = 6 * kv_h + (wave << 1) + (qf2 >> 3);
                    const unsigned rb =
                        (((unsigned)qr * 24u + (unsigned)h)
                             * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                    float* rec = out + rb;
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        rec[2 + dc * 16 + ml] = Ofr[dc][j];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_gfx1201(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    // tiles_total is unknown without seq_len; the body derives seq_len
    // from positions[] and clamps, so pass an unbounded range.
    fa2_gqa_body<false>(q, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 0x3fffffff, 0, 1);
}

extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_partial_gfx1201(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn,
    int n_splits)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    const int split = blockIdx.z;
    if (split >= n_splits)
        return;
    // Key-range partition needs seq_len, which lives in positions[] on
    // the host side of the launcher... instead partition by tile index
    // with an unbounded range: split s owns tiles [s*T/S,(s+1)*T/S) of
    // the clamped range. The body cannot know T yet, so approximate with
    // a fixed upper bound derived from the 32768 max_ctx_len gate:
    // T_MAX = 32768/64 = 512 tiles; per = ceil(512/S).
    const int per = (512 + n_splits - 1) / n_splits;
    const int t0 = split * per;
    const int t1 = t0 + per;
    fa2_gqa_body<true>(q, k_cache, v_cache, partials, positions, batch_size,
                       scale_attn, kv_h, q_base, t0, t1, split, n_splits);
}

extern "C" __global__ __launch_bounds__(256) void attention_q8_0_fa2_gqa_merge_gfx1201(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int batch_size,
    int n_heads,
    int head_dim,
    int n_splits)
{
    if (n_heads != 24 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rec = blockIdx.x * 8 + wave;
    if (rec >= batch_size * n_heads)
        return;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        const float* pr =
            partials + ((long long)rec * n_splits + s) * 258;
        m = fmaxf(m, pr[0]);
    }
    for (int d = lane; d < head_dim; d += 32) {
        float num = 0.0f;
        float den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float* pr =
                partials + ((long long)rec * n_splits + s) * 258;
            const float w =
                (pr[1] > 0.0f) ? __expf(pr[0] - m) : 0.0f;
            den += w * pr[1];
            num += w * pr[2 + d];
        }
        out[(long long)rec * head_dim + d] =
            (den > 0.0f) ? num / den : 0.0f;
    }
}
