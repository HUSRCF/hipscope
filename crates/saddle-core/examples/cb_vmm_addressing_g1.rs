// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// PLAN G1 — per-request VMM KV addressing gate (Slice1A).
//
// Two independently reserved per-request VMM KV owners of unequal length
// (A crosses a physical-chunk/granule boundary, B is short) are written and
// attended by the `_vmm` kernels in ONE mixed launch, then compared
// byte-for-byte against isolated single-request legacy contiguous caches
// written/attended by the existing kernels. Also: pointer/owner/mapped-prefix
// receipt, forced multi-row verify rows, poisoned rejected tail, capacity
// refusal per admission (logical bound + budget, mapped prefix unchanged),
// wrong-epoch / wrong-row / out-of-prefix / masked-slot host negative
// controls, abort + slot reuse with a new owner generation and no bleed into
// the surviving request.
//
// Exit 0 only if every check passes. Usage: cb_vmm_addressing_g1 [q8|fp8|all]

use hip_bridge::HipResult;
use rdna_compute::attention::{VmmKvFormat, VmmKvSide};
use rdna_compute::kv_slots::{validate_vmm_rows, VmmKvSlotDesc};
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_core::kv::{KvBackend, KvCache, KvDims, KvLayers, KvMode};

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const MAX_SEQ: usize = 4096;
const LAYERS: [bool; 4] = [false, true, false, true];
const LEN_A: usize = 2100;
const LEN_B: usize = 300;
const LEN_C: usize = 517;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn err(msg: impl Into<String>) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, &msg.into())
}

fn check(ok: bool, what: &str, fails: &mut Vec<String>) {
    println!("[{}] {what}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        fails.push(what.to_string());
    }
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up_i32(gpu: &Gpu, v: &[i32]) -> HipResult<GpuTensor> {
    gpu.upload_raw(&i32_bytes(v), &[v.len() * 4])
}

/// Per-request source rows (K and V) for every KV layer, deterministic.
struct ReqData {
    k: Vec<Vec<f32>>, // [layer] -> len*kv_dim
    v: Vec<Vec<f32>>,
}

fn req_data(seed: u64, len: usize) -> ReqData {
    let mut r = Lcg(seed);
    let kv_dim = N_KV * HD;
    ReqData {
        k: LAYERS.iter().map(|_| r.vec(len * kv_dim)).collect(),
        v: LAYERS.iter().map(|_| r.vec(len * kv_dim)).collect(),
    }
}

fn new_owner(gpu: &mut Gpu, mode: KvMode) -> HipResult<KvCache> {
    let dims = KvDims {
        layers: KvLayers::Mask(LAYERS.to_vec()),
        n_kv_heads: N_KV,
        head_dim: HD,
        max_seq: MAX_SEQ,
        physical_cap: Some(MAX_SEQ),
    };
    KvCache::from_mode_with_backend(mode, KvBackend::Vmm, gpu, &dims)
}

/// One legacy contiguous per-request cache (isolated reference).
struct Legacy {
    k: Vec<Option<GpuTensor>>,
    v: Vec<Option<GpuTensor>>,
}

fn legacy_write(
    gpu: &mut Gpu,
    fmt: VmmKvFormat,
    data: &ReqData,
    rows: std::ops::Range<usize>,
    row: usize,
) -> HipResult<Legacy> {
    let kv_dim = N_KV * HD;
    let mut out = Legacy { k: vec![], v: vec![] };
    let pos: Vec<i32> = rows.clone().map(|p| p as i32).collect();
    let pos_t = up_i32(gpu, &pos)?;
    for (l, &is_kv) in LAYERS.iter().enumerate() {
        if !is_kv {
            out.k.push(None);
            out.v.push(None);
            continue;
        }
        let mut sides = vec![];
        for src in [&data.k[l], &data.v[l]] {
            let dst = gpu.zeros(&[MAX_SEQ * row / 4], DType::F32)?;
            let s = gpu.upload_f32(&src[rows.start * kv_dim..rows.end * kv_dim], &[pos.len() * kv_dim])?;
            match fmt {
                VmmKvFormat::Q8 => gpu.kv_cache_write_q8_0_batched(&dst, &s, &pos_t, N_KV, HD, pos.len())?,
                VmmKvFormat::Fp8E4M3 => {
                    gpu.kv_cache_write_fp8_e4m3_batched(&dst, &s, &pos_t, N_KV, HD, pos.len())?
                }
            }
            gpu.free_tensor(s)?;
            sides.push(dst);
        }
        let v = sides.pop();
        let k = sides.pop();
        out.k.push(k);
        out.v.push(v);
    }
    gpu.free_tensor(pos_t)?;
    Ok(out)
}

/// One mixed `_vmm` write launch per layer and side over rows of several
/// requests: `rows` = (slot, position, request data).
fn vmm_write(
    gpu: &mut Gpu,
    fmt: VmmKvFormat,
    layer_descs: &[Vec<VmmKvSlotDesc>],
    rows: &[(usize, usize, &ReqData)],
    generations: &[u64],
) -> HipResult<()> {
    let kv_dim = N_KV * HD;
    let pos: Vec<i32> = rows.iter().map(|r| r.1 as i32).collect();
    let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let pos_t = up_i32(gpu, &pos)?;
    let slot_t = up_i32(gpu, &slot)?;
    for (l, &is_kv) in LAYERS.iter().enumerate() {
        if !is_kv {
            continue;
        }
        let descs: Vec<VmmKvSlotDesc> = layer_descs.iter().map(|d| d[l]).collect();
        let row_gen: Vec<u64> = rows.iter().map(|r| generations[r.0]).collect();
        validate_vmm_rows(&descs, &slot, &pos, &row_gen).map_err(err)?;
        let d_t = gpu.upload_raw(VmmKvSlotDesc::as_bytes(&descs), &[descs.len() * 32])?;
        for side in [VmmKvSide::K, VmmKvSide::V] {
            let mut src = Vec::with_capacity(rows.len() * kv_dim);
            for &(_, p, data) in rows {
                let s = if side == VmmKvSide::K { &data.k[l] } else { &data.v[l] };
                src.extend_from_slice(&s[p * kv_dim..(p + 1) * kv_dim]);
            }
            let s_t = gpu.upload_f32(&src, &[src.len()])?;
            gpu.kv_cache_write_batched_vmm(fmt, side, &s_t, &pos_t, N_KV, HD, rows.len(), &d_t, &slot_t)?;
            gpu.free_tensor(s_t)?;
        }
        gpu.free_tensor(d_t)?;
    }
    gpu.free_tensor(pos_t)?;
    gpu.free_tensor(slot_t)?;
    Ok(())
}

fn vmm_prefix(gpu: &Gpu, t: &GpuTensor, len: usize, row: usize) -> HipResult<Vec<u8>> {
    let mut out = vec![0u8; len * row];
    gpu.hip.memcpy_dtoh_at(&mut out, &t.buf, 0)?;
    Ok(out)
}

fn legacy_prefix(gpu: &Gpu, t: &GpuTensor, len: usize, row: usize) -> HipResult<Vec<u8>> {
    let mut out = vec![0u8; len * row];
    gpu.hip.memcpy_dtoh_at(&mut out, &t.buf, 0)?;
    Ok(out)
}

fn kv_bytes_equal(gpu: &Gpu, owner: &KvCache, legacy: &Legacy, len: usize, row: usize) -> HipResult<bool> {
    for (l, &is_kv) in LAYERS.iter().enumerate() {
        if !is_kv {
            continue;
        }
        let lk = legacy.k[l].as_ref().unwrap();
        let lv = legacy.v[l].as_ref().unwrap();
        if vmm_prefix(gpu, &owner.k_gpu[l], len, row)? != legacy_prefix(gpu, lk, len, row)?
            || vmm_prefix(gpu, &owner.v_gpu[l], len, row)? != legacy_prefix(gpu, lv, len, row)?
        {
            println!("    layer {l} KV bytes differ");
            return Ok(false);
        }
    }
    Ok(true)
}

/// Reference attention for one request's rows on its legacy cache.
fn legacy_attn(
    gpu: &mut Gpu,
    fmt: VmmKvFormat,
    legacy: &Legacy,
    layer: usize,
    q: &[f32],
    pos: &[i32],
    max_ctx: usize,
) -> HipResult<Vec<f32>> {
    let q_t = gpu.upload_f32(q, &[q.len()])?;
    let o_t = gpu.zeros(&[q.len()], DType::F32)?;
    let p_t = up_i32(gpu, pos)?;
    let (k, v) = (legacy.k[layer].as_ref().unwrap(), legacy.v[layer].as_ref().unwrap());
    match fmt {
        VmmKvFormat::Q8 => gpu.attention_q8_0_kv_batched(
            &q_t, k, v, &o_t, &p_t, N_HEADS, N_KV, HD, MAX_SEQ, max_ctx, pos.len(),
        )?,
        VmmKvFormat::Fp8E4M3 => gpu.attention_fp8_e4m3_kv_batched(
            &q_t, k, v, &o_t, &p_t, N_HEADS, N_KV, HD, MAX_SEQ, max_ctx, pos.len(), None, 0, 0,
        )?,
    }
    gpu.hip.device_synchronize()?;
    let out = gpu.download_f32(&o_t)?;
    for t in [q_t, o_t, p_t] {
        gpu.free_tensor(t)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn vmm_attn(
    gpu: &mut Gpu,
    fmt: VmmKvFormat,
    descs: &[VmmKvSlotDesc],
    q: &[f32],
    pos: &[i32],
    slot: &[i32],
    row_gen: &[u64],
    max_ctx: usize,
) -> HipResult<Vec<f32>> {
    validate_vmm_rows(descs, slot, pos, row_gen).map_err(err)?;
    let q_t = gpu.upload_f32(q, &[q.len()])?;
    let o_t = gpu.zeros(&[q.len()], DType::F32)?;
    let p_t = up_i32(gpu, pos)?;
    let s_t = up_i32(gpu, slot)?;
    let d_t = gpu.upload_raw(VmmKvSlotDesc::as_bytes(descs), &[descs.len() * 32])?;
    gpu.attention_kv_batched_vmm(fmt, &q_t, &o_t, &p_t, N_HEADS, N_KV, HD, max_ctx, pos.len(), &d_t, &s_t)?;
    gpu.hip.device_synchronize()?;
    let out = gpu.download_f32(&o_t)?;
    for t in [q_t, o_t, p_t, s_t, d_t] {
        gpu.free_tensor(t)?;
    }
    Ok(out)
}

fn free_legacy(gpu: &mut Gpu, l: Legacy) -> HipResult<()> {
    for t in l.k.into_iter().chain(l.v).flatten() {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn run(gpu: &mut Gpu, mode: KvMode, fails: &mut Vec<String>) -> HipResult<()> {
    println!("=== G1 mode={mode:?} arch={} ===", gpu.arch);
    let mut a = new_owner(gpu, mode)?;
    let mut b = new_owner(gpu, mode)?;
    let fmt = a.vmm_kv_format()?;
    let row = fmt.bytes_per_token(N_KV, HD);
    let qd = N_HEADS * HD;
    let budget = usize::MAX;

    // ---- capacity refusal per admission --------------------------------
    let before = a.mapped_token_capacity()?.unwrap_or(0);
    let over_bound = a.provision_vmm_positions(gpu, MAX_SEQ + 1, budget);
    check(over_bound.is_err(), "refuse positions > logical bound (no silent max_seq reduction)", fails);
    if let Err(e) = &over_bound {
        println!("    refusal: {e}");
    }
    let need = a.vmm_admission_bytes(gpu, LEN_A)?;
    let over_budget = a.provision_vmm_positions(gpu, LEN_A, need - 1);
    check(over_budget.is_err(), "refuse admission over physical budget", fails);
    if let Err(e) = &over_budget {
        println!("    refusal: {e}");
    }
    check(
        a.mapped_token_capacity()?.unwrap_or(0) == before,
        &format!("refusal maps nothing (mapped prefix stays {before})"),
        fails,
    );

    // ---- A: provision first chunk, write, grow across the boundary -----
    a.provision_vmm_positions(gpu, 1, budget)?;
    let descs_a0 = a.vmm_slot_descs(gpu)?;
    let m1 = descs_a0[1].mapped_positions as usize;
    println!("    A first chunk mapped_positions={m1} (row {row} B)");
    check(m1 < LEN_A, "A length crosses its first mapped chunk", fails);
    b.provision_vmm_positions(gpu, LEN_B, budget)?;
    let data_a = req_data(0xA, LEN_A + 8);
    let data_b = req_data(0xB, LEN_B);

    // generation per slot (epoch): slot 0 = A, slot 1 = B
    let gen = |d: &[VmmKvSlotDesc]| d[1].owner_generation;
    let descs_b = b.vmm_slot_descs(gpu)?;
    let gens = vec![gen(&descs_a0), gen(&descs_b)];
    // First mixed write: A [0, m1) interleaved with B [0, LEN_B).
    let mut rows: Vec<(usize, usize, &ReqData)> = vec![];
    for i in 0..m1.max(LEN_B) {
        if i < m1 {
            rows.push((0, i, &data_a));
        }
        if i < LEN_B {
            rows.push((1, i, &data_b));
        }
    }
    vmm_write(gpu, fmt, &[descs_a0.clone(), descs_b.clone()], &rows, &gens)?;

    // Grow A (outside any capture) and write the straddling remainder.
    let grown = a.provision_vmm_positions(gpu, LEN_A + 8, budget)?;
    let descs_a = a.vmm_slot_descs(gpu)?;
    let stable = LAYERS.iter().enumerate().all(|(l, _)| {
        descs_a[l].k_base == descs_a0[l].k_base && descs_a[l].v_base == descs_a0[l].v_base
    });
    check(stable && grown > 0, &format!("A grew by {grown} B at a stable VA"), fails);
    check(
        descs_a[1].mapped_positions as usize >= LEN_A + 8 && gen(&descs_a) == gens[0],
        "A mapped prefix covers LEN_A+8, owner generation unchanged",
        fails,
    );
    let rows: Vec<(usize, usize, &ReqData)> = (m1..LEN_A).map(|p| (0, p, &data_a)).collect();
    vmm_write(gpu, fmt, &[descs_a.clone(), descs_b.clone()], &rows, &gens)?;
    gpu.hip.device_synchronize()?;

    // ---- receipt --------------------------------------------------------
    for (name, d) in [("A", &descs_a), ("B", &descs_b)] {
        for (l, x) in d.iter().enumerate().filter(|(l, _)| LAYERS[*l]) {
            println!(
                "    receipt {name} layer {l}: k_base=0x{:x} v_base=0x{:x} mapped_positions={} logical_bound={} owner_generation={}",
                x.k_base, x.v_base, x.mapped_positions, x.logical_bound, x.owner_generation
            );
        }
    }
    check(
        descs_a.iter().zip(&descs_b).enumerate().all(|(l, (x, y))| {
            !LAYERS[l] || (x.k_base != y.k_base && x.v_base != y.v_base && x.owner_generation != y.owner_generation)
        }),
        "A and B own distinct VAs and generations",
        fails,
    );

    // ---- KV byte parity vs isolated legacy ------------------------------
    let leg_a = legacy_write(gpu, fmt, &data_a, 0..LEN_A, row)?;
    let leg_b = legacy_write(gpu, fmt, &data_b, 0..LEN_B, row)?;
    gpu.hip.device_synchronize()?;
    check(kv_bytes_equal(gpu, &a, &leg_a, LEN_A, row)?, "A KV bytes == isolated legacy", fails);
    check(kv_bytes_equal(gpu, &b, &leg_b, LEN_B, row)?, "B KV bytes == isolated legacy", fails);

    // ---- mixed attention: decode + forced verify rows + boundary rows ---
    let pos_a: Vec<i32> = vec![
        (LEN_A - 5) as i32, (LEN_A - 4) as i32, (LEN_A - 3) as i32, (LEN_A - 2) as i32,
        (LEN_A - 1) as i32, (m1 - 1) as i32, m1 as i32, 0,
    ];
    let pos_b: Vec<i32> = vec![(LEN_B - 1) as i32, 0, 17];
    let mut qr = Lcg(0x5EED);
    let q_a = qr.vec(pos_a.len() * qd);
    let q_b = qr.vec(pos_b.len() * qd);
    // Interleave rows: B0 A0 A1 B1 A2.. to prove row_slot routing.
    let mut order: Vec<(usize, usize)> = vec![];
    let (mut ia, mut ib) = (0, 0);
    while ia < pos_a.len() || ib < pos_b.len() {
        if ib < pos_b.len() {
            order.push((1, ib));
            ib += 1;
        }
        for _ in 0..2 {
            if ia < pos_a.len() {
                order.push((0, ia));
                ia += 1;
            }
        }
    }
    let max_ctx = LEN_A + 8;
    let mut attn_check = |gpu: &mut Gpu, label: &str, descs_a: &[VmmKvSlotDesc], fails: &mut Vec<String>| -> HipResult<()> {
        for (l, &is_kv) in LAYERS.iter().enumerate() {
            if !is_kv {
                continue;
            }
            let ref_a = legacy_attn(gpu, fmt, &leg_a, l, &q_a, &pos_a, max_ctx)?;
            let ref_b = legacy_attn(gpu, fmt, &leg_b, l, &q_b, &pos_b, max_ctx)?;
            let descs = vec![descs_a[l], descs_b[l]];
            let (mut q, mut pos, mut slot, mut rg) = (vec![], vec![], vec![], vec![]);
            for &(s, i) in &order {
                let (qq, pp) = if s == 0 { (&q_a, &pos_a) } else { (&q_b, &pos_b) };
                q.extend_from_slice(&qq[i * qd..(i + 1) * qd]);
                pos.push(pp[i]);
                slot.push(s as i32);
                rg.push(gens[s]);
            }
            let got = vmm_attn(gpu, fmt, &descs, &q, &pos, &slot, &rg, max_ctx)?;
            let mut ok = true;
            for (r, &(s, i)) in order.iter().enumerate() {
                let want = if s == 0 { &ref_a } else { &ref_b };
                ok &= bits_equal(&got[r * qd..(r + 1) * qd], &want[i * qd..(i + 1) * qd]);
            }
            check(ok, &format!("{label}: layer {l} mixed VMM attention bitwise == isolated legacy"), fails);
        }
        Ok(())
    };
    attn_check(gpu, "mixed", &descs_a, fails)?;

    // ---- poisoned rejected tail (rows past every live query) ------------
    let poison = ReqData {
        k: data_a.k.iter().map(|v| v.iter().map(|x| x * 1.0e3 + 7.0).collect()).collect(),
        v: data_a.v.iter().map(|v| v.iter().map(|x| -x * 1.0e3).collect()).collect(),
    };
    let rows: Vec<(usize, usize, &ReqData)> = (LEN_A..LEN_A + 8).map(|p| (0, p, &poison)).collect();
    vmm_write(gpu, fmt, &[descs_a.clone(), descs_b.clone()], &rows, &gens)?;
    gpu.hip.device_synchronize()?;
    attn_check(gpu, "poisoned tail", &descs_a, fails)?;

    // ---- host negative controls -----------------------------------------
    let d1 = vec![descs_a[1], descs_b[1]];
    let wrong_epoch = validate_vmm_rows(&d1, &[0], &[5], &[gens[0] + 1000]);
    check(wrong_epoch.is_err(), "wrong-epoch row rejected", fails);
    let wrong_row = validate_vmm_rows(&d1, &[1], &[5], &[gens[0]]);
    check(wrong_row.is_err(), "wrong-row (A row pointed at B slot) rejected", fails);
    let past = validate_vmm_rows(&d1, &[1], &[descs_b[1].mapped_positions as i32], &[gens[1]]);
    check(past.is_err(), "position == mapped_positions rejected", fails);
    let oob_slot = validate_vmm_rows(&d1, &[2], &[0], &[gens[0]]);
    check(oob_slot.is_err(), "slot index out of range rejected", fails);
    let masked = validate_vmm_rows(&[VmmKvSlotDesc::MASKED], &[0], &[0], &[0]);
    check(masked.is_err(), "masked/idle slot rejected", fails);
    let masked_layer = vmm_attn(gpu, fmt, &[descs_a[0], descs_b[0]], &q_a[..qd], &[0], &[0], &[gens[0]], 8);
    check(masked_layer.is_err(), "non-KV layer descriptor refused before launch", fails);

    // ---- abort A, reuse slot 0 with a new owner C -----------------------
    gpu.hip.device_synchronize()?;
    let event = gpu.hip.event_create()?;
    gpu.hip.event_record(&event, gpu.active_stream.as_ref())?;
    a.release_vmm_after(gpu, Some(&event))?;
    gpu.hip.event_destroy(event)?;
    free_legacy(gpu, leg_a)?;
    let mut c = new_owner(gpu, mode)?;
    c.provision_vmm_positions(gpu, LEN_C, budget)?;
    let descs_c = c.vmm_slot_descs(gpu)?;
    check(gen(&descs_c) != gens[0], "reused slot gets a new owner generation", fails);
    let stale = validate_vmm_rows(&[descs_c[1], descs_b[1]], &[0], &[3], &[gens[0]]);
    check(stale.is_err(), "stale A epoch rejected against reused slot C", fails);
    let data_c = req_data(0xC, LEN_C);
    let gens_c = vec![gen(&descs_c), gens[1]];
    // C prefill rows mixed with one B decode-row rewrite of B's last position
    // (same data, so B's bytes must still equal its isolated reference).
    let mut rows: Vec<(usize, usize, &ReqData)> = (0..LEN_C).map(|p| (0, p, &data_c)).collect();
    rows.push((1, LEN_B - 1, &data_b));
    vmm_write(gpu, fmt, &[descs_c.clone(), descs_b.clone()], &rows, &gens_c)?;
    gpu.hip.device_synchronize()?;
    let leg_c = legacy_write(gpu, fmt, &data_c, 0..LEN_C, row)?;
    gpu.hip.device_synchronize()?;
    check(kv_bytes_equal(gpu, &c, &leg_c, LEN_C, row)?, "C (reused slot) KV bytes == isolated legacy", fails);
    check(kv_bytes_equal(gpu, &b, &leg_b, LEN_B, row)?, "B unaffected by abort/reuse (no bleed)", fails);
    for (l, &is_kv) in LAYERS.iter().enumerate() {
        if !is_kv {
            continue;
        }
        let pos_c = vec![(LEN_C - 1) as i32, 100];
        let q_c = qr.vec(pos_c.len() * qd);
        let want = legacy_attn(gpu, fmt, &leg_c, l, &q_c, &pos_c, LEN_C)?;
        let got = vmm_attn(gpu, fmt, &[descs_c[l], descs_b[l]], &q_c, &pos_c, &[0, 0], &[gens_c[0]; 2], LEN_C)?;
        check(bits_equal(&got, &want), &format!("C layer {l} VMM attention == isolated legacy"), fails);
    }

    free_legacy(gpu, leg_b)?;
    free_legacy(gpu, leg_c)?;
    b.release_vmm_after(gpu, None)?;
    c.release_vmm_after(gpu, None)?;
    Ok(())
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let mut gpu = Gpu::init().expect("gpu init");
    let modes: Vec<KvMode> = match which.as_str() {
        "q8" => vec![KvMode::Q8],
        "fp8" => vec![KvMode::Fp8],
        _ => vec![KvMode::Q8, KvMode::Fp8],
    };
    let mut fails = vec![];
    for mode in modes {
        if let Err(e) = run(&mut gpu, mode, &mut fails) {
            println!("[FAIL] mode {mode:?} aborted: {e}");
            fails.push(format!("{mode:?} aborted: {e}"));
        }
    }
    if fails.is_empty() {
        println!("G1 RESULT: PASS");
    } else {
        println!("G1 RESULT: FAIL ({} checks)", fails.len());
        std::process::exit(1);
    }
}
