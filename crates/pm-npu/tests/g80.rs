// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! G80 (`gemm_g80::design_g80`) in the whole-array simulator: a 256-row x 1280-column wave (128x80 core tile, a vertical core
//! pair shares one 128-row A block through E/O halves, N fixed at 1280, int8 SRS epilogue, `K = 64*kc`, `kc <= 40`).
//!
//! Proven here, for the generated PDI/TXN only:
//!   * every submit is CPU-exact (`clamp(floor(c / 2^shift), -128, 127)`, computed independently of the design), leaves the
//!     original and the packed operands unchanged and moves exactly the per-channel shim bytes of the layout
//!     (MM2S0 = B `kc*10240` per column, MM2S1 = A `waves*kc*4096` on columns 0..4 only, S2MM0/1 = `waves*2*10240`);
//!   * every core tile of every wave is delivered; the tile ranges tile the C argument exactly;
//!   * padded `M`, odd and even `kc*waves`, three seeds on one reused `Config`, signed extremes under every shift;
//!   * `no_compute` / `repeat` probes are not GEMMs but complete, with the same finite byte traffic;
//!   * explicit shape / epilogue / probe-layout boundaries are rejected, accepted boundaries build;
//!   * the grouped launch API (`append_run_body` with arena offsets): three one-wave experts M = 64 / 160 / 200 in ONE TXN
//!     equal their eager fresh-`Config` bytes and leave the retained state of a single full run;
//!   * lean submits (`lean_insts`, `lean_insts_requeue`, `append_lean_run_body`): full -> 3 lean submits with fresh operands
//!     are exact, byte-equal to a fresh full run with the same operands and leave `Config::state_snapshot` equal to a
//!     fresh full run; full/lean interleaves; the E3 group with lean tail bodies equals the all-full group (bytes and state).
//!
//! Simulator limitations (not faked): `Config` has no consumer-delay / asymmetric drain pacing knob and no mid-run state
//! inspection (`state_snapshot` is a between-submit fingerprint, `submit_with`'s producer only publishes host bytes), and
//! no TXN op can inject into the middle of a body. The two-stream B readiness ordering (even / odd N tile, K chunk, replay
//! per M wave) is therefore proven through *distinct per-parity, per-chunk B patterns* whose every misordering changes C
//! (`pattern_b_*`), at one and two M waves, reusing one `Config` with the parity pattern flipped between submits. A
//! negative perturbed ready-credit run is not accessible for the same reason.
use std::{hash::Hasher,ops::Range,panic::{catch_unwind,AssertUnwindSafe}};
use pm_npu::dma::Direction as DmaDirection;
use pm_npu::kernels::{gemm_array::ArrayDesign,gemm_core::{Control,Epilogue,Probe},gemm_g80::design_g80,gemm_i8::{cpu_reference,ArgKind,ArgSpec}};
use pm_npu::txn::Txn;
use pm_npu::kernels::{experts::{grouped,grouped_all_full,ExpertPlan,GroupedDesign},ring::{RingLayout,SENTINEL}};
use std::cell::Cell;
use pm_npu::sim::{config::{parse_txn,Config,TxnOp},dma::Direction};

const N:usize=1280;
const WAVE_M:usize=256;
const COLS:usize=8;
const A_COLS:usize=4;
/// One K chunk of the A half a column injects (128 rows x 64 / 2), of the B segment of one column (two 80-wide N tiles x 64),
/// and one core C tile (128 x 80 int8).
const A_CHUNK:usize=4096;
const B_CHUNK:usize=10240;
const C_TILE:usize=10240;
const POISON:u8=0xa5;
const GUARD:u8=0xcc;
const GAP:usize=256;
const SEEDS:[u32;3]=[0x12345678,0x98765432,0x2468ace1];
const BOTH:[Control;2]=[Control::Fast,Control::Slow];
/// (m, k) with kc*waves = 1, 2, 3, 8, 9, 12 (odd and even) plus padded 200 rows (2) and 513 rows (3 waves, kc 1).
const REUSE_SHAPES:[(usize,usize);8]=[(256,64),(256,128),(768,64),(512,256),(768,192),(1024,192),(200,128),(513,64)];

fn int8(shift:u8)->Epilogue {Epilogue::Int8{shift}}
fn design(m:usize,k:usize,ctl:Control,shift:u8)->ArrayDesign {design_g80(m,N,k,int8(shift),ctl)}
fn signature(bytes:&[u8])->u64 {let mut h=std::collections::hash_map::DefaultHasher::new();h.write(bytes);h.finish()}
fn first_diff(a:&[u8],b:&[u8])->Option<usize> {if a.len()!=b.len() {return Some(a.len().min(b.len()))}(0..a.len()).find(|&i|a[i]!=b[i])}
/// Independent reference: exact int32 GEMM, arithmetic shift (floor), saturate to int8.
fn want(a:&[i8],b:&[i8],m:usize,k:usize,shift:u8)->Vec<i32> {
    cpu_reference(a,b,m,N,k).into_iter().map(|c|((c as i64)>>shift).clamp(-128,127) as i32).collect()
}

// ---------------------------------------------------------------- operands
fn random(m:usize,k:usize,seed:u32)->(Vec<i8>,Vec<i8>) {
    let mut state=seed;let mut next=|| {state^=state<<13;state^=state>>17;state^=state<<5;state as i8};
    let mut a:Vec<_>=(0..m*k).map(|_|next()).collect();let mut b:Vec<_>=(0..k*N).map(|_|next()).collect();a[0]=-128;b[0]=-128;(a,b)
}
/// Row and column classes at the signed extremes: -128 / 127 / -1 / 1 / 0 / alternating / pseudo-random, so products reach
/// positive and negative saturation, exact boundaries and small negative sums whose floor differs from truncation.
fn extremes(m:usize,k:usize,rot:usize)->(Vec<i8>,Vec<i8>) {
    let a=(0..m*k).map(|x|{let (i,kk)=(x/k,x%k);match (i+rot)%8 {
        0=>-128,1=>127,2=>-1,3=>1,4=>if kk%2==0 {-128} else {127},5=>if kk%3==0 {127} else {-1},6=>(kk*37+i) as i8,_=>0}}).collect();
    let b=(0..k*N).map(|x|{let (kk,j)=(x/N,x%N);match (j+rot)%8 {
        0=>-128,1=>127,2=>-1,3=>1,4=>if kk%2==0 {127} else {-128},5=>if kk%3==1 {-128} else {1},6=>(kk*11+j*5) as i8,_=>0}}).collect();
    (a,b)
}
/// Every K chunk and each of the two N-tile parities of a core column (tile = j/80, parity = tile%2, flipped by `flip`) gets
/// its own value stream, so a consumer that takes the wrong parity, the wrong chunk or a stale chunk produces another C.
fn pattern_b(m:usize,k:usize,flip:bool)->(Vec<i8>,Vec<i8>) {
    let a=(0..m*k).map(|x|{let (i,kk)=(x/k,x%k);(((kk/64)*29+(kk%64)*3+i*7+(i/128)*11)%255) as i32-127}).map(|v|v as i8).collect();
    let b=(0..k*N).map(|x|{let (kk,j)=(x/N,x%N);let t=j/80;let parity=(t%2)^usize::from(flip);
        ((((kk/64)*53+parity*101+(t/2)*17+(kk%64)*5+(j%80)*3+1)%255) as i32-127) as i8}).collect();
    (a,b)
}

// ---------------------------------------------------------------- layout and traffic
type Traffic=Vec<(u8,Direction,u8,u64)>;
fn traffic(sim:&Config)->Traffic {sim.shim_stats().map(|c|(c.col,c.direction,c.channel,c.dram_bytes())).collect()}
fn core_sum(sim:&Config)->(u64,u64) {sim.core_stats().fold((0,0),|(busy,wait),c|(busy+c.busy,wait+c.lock_wait))}
/// Expected bytes of one shim channel over `runs` bodies moving `waves` waves in total.
fn expected_bytes(dir:Direction,ch:u8,col:u8,runs:usize,waves:usize,kc:usize)->u64 {
    (match (dir,ch) {
        (Direction::Mm2s,0)=>runs*kc*B_CHUNK,
        (Direction::Mm2s,1)=>if (col as usize)<A_COLS {waves*kc*A_CHUNK} else {0},
        (Direction::S2mm,0)|(Direction::S2mm,1)=>waves*2*C_TILE,
        _=>panic!("unexpected shim channel {dir:?} {ch}"),
    }) as u64
}
/// Per-channel deltas equal the layout formula; returns (read, written) bytes of the delta.
fn assert_traffic(before:&Traffic,after:&Traffic,runs:usize,waves:usize,kc:usize,label:&str)->(u64,u64) {
    assert_eq!(before.len(),after.len(),"{label}");
    let (mut read,mut written)=(0,0);
    for (&(col,dir,ch,was),&(col1,dir1,ch1,now)) in before.iter().zip(after) {
        assert_eq!((col,dir,ch),(col1,dir1,ch1),"{label}");let delta=now-was;
        assert_eq!(delta,expected_bytes(dir,ch,col,runs,waves,kc),"{label}: shim col {col} {dir:?} ch {ch} bytes");
        match dir {Direction::Mm2s=>read+=delta,Direction::S2mm=>written+=delta}
    }
    (read,written)
}
/// Argument sizes, host byte totals, wave grid and the exact tiling of C by the per-tile ranges.
fn check_layout(d:&ArrayDesign,m:usize,k:usize,label:&str) {
    let (kc,waves)=(k/64,m.div_ceil(WAVE_M));
    assert_eq!(d.waves(),waves,"{label}: waves");assert_eq!(d.wave_m(),WAVE_M,"{label}: wave_m");
    let sizes=[A_COLS*waves*kc*A_CHUNK,COLS*kc*B_CHUNK,COLS*waves*4*C_TILE];
    assert_eq!(d.args,vec![ArgSpec{bytes:sizes[0],kind:ArgKind::In},ArgSpec{bytes:sizes[1],kind:ArgKind::In},ArgSpec{bytes:sizes[2],kind:ArgKind::Out}],"{label}: args");
    assert_eq!(d.host_bytes_read(),(sizes[0]+sizes[1]) as u64,"{label}");assert_eq!(d.host_bytes_written(),sizes[2] as u64,"{label}");
    let tiles=d.output_tile_ranges();assert_eq!(tiles.len(),COLS*4,"{label}: output tiles");
    let mut all:Vec<Range<usize>>=Vec::new();
    for ((col,row),ranges) in &tiles {
        assert_eq!(ranges.len(),waves,"{label}: tile ({col},{row}) ranges");
        for r in ranges {assert_eq!(r.len(),C_TILE,"{label}: tile ({col},{row}) range length");all.push(r.clone())}
    }
    all.sort_by_key(|r|r.start);let mut end=0;
    for r in &all {assert_eq!(r.start,end,"{label}: C tile ranges must tile arg2 exactly");end=r.end}
    assert_eq!(end,sizes[2],"{label}: C tile ranges cover arg2");
}

// ---------------------------------------------------------------- one checked submit
struct Case {label:String,m:usize,k:usize,shift:u8}
fn case(label:impl Into<String>,m:usize,k:usize,shift:u8)->Case {Case{label:label.into(),m,k,shift}}
struct Run {ticks:usize,c:Vec<u8>}
/// One submit of `insts` (a full or lean TXN of `d`) on a possibly reused `Config`. `values`: C must equal the CPU reference
/// (false for timing probes: only completion, traffic and delivery are checked). Counters are cumulative, so everything is
/// a per-submit delta.
fn submit_checked(sim:&mut Config,d:&ArrayDesign,insts:&[u8],c:&Case,a:&[i8],b:&[i8],values:bool)->Run {
    let Case{label,m,k,shift}=c;let (m,k,shift)=(*m,*k,*shift);let (kc,waves)=(k/64,m.div_ceil(WAVE_M));
    let [a_host,b_host]=d.pack_in(a,b);
    assert_eq!((a_host.len(),b_host.len()),(d.args[0].bytes,d.args[1].bytes),"{label}: packed sizes");
    let (a0,b0)=(a.to_vec(),b.to_vec());let before=[signature(&a_host),signature(&b_host)];
    let mut args=vec![a_host,b_host,vec![POISON;d.args[2].bytes]];
    let (t0,ticks0,(busy0,wait0))=(traffic(sim),sim.ticks,core_sum(sim));
    sim.submit(insts,&mut args).unwrap_or_else(|e|panic!("{label} {m}x{N}x{k}: submit failed: {e}"));
    let ticks=sim.ticks-ticks0;let (busy1,wait1)=core_sum(sim);
    if values {
        let w=want(a,b,m,k,shift);assert_eq!(d.reference(a,b),w,"{label}: design reference != independent reference");
        let got=d.unpack_out(&args[2]);assert_eq!(got.len(),w.len(),"{label}");
        if let Some(i)=(0..got.len()).find(|&i|got[i]!=w[i]) {panic!("{label} {m}x{N}x{k} shift {shift}: first mismatch at ({},{}) got {} want {}",i/N,i%N,got[i],w[i])}
    }
    assert_eq!([signature(&args[0]),signature(&args[1])],before,"{label}: packed inputs changed");
    assert!(a==&a0[..] && b==&b0[..],"{label}: operands changed");
    let (read,written)=assert_traffic(&t0,&traffic(sim),1,waves,kc,label);
    assert_eq!((read,written),(d.host_bytes_read(),d.host_bytes_written()),"{label}: host byte totals");
    for ((col,row),ranges) in d.output_tile_ranges() {for (w,r) in ranges.iter().enumerate() {
        assert!(args[2][r.clone()].iter().any(|&x|x!=POISON),"{label}: tile ({col},{row}) wave {w} not delivered");
    }}
    eprintln!("PASS {label} {m}x{N}x{k} shift {shift}: {}; {ticks} functional ticks (not hardware cycles); core busy={} lock_wait={}; DRAM read={read} write={written} \
        (per column MM2S0 B={} MM2S1 A={}(cols<4) S2MM0/1={} B)",if values {format!("{} CPU-exact words",m*N)} else {"timing probe, completion + bytes only".into()},
        busy1-busy0,wait1-wait0,kc*B_CHUNK,waves*kc*A_CHUNK,waves*2*C_TILE);
    Run{ticks,c:args.swap_remove(2)}
}
fn fresh(d:&ArrayDesign)->Config {Config::from_pdi(&d.pdi).unwrap()}
/// One fresh-`Config` full submit.
fn once(d:&ArrayDesign,c:&Case,a:&[i8],b:&[i8])->Run {submit_checked(&mut fresh(d),d,&d.insts,c,a,b,true)}

// ---------------------------------------------------------------- exact shapes
fn shape(m:usize,k:usize,ctl:Control) {
    let d=design(m,k,ctl,12);check_layout(&d,m,k,"layout");let (a,b)=random(m,k,0xc001d00d);
    let run=once(&d,&case(format!("G80 {ctl:?}"),m,k,12),&a,&b);
    let lean=d.lean_insts().map(|l|format!("{} B",l.len())).unwrap_or_else(|e|format!("n/a ({e})"));
    eprintln!("MEASURED G80 {ctl:?} {m}x{N}x{k}: {} ticks; PDI {} B, full TXN {} B, lean TXN {lean}; args A={} B={} C={} B; DRAM read {} write {}",
        run.ticks,d.pdi.len(),d.insts.len(),d.args[0].bytes,d.args[1].bytes,d.args[2].bytes,d.host_bytes_read(),d.host_bytes_written());
}
fn shape_both(m:usize,k:usize) {for ctl in BOTH {shape(m,k,ctl)}}
#[test] fn g80_exact_256_1280_64() {shape_both(256,64)}
#[test] fn g80_exact_256_1280_128() {shape_both(256,128)}
#[test] fn g80_exact_160_1280_2560() {shape_both(160,2560)}
#[test] fn g80_exact_64_1280_2560() {shape_both(64,2560)}
#[test] fn g80_exact_512_1280_640() {shape_both(512,640)}
#[test] fn g80_exact_4096_1280_2560_fast() {shape(4096,2560,Control::Fast)}
#[test] fn g80_exact_4096_1280_2560_slow() {shape(4096,2560,Control::Slow)}
/// Padded M: partial last wave (200, 255, 257, 513), a single row, and exact wave multiples.
#[test] fn g80_exact_padded_m() {for (m,k) in [(200,192),(513,128),(255,64),(257,64),(1,64),(200,2560)] {shape_both(m,k)}}
/// Int8 shift sweep on random data (floor + saturate), one padded wave.
#[test] fn g80_exact_random_shift_sweep() {
    for shift in [0u8,1,6,12,20,31] {for ctl in BOTH {
        let (m,k)=(200,128);let d=design(m,k,ctl,shift);let (a,b)=random(m,k,0xfeed0001);
        once(&d,&case(format!("G80 {ctl:?} random"),m,k,shift),&a,&b);
    }}
}
/// Signed extremes: floor of small negative sums, exact saturation boundary (shift 14, K = 128: 2^21 / 2^14 = 128) and both
/// saturations, through the reverse int8 epilogue overlaying its own C accumulator, at every shift, two padded waves.
#[test] fn g80_extremes_floor_shift_saturation() {
    for shift in [0u8,1,6,12,14,20,31] {for ctl in BOTH {for rot in [0usize,3] {
        let (m,k)=(300,128);let d=design(m,k,ctl,shift);let (a,b)=extremes(m,k,rot);
        let raw=cpu_reference(&a,&b,m,N,k);
        if shift<=12 {assert!(raw.iter().any(|&c|(c as i64)>>shift>=128)&&raw.iter().any(|&c|(c as i64)>>shift<=-129),"data must saturate both ways at shift {shift}")}
        if shift>=1 {assert!(raw.iter().any(|&c|c<0&&(c as i64)&((1i64<<shift)-1)!=0),"data must hold negative non-multiples (floor != truncation) at shift {shift}")}
        assert!(want(&a,&b,m,k,shift).contains(&-128)||shift>12,"shift {shift}: negative saturation must be exercised");
        once(&d,&case(format!("G80 {ctl:?} extremes rot {rot}"),m,k,shift),&a,&b);
    }}}
}

// ---------------------------------------------------------------- context reuse
/// Three seeds on one `Config`: finite queues, lock initial values, iteration counters and ring parities must reset every
/// submit, for odd and even `kc*waves`.
#[test] fn g80_reuses_context_three_seeds_odd_and_even() {
    for (m,k) in REUSE_SHAPES {for ctl in BOTH {
        let d=design(m,k,ctl,12);let mut sim=fresh(&d);
        for seed in SEEDS {let (a,b)=random(m,k,seed);submit_checked(&mut sim,&d,&d.insts,&case(format!("G80 reuse {ctl:?} seed {seed:#x}"),m,k,12),&a,&b,true);}
        eprintln!("PASS G80 same-context {ctl:?} {m}x{N}x{k} (kc*waves={}) three submits, {} functional ticks total",k/64*d.waves(),sim.ticks);
    }}
}

// ---------------------------------------------------------------- B stream readiness / ordering
/// Distinct B per parity and chunk, at one and two M waves (the second M wave replays B from the memtile), same `Config`
/// with the parity pattern flipped and flipped back.
fn pattern_b_case(kc:usize,mw:usize,ctl:Control) {
    let (m,k)=(WAVE_M*mw,64*kc);let d=design(m,k,ctl,12);let mut sim=fresh(&d);
    let (a,b0)=pattern_b(m,k,false);let (_,b1)=pattern_b(m,k,true);
    assert_ne!(want(&a,&b0,m,k,12),want(&a,&b1,m,k,12),"pattern must be parity sensitive");
    for (i,flip) in [false,true,false].into_iter().enumerate() {
        let (a,b)=pattern_b(m,k,flip);
        submit_checked(&mut sim,&d,&d.insts,&case(format!("G80 patternB {ctl:?} kc{kc} mw{mw} flip{i}"),m,k,12),&a,&b,true);
    }
}
#[test] fn g80_pattern_b_parities_and_chunks_at_one_and_two_m_waves() {for kc in [1,2,3,5,8] {for mw in [1,2] {for ctl in BOTH {pattern_b_case(kc,mw,ctl)}}}}
/// Full 40-chunk segment (5120 B per parity half-queue, K = 2560) at one and two M waves.
#[test] fn g80_pattern_b_full_kc40() {for mw in [1,2] {pattern_b_case(40,mw,Control::Fast)}}

// ---------------------------------------------------------------- timing probes
/// Not GEMMs (`no_compute` / `repeat`): the lock / DMA protocol is unchanged, so the run completes, moves the same bytes,
/// delivers every tile and leaves the operands untouched; two submits on one `Config` prove the finite queues requeue.
fn probe_case(m:usize,k:usize,p:Probe) {
    let d=design_g80(m,N,k,int8(12),Control::Probe(p));check_layout(&d,m,k,"probe layout");
    let mut sim=fresh(&d);let mut ticks=Vec::new();
    for seed in [SEEDS[0],SEEDS[1]] {
        let (a,b)=random(m,k,seed);
        ticks.push(submit_checked(&mut sim,&d,&d.insts,&case(format!("G80 probe {p:?} seed {seed:#x}"),m,k,12),&a,&b,false).ticks);
    }
    let (a,b)=random(m,k,SEEDS[0]);let fast=once(&design(m,k,Control::Fast,12),&case("G80 Fast reference",m,k,12),&a,&b);
    eprintln!("MEASURED G80 probe {p:?} {m}x{N}x{k}: {} / {} ticks per submit vs Fast {} ticks",ticks[0],ticks[1],fast.ticks);
}
#[test] fn g80_probe_no_compute_completes_with_exact_bytes() {
    for (m,k) in [(256,128),(512,192)] {probe_case(m,k,Probe{no_compute:true,..Probe::default()});probe_case(m,k,Probe{no_compute:true,serial:true,..Probe::default()})}
}
#[test] fn g80_probe_repeat_completes_with_exact_bytes() {for (m,k) in [(256,128),(300,64)] {probe_case(m,k,Probe{repeat:3,..Probe::default()})}}
/// The probe spellings of the two plain controls are the plain controls.
#[test] fn g80_probe_default_and_serial_are_still_exact_gemms() {
    let (m,k)=(200,128);let (a,b)=random(m,k,SEEDS[2]);
    for p in [Probe::default(),Probe{serial:true,..Probe::default()}] {
        let d=design_g80(m,N,k,int8(12),Control::Probe(p));submit_checked(&mut fresh(&d),&d,&d.insts,&case(format!("G80 {p:?}"),m,k,12),&a,&b,true);
    }
}

// ---------------------------------------------------------------- unsupported shapes and boundaries
fn rejects(f:impl FnOnce()) -> bool {catch_unwind(AssertUnwindSafe(f)).is_err()}
fn assert_all_rejected(cases:Vec<(String,bool)>) {
    let accepted:Vec<_>=cases.iter().filter(|c|!c.1).map(|c|c.0.as_str()).collect();assert!(accepted.is_empty(),"unsupported but accepted: {accepted:?}");
}
#[test] fn g80_rejects_n_other_than_1280() {
    assert_all_rejected([0usize,1,640,1279,1281,2560,2561].into_iter().map(|n|(format!("N={n}"),rejects(||{design_g80(256,n,128,int8(12),Control::Fast);}))).collect());
}
#[test] fn g80_rejects_k_outside_multiples_of_64_up_to_2560() {
    assert_all_rejected([0usize,1,63,65,100,2559,2561,2624,5120].into_iter().map(|k|(format!("K={k}"),rejects(||{design_g80(256,N,k,int8(12),Control::Fast);}))).collect());
}
#[test] fn g80_rejects_empty_and_more_than_256_waves() {
    assert_all_rejected([0usize,256*256+1,256*257].into_iter().map(|m|(format!("M={m}"),rejects(||{design_g80(m,N,64,int8(12),Control::Fast);}))).collect());
}
#[test] fn g80_rejects_i32_epilogue() {assert!(rejects(||{design_g80(256,N,64,Epilogue::I32,Control::Fast);}))}
#[test] fn g80_rejects_probe_layouts_other_than_0() {
    assert_all_rejected((1u8..=3).map(|layout|(format!("layout={layout}"),rejects(||{design_g80(256,N,64,int8(12),Control::Probe(Probe{layout,..Probe::default()}));}))).collect());
}
/// `pack_in` checks the operand lengths.
#[test] fn g80_pack_rejects_wrong_operand_lengths() {
    let (m,k)=(200,128);let d=design(m,k,Control::Fast,12);let (a,b)=random(m,k,SEEDS[0]);
    assert_all_rejected(vec![
        ("A short".into(),rejects(||{d.pack_in(&a[..a.len()-1],&b);})),("A long".into(),rejects(||{let mut x=a.clone();x.push(0);d.pack_in(&x,&b);})),
        ("B short".into(),rejects(||{d.pack_in(&a,&b[..b.len()-1]);})),("B long".into(),rejects(||{let mut x=b.clone();x.push(0);d.pack_in(&a,&x);})),
        ("A empty".into(),rejects(||{d.pack_in(&[],&b);})),
    ]);
}
/// Accepted extremes build with the documented argument sizes: one row, kc = 1, kc = 40, exactly 256 waves.
#[test] fn g80_accepts_boundary_shapes() {
    for (m,k) in [(1,64),(1,2560),(256,2560),(257,64),(256*256,64),(256*256,2560)] {
        for ctl in BOTH {check_layout(&design(m,k,ctl,12),m,k,&format!("{ctl:?} boundary"))}
    }
}

// ---------------------------------------------------------------- grouped launch (`append_run_body` with arena offsets)
struct Expert {m:usize,k:usize,shift:u8,a:Vec<i8>,b:Vec<i8>,d:ArrayDesign,packed:[Vec<u8>;2],eager_c:Vec<u8>,eager:Config}
/// Experts with distinct M that all pad to the same wave count as expert 0: operands and the eager fresh-`Config` full run
/// (itself CPU-exact).
fn experts(k:usize,ms:&[usize],ctl:Control,shift:u8,seed0:u32)->Vec<Expert> {
    let waves=ms[0].div_ceil(WAVE_M);
    ms.iter().enumerate().map(|(e,&m)|{
        let d=design(m,k,ctl,shift);check_layout(&d,m,k,"expert");assert_eq!(d.waves(),waves,"expert M={m} must pad to {waves} wave(s)");
        let (a,b)=random(m,k,seed0.wrapping_add(e as u32*0x9e37));let mut eager=fresh(&d);
        let run=submit_checked(&mut eager,&d,&d.insts,&case(format!("G80 eager expert {e} {ctl:?}"),m,k,shift),&a,&b,true);
        let packed=d.pack_in(&a,&b);Expert{m,k,shift,a,b,d,packed,eager_c:run.c,eager}
    }).collect()
}
/// Arena layout with `GAP` guard bytes around every region; A forward, B and C reversed (non-monotonic).
fn arena_off(ex:&[Expert],e:usize,arg:usize)->u64 {
    let stride=ex[0].d.args[arg].bytes+GAP;let pos=if arg==0 {e} else {ex.len()-1-e};(GAP+pos*stride) as u64
}
fn arena_len(ex:&[Expert],arg:usize)->usize {GAP+ex.len()*(ex[0].d.args[arg].bytes+GAP)}
fn group_txn(ex:&[Expert],lean_tail:bool)->Vec<u8> {
    let mut t=Txn::aie2p_8col();
    for (e,x) in ex.iter().enumerate() {
        let base=[arena_off(ex,e,0),arena_off(ex,e,1),arena_off(ex,e,2)];
        if lean_tail&&e>0 {x.d.append_lean_run_body(&mut t,base,&[]).unwrap_or_else(|err|panic!("lean body of expert {e}: {err}"))} else {x.d.append_run_body(&mut t,base)}
    }
    t.to_bytes()
}
fn group_args(ex:&[Expert])->Vec<Vec<u8>> {
    let mut args=vec![vec![GUARD;arena_len(ex,0)],vec![GUARD;arena_len(ex,1)],vec![POISON;arena_len(ex,2)]];
    for (e,x) in ex.iter().enumerate() {for arg in 0..2 {let at=arena_off(ex,e,arg) as usize;args[arg][at..at+x.packed[arg].len()].copy_from_slice(&x.packed[arg]);}}
    args
}
/// A fresh `Config` after ONE full body of the last expert at its arena offsets (the state a group must end in: the shim DDR
/// patch registers of the final body are part of the retained state, so a standalone zero-base run is not comparable).
fn last_body_only(ex:&[Expert])->Config {
    let e=ex.len()-1;let mut t=Txn::aie2p_8col();ex[e].d.append_run_body(&mut t,[arena_off(ex,e,0),arena_off(ex,e,1),arena_off(ex,e,2)]);
    let mut sim=fresh(&ex[0].d);let mut args=group_args(ex);sim.submit(&t.to_bytes(),&mut args).unwrap();sim
}
/// Submits the group TXN on `sim`: every expert's C region is byte-equal to its eager fresh-`Config` bytes and equals the
/// independent reference, operands are untouched, every byte outside the C regions is still poison, and the shim traffic is
/// the sum of the experts'. Returns the final args.
fn run_group(sim:&mut Config,ex:&[Expert],txn:&[u8],what:&str)->Vec<Vec<u8>> {
    let mut args=group_args(ex);let before=args.clone();let (t0,ticks0)=(traffic(sim),sim.ticks);
    sim.submit(txn,&mut args).unwrap_or_else(|e|panic!("{what}: group submit failed: {e}"));
    let kc=ex[0].k/64;let (read,written)=assert_traffic(&t0,&traffic(sim),ex.len(),ex.len(),kc,what);
    let mut owned=vec![false;args[2].len()];
    for (e,x) in ex.iter().enumerate() {
        let at=arena_off(ex,e,2) as usize;let r=at..at+x.d.args[2].bytes;
        if let Some(i)=first_diff(&args[2][r.clone()],&x.eager_c) {panic!("{what}: expert {e} (M={}) C bytes != eager at byte {i}",x.m)}
        assert_eq!(x.d.unpack_out(&args[2][r.clone()]),want(&x.a,&x.b,x.m,x.k,x.shift),"{what}: expert {e} (M={}) != CPU reference",x.m);
        for o in &mut owned[r] {*o=true}
    }
    assert!(args[0]==before[0]&&args[1]==before[1],"{what}: operand arenas (incl. guards) changed");
    for (i,&o) in owned.iter().enumerate() {if !o {assert_eq!(args[2][i],POISON,"{what}: stray write outside every C region at byte {i}")}}
    assert_eq!((read,written),(ex.iter().map(|x|x.d.host_bytes_read()).sum::<u64>(),ex.iter().map(|x|x.d.host_bytes_written()).sum::<u64>()),"{what}: summed host bytes");
    eprintln!("PASS {what}: E={} M={:?}: every expert byte-equal to its eager fresh-Config C and CPU-exact; {} group ticks; DRAM read={read} write={written}",
        ex.len(),ex.iter().map(|x|x.m).collect::<Vec<_>>(),sim.ticks-ticks0);
    args
}
/// E3 mixed M = 64 / 160 / 200 (one wave each, shared PDI core wave counter) in ONE TXN of full bodies.
fn group_exact(k:usize,ctl:Control) {
    let ms=[64,160,200];let ex=experts(k,&ms,ctl,12,0xc0ffee01);
    for x in &ex[1..] {assert_eq!(x.d.pdi,ex[0].d.pdi,"one-wave experts must share one PDI")}
    let txn=group_txn(&ex,false);let mut sim=fresh(&ex[0].d);let what=format!("G80 group {ctl:?} K={k}");
    run_group(&mut sim,&ex,&txn,&what);
    let eager_ticks:usize=ex.iter().map(|x|x.eager.ticks).sum();
    eprintln!("MEASURED {what}: group {} ticks vs eager sum {eager_ticks} ticks; TXN {} B vs standalone sum {} B",sim.ticks,txn.len(),ex.iter().map(|x|x.d.insts.len()).sum::<usize>());
    // The group leaves the retained state of one full run.
    assert!(sim.state_snapshot()==last_body_only(&ex).state_snapshot(),"{what}: retained state after the group != one fresh full run of the last body at its arena offsets");
    // The same Config again with new operands (finite queues and iteration counters reset), then a standalone full TXN.
    let ex2=experts(k,&ms,ctl,12,0xbadc0de1);assert_eq!(group_txn(&ex2,false),txn,"{what}: group TXN depends on operands");
    run_group(&mut sim,&ex2,&txn,&format!("{what} resubmit"));
    let y=&ex2[0];submit_checked(&mut sim,&y.d,&y.d.insts,&case(format!("{what} standalone after group"),y.m,y.k,12),&y.a,&y.b,true);
    assert!(sim.state_snapshot()==ex[0].eager.state_snapshot(),"{what}: retained state after the standalone run != after one full run");
}
#[test] fn g80_group_e3_mixed_m_exact_fast_k192() {group_exact(192,Control::Fast)}
#[test] fn g80_group_e3_mixed_m_exact_slow_k192() {group_exact(192,Control::Slow)}
#[test] fn g80_group_e3_mixed_m_exact_fast_k64() {group_exact(64,Control::Fast)}
#[test] fn g80_group_e3_mixed_m_exact_fast_k2560() {group_exact(2560,Control::Fast)}

// ---------------------------------------------------------------- lean submits
fn zero_args(d:&ArrayDesign)->Vec<Vec<u8>> {vec![vec![0;d.args[0].bytes],vec![0;d.args[1].bytes],vec![POISON;d.args[2].bytes]]}
/// A fresh `Config` after one completed full run with zero operands: the state every lean submit assumes.
fn primed(d:&ArrayDesign)->Config {
    let mut sim=fresh(d);let mut args=zero_args(d);sim.submit(&d.insts,&mut args).unwrap();
    assert!(args[2].iter().all(|&x|x==0),"zero operands must give zero C");sim
}
fn lean_of(d:&ArrayDesign,what:&str)->Vec<u8> {d.lean_insts().unwrap_or_else(|e|panic!("{what}: lean_insts: {e}"))}
/// One lean submit of `insts` on `sim` for seed operands: exact, byte-equal to a fresh full run (whose own retained state
/// also equals `initial`), and the retained state afterwards equals `initial`.
macro_rules! lean_step {
    ($sim:expr,$d:expr,$insts:expr,$initial:expr,$m:expr,$k:expr,$seed:expr,$what:expr) => {{
        let (a,b)=random($m,$k,$seed);let c=case($what.to_string(),$m,$k,12);
        let mut full=fresh($d);let reference=submit_checked(&mut full,$d,&$d.insts,&c,&a,&b,true);
        assert!(full.state_snapshot()==$initial,"{}: fresh full run state != initial full state",$what);
        let run=submit_checked($sim,$d,$insts,&c,&a,&b,true);
        if let Some(i)=first_diff(&run.c,&reference.c) {panic!("{}: C bytes != fresh full TXN bytes at byte {i}",$what)}
        assert!($sim.state_snapshot()==$initial,"{}: retained state != fresh full state",$what);
        eprintln!("MEASURED {}: lean {} ticks vs full {} ticks",$what,run.ticks,reference.ticks);
    }};
}
/// Full (zero operands) -> three lean submits with fresh operands on the SAME Config.
fn lean_repeat(m:usize,k:usize,ctl:Control) {
    let d=design(m,k,ctl,12);let what=format!("G80 {ctl:?} {m}x{N}x{k}");let lean=lean_of(&d,&what);
    let mut sim=primed(&d);let initial=sim.state_snapshot();
    for (i,seed) in SEEDS.into_iter().enumerate() {lean_step!(&mut sim,&d,&lean,initial,m,k,seed,format!("{what} lean #{i}"))}
    eprintln!("PASS lean x3 {what}: exact + byte-equal to fresh full + state == fresh full; full {} B lean {} B",d.insts.len(),lean.len());
}
/// full -> lean -> full -> lean on one Config.
fn lean_interleave(m:usize,k:usize,ctl:Control) {
    let d=design(m,k,ctl,12);let what=format!("G80 {ctl:?} {m}x{N}x{k}");let lean=lean_of(&d,&what);
    let mut sim=primed(&d);let initial=sim.state_snapshot();
    for (i,(seed,is_lean)) in [(SEEDS[0],false),(SEEDS[1],true),(SEEDS[2],false),(0xdeadbeef,true)].into_iter().enumerate() {
        let label=format!("{what} interleave #{i} {}",if is_lean {"lean"} else {"full"});
        lean_step!(&mut sim,&d,if is_lean {&lean[..]} else {&d.insts[..]},initial,m,k,seed,label);
    }
    eprintln!("PASS full->lean->full->lean {what}");
}
#[test] fn g80_lean_three_submits_exact_and_state_restored() {for (m,k) in REUSE_SHAPES {for ctl in BOTH {lean_repeat(m,k,ctl)}}}
#[test] fn g80_lean_full_lean_interleave_exact_and_state_restored() {for (m,k) in REUSE_SHAPES {for ctl in BOTH {lean_interleave(m,k,ctl)}}}
#[test] fn g80_lean_large_k_and_waves() {for (m,k) in [(160,2560),(512,640)] {lean_repeat(m,k,Control::Fast);lean_interleave(m,k,Control::Fast)}}
const FORCE_B:(u32,DmaDirection,u32)=(0,DmaDirection::S2mm,4);
const FORCE_C:(u32,DmaDirection,u32)=(0,DmaDirection::Mm2s,0);
/// The persistent ring's POLL / DONE clobber column 0 memtile S2MM4 (B fill) and MM2S0 (C drain): the forced reset +
/// requeue keeps every finite queue and still gives exact, byte-equal results and the fresh full retained state.
fn forced_requeue(m:usize,k:usize,ctl:Control) {
    let d=design(m,k,ctl,12);let what=format!("G80 forced {ctl:?} {m}x{N}x{k}");let plain=lean_of(&d,&what);
    let sets:[(&str,Vec<(u32,DmaDirection,u32)>);3]=[("B S2MM4",vec![FORCE_B]),("C MM2S0",vec![FORCE_C]),("B S2MM4 + C MM2S0",vec![FORCE_B,FORCE_C])];
    for (label,extra) in sets {
        let forced=d.lean_insts_requeue(&extra).unwrap_or_else(|e|panic!("{what} {label}: lean_insts_requeue: {e}"));
        let mut sim=primed(&d);let initial=sim.state_snapshot();
        for (i,seed) in SEEDS.into_iter().enumerate() {lean_step!(&mut sim,&d,&forced,initial,m,k,seed,format!("{what} {label} #{i}"))}
        lean_step!(&mut sim,&d,&plain,initial,m,k,0xdeadbeef,format!("{what} plain lean after {label}"));
        eprintln!("PASS forced requeue {label} {what}: lean {} B forced {} B",plain.len(),forced.len());
    }
}
#[test] fn g80_lean_forced_requeue_one_and_two_waves() {for (m,k) in [(256,64),(256,128),(512,192),(200,128)] {for ctl in BOTH {forced_requeue(m,k,ctl)}}}
/// Unsupported forced channels are rejected before any op is appended.
#[test] fn g80_lean_forced_requeue_rejects_unsupported_channels() {
    let d=design(256,128,Control::Fast,12);
    for extra in [(8u32,DmaDirection::S2mm,4u32),(0,DmaDirection::S2mm,5),(0,DmaDirection::Mm2s,4)] {
        assert!(d.lean_insts_requeue(&[extra]).is_err(),"forced {extra:?} must be rejected");
        let mut t=Txn::aie2p_8col();let empty=t.to_bytes();assert!(d.append_lean_run_body(&mut t,[0;3],&[extra]).is_err());assert_eq!(t.to_bytes(),empty,"txn must be unchanged on Err");
    }
}
/// E3 mixed M group: expert 0 full body then lean bodies for experts 1, 2 equals the all-full group in every output byte and
/// in the retained state, and is shorter.
fn group_lean(k:usize,ctl:Control) {
    let ex=experts(k,&[64,160,200],ctl,12,0xc0ffee02);let (full,lean)=(group_txn(&ex,false),group_txn(&ex,true));
    let what=format!("G80 lean group {ctl:?} K={k}");
    let (mut sim_l,mut sim_f)=(fresh(&ex[0].d),fresh(&ex[0].d));
    let args_l=run_group(&mut sim_l,&ex,&lean,&what);let args_f=run_group(&mut sim_f,&ex,&full,&format!("{what} (all-full oracle)"));
    for i in 0..3 {if let Some(at)=first_diff(&args_l[i],&args_f[i]) {panic!("{what}: arg{i} differs from the all-full group at byte {at}")}}
    assert!(sim_l.state_snapshot()==sim_f.state_snapshot(),"{what}: retained state after the lean group != the all-full group");
    assert!(sim_l.state_snapshot()==last_body_only(&ex).state_snapshot(),"{what}: retained state != one fresh full run of the last body at its arena offsets");
    // Reuse: a second lean group on the same Config with new operands (every body, including expert 0, starts after a completed run).
    let ex2=experts(k,&[64,160,200],ctl,12,0xbadc0de2);run_group(&mut sim_l,&ex2,&lean,&format!("{what} resubmit"));
    assert!(sim_l.state_snapshot()==sim_f.state_snapshot(),"{what}: state after the second lean group");
    eprintln!("MEASURED {what}: lean group TXN {} B vs all-full {} B; cumulative ticks after two lean groups {} vs one all-full group {} (one group = {})",lean.len(),full.len(),sim_l.ticks,sim_f.ticks,sim_l.ticks/2);
}
#[test] fn g80_group_lean_tail_equals_all_full_fast_k192() {group_lean(192,Control::Fast)}
#[test] fn g80_group_lean_tail_equals_all_full_slow_k128() {group_lean(128,Control::Slow)}
#[test] fn g80_group_lean_tail_equals_all_full_fast_k2560() {group_lean(2560,Control::Fast)}

// ---------------------------------------------------------------- `experts::grouped` with a G80 design
// The production grouped API (expert 0 FULL body, experts 1.. LEAN bodies, optional persistent ring once around the group)
// built from ONE one-wave G80 design (`wave_m = 256`), against `grouped_all_full` (every body FULL) and the eager designs.
const DONE_MAGIC:u32=0x454e4f44;
const LINE:usize=64;
const FIRST_AT:usize=37;
const OPERAND_TO_SEQ:usize=3;
fn wr32(b:&mut [u8],at:usize,v:u32) {b[at..at+4].copy_from_slice(&v.to_le_bytes())}
fn wr64(b:&mut [u8],at:usize,v:u64) {b[at..at+8].copy_from_slice(&v.to_le_bytes())}
fn done_bytes(slot:usize,seq:u32)->[u8;LINE] {let mut l=[0;LINE];wr32(&mut l,0,seq);wr32(&mut l,4,slot as u32);wr32(&mut l,8,DONE_MAGIC);l}
fn plans_of(ex:&[Expert])->Vec<ExpertPlan> {
    (0..ex.len()).map(|e|ExpertPlan{m:ex[e].m,a_off:arena_off(ex,e,0),b_off:arena_off(ex,e,1),c_off:arena_off(ex,e,2)}).collect()
}
/// `(grouped, grouped_all_full)` of the same plans on the G80 design of `round_up(max m, 256)` rows (one or more waves),
/// passed with `wave_m` = all its rows (the npu-experts construction).
fn g80_groups(ex:&[Expert],ctl:Control,k:usize,ring:Option<(RingLayout,usize,u32)>)->(GroupedDesign,GroupedDesign) {
    let plans=plans_of(ex);let rows=ex.iter().map(|x|x.m).max().unwrap().div_ceil(WAVE_M)*WAVE_M;
    let d=design(rows,k,ctl,12);let wave_m=d.waves()*d.wave_m();assert_eq!(wave_m,rows);
    (grouped(d,wave_m,N,k,&plans,ring),grouped_all_full(design(rows,k,ctl,12),wave_m,N,k,&plans,ring))
}
/// Guard-filled A / B arenas, poison C arena (and an initialised ring arg).
fn blank_g(g:&GroupedDesign,layout:Option<RingLayout>)->Vec<Vec<u8>> {
    let mut v:Vec<Vec<u8>>=g.args.iter().map(|s|vec![GUARD;s.bytes]).collect();v[2].fill(POISON);
    if let Some(l)=layout {assert_eq!(v.len(),4);assert_eq!(v[3].len(),l.bytes());l.initialize(&mut v[3])} else {assert_eq!(v.len(),3)}
    v
}
fn put_g(args:&mut [Vec<u8>],plans:&[ExpertPlan],ex:&[Expert]) {
    for (p,x) in plans.iter().zip(ex) {
        let (a,b)=(p.a_off as usize,p.b_off as usize);
        args[0][a..a+x.packed[0].len()].copy_from_slice(&x.packed[0]);args[1][b..b+x.packed[1].len()].copy_from_slice(&x.packed[1]);
    }
}
fn c_range_g(g:&GroupedDesign,p:&ExpertPlan,e:usize)->Range<usize> {p.c_off as usize..p.c_off as usize+g.c_bytes(e)}
/// Every expert's C is byte-equal to its eager fresh-`Config` bytes and equals the independent reference, the operand arenas
/// (incl. guards) are untouched and no byte outside the C regions was written.
fn check_group_g(g:&GroupedDesign,plans:&[ExpertPlan],ex:&[Expert],args:&[Vec<u8>],layout:Option<RingLayout>,what:&str) {
    let mut want_args=blank_g(g,layout);put_g(&mut want_args,plans,ex);
    if let Some(i)=first_diff(&args[0],&want_args[0]) {panic!("{what}: A arena modified at byte {i}")}
    if let Some(i)=first_diff(&args[1],&want_args[1]) {panic!("{what}: B arena modified at byte {i}")}
    let mut covered=vec![false;args[2].len()];
    for (e,(p,x)) in plans.iter().zip(ex).enumerate() {
        let c=c_range_g(g,p,e);
        if let Some(i)=first_diff(&args[2][c.clone()],&x.eager_c) {panic!("{what}: expert {e} (m={}) C != eager G80 bytes of a fresh Config (byte {i})",p.m)}
        assert_eq!(g.unpack_out(e,&args[2][c.clone()]),want(&x.a,&x.b,x.m,x.k,x.shift),"{what}: expert {e} C != CPU reference");
        covered[c].fill(true);
    }
    assert!(args[2].iter().zip(&covered).all(|(&b,&c)|c||b==POISON),"{what}: C arena written outside the expert regions");
}
/// Packing, reference, unpacking, sizes and image of the group equal the per-expert eager design.
fn check_api_g(g:&GroupedDesign,ex:&[Expert]) {
    assert_eq!(g.experts(),ex.len());
    for (e,x) in ex.iter().enumerate() {
        assert_eq!(g.pack_in(e,&x.a,&x.b),x.packed,"expert {e} (m={}): grouped packing == eager packing",x.m);
        assert_eq!(g.reference(e,&x.a,&x.b),want(&x.a,&x.b,x.m,x.k,x.shift),"expert {e}: grouped reference == independent reference");
        assert_eq!(g.unpack_out(e,&x.eager_c),want(&x.a,&x.b,x.m,x.k,x.shift),"expert {e}: grouped unpack of the eager C");
        assert_eq!((g.a_bytes(e),g.b_bytes(e),g.c_bytes(e)),(x.d.args[0].bytes,x.d.args[1].bytes,x.d.args[2].bytes),"expert {e}: sizes");
        assert!(g.pdi==x.d.pdi,"expert {e}: one shared PDI");
    }
}
/// Lean group vs all-full group on the same `Config` type, two rounds of fresh operands (the second submit starts from the
/// end state of the first): exact C, equal arenas, per-channel shim bytes of `E` bodies, equal retained state.
fn grouped_api_case(ms:&[usize],k:usize,ctl:Control) {
    let ex=experts(k,ms,ctl,12,0xc0ffee11);let plans=plans_of(&ex);let (g,f)=g80_groups(&ex,ctl,k,None);
    assert!(g.patch_sites.is_empty()&&f.patch_sites.is_empty());
    assert!(g.pdi==f.pdi,"lean and all-full group load the same PDI");
    check_api_g(&g,&ex);
    let (mut sg,mut sf)=(Config::from_pdi(&g.pdi).unwrap(),Config::from_pdi(&f.pdi).unwrap());
    let ex2=experts(k,ms,ctl,12,0xbadc0de3);
    for (round,x) in [&ex,&ex2].into_iter().enumerate() {
        let (mut ag,mut af)=(blank_g(&g,None),blank_g(&f,None));put_g(&mut ag,&plans,x);put_g(&mut af,&plans,x);
        let (tg,tf,(g0,f0))=(traffic(&sg),traffic(&sf),(sg.ticks,sf.ticks));
        sg.submit(&g.insts,&mut ag).unwrap_or_else(|e|panic!("lean group round {round}: {e}"));sf.submit(&f.insts,&mut af).unwrap_or_else(|e|panic!("all-full group round {round}: {e}"));
        check_group_g(&g,&plans,x,&ag,None,&format!("lean group round {round}"));check_group_g(&f,&plans,x,&af,None,&format!("all-full group round {round}"));
        for i in 0..3 {if let Some(at)=first_diff(&ag[i],&af[i]) {panic!("round {round}: lean group arg{i} != all-full group arg{i} at byte {at}")}}
        let (kc,w)=(k/64,ex[0].d.waves());let (rg,wg)=assert_traffic(&tg,&traffic(&sg),ms.len(),w*ms.len(),kc,&format!("lean group round {round}"));
        let (rf,wf)=assert_traffic(&tf,&traffic(&sf),ms.len(),w*ms.len(),kc,&format!("all-full group round {round}"));assert_eq!((rg,wg),(rf,wf));
        assert!(sg.state_snapshot()==sf.state_snapshot(),"round {round}: retained state after the lean group != the all-full group");
        eprintln!("MEASURED G80 grouped API {ctl:?} E={} M={ms:?} K={k} round {round}: PDI {} B; TXN lean {} B vs all-full {} B; {} vs {} functional ticks; DRAM read={rg} write={wg}; \
            per column MM2S0 B={} MM2S1 A={}(cols<4) S2MM0/1={} B",ms.len(),g.pdi.len(),g.insts.len(),f.insts.len(),sg.ticks-g0,sf.ticks-f0,ms.len()*kc*B_CHUNK,w*ms.len()*kc*A_CHUNK,w*ms.len()*2*C_TILE);
    }
}
#[test] fn g80_grouped_api_e3_mixed_m_fast_k192() {grouped_api_case(&[64,160,200],192,Control::Fast)}
#[test] fn g80_grouped_api_e3_mixed_m_slow_k128() {grouped_api_case(&[64,160,200],128,Control::Slow)}
#[test] fn g80_grouped_api_e3_mixed_m_fast_k64_kc1() {grouped_api_case(&[64,160,200],64,Control::Fast)}
#[test] fn g80_grouped_api_e3_mixed_m_fast_k2560() {grouped_api_case(&[64,160,200],2560,Control::Fast)}
#[test] fn g80_grouped_api_e4_boundary_m_fast_k128() {grouped_api_case(&[1,256,200,64],128,Control::Fast)}
// Two-wave groups (npu-experts `--design G80 --m 257..=512`: `design_g80(512, ..)`, `wave_m = 512`): lean tails exact and
// state-equal to the all-full group at kc 1 / 2 / 3 / 40, Fast and Slow, mixed M, with and without the ring.
#[test] fn g80_grouped_api_two_waves_e3_fast_k192() {grouped_api_case(&[300,512,257],192,Control::Fast)}
#[test] fn g80_grouped_api_two_waves_e3_slow_k128() {grouped_api_case(&[512,300,400],128,Control::Slow)}
#[test] fn g80_grouped_api_two_waves_e2_fast_k2560() {grouped_api_case(&[512,384],2560,Control::Fast)}
#[test] fn g80_grouped_ring_two_waves_e3_late_producer() {grouped_ring_case(&[512,300,257],64,Control::Fast,RingLayout{nslots:2},1,5,2_000)}
/// The persistent ring once around the whole group; the producer publishes operands, then `seq`, late; C must stay poison
/// until then; the ring's DONE line is exact and the lean ring group equals the all-full ring group (bytes and state).
fn grouped_ring_case(ms:&[usize],k:usize,ctl:Control,layout:RingLayout,slot:usize,seq:u32,delay:usize) {
    let ex=experts(k,ms,ctl,12,0xc0ffee21);let plans=plans_of(&ex);let ring=Some((layout,slot,seq));let (g,f)=g80_groups(&ex,ctl,k,ring);
    assert_eq!(g.args.len(),4);assert_eq!(g.args[3].bytes,layout.bytes());check_api_g(&g,&ex);
    let run=|d:&GroupedDesign|->(Vec<Vec<u8>>,Config) {
        let mut args=blank_g(d,Some(layout));let mut sim=Config::from_pdi(&d.pdi).unwrap();let mut shadow=args[3].clone();
        let (mut stage,mut at,mut calls)=(0u8,FIRST_AT,0usize);
        sim.submit_with(&d.insts,&mut args,&mut |t,a|{
            calls+=1;
            if stage<2 && calls%16==1 {for (e,p) in plans.iter().enumerate() {
                assert!(a[2][c_range_g(d,p,e)].iter().all(|&b|b==POISON),"expert {e} C written before the group was published (tick {t})");
            }}
            if stage==2 || t<at {return}
            if stage==0 {put_g(a,&plans,&ex);stage=1;at=t+OPERAND_TO_SEQ+delay;return}
            for (e,p) in plans.iter().enumerate() {assert!(a[2][c_range_g(d,p,e)].iter().all(|&b|b==POISON),"expert {e}: C computed from operands whose seq was not yet published (tick {t})")}
            let line=layout.slot_line(slot);let p=&plans[0];
            let mut fields=[0u8;LINE-4];wr32(&mut fields,0,9);wr32(&mut fields,4,p.m as u32);wr32(&mut fields,8,0);
            wr64(&mut fields,12,p.a_off);wr64(&mut fields,20,p.b_off);wr64(&mut fields,28,p.c_off);
            for arg in [&mut a[3],&mut shadow] {arg[line+4..line+LINE].copy_from_slice(&fields);arg[line..line+4].copy_from_slice(&seq.to_le_bytes())}
            stage=2;
        }).unwrap();
        assert_eq!(stage,2,"the producer never published");assert!(sim.ticks>delay,"the group finished ({} ticks) before the late publication ({delay})",sim.ticks);
        if let Some(i)=first_diff(&args[3][..layout.done_line(0)],&shadow[..layout.done_line(0)]) {panic!("ring header / slot lines != initial + producer writes at byte {i}")}
        for s in 0..layout.nslots {
            let at=layout.done_line(s);let want_line=if s==slot {done_bytes(slot,seq)} else {[0;LINE]};
            assert_eq!(&args[3][at..at+LINE],&want_line[..],"done line of slot {s}");
        }
        (args,sim)
    };
    let ((ag,sg),(af,sf))=(run(&g),run(&f));
    check_group_g(&g,&plans,&ex,&ag,Some(layout),"lean ring group");check_group_g(&f,&plans,&ex,&af,Some(layout),"all-full ring group");
    for i in 0..4 {if let Some(at)=first_diff(&ag[i],&af[i]) {panic!("lean ring group arg{i} != all-full ring group arg{i} at byte {at}")}}
    assert!(sg.state_snapshot()==sf.state_snapshot(),"retained state after the lean ring group != the all-full ring group");
    eprintln!("MEASURED G80 grouped ring {ctl:?} E={} M={ms:?} K={k} slot={slot} seq={seq} delay={delay}: PDI {} B; TXN lean {} B vs all-full {} B; {} vs {} functional ticks",
        ms.len(),g.pdi.len(),g.insts.len(),f.insts.len(),sg.ticks,sf.ticks);
}
#[test] fn g80_grouped_ring_e3_late_producer() {grouped_ring_case(&[64,160,200],128,Control::Fast,RingLayout{nslots:2},0,1,20_000)}
#[test] fn g80_grouped_ring_e4_late_producer_slot_seq() {grouped_ring_case(&[160,200,64,64],192,Control::Slow,RingLayout{nslots:4},3,7,400)}
/// Group boundaries: M = 1 and one full wave are exact; empty, M = 0, M above the wave and unaligned offsets are rejected.
#[test] fn g80_grouped_boundaries() {
    grouped_api_case(&[1,256],64,Control::Fast);
    let (k,wave_m)=(64usize,WAVE_M);let one=|plans:&[ExpertPlan]|{grouped(design(WAVE_M,k,Control::Fast,12),wave_m,N,k,plans,None);};
    let plan=|m,a,b,c|ExpertPlan{m,a_off:a,b_off:b,c_off:c};
    assert_all_rejected(vec![
        ("empty group".into(),rejects(||one(&[]))),("m=0".into(),rejects(||one(&[plan(0,0,0,0)]))),("m=257".into(),rejects(||one(&[plan(257,0,0,0)]))),
        ("unaligned A".into(),rejects(||one(&[plan(64,2,0,0)]))),("unaligned C".into(),rejects(||one(&[plan(64,0,0,0),plan(64,0,0,6)]))),
        ("multi-wave design".into(),rejects(||{grouped(design(512,k,Control::Fast,12),wave_m,N,k,&[plan(64,0,0,0)],None);})),
    ]);
}

// ---------------------------------------------------------------- adversarial B fill order (parity 1 ready before parity 0)
/// TEST ONLY. The memtile B fill is S2MM4: BD4 (parity 0 region, releases ready lock 4) -> BD5 (parity 1 region, releases ready
/// lock 6), each one 5120 B chunk half, repeated `kc` times. The two guarded readers each acquire their own ready lock and read
/// their own parity region, so the guard order is independent of the fill order. Here the fill order is reversed: BD4 and BD5
/// exchange address and lock words in every column's descriptor rewrite of the full TXN (BD4 keeps `next = 5`, BD5 none, both
/// keep length and iteration), so parity 1 is filled and signalled first, and the host B halves `[even, odd]` of every chunk
/// are exchanged to `[odd, even]` so every byte still lands in its original region. The result must stay CPU-exact.
const B_HALF:usize=5120;
const BD_FILL:[u32;2]=[4,5];
fn mem_bd_addr(col:u32,id:u32)->u32 {pm_npu::regs::tile(col,1,0xa0000+id*0x20)}
/// `d.insts` with the B fill descriptors of all 8 columns exchanged (panics if a column's BD4 / BD5 rewrite is not found).
fn reverse_fill_ops(d:&ArrayDesign)->Vec<TxnOp> {
    let mut ops=parse_txn(&d.insts).unwrap();
    for col in 0..COLS as u32 {
        let at=BD_FILL.map(|id|ops.iter().position(|o|matches!(o,TxnOp::BlockWrite(a,w) if *a==mem_bd_addr(col,id)&&w.len()==8))
            .unwrap_or_else(||panic!("col {col}: no 8-word BlockWrite of memtile BD{id} in the full TXN")));
        let words=at.map(|i|match &ops[i] {TxnOp::BlockWrite(_,w)=>w.clone(),_=>unreachable!()});
        let (w4,w5)=(&words[0],&words[1]);
        assert!(w4[1]>>19&1==1&&(w4[1]>>20)&0x3f==BD_FILL[1]&&w5[1]>>19&1==0,"col {col}: BD4 must chain to BD5 and BD5 end the chain");
        assert!(w4[0]==w5[0]&&w4[6]==w5[6]&&w4[2..6]==w5[2..6],"col {col}: BD4/BD5 differ in more than address and locks");
        assert!(w4[1]&0x7ffff!=w5[1]&0x7ffff&&w4[7]!=w5[7],"col {col}: BD4/BD5 must differ in address and locks");
        let (mut n4,mut n5)=(w4.clone(),w5.clone());
        n4[1]=(w4[1]&!0x7ffff)|(w5[1]&0x7ffff);n5[1]=(w5[1]&!0x7ffff)|(w4[1]&0x7ffff);n4[7]=w5[7];n5[7]=w4[7];
        ops[at[0]]=TxnOp::BlockWrite(mem_bd_addr(col,BD_FILL[0]),n4);ops[at[1]]=TxnOp::BlockWrite(mem_bd_addr(col,BD_FILL[1]),n5);
    }
    ops
}
/// Host B with the `[even, odd]` halves of every column chunk exchanged to `[odd, even]`.
fn swap_b_halves(b:&mut [u8],kc:usize) {
    for col in 0..COLS {for ch in 0..kc {let base=(col*kc+ch)*2*B_HALF;for i in 0..B_HALF {b.swap(base+i,base+B_HALF+i)}}}
}
/// Reversed-order TXN: CPU-exact, A / reversed B unchanged, exact per-channel shim bytes, on a REUSED `Config` (two submits).
/// `control`: the same reversed descriptors with the unexchanged host B must NOT give the CPU result (the patch is live).
fn reversed_fill_case(m:usize,k:usize,ctl:Control) {
    let d=design(m,k,ctl,12);let (kc,waves)=(k/64,m.div_ceil(WAVE_M));let ops=reverse_fill_ops(&d);let what=format!("G80 reversed B fill {ctl:?} {m}x{N}x{k}");
    let mut sim=fresh(&d);
    for (i,seed) in [SEEDS[0],SEEDS[1]].into_iter().enumerate() {
        let (a,b)=random(m,k,seed);let [a_host,mut b_host]=d.pack_in(&a,&b);swap_b_halves(&mut b_host,kc);
        let before=[signature(&a_host),signature(&b_host)];let mut args=vec![a_host,b_host,vec![POISON;d.args[2].bytes]];
        let (t0,ticks0)=(traffic(&sim),sim.ticks);
        sim.execute(&ops,&mut args).unwrap_or_else(|e|panic!("{what} #{i}: {e}"));
        let w=want(&a,&b,m,k,12);let got=d.unpack_out(&args[2]);
        if let Some(j)=(0..w.len()).find(|&j|got[j]!=w[j]) {panic!("{what} #{i}: first mismatch at ({},{}) got {} want {}",j/N,j%N,got[j],w[j])}
        assert_eq!([signature(&args[0]),signature(&args[1])],before,"{what} #{i}: reversed inputs changed");
        let (read,written)=assert_traffic(&t0,&traffic(&sim),1,waves,kc,&what);
        eprintln!("MEASURED {what} #{i}: CPU-exact; {} ticks; DRAM read={read} write={written}",sim.ticks-ticks0);
    }
    let (a,b)=random(m,k,SEEDS[2]);let [a_host,b_host]=d.pack_in(&a,&b);
    let mut args=vec![a_host,b_host,vec![POISON;d.args[2].bytes]];let mut control=fresh(&d);
    let outcome=control.execute(&ops,&mut args);
    assert!(outcome.is_err()||d.unpack_out(&args[2])!=want(&a,&b,m,k,12),"{what}: control (reversed descriptors, unexchanged B) must differ from the CPU result");
}
#[test] fn g80_reversed_b_fill_order_exact_two_parities_and_chunks() {
    for (m,k) in [(256,128),(256,192),(512,192),(768,64),(512,640),(300,320)] {for ctl in BOTH {reversed_fill_case(m,k,ctl)}}
}
#[test] fn g80_reversed_b_fill_order_exact_kc40_two_m_waves() {reversed_fill_case(512,2560,Control::Fast)}

// ---------------------------------------------------------------- negative: the finite requeues / forced resets are load-bearing
// The lean body must requeue every finite memtile task (B fill BD4, guarded readers BD30 / BD42, whole-segment replays BD31 /
// BD43 for MW > 1) after a completed full run, and a ring POLL's self-looping S2MM4 task must be reset by the group's first
// body. Each omission below is a TEST-ONLY TXN derived from valid ops; the run must hang (bounded `tick_limit`) or give a wrong
// C. The unmodified ops are the control in every case.
/// Memtile DMA start-queue register offset: S2MM `ch` at `0xa0604 + 8*ch`, MM2S `ch` after the six S2MM channels.
fn mem_queue_off(mm2s:bool,ch:u32)->u32 {0xa0604+8*(if mm2s {6+ch} else {ch})}
/// Removes only the FIRST write of the task queue register `off` of column `col`'s memtile that queues BD `bd`.
fn drop_one_queue_write(col:u32,off:u32,bd:u32)->impl Fn(&TxnOp)->bool {
    let dropped=Cell::new(false);
    move |op| match *op {
        TxnOp::Write(a,v) if !dropped.get()&&a>>25==col&&(a>>20)&31==1&&a&0xfffff==off&&v&0xffff==bd => {dropped.set(true);true}
        _=>false,
    }
}
#[derive(Debug)] #[allow(dead_code)]
enum Detected {Failed(String),WrongC(usize)}
/// The valid lean ops of `d` minus the ops selected by `drop`; shim programming, core control and every SYNC are retained.
fn forced_ops(d:&ArrayDesign,drop:&dyn Fn(&TxnOp)->bool)->(Vec<TxnOp>,usize) {
    let lean=parse_txn(&d.lean_insts().unwrap()).unwrap();
    let kept:Vec<TxnOp>=lean.iter().filter(|o|!drop(*o)).cloned().collect();let removed=lean.len()-kept.len();
    assert_eq!(kept.iter().filter(|o|matches!(o,TxnOp::Sync{..})).count(),lean.iter().filter(|o|matches!(o,TxnOp::Sync{..})).count());
    assert!(kept.iter().any(|o|matches!(o,TxnOp::DdrPatch{..})));
    (kept,removed)
}
/// Runs `ops` up to three times on a Config that completed one full zero-operand run, with the deadlock limit tightened to a
/// multiple of that run so a stuck SYNC reports quickly. `None` = it behaved like a lean TXN.
fn run_forced(d:&ArrayDesign,m:usize,k:usize,ops:&[TxnOp])->Option<Detected> {
    let mut sim=primed(d);sim.tick_limit=sim.ticks*2+4096;
    for seed in SEEDS {
        let (a,b)=random(m,k,seed);let [a_host,b_host]=d.pack_in(&a,&b);let mut args=vec![a_host,b_host,vec![POISON;d.args[2].bytes]];
        if let Err(e)=sim.execute(ops,&mut args) {return Some(Detected::Failed(e))}
        let (got,w)=(d.unpack_out(&args[2]),want(&a,&b,m,k,12));
        if let Some(i)=(0..got.len()).find(|&i|got[i]!=w[i]) {return Some(Detected::WrongC(i))}
    }
    None
}
fn omitted_queue_detected(m:usize,k:usize,ctl:Control,mm2s:bool,ch:u32,bd:u32,what_dropped:&str) {
    let d=design(m,k,ctl,12);let what=format!("G80 {ctl:?} {m}x{N}x{k}");
    let (control,removed)=forced_ops(&d,&|_|false);assert_eq!(removed,0);
    assert!(run_forced(&d,m,k,&control).is_none(),"{what}: the unmodified lean ops must behave like lean");
    let (ops,removed)=forced_ops(&d,&drop_one_queue_write(0,mem_queue_off(mm2s,ch),bd));
    assert_eq!(removed,1,"{what}: expected exactly one {what_dropped} queue write to omit");
    let got=run_forced(&d,m,k,&ops);
    assert!(got.is_some(),"{what}: omitting the col 0 {what_dropped} queue write still behaves like lean: the requeue is not load-bearing");
    eprintln!("NEGATIVE {what}: omitting col 0 {what_dropped} queue write -> {:?}",got.unwrap());
}
/// One M wave (no whole-segment replay) and two M waves.
const OMIT_SHAPES:[(usize,usize);2]=[(256,128),(512,192)];
#[test] fn g80_omitting_b_fill_bd4_requeue_is_detected() {for (m,k) in OMIT_SHAPES {for ctl in BOTH {omitted_queue_detected(m,k,ctl,false,4,4,"B fill BD4")}}}
#[test] fn g80_omitting_guard_bd30_requeue_is_detected() {for (m,k) in OMIT_SHAPES {for ctl in BOTH {omitted_queue_detected(m,k,ctl,true,1,30,"guarded reader BD30")}}}
#[test] fn g80_omitting_guard_bd42_requeue_is_detected() {for (m,k) in OMIT_SHAPES {for ctl in BOTH {omitted_queue_detected(m,k,ctl,true,5,42,"guarded reader BD42")}}}
#[test] fn g80_omitting_whole_bd31_requeue_is_detected_mw2() {for ctl in BOTH {omitted_queue_detected(512,192,ctl,true,1,31,"whole-segment replay BD31")}}
#[test] fn g80_omitting_whole_bd43_requeue_is_detected_mw2() {for ctl in BOTH {omitted_queue_detected(512,192,ctl,true,5,43,"whole-segment replay BD43")}}
/// Real grouped ring (E3 mixed M, one wave each): the POLL leaves a self-looping task on column 0 memtile S2MM4. The group's
/// first (full) body must reset that channel; with its S2MM4 CTRL (reset) writes omitted the fill never starts and the run
/// must hang within the bounded `tick_limit` or give a wrong C. The unmodified ring group is the control.
fn ring_poll_reset_omission_detected(ms:&[usize],k:usize,ctl:Control,layout:RingLayout,slot:usize,seq:u32) {
    let ex=experts(k,ms,ctl,12,0xc0ffee31);let plans=plans_of(&ex);let (g,_)=g80_groups(&ex,ctl,k,Some((layout,slot,seq)));
    let what=format!("G80 grouped ring {ctl:?} E={} M={ms:?} K={k}",ms.len());
    let run=|ops:&[TxnOp],tick_limit:Option<usize>|->(Result<(),String>,Vec<Vec<u8>>,usize) {
        let mut args=blank_g(&g,Some(layout));let mut sim=Config::from_pdi(&g.pdi).unwrap();
        if let Some(l)=tick_limit {sim.tick_limit=l}
        let (mut stage,mut next)=(0u8,FIRST_AT);
        let res=sim.execute_with(ops,&mut args,&mut |t,a|{
            if stage==2||t<next {return}
            if stage==0 {put_g(a,&plans,&ex);stage=1;next=t+OPERAND_TO_SEQ;return}
            let line=layout.slot_line(slot);let p=&plans[0];
            let mut fields=[0u8;LINE-4];wr32(&mut fields,0,9);wr32(&mut fields,4,p.m as u32);wr32(&mut fields,8,0);
            wr64(&mut fields,12,p.a_off);wr64(&mut fields,20,p.b_off);wr64(&mut fields,28,p.c_off);
            a[3][line+4..line+LINE].copy_from_slice(&fields);a[3][line..line+4].copy_from_slice(&seq.to_le_bytes());stage=2;
        });
        (res,args,sim.ticks)
    };
    let ops=parse_txn(&g.insts).unwrap();
    let (res,args,ticks)=run(&ops,None);res.unwrap_or_else(|e|panic!("{what}: control ring group failed: {e}"));
    check_group_g(&g,&plans,&ex,&args,Some(layout),&format!("{what} control"));
    let drain=ops.iter().position(|o|matches!(o,TxnOp::MaskPoll(_,_,v) if *v==SENTINEL)).expect("ring drain poll");
    let ctrl=pm_npu::regs::tile(0,1,0xa0600+4*8);let mut removed=0usize;
    let kept:Vec<TxnOp>=ops.iter().enumerate().filter(|(i,o)|{let hit=*i>drain&&matches!(o,TxnOp::MaskWrite(a,..) if *a==ctrl);if hit {removed+=1}!hit}).map(|(_,o)|o.clone()).collect();
    assert!(removed>0,"{what}: no col 0 S2MM4 CTRL write in the group body to omit");
    let (res,args,_)=run(&kept,Some(ticks*2+4096));
    let wrong=plans.iter().zip(&ex).enumerate().any(|(e,(p,x))|args[2][c_range_g(&g,p,e)]!=x.eager_c[..]);
    assert!(res.is_err()||wrong,"{what}: omitting the {removed} col 0 S2MM4 reset writes of the group body still gave the exact result: the POLL self-loop reset is not load-bearing");
    eprintln!("NEGATIVE {what}: omitting {removed} col 0 S2MM4 CTRL writes of the group body -> {}",match &res {Err(e)=>format!("failed: {e}"),Ok(())=>"wrong C".into()});
}
#[test] fn g80_grouped_ring_omitting_col0_s2mm4_reset_is_detected_e3() {
    for ctl in BOTH {ring_poll_reset_omission_detected(&[64,160,200],128,ctl,RingLayout{nslots:2},0,1)}
}
