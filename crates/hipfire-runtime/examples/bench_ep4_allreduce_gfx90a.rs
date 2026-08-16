// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare RCCL with hipfire's N-rank peer-copy all-reduce at the exact
//! DeepSeek V4 EP decode payload: 4096 f32 values (16 KiB) on four gfx90a GCDs.

use hip_bridge::DeviceBuffer;
use hipfire_runtime::multi_gpu::Gpus;
use std::time::Instant;

const COUNT: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 200;

fn sync_all(gpus: &mut Gpus) {
    for dev in &mut gpus.devices {
        dev.bind_thread().expect("bind");
        dev.hip
            .stream_synchronize(dev.active_stream.as_ref().expect("active stream"))
            .expect("stream sync");
    }
}

fn upload(gpus: &mut Gpus, rank: usize, buffer: &DeviceBuffer, values: &[f32]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    gpus.devices[rank].bind_thread().expect("bind upload");
    gpus.devices[rank]
        .hip
        .memcpy_htod(buffer, bytes)
        .expect("upload");
}

fn download(gpus: &mut Gpus, rank: usize, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; COUNT * std::mem::size_of::<f32>()];
    gpus.devices[rank].bind_thread().expect("bind download");
    gpus.devices[rank]
        .hip
        .memcpy_dtoh(&mut bytes, buffer)
        .expect("download");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn verify_nonzero(gpus: &mut Gpus, bytes: usize) {
    let mut rccl = Vec::with_capacity(4);
    let mut async_root = Vec::with_capacity(4);
    let mut inputs = Vec::with_capacity(4);
    for rank in 0..4 {
        gpus.devices[rank].bind_thread().expect("bind alloc");
        rccl.push(gpus.devices[rank].hip.malloc(bytes).expect("malloc RCCL"));
        async_root.push(
            gpus.devices[rank]
                .hip
                .malloc(bytes)
                .expect("malloc async root"),
        );
        let scale = [0.001f32, 1.0, 1000.0, 0.1][rank];
        let values: Vec<f32> = (0..COUNT)
            .map(|i| (((i * 37 + rank * 101) % 1009) as f32 - 504.0) * scale)
            .collect();
        upload(gpus, rank, &rccl[rank], &values);
        upload(gpus, rank, &async_root[rank], &values);
        inputs.push(values);
    }
    let rccl_refs: Vec<&DeviceBuffer> = rccl.iter().collect();
    gpus.all_reduce_sum_f32(&rccl_refs, COUNT)
        .expect("RCCL oracle");
    let async_refs: Vec<&DeviceBuffer> = async_root.iter().collect();
    gpus.all_reduce_sum_f32_peer_root_async(&async_refs, COUNT, 0)
        .expect("async-root oracle");
    sync_all(gpus);

    let rccl0 = download(gpus, 0, &rccl[0]);
    let async0 = download(gpus, 0, &async_root[0]);
    let mut max_abs = 0.0f32;
    let mut squared = 0.0f64;
    for (&a, &b) in rccl0.iter().zip(&async0) {
        let delta = (a - b).abs();
        max_abs = max_abs.max(delta);
        squared += f64::from(delta) * f64::from(delta);
    }
    for rank in 1..4 {
        let rccl_output = download(gpus, rank, &rccl[rank]);
        let rccl_exact = rccl0
            .iter()
            .zip(&rccl_output)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!("oracle rccl-rank-{rank}-bit-exact={rccl_exact}");
        let output = download(gpus, rank, &async_root[rank]);
        assert_eq!(
            async0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "async-root rank {rank} differs from root"
        );
    }
    println!(
        "oracle async-ranks-bit-exact=true rccl-max-abs={:.9e} rccl-rms={:.9e}",
        max_abs,
        (squared / COUNT as f64).sqrt()
    );

    let mut best_seq = (0usize, [0usize; 4]);
    let mut best_pair = (0usize, [0usize; 4]);
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let order = [a, b, c, d];
                    if order
                        .iter()
                        .enumerate()
                        .any(|(i, rank)| order[..i].contains(rank))
                    {
                        continue;
                    }
                    let mut seq_hits = 0usize;
                    let mut pair_hits = 0usize;
                    for i in 0..COUNT {
                        let seq = ((inputs[a][i] + inputs[b][i]) + inputs[c][i]) + inputs[d][i];
                        let pair = (inputs[a][i] + inputs[b][i]) + (inputs[c][i] + inputs[d][i]);
                        seq_hits += usize::from(seq.to_bits() == rccl0[i].to_bits());
                        pair_hits += usize::from(pair.to_bits() == rccl0[i].to_bits());
                    }
                    if seq_hits > best_seq.0 {
                        best_seq = (seq_hits, order);
                    }
                    if pair_hits > best_pair.0 {
                        best_pair = (pair_hits, order);
                    }
                }
            }
        }
    }
    println!(
        "oracle rccl-order best-sequential={:?} exact={}/{} best-pairwise={:?} exact={}/{}",
        best_seq.1, best_seq.0, COUNT, best_pair.1, best_pair.0, COUNT
    );
}

fn measure(gpus: &mut Gpus, buffers: &[DeviceBuffer], mode: &str) -> (f64, f64, f64) {
    let refs: Vec<&DeviceBuffer> = buffers.iter().collect();
    for _ in 0..WARMUP {
        match mode {
            "rccl" => gpus.all_reduce_sum_f32(&refs, COUNT).expect("RCCL warmup"),
            "peer" => gpus
                .all_reduce_sum_f32_peer(&refs, COUNT)
                .expect("peer warmup"),
            "async" => gpus
                .all_reduce_sum_f32_peer_root_async(&refs, COUNT, 0)
                .expect("async root peer warmup"),
            _ => unreachable!(),
        }
        sync_all(gpus);
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let started = Instant::now();
        match mode {
            "rccl" => gpus
                .all_reduce_sum_f32(&refs, COUNT)
                .expect("RCCL allreduce"),
            "peer" => gpus
                .all_reduce_sum_f32_peer(&refs, COUNT)
                .expect("peer allreduce"),
            "async" => gpus
                .all_reduce_sum_f32_peer_root_async(&refs, COUNT, 0)
                .expect("async root peer allreduce"),
            _ => unreachable!(),
        }
        sync_all(gpus);
        samples.push(started.elapsed().as_secs_f64() * 1.0e6);
    }
    samples.sort_by(f64::total_cmp);
    (
        samples[ITERS / 2],
        samples[ITERS / 10],
        samples[ITERS * 9 / 10],
    )
}

fn main() {
    let mut gpus = Gpus::init_uniform(4, 4).expect("init four ranks");
    assert!(
        gpus.devices.iter().all(|gpu| gpu.arch == "gfx90a"),
        "gfx90a only"
    );
    assert!(gpus.enable_peer_all().expect("enable peer access"));
    for dev in &mut gpus.devices {
        dev.bind_thread().expect("bind");
        dev.active_stream = Some(dev.hip.stream_create().expect("stream"));
    }

    let bytes = COUNT * std::mem::size_of::<f32>();
    let mut buffers = Vec::with_capacity(4);
    for dev in &mut gpus.devices {
        dev.bind_thread().expect("bind");
        let buffer = dev.hip.malloc(bytes).expect("malloc");
        dev.hip.memset(&buffer, 0, bytes).expect("zero");
        buffers.push(buffer);
    }

    verify_nonzero(&mut gpus, bytes);

    let rccl = measure(&mut gpus, &buffers, "rccl");
    let peer = measure(&mut gpus, &buffers, "peer");
    let async_root = measure(&mut gpus, &buffers, "async");
    println!("payload={} bytes ranks=4", bytes);
    println!(
        "rccl median={:.3} us p10={:.3} p90={:.3}",
        rccl.0, rccl.1, rccl.2
    );
    println!(
        "peer median={:.3} us p10={:.3} p90={:.3} speedup={:.3}x",
        peer.0,
        peer.1,
        peer.2,
        rccl.0 / peer.0
    );
    println!(
        "async-root median={:.3} us p10={:.3} p90={:.3} speedup={:.3}x",
        async_root.0,
        async_root.1,
        async_root.2,
        rccl.0 / async_root.0
    );
}
