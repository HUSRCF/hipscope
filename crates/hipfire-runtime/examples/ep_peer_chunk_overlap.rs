use hipfire_runtime::multi_gpu::Gpus;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 4096;
    const CHUNK: usize = 1024;

    let mut gpus = Gpus::init_tp(2, 1)?;
    if !gpus.enable_peer_all()? {
        return Err("peer access unavailable".into());
    }

    let mut tensors = Vec::new();
    for rank in 0..2 {
        let values = vec![(rank + 1) as f32; N];
        let tensor = gpus.devices[rank].upload_f32(&values, &[N])?;
        gpus.devices[rank].active_stream = Some(gpus.devices[rank].hip.stream_create()?);
        tensors.push(tensor);
    }

    let refs = [&tensors[0].buf, &tensors[1].buf];
    let started = std::time::Instant::now();
    let mut done = Vec::new();
    for row_base in (0..N).step_by(CHUNK) {
        gpus.devices[0].bind_thread()?;
        let ready0 = gpus.devices[0].hip.event_create()?;
        gpus.devices[1].bind_thread()?;
        let ready1 = gpus.devices[1].hip.event_create()?;
        gpus.devices[0].bind_thread()?;
        gpus.devices[0]
            .hip
            .event_record(&ready0, gpus.devices[0].active_stream.as_ref())?;
        gpus.devices[1].bind_thread()?;
        gpus.devices[1]
            .hip
            .event_record(&ready1, gpus.devices[1].active_stream.as_ref())?;
        done.push(gpus.all_reduce_sum_f32_peer_chunk_async(
            &refs,
            row_base,
            CHUNK.min(N - row_base),
            [ready0, ready1],
        )?);
    }
    gpus.finish_peer_chunks(done)?;
    for rank in 0..2 {
        let stream = gpus.devices[rank].active_stream.as_ref().unwrap();
        gpus.devices[rank].hip.stream_synchronize(stream)?;
    }
    let elapsed_us = started.elapsed().as_secs_f64() * 1e6;

    for rank in 0..2 {
        let values = gpus.devices[rank].download_f32(&tensors[rank])?;
        let max_err = values
            .iter()
            .map(|value| (value - 3.0).abs())
            .fold(0.0f32, f32::max);
        println!("rank={rank} max_err={max_err:.3e}");
        if max_err != 0.0 {
            return Err(format!("rank {rank} mismatch").into());
        }
    }
    println!("PASS chunks={} total_us={elapsed_us:.3}", N.div_ceil(CHUNK));
    Ok(())
}
