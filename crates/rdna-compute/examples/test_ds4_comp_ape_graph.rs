// SPDX-License-Identifier: Apache-2.0

//! Graph-replay correctness test for DeepSeek V4 compressor APE row selection.

use rdna_compute::{DType, Gpu};

fn as_i32_bytes(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn as_f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn run_ratio(gpu: &mut Gpu, ratio: usize) {
    const DIM: usize = 64;
    let ape_host: Vec<f32> = (0..ratio)
        .flat_map(|row| (0..DIM).map(move |d| row as f32 * 10.0 + d as f32 * 0.001))
        .collect();
    let zero = vec![0.0f32; DIM];
    let ape = gpu.upload_f32(&ape_host, &[ratio, DIM]).unwrap();
    let score = gpu.zeros(&[DIM], DType::F32).unwrap();
    let mut pos_host = Box::new([ratio as i32 - 1]);
    let pos_buf = gpu
        .upload_raw(as_i32_bytes(pos_host.as_ref()), &[1])
        .unwrap();

    // Compile and validate the direct-dispatch path before capture.
    gpu.compressor_add_ape_pos_buf_f32(&score, &ape, &pos_buf, ratio as i32, DIM as i32)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    assert_eq!(
        gpu.download_f32(&score).unwrap(),
        ape_host[(ratio - 1) * DIM..ratio * DIM]
    );
    gpu.hip
        .memcpy_htod(&score.buf, as_f32_bytes(&zero))
        .unwrap();

    gpu.graphs
        .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
        .unwrap();
    gpu.memcpy_htod_auto(&pos_buf.buf, as_i32_bytes(pos_host.as_ref()))
        .unwrap();
    gpu.compressor_add_ape_pos_buf_f32(&score, &ape, &pos_buf, ratio as i32, DIM as i32)
        .unwrap();
    gpu.graphs
        .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
        .unwrap();

    // Cross the wrap boundary. A captured host-selected row would keep
    // returning ratio-1 here; the graph-safe kernel must select 0 then 1.
    for pos in [ratio as i32 - 1, ratio as i32, ratio as i32 + 1] {
        pos_host[0] = pos;
        gpu.hip
            .memcpy_htod(&score.buf, as_f32_bytes(&zero))
            .unwrap();
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .unwrap();
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();

        let got = gpu.download_f32(&score).unwrap();
        let row = pos as usize % ratio;
        let expected = &ape_host[row * DIM..(row + 1) * DIM];
        assert_eq!(
            got, expected,
            "ratio={ratio} pos={pos} selected wrong APE row"
        );
        eprintln!("ratio={ratio} pos={pos} row={row}: bit-exact");
    }

    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "=== DeepSeek V4 compressor APE hipGraph replay: {} ===",
        gpu.arch
    );
    gpu.active_stream = Some(gpu.hip.stream_create().expect("stream create"));

    run_ratio(&mut gpu, 4);
    run_ratio(&mut gpu, 128);

    let stream = gpu.active_stream.take().unwrap();
    gpu.hip.stream_destroy(stream).unwrap();
    eprintln!("ALL PASS");
}
