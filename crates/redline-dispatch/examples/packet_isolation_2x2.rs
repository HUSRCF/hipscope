// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GFX12 packet-isolation diagnostic for COPY_DATA × CS_PARTIAL_FLUSH.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use redline_dispatch::aql::SingleQueuePm4Ib;
use redline_rocr::{
    Executable, Gfx12CopyDataDstScope, Gfx12DispatchTimestampConfig, Gfx12Pm4CommandBuffer,
    GpuSelector, KernargBuffer, KernargPool, LaunchGeometry, Runtime, load_symbols,
};

const SENTINEL: u64 = 0xd1a6_7057_dead_beef;
const ARMS: [(&str, bool, bool); 4] = [
    ("T00", false, false),
    ("T10", false, true),
    ("T01", true, false),
    ("T11", true, true),
];
const ORDERS: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

struct Args {
    hsaco: PathBuf,
    gpu: usize,
    replays_per_sample: usize,
    spin_iterations: u32,
    warmups: usize,
    cycles: usize,
}

struct ArmGraph {
    name: &'static str,
    copy_data: bool,
    wait_compute_idle: bool,
    graph: SingleQueuePm4Ib,
    timestamps: KernargBuffer,
    command_dwords: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut values = std::env::args().skip(1);
    let mut hsaco = None;
    let mut gpu = 0;
    let mut replays_per_sample = 307;
    let mut spin_iterations = 256;
    let mut warmups = 10;
    let mut cycles = 40;
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--hsaco" => hsaco = Some(PathBuf::from(value)),
            "--gpu" => gpu = value.parse().map_err(|_| "invalid --gpu")?,
            "--replays-per-sample" => {
                replays_per_sample = value.parse().map_err(|_| "invalid --replays-per-sample")?
            }
            "--spin-iterations" => {
                spin_iterations = value.parse().map_err(|_| "invalid --spin-iterations")?
            }
            "--warmups" => warmups = value.parse().map_err(|_| "invalid --warmups")?,
            "--cycles" => cycles = value.parse().map_err(|_| "invalid --cycles")?,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if replays_per_sample == 0 || cycles == 0 {
        return Err("--replays-per-sample and --cycles must be positive".to_owned());
    }
    Ok(Args {
        hsaco: hsaco.ok_or_else(|| "--hsaco is required".to_owned())?,
        gpu,
        replays_per_sample,
        spin_iterations,
        warmups,
        cycles,
    })
}

fn reset_timestamps(buffer: &mut KernargBuffer) {
    for word in buffer.as_mut_bytes().chunks_exact_mut(8) {
        word.copy_from_slice(&SENTINEL.to_le_bytes());
    }
}

fn validate_timestamps(buffer: &mut KernargBuffer, expect_factor: bool) -> (bool, bool, u64, u64) {
    let ticks = buffer
        .as_mut_bytes()
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let complete =
        ticks[0] != SENTINEL && ticks[1] != SENTINEL && (ticks[2] != SENTINEL) == expect_factor;
    let monotonic =
        ticks[1] >= ticks[0] && (!expect_factor || (ticks[2] >= ticks[0] && ticks[1] >= ticks[2]));
    (complete, monotonic, ticks[0], ticks[1])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|reason| format!("packet-isolation: {reason}"))?;
    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::Ordinal(args.gpu))?;
    if !device.name().starts_with("gfx12") {
        return Err(format!("packet-isolation requires gfx12, got {}", device.name()).into());
    }
    let pool = KernargPool::discover(&device)?;
    let code: Arc<[u8]> = std::fs::read(&args.hsaco)?.into();
    let executable = Executable::load(&device, code)?;
    let kernel = executable.kernel("redline_packet_isolation_spin.kd")?;
    let geometry = LaunchGeometry::from_hip_workgroups([64, 1, 1], [64, 1, 1])?;

    let mut output = pool.allocate_executable_bytes(8)?;
    output.as_mut_bytes().fill(0);
    let mut kernarg = pool.allocate_for(kernel.metadata())?;
    let bytes = kernarg.as_mut_bytes();
    if bytes.len() < 12 {
        return Err(format!("unexpected kernarg size {}", bytes.len()).into());
    }
    bytes.fill(0);
    let address = output.address() as usize as u64;
    bytes[..8].copy_from_slice(&address.to_ne_bytes());
    bytes[8..12].copy_from_slice(&args.spin_iterations.to_ne_bytes());

    let timestamp_config = Gfx12DispatchTimestampConfig {
        dst_scope: Gfx12CopyDataDstScope::System,
        per_write_confirm: true,
        tail_release: false,
    };
    let timestamp_frequency_hz = device.gpu_timestamp_frequency_hz()?;
    let mut arms = Vec::with_capacity(ARMS.len());
    for (name, copy_data, wait_compute_idle) in ARMS {
        let mut timestamps = pool.allocate_executable_bytes(24)?;
        reset_timestamps(&mut timestamps);
        let timestamp_base = timestamps.address() as usize as u64;
        let mut commands = Gfx12Pm4CommandBuffer::new_stateful();
        commands.acquire_system_gfx12();
        commands.dispatch(&kernel, geometry, 0, kernarg.address())?;
        commands.diagnostic_packet_isolation_cell(Some(timestamp_base), false, timestamp_config);
        commands.diagnostic_packet_isolation_cell(
            copy_data.then_some(timestamp_base + 16),
            wait_compute_idle,
            timestamp_config,
        );
        commands.diagnostic_packet_isolation_cell(
            Some(timestamp_base + 8),
            false,
            timestamp_config,
        );
        // Outside the measured start/end window: safely complete compute before
        // the enclosing AQL PM4-IB packet publishes its completion signal.
        commands.wait_compute_idle();
        let command_dwords = commands.len_dwords();
        let graph = SingleQueuePm4Ib::create(&device, &pool, &commands)?;
        arms.push(ArmGraph {
            name,
            copy_data,
            wait_compute_idle,
            graph,
            timestamps,
            command_dwords,
        });
    }
    if arms
        .iter()
        .map(|arm| arm.command_dwords)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != 1
    {
        return Err("2x2 arms do not have equal command DWORD counts".into());
    }

    println!(
        "META\tdevice\t{}\tgpu\t{}\treplays_per_sample\t{}\tspin_iterations\t{}\twarmups\t{}\tcycles\t{}\tcommand_dwords\t{}",
        device.name(),
        args.gpu,
        args.replays_per_sample,
        args.spin_iterations,
        args.warmups,
        args.cycles,
        arms[0].command_dwords,
    );
    for _ in 0..args.warmups {
        for arm in &mut arms {
            for _ in 0..args.replays_per_sample {
                reset_timestamps(&mut arm.timestamps);
                // SAFETY: executable, kernargs, output, and timestamp buffers
                // remain live until every graph is dropped below.
                unsafe { arm.graph.replay_and_wait()? };
            }
        }
    }

    let mut seen = BTreeMap::new();
    for cycle in 0..args.cycles {
        for (position, arm_index) in ORDERS[cycle % ORDERS.len()].iter().copied().enumerate() {
            let arm = &mut arms[arm_index];
            let started = Instant::now();
            let mut gpu_ns = 0_u64;
            let mut complete = true;
            let mut monotonic = true;
            for _ in 0..args.replays_per_sample {
                reset_timestamps(&mut arm.timestamps);
                // SAFETY: all addresses encoded in the retained IB remain live.
                unsafe { arm.graph.replay_and_wait()? };
                let validation = validate_timestamps(&mut arm.timestamps, arm.copy_data);
                complete &= validation.0;
                monotonic &= validation.1;
                gpu_ns = gpu_ns.saturating_add(
                    validation
                        .3
                        .saturating_sub(validation.2)
                        .saturating_mul(1_000_000_000)
                        / timestamp_frequency_hz,
                );
            }
            let host_ns = started.elapsed().as_nanos();
            if !complete || !monotonic {
                return Err(format!(
                    "{} timestamp validation failed: complete={complete} monotonic={monotonic}",
                    arm.name
                )
                .into());
            }
            println!(
                "SAMPLE\t{cycle}\t{position}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                arm.name,
                u8::from(arm.copy_data),
                u8::from(arm.wait_compute_idle),
                gpu_ns,
                host_ns,
                u8::from(complete),
                u8::from(monotonic),
            );
            seen.insert((cycle, arm.name), ());
        }
    }
    if seen.len() != args.cycles * ARMS.len() {
        return Err("balanced matrix did not emit every cycle/arm pair".into());
    }
    Ok(())
}
