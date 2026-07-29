#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Run and analyze the GFX12 COPY_DATA × CS_PARTIAL_FLUSH microbenchmark."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
ARMS = ("T00", "T10", "T01", "T11")


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def statistics_for(values):
    mean = statistics.fmean(values)
    return {
        "min_ns": min(values),
        "median_ns": statistics.median(values),
        "p90_ns": percentile(values, 0.90),
        "max_ns": max(values),
        "cv": statistics.pstdev(values) / abs(mean) if len(values) > 1 and mean else 0.0,
    }


def parse_runner_output(text):
    metadata = None
    samples = []
    for line in text.splitlines():
        fields = line.split("\t")
        if fields[0] == "META":
            if len(fields) < 3 or len(fields[1:]) % 2:
                raise ValueError("malformed META line")
            metadata = dict(zip(fields[1::2], fields[2::2]))
        elif fields[0] == "SAMPLE":
            if len(fields) != 10:
                raise ValueError("malformed SAMPLE line")
            samples.append(
                {
                    "cycle": int(fields[1]),
                    "position": int(fields[2]),
                    "arm": fields[3],
                    "copy_data": bool(int(fields[4])),
                    "wait_compute_idle": bool(int(fields[5])),
                    "gpu_ns": int(fields[6]),
                    "host_ns": int(fields[7]),
                    "timestamps_complete": bool(int(fields[8])),
                    "timestamps_monotonic": bool(int(fields[9])),
                }
            )
    if metadata is None:
        raise ValueError("runner emitted no META line")
    return metadata, samples


def validate_samples(samples):
    if not samples:
        raise ValueError("runner emitted no samples")
    cycles = sorted({sample["cycle"] for sample in samples})
    if cycles != list(range(len(cycles))):
        raise ValueError("cycle indices are not contiguous")
    positions = {arm: [] for arm in ARMS}
    for cycle in cycles:
        rows = [sample for sample in samples if sample["cycle"] == cycle]
        if len(rows) != len(ARMS) or {row["arm"] for row in rows} != set(ARMS):
            raise ValueError(f"cycle {cycle} does not contain exactly one of every arm")
        if sorted(row["position"] for row in rows) != list(range(len(ARMS))):
            raise ValueError(f"cycle {cycle} positions are not contiguous")
        for row in rows:
            if not row["timestamps_complete"] or not row["timestamps_monotonic"]:
                raise ValueError(f"{row['arm']} failed timestamp validation")
            positions[row["arm"]].append(row["position"])
    for arm, arm_positions in positions.items():
        counts = [arm_positions.count(position) for position in range(len(ARMS))]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"{arm} is not position-balanced")


def paired_effects(samples):
    validate_samples(samples)
    cycles = sorted({sample["cycle"] for sample in samples})
    effects = {
        "flush_without_copy": [],
        "flush_with_copy": [],
        "copy_without_flush": [],
        "copy_with_flush": [],
        "interaction": [],
    }
    for cycle in cycles:
        values = {
            row["arm"]: row["gpu_ns"]
            for row in samples
            if row["cycle"] == cycle
        }
        effects["flush_without_copy"].append(values["T10"] - values["T00"])
        effects["flush_with_copy"].append(values["T11"] - values["T01"])
        effects["copy_without_flush"].append(values["T01"] - values["T00"])
        effects["copy_with_flush"].append(values["T11"] - values["T10"])
        effects["interaction"].append(
            values["T11"] - values["T10"] - values["T01"] + values["T00"]
        )
    return effects


def analyze(metadata, samples):
    validate_samples(samples)
    replays_per_sample = int(metadata["replays_per_sample"])
    arms = {}
    for arm in ARMS:
        rows = [sample for sample in samples if sample["arm"] == arm]
        arms[arm] = {
            "copy_data": rows[0]["copy_data"],
            "wait_compute_idle": rows[0]["wait_compute_idle"],
            "gpu_total": statistics_for([row["gpu_ns"] for row in rows]),
            "host_total": statistics_for([row["host_ns"] for row in rows]),
        }
    effects = {}
    for name, values in paired_effects(samples).items():
        effects[name] = {
            **statistics_for(values),
            "per_boundary_median_ns": statistics.median(values) / replays_per_sample,
            "paired_cycle_values_ns": values,
        }
    return {"arms": arms, "effects": effects}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    return subprocess.run(
        command,
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def resolve_hipcc():
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidate = Path(rocm_path) / "bin" / "hipcc"
        if candidate.is_file():
            return candidate
    found = shutil.which("hipcc")
    if found:
        return Path(found)
    raise RuntimeError("hipcc not found; set ROCM_PATH or add hipcc to PATH")


def main():
    parser = argparse.ArgumentParser(
        description="GFX12 packet-isolated COPY_DATA x CS_PARTIAL_FLUSH diagnostic"
    )
    parser.add_argument("--arch", default="gfx1201")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--replays-per-sample", type=int, default=307)
    parser.add_argument("--spin-iterations", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument(
        "--out", default=str(REPO / ".redline-work/packet-isolation-2x2.json")
    )
    parser.add_argument(
        "--log", default=str(REPO / ".redline-work/packet-isolation-2x2.log")
    )
    parser.add_argument(
        "--work-dir", default=str(REPO / ".redline-work/packet-isolation-build")
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if (
        args.replays_per_sample <= 0
        or args.spin_iterations <= 0
        or args.warmups < 0
        or args.cycles < 4
    ):
        parser.error("replays-per-sample/spin-iterations must be positive, cycles >= 4")

    output = Path(args.out).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    source = REPO / "kernels/src/redline_packet_isolation_spin.hip"
    hsaco = work_dir / f"redline_packet_isolation_spin_{args.arch}.hsaco"
    binary = REPO / "target/release/examples/packet_isolation_2x2"
    hipcc = resolve_hipcc()

    if not args.skip_build:
        subprocess.run(
            [
                str(hipcc),
                "--genco",
                "-O3",
                f"--offload-arch={args.arch}",
                str(source),
                "-o",
                str(hsaco),
            ],
            cwd=REPO,
            check=True,
        )
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "-p",
                "redline-dispatch",
                "--example",
                "packet_isolation_2x2",
            ],
            cwd=REPO,
            check=True,
        )
    if not hsaco.is_file() or not binary.is_file():
        parser.error("--skip-build requires existing HSACO and runner binary")

    command = [
        str(binary),
        "--hsaco",
        str(hsaco),
        "--gpu",
        str(args.gpu),
        "--replays-per-sample",
        str(args.replays_per_sample),
        "--spin-iterations",
        str(args.spin_iterations),
        "--warmups",
        str(args.warmups),
        "--cycles",
        str(args.cycles),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_path.write_text(completed.stderr)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        return completed.returncode
    metadata, samples = parse_runner_output(completed.stdout)
    analysis = analyze(metadata, samples)
    report = {
        "schema_version": SCHEMA_VERSION,
        "type": "hipfire_redline_packet_isolation_2x2",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "design": {
            "factors": ["copy_data_gpu_clock", "cs_partial_flush"],
            "arms": {
                "T00": {"copy_data_gpu_clock": False, "cs_partial_flush": False},
                "T10": {"copy_data_gpu_clock": False, "cs_partial_flush": True},
                "T01": {"copy_data_gpu_clock": True, "cs_partial_flush": False},
                "T11": {"copy_data_gpu_clock": True, "cs_partial_flush": True},
            },
            "equal_dword_nop_padding": True,
            "gpu_window_brackets_factor_cell_only": True,
            "common_tail_compute_idle_outside_window": True,
            "cycle_position_balanced": True,
            "independent_dispatch_outputs": True,
        },
        "request": {
            "arch": args.arch,
            "gpu": args.gpu,
            "replays_per_sample": args.replays_per_sample,
            "spin_iterations": args.spin_iterations,
            "warmups": args.warmups,
            "cycles": args.cycles,
        },
        "runner_metadata": metadata,
        "build": {
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "cargo": command_output(["cargo", "--version"]),
            "rustc": command_output(["rustc", "--version", "--verbose"]),
            "hipcc": command_output([str(hipcc), "--version"]),
            "kernel_sha256": sha256_file(hsaco),
            "runner_sha256": sha256_file(binary),
        },
        "timestamp_validation": {
            "all_complete": all(row["timestamps_complete"] for row in samples),
            "all_monotonic": all(row["timestamps_monotonic"] for row in samples),
        },
        "samples": samples,
        **analysis,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("arm  copy  flush  gpu_median_us  p90_us  cv")
    for arm in ARMS:
        row = report["arms"][arm]
        gpu = row["gpu_total"]
        print(
            f"{arm:3s}  {int(row['copy_data']):4d}  "
            f"{int(row['wait_compute_idle']):5d}  "
            f"{gpu['median_ns'] / 1_000:13.3f}  "
            f"{gpu['p90_ns'] / 1_000:7.3f}  {gpu['cv']:.4f}"
        )
    print("effect                   median_us  ns/boundary")
    for name in (
        "flush_without_copy",
        "flush_with_copy",
        "copy_without_flush",
        "copy_with_flush",
        "interaction",
    ):
        row = report["effects"][name]
        print(
            f"{name:24s} {row['median_ns'] / 1_000:9.3f}  "
            f"{row['per_boundary_median_ns']:11.3f}"
        )
    print(f"report={output}")
    print(f"runner_log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
