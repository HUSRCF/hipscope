# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/redline_packet_isolation.py"
SPEC = importlib.util.spec_from_file_location("redline_packet_isolation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_output(cycles=4):
    orders = (
        ("T00", "T10", "T11", "T01"),
        ("T10", "T01", "T00", "T11"),
        ("T01", "T11", "T10", "T00"),
        ("T11", "T00", "T01", "T10"),
    )
    values = {"T00": 1000, "T10": 1300, "T01": 1200, "T11": 1700}
    flags = {
        "T00": (0, 0),
        "T10": (0, 1),
        "T01": (1, 0),
        "T11": (1, 1),
    }
    lines = [
        "META\tdevice\tgfx1201\tgpu\t0\treplays_per_sample\t10\t"
        "spin_iterations\t256\twarmups\t1\tcycles\t4\tcommand_dwords\t99"
    ]
    for cycle in range(cycles):
        for position, arm in enumerate(orders[cycle % len(orders)]):
            copy_data, wait = flags[arm]
            lines.append(
                f"SAMPLE\t{cycle}\t{position}\t{arm}\t{copy_data}\t{wait}\t"
                f"{values[arm]}\t{values[arm] + 10}\t1\t1"
            )
    return "\n".join(lines)


def test_parser_and_factorial_effects_are_paired_by_cycle():
    metadata, samples = MODULE.parse_runner_output(synthetic_output())
    analysis = MODULE.analyze(metadata, samples)
    assert analysis["effects"]["flush_without_copy"]["median_ns"] == 300
    assert analysis["effects"]["copy_without_flush"]["median_ns"] == 200
    assert analysis["effects"]["interaction"]["median_ns"] == 200
    assert analysis["effects"]["interaction"]["per_boundary_median_ns"] == 20
    assert MODULE.statistics_for([-3, -2, -1])["cv"] >= 0


def test_validation_rejects_incomplete_timestamp_writes():
    metadata, samples = MODULE.parse_runner_output(synthetic_output())
    samples[0]["timestamps_complete"] = False
    try:
        MODULE.analyze(metadata, samples)
    except ValueError as error:
        assert "timestamp validation" in str(error)
    else:
        raise AssertionError("incomplete timestamp set was accepted")
