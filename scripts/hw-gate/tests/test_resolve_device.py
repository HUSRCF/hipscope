"""A gate lane must benchmark the card it pinned, not an index.

HIP enumerates in KFD-node order, so `device: "3"` on hiptrx pointed at
`4b:00.0` -- the Thunderbolt eGPU -- and moved again when that card
re-enumerated from `7b`. These tests use a synthetic topology tree, so they
assert the mapping rule rather than this host's hardware.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "resolve_device", Path(__file__).resolve().parent.parent / "resolve_device.py"
)
resolve_device = importlib.util.module_from_spec(_SPEC)
sys.modules["resolve_device"] = resolve_device
_SPEC.loader.exec_module(resolve_device)


def _topology(tmp_path: Path, nodes: list[tuple[int, str | None]]) -> str:
    """nodes = [(node_number, "bb:dd.f" or None for the CPU node)]."""
    root = tmp_path / "nodes"
    root.mkdir(parents=True)
    for number, addr in nodes:
        node = root / str(number)
        node.mkdir()
        if addr is None:
            (node / "properties").write_text("cpu_cores_count 64\nsimd_count 0\n")
            continue
        bus, rest = addr.split(":")
        dev = rest.split(".")[0]
        location = (int(bus, 16) << 8) | int(dev, 16)
        (node / "properties").write_text(f"simd_count 128\nlocation_id {location}\ngfx_target_version 120100\n")
    return str(root)


# hiptrx as it actually enumerated on 2026-09-04: KFD order is not bus order,
# and the eGPU sits at index 3.
HIPTRX = [(0, None), (1, "03:00.0"), (2, "c3:00.0"), (3, "e3:00.0"), (4, "4b:00.0"), (5, "13:00.0")]


def test_index_follows_kfd_node_order_not_bus_order(tmp_path):
    root = _topology(tmp_path, HIPTRX)
    # bus order would say c3 is 3rd of five; KFD order puts it at 1
    assert resolve_device.resolve("0000:c3:00.0", root) == 1
    assert resolve_device.resolve("0000:13:00.0", root) == 4


def test_the_historical_index_3_was_the_egpu(tmp_path):
    """Regression pin for the defect itself, so nobody reintroduces a literal index."""
    root = _topology(tmp_path, HIPTRX)
    assert resolve_device.resolve("0000:4b:00.0", root) == 3


def test_cpu_node_does_not_consume_an_index(tmp_path):
    root = _topology(tmp_path, HIPTRX)
    assert resolve_device.resolve("0000:03:00.0", root) == 0


def test_absent_card_is_a_hard_error(tmp_path):
    """Silently benchmarking whichever GPU holds the index is the bug being fixed."""
    root = _topology(tmp_path, HIPTRX)
    with pytest.raises(LookupError) as exc:
        resolve_device.resolve("0000:ff:00.0", root)
    assert "not among" in str(exc.value)
    assert "c3:00.0" in str(exc.value)  # error names what IS present


def test_reenumeration_moves_the_index(tmp_path):
    """7b -> 4b shifted every index after it; the address must still resolve."""
    before = _topology(tmp_path / "before", [(0, None), (1, "03:00.0"), (2, "7b:00.0"), (3, "c3:00.0")])
    after = _topology(tmp_path / "after", [(0, None), (1, "03:00.0"), (2, "c3:00.0"), (3, "4b:00.0")])
    assert resolve_device.resolve("0000:c3:00.0", before) == 2
    assert resolve_device.resolve("0000:c3:00.0", after) == 1


def test_address_forms_and_bad_input(tmp_path):
    root = _topology(tmp_path, HIPTRX)
    assert resolve_device.resolve("c3:00.0", root) == 1  # domain optional
    assert resolve_device.resolve("0000:C3:00.0", root) == 1  # case-insensitive
    with pytest.raises(ValueError):
        resolve_device.resolve("gpu3", root)


def test_cli_emits_a_github_env_line(tmp_path, capsys):
    root = _topology(tmp_path, HIPTRX)
    assert resolve_device.main(["--pci", "0000:c3:00.0", "--topology-root", root]) == 0
    assert capsys.readouterr().out.strip() == "HW_GATE_DEVICE=1"


def test_cli_fails_nonzero_on_absent_card(tmp_path, capsys):
    root = _topology(tmp_path, HIPTRX)
    assert resolve_device.main(["--pci", "0000:ff:00.0", "--topology-root", root]) == 2
    out = capsys.readouterr()
    assert out.out == ""  # nothing appended to GITHUB_ENV
    assert "resolve_device:" in out.err


def test_missing_topology_is_an_error_not_index_zero(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_device.resolve("0000:c3:00.0", str(tmp_path / "absent"))
