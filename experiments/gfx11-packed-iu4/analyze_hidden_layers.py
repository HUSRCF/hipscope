#!/usr/bin/env python3
"""Compare two HIPFIRE_DUMP_HIDDEN layer streams."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def read_layers(path: Path, dim: int) -> dict[int, tuple[float, ...]]:
    record_bytes = 4 + dim * 4
    raw = path.read_bytes()
    if len(raw) % record_bytes:
        raise ValueError(
            f"{path}: {len(raw)} bytes is not divisible by record size {record_bytes}"
        )
    layers: dict[int, tuple[float, ...]] = {}
    for offset in range(0, len(raw), record_bytes):
        layer = struct.unpack_from("<I", raw, offset)[0]
        if layer in layers:
            raise ValueError(f"{path}: duplicate layer record {layer}")
        values = struct.unpack_from(f"<{dim}f", raw, offset + 4)
        layers[layer] = values
    return layers


def metrics(reference: tuple[float, ...], candidate: tuple[float, ...]) -> tuple[float, float, float]:
    dot = sum(a * b for a, b in zip(reference, candidate))
    ref_sq = sum(a * a for a in reference)
    cand_sq = sum(b * b for b in candidate)
    err_sq = sum((a - b) ** 2 for a, b in zip(reference, candidate))
    cosine = dot / math.sqrt(ref_sq * cand_sq) if ref_sq and cand_sq else float("nan")
    rel_l2 = math.sqrt(err_sq / ref_sq) if ref_sq else float("nan")
    max_abs = max(abs(a - b) for a, b in zip(reference, candidate))
    return cosine, rel_l2, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dim", type=int, required=True)
    args = parser.parse_args()

    reference = read_layers(args.reference, args.dim)
    candidate = read_layers(args.candidate, args.dim)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise SystemExit(f"layer sets differ: missing={missing} extra={extra}")
    common = sorted(reference)

    print("layer\tcosine\trel_l2\tmax_abs\trel_l2_delta")
    previous = 0.0
    for layer in common:
        cosine, rel_l2, max_abs = metrics(reference[layer], candidate[layer])
        print(f"{layer}\t{cosine:.9f}\t{rel_l2:.9f}\t{max_abs:.9f}\t{rel_l2 - previous:+.9f}")
        previous = rel_l2


if __name__ == "__main__":
    main()
