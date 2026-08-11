#!/usr/bin/env python3
"""Summarize the final warmed PP8192 pass in a rocprofv3 runtime trace."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def category(name: str) -> str:
    if "gemm_hfq4g256_mmq" in name and "full_set" in name:
        return "packed_mq4_set"
    if "gemm_hfq4g256_mmq" in name and "full_add" in name:
        return "packed_mq4_add"
    if (
        "ck_tile::kentry" in name
        or "hipfire::ck_attention::predecode" in name
        or "rotate_q_givens_f32_to_f16" in name
        or "convert_f16_to_f32" in name
    ):
        return "ck_attention_and_bridges"
    if name == "gated_delta_net_q8_fast":
        return "gated_delta_net_core"
    if name == "conv1d_silu_split_f32":
        return "conv1d_silu"
    if "fused_silu_mul_mq_rotate" in name:
        return "fused_swiglu_rotate"
    if name.startswith("quantize_q8_1_mmq"):
        return "q8_activation_quantization"
    if name.startswith("__amd_rocclr_fillBuffer"):
        return "buffer_fill"
    if any(
        token in name
        for token in (
            "fused_gate_up_hfq4g256",
            "gemm_gate_up_hfq4g256",
            "fused_qkvza_hfq4g256",
            "fused_qkv_hfq4g256",
            "gemv_hfq4g256",
        )
    ):
        return "packed_mq4_tail_and_lm_head"
    return "other"


def merged_duration(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with args.trace.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["Start_Timestamp"]))

    chunk_markers = [row for row in rows if row["Kernel_Name"] == "embedding_q8_batched"]
    if len(chunk_markers) < args.chunks:
        raise SystemExit(
            f"expected at least {args.chunks} embedding_q8_batched markers, "
            f"found {len(chunk_markers)}"
        )
    start_marker = chunk_markers[-args.chunks]
    start = int(start_marker["Start_Timestamp"])
    agent_id = start_marker["Agent_Id"]

    decode_starts = [
        int(row["Start_Timestamp"])
        for row in rows
        if row["Agent_Id"] == agent_id
        and row["Kernel_Name"] == "embedding_q8"
        and int(row["Start_Timestamp"]) > start
    ]
    if not decode_starts:
        raise SystemExit("could not find a decode embedding marker after final prefill")
    decode_start = min(decode_starts)
    lm_heads = [
        int(row["End_Timestamp"])
        for row in rows
        if row["Agent_Id"] == agent_id
        and row["Kernel_Name"] == "gemv_hfq4g256"
        and start <= int(row["Start_Timestamp"]) < decode_start
    ]
    if not lm_heads:
        raise SystemExit("could not find the final prefill gemv_hfq4g256 marker")
    end = max(lm_heads)

    by_name: dict[str, int] = defaultdict(int)
    by_name_calls: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_category_calls: dict[str, int] = defaultdict(int)
    intervals: list[tuple[int, int]] = []
    dispatches = 0
    for row in rows:
        if row["Agent_Id"] != agent_id:
            continue
        row_start = int(row["Start_Timestamp"])
        row_end = int(row["End_Timestamp"])
        clipped_start = max(start, row_start)
        clipped_end = min(end, row_end)
        if clipped_end <= clipped_start:
            continue
        duration = clipped_end - clipped_start
        name = row["Kernel_Name"]
        group = category(name)
        dispatches += 1
        intervals.append((clipped_start, clipped_end))
        by_name[name] += duration
        by_name_calls[name] += 1
        by_category[group] += duration
        by_category_calls[group] += 1

    wall_ns = end - start
    busy_ns = merged_duration(intervals)
    print(f"window_start_ns={start}")
    print(f"window_end_ns={end}")
    print(f"agent_id={agent_id}")
    print(f"window_ms={wall_ns / 1e6:.3f}")
    print(f"kernel_busy_ms={busy_ns / 1e6:.3f}")
    print(f"no_kernel_gap_ms={(wall_ns - busy_ns) / 1e6:.3f}")
    print(f"dispatches={dispatches}")
    print()
    print("category\tcalls\tms\twall_pct")
    for name, duration in sorted(by_category.items(), key=lambda item: item[1], reverse=True):
        print(
            f"{name}\t{by_category_calls[name]}\t{duration / 1e6:.3f}\t"
            f"{100.0 * duration / wall_ns:.2f}"
        )
    print()
    print("top_kernel\tcalls\tms\twall_pct")
    for name, duration in sorted(by_name.items(), key=lambda item: item[1], reverse=True)[: args.top]:
        print(
            f"{name}\t{by_name_calls[name]}\t{duration / 1e6:.3f}\t"
            f"{100.0 * duration / wall_ns:.2f}"
        )


if __name__ == "__main__":
    main()
