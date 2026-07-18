#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize RDNA3 QKVZA cold-prefill fresh-process A/B cells."""

import argparse
import csv
import math
import pathlib
import statistics


def read_tsv(path: pathlib.Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def process_key(row):
    """Identify one fresh process while allowing multiple timed samples."""
    return (
        int(row["length"]),
        row.get("pair") or "",
        row.get("order") or "",
        row["mode"],
    )


def collect_pairs(result_dir: pathlib.Path, rows):
    if rows and all(row.get("pair") for row in rows):
        grouped = {}
        for row in rows:
            key = (int(row["length"]), int(row["pair"]))
            grouped.setdefault(key, []).append(row)
        pairs = []
        for (length, pair), pair_rows in sorted(grouped.items()):
            by_mode = {}
            for row in pair_rows:
                by_mode.setdefault(row["mode"], []).append(float(row["prefill_tok_s"]))
            if not by_mode.get("off") or not by_mode.get("on"):
                raise SystemExit(f"incomplete pair in raw.tsv: prefill={length} pair={pair}")
            off = statistics.median(by_mode["off"])
            on = statistics.median(by_mode["on"])
            order = pair_rows[0].get("order") or "unknown"
            pairs.append(
                {
                    "length": length,
                    "pair": pair,
                    "order": order,
                    "off": off,
                    "on": on,
                    "delta": (on / off - 1.0) * 100.0,
                }
            )
        return pairs

    # Legacy results did not carry pair identity in raw.tsv. Recover it from
    # the per-pair cells, but fail below if those artifacts are unavailable.
    pairs = []
    for path in sorted((result_dir / "cells").glob("pp*/pair*/summary.tsv")):
        length = int(path.parent.parent.name.removeprefix("pp"))
        pair = int(path.parent.name.removeprefix("pair"))
        rows = read_tsv(path)
        by_mode = {}
        order = []
        for row in rows:
            mode = row["mode"]
            if mode not in by_mode:
                order.append(mode)
            by_mode.setdefault(mode, []).append(float(row["prefill_tok_s"]))
        if not by_mode.get("off") or not by_mode.get("on"):
            continue
        off = statistics.median(by_mode["off"])
        on = statistics.median(by_mode["on"])
        pairs.append(
            {
                "length": length,
                "pair": pair,
                "order": "-".join(order),
                "off": off,
                "on": on,
                "delta": (on / off - 1.0) * 100.0,
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=pathlib.Path)
    parser.add_argument("thresholds", type=int, nargs="+")
    args = parser.parse_args()

    result_dir = args.result_dir
    rows = read_tsv(result_dir / "raw.tsv")
    route_rows = read_tsv(result_dir / "routes.tsv")

    # raw.tsv has one row per timed prefill sample, while routes.tsv has one
    # row per fresh process. Modern results carry pair identity and can be
    # joined strictly. Legacy artifacts can only be validated at cell level.
    has_process_identity = rows and route_rows and all(
        row.get("pair") for row in rows + route_rows
    )
    route_by_cell = {}
    route_processes = set()
    for route_index, route_row in enumerate(route_rows, start=1):
        key = process_key(route_row)
        if has_process_identity and key in route_processes:
            raise SystemExit(f"duplicate route diagnostics for process {key}")
        mode = route_row["mode"]
        eligible = int(route_row["eligible_events"])
        hits = int(route_row["route_hit_events"])
        if mode == "off" and (eligible != 0 or hits != 0):
            raise SystemExit(f"off route {route_index} unexpectedly eligible/hit")
        if mode == "on" and (eligible == 0 or hits == 0):
            raise SystemExit(f"active route {route_index} missed eligibility/route")
        if mode not in ("off", "on"):
            raise SystemExit(f"unknown route mode at row {route_index}: {mode}")
        route_processes.add(key)
        cell = (key[0], key[3])
        total_eligible, total_hits = route_by_cell.get(cell, (0, 0))
        route_by_cell[cell] = (
            total_eligible + eligible,
            total_hits + hits,
        )

    if has_process_identity:
        perf_processes = {process_key(row) for row in rows}
        missing_routes = perf_processes - route_processes
        extra_routes = route_processes - perf_processes
        mismatch_label = "process identity"
    else:
        perf_cells = {(int(row["length"]), row["mode"]) for row in rows}
        route_cells = set(route_by_cell)
        missing_routes = perf_cells - route_cells
        extra_routes = route_cells - perf_cells
        mismatch_label = "legacy cell identity"
    if missing_routes or extra_routes:
        raise SystemExit(
            f"raw/routes {mismatch_label} mismatch: "
            f"missing_routes={sorted(missing_routes)} "
            f"extra_routes={sorted(extra_routes)}"
        )

    by_cell = {}
    for row in rows:
        key = (int(row["length"]), row["mode"])
        by_cell.setdefault(key, []).append(float(row["prefill_tok_s"]))

    pairs = collect_pairs(result_dir, rows)
    pairs_by_length = {}
    for pair in pairs:
        pairs_by_length.setdefault(pair["length"], []).append(pair)
    pair_ids_by_length = {}
    for row in rows:
        if row.get("pair"):
            pair_ids_by_length.setdefault(int(row["length"]), set()).add(int(row["pair"]))

    with (result_dir / "paired_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "prefill_tokens",
                "pair",
                "order",
                "off_tok_s",
                "active_tok_s",
                "delta_pct",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    pair["length"],
                    pair["pair"],
                    pair["order"],
                    f'{pair["off"]:.3f}',
                    f'{pair["on"]:.3f}',
                    f'{pair["delta"]:.3f}',
                ]
            )

    length_rows = []
    for length in sorted({key[0] for key in by_cell}):
        off = by_cell.get((length, "off"), [])
        on = by_cell.get((length, "on"), [])
        if not off or not on:
            raise SystemExit(f"missing off/on samples for prefill={length}")
        off_med = statistics.median(off)
        on_med = statistics.median(on)
        delta = (on_med / off_med - 1.0) * 100.0
        off_eligible, off_hits = route_by_cell.get((length, "off"), (0, 0))
        on_eligible, on_hits = route_by_cell.get((length, "on"), (0, 0))
        if off_eligible != 0 or off_hits != 0:
            raise SystemExit(f"off route unexpectedly eligible/hit at prefill={length}")
        if on_eligible == 0 or on_hits == 0:
            raise SystemExit(f"active route did not report eligibility/hit at prefill={length}")
        length_pairs = pairs_by_length.get(length, [])
        expected_pairs = len(pair_ids_by_length.get(length, ())) or len(length_pairs)
        if len(length_pairs) != expected_pairs:
            raise SystemExit(
                f"paired artifacts incomplete at prefill={length}: "
                f"expected {expected_pairs}, found {len(length_pairs)}"
            )
        pair_deltas = [pair["delta"] for pair in length_pairs]
        length_rows.append(
            {
                "length": length,
                "off": off_med,
                "on": on_med,
                "delta": delta,
                "off_p25": percentile(off, 0.25),
                "off_p75": percentile(off, 0.75),
                "on_p25": percentile(on, 0.25),
                "on_p75": percentile(on, 0.75),
                "samples": min(len(off), len(on)),
                "active_eligible": on_eligible,
                "active_hits": on_hits,
                "paired_median": statistics.median(pair_deltas) if pair_deltas else None,
                "positive_pairs": sum(delta > 0.0 for delta in pair_deltas),
                "pair_count": len(pair_deltas),
            }
        )

    with (result_dir / "length_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "prefill_tokens",
                "off_median_tok_s",
                "active_median_tok_s",
                "delta_pct",
                "paired_median_delta_pct",
                "positive_pairs",
                "pair_count",
                "off_p25",
                "off_p75",
                "active_p25",
                "active_p75",
                "samples_per_mode",
                "active_eligible_events",
                "active_route_hit_events",
            ]
        )
        for row in length_rows:
            writer.writerow(
                [
                    row["length"],
                    f'{row["off"]:.3f}',
                    f'{row["on"]:.3f}',
                    f'{row["delta"]:.3f}',
                    "" if row["paired_median"] is None else f'{row["paired_median"]:.3f}',
                    row["positive_pairs"],
                    row["pair_count"],
                    f'{row["off_p25"]:.3f}',
                    f'{row["off_p75"]:.3f}',
                    f'{row["on_p25"]:.3f}',
                    f'{row["on_p75"]:.3f}',
                    row["samples"],
                    row["active_eligible"],
                    row["active_hits"],
                ]
            )

    policy_rows = []
    for threshold in args.thresholds:
        active = [row for row in length_rows if row["length"] >= threshold]
        ratios = [
            (row["on"] / row["off"]) if row["length"] >= threshold else 1.0
            for row in length_rows
        ]
        geometric = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        active_deltas = [row["delta"] for row in active]
        policy_rows.append(
            {
                "threshold": threshold,
                "active_lengths": ",".join(str(row["length"]) for row in active) or "none",
                "active_points": len(active),
                "median_active": statistics.median(active_deltas) if active_deltas else 0.0,
                "worst_active": min(active_deltas) if active_deltas else 0.0,
                "regressions": sum(value < 0.0 for value in active_deltas),
                "all_geomean": (geometric - 1.0) * 100.0,
            }
        )

    with (result_dir / "threshold_projection.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "threshold",
                "active_lengths",
                "active_points",
                "median_active_delta_pct",
                "worst_active_delta_pct",
                "regressed_active_points",
                "equal_weight_all_length_geomean_delta_pct",
            ]
        )
        for row in policy_rows:
            writer.writerow(
                [
                    row["threshold"],
                    row["active_lengths"],
                    row["active_points"],
                    f'{row["median_active"]:.3f}',
                    f'{row["worst_active"]:.3f}',
                    row["regressions"],
                    f'{row["all_geomean"]:.3f}',
                ]
            )

    with (result_dir / "report.md").open("w") as handle:
        handle.write("# RDNA3 QKVZA long-uncached-prefill crossover\n\n")
        handle.write("## Measured length A/B\n\n")
        handle.write(
            "| Prefill tokens | Off median tok/s | Active median tok/s | Delta | "
            "Paired median delta | Positive pairs | Off IQR | Active IQR | Route hits |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in length_rows:
            paired = "n/a" if row["paired_median"] is None else f'{row["paired_median"]:+.2f}%'
            handle.write(
                f'| {row["length"]} | {row["off"]:.1f} | {row["on"]:.1f} | '
                f'{row["delta"]:+.2f}% | {paired} | '
                f'{row["positive_pairs"]}/{row["pair_count"]} | '
                f'{row["off_p25"]:.1f}-{row["off_p75"]:.1f} | '
                f'{row["on_p25"]:.1f}-{row["on_p75"]:.1f} | '
                f'{row["active_hits"]} |\n'
            )
        handle.write("\n## Threshold policy projection\n\n")
        handle.write(
            "Rows below each threshold use the measured off route; rows at or above "
            "it use the measured active route. These are policy projections from the "
            "measured cells above, not redundant benchmark reruns.\n\n"
        )
        handle.write(
            "| Threshold | Activated tested lengths | Median active delta | "
            "Worst active delta | Regressed active points | All-length geomean delta |\n"
        )
        handle.write("|---:|:---|---:|---:|---:|---:|\n")
        for row in policy_rows:
            handle.write(
                f'| {row["threshold"]} | {row["active_lengths"]} | '
                f'{row["median_active"]:+.2f}% | {row["worst_active"]:+.2f}% | '
                f'{row["regressions"]} | {row["all_geomean"]:+.2f}% |\n'
            )

    print((result_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
