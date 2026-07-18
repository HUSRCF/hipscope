import csv
import pathlib
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]
ANALYZER = REPO / "scripts" / "analyze_qkvza_long_uncached.py"


def write_tsv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_multiple_samples_share_one_route_row_per_process(tmp_path):
    raw_rows = []
    route_rows = []
    for pair, order in ((1, "off-on"), (2, "on-off")):
        for mode, base in (("off", 100.0), ("on", 105.0)):
            for run in range(1, 4):
                raw_rows.append(
                    {
                        "length": 4096,
                        "pair": pair,
                        "order": order,
                        "mode": mode,
                        "run": run,
                        "prefill_ms": 10.0,
                        "prefill_tok_s": base + pair + run / 10,
                    }
                )
            route_rows.append(
                {
                    "length": 4096,
                    "pair": pair,
                    "order": order,
                    "mode": mode,
                    "eligible_events": int(mode == "on"),
                    "route_hit_events": int(mode == "on"),
                }
            )

    write_tsv(
        tmp_path / "raw.tsv",
        ["length", "pair", "order", "mode", "run", "prefill_ms", "prefill_tok_s"],
        raw_rows,
    )
    write_tsv(
        tmp_path / "routes.tsv",
        ["length", "pair", "order", "mode", "eligible_events", "route_hit_events"],
        route_rows,
    )

    subprocess.run(
        [sys.executable, str(ANALYZER), str(tmp_path), "4096"],
        check=True,
        cwd=REPO,
    )

    with (tmp_path / "length_summary.tsv").open(newline="") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    assert len(summary) == 1
    assert summary[0]["samples_per_mode"] == "6"
    assert summary[0]["pair_count"] == "2"
    assert summary[0]["active_route_hit_events"] == "2"


def test_legacy_rows_allow_multiple_process_diagnostics_per_cell(tmp_path):
    raw_rows = []
    route_rows = []
    for mode, base in (("off", 100.0), ("on", 105.0)):
        for run in range(1, 4):
            raw_rows.append(
                {
                    "length": 4096,
                    "mode": mode,
                    "run": run,
                    "prefill_ms": 10.0,
                    "prefill_tok_s": base + run / 10,
                }
            )
        for _ in range(2):
            route_rows.append(
                {
                    "length": 4096,
                    "mode": mode,
                    "eligible_events": int(mode == "on"),
                    "route_hit_events": int(mode == "on"),
                }
            )

    write_tsv(
        tmp_path / "raw.tsv",
        ["length", "mode", "run", "prefill_ms", "prefill_tok_s"],
        raw_rows,
    )
    write_tsv(
        tmp_path / "routes.tsv",
        ["length", "mode", "eligible_events", "route_hit_events"],
        route_rows,
    )
    pair_dir = tmp_path / "cells" / "pp4096" / "pair01"
    pair_dir.mkdir(parents=True)
    write_tsv(
        pair_dir / "summary.tsv",
        ["process", "mode", "run", "prefill_ms", "prefill_tok_s"],
        [
            {"process": 1, "mode": row["mode"], "run": row["run"], "prefill_ms": row["prefill_ms"], "prefill_tok_s": row["prefill_tok_s"]}
            for row in raw_rows
        ],
    )

    subprocess.run(
        [sys.executable, str(ANALYZER), str(tmp_path), "4096"],
        check=True,
        cwd=REPO,
    )

    with (tmp_path / "length_summary.tsv").open(newline="") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    assert summary[0]["pair_count"] == "1"
    assert summary[0]["active_route_hit_events"] == "2"
