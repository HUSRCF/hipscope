#!/usr/bin/env python3
"""kld_reduce.py — canonical reducer for hipfire quant-quality eval.

Reads per-sequence-KLD files (one per (variant, arch)), aggregates,
computes 95% bootstrap CI on the slice-mean, emits the result table
as markdown + a JSON sidecar for plotting.

Inputs:
  --result-dir <dir>     directory of per-sequence-KLD files (HFKSEQ format)
                         filename convention: <variant>__<arch>.kldseq
                         e.g., qwen3.5-9b-mq3-uniform__gfx1100.kldseq

Output:
  result-table.md        markdown table with mean ± CI, p99, etc.
  result-data.json       same data as JSON for downstream plot scripts

Run:
  python3 kld_reduce.py --result-dir results/2026-05-XX/per-seq/

Spec: docs/plans/issue-113-quant-quality-eval.md §"Result table format".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Local import — kldref_format.py lives next to this script.
sys.path.insert(0, str(Path(__file__).parent))
from kldref_format import read_per_seq_kld  # noqa: E402


@dataclass
class Row:
    variant: str
    arch: str
    n_chunks: int
    mean_kld: float
    mean_kld_ci_lo: float    # 95% bootstrap lower
    mean_kld_ci_hi: float
    p99_kld: float
    notes: str = ""


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) where CI is 2.5th/97.5th percentile of
    bootstrapped resample-means."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def parse_filename(name: str) -> tuple[str, str]:
    """qwen3.5-9b.mq3-uniform__gfx1100.kldseq → ('qwen3.5-9b.mq3-uniform', 'gfx1100')"""
    stem = Path(name).stem  # strip .kldseq
    if "__" not in stem:
        raise ValueError(f"filename {name!r} doesn't match <variant>__<arch>.kldseq convention")
    variant, arch = stem.rsplit("__", 1)
    return variant, arch


def reduce_one(path: Path) -> Row:
    means, p99s = read_per_seq_kld(path)
    variant, arch = parse_filename(path.name)
    means_arr = np.asarray(means, dtype=np.float64)
    p99s_arr = np.asarray(p99s, dtype=np.float64)
    mean, ci_lo, ci_hi = bootstrap_mean_ci(means_arr)
    p99 = float(np.percentile(p99s_arr, 99))
    return Row(
        variant=variant, arch=arch, n_chunks=len(means),
        mean_kld=mean, mean_kld_ci_lo=ci_lo, mean_kld_ci_hi=ci_hi,
        p99_kld=p99,
    )


def render_markdown_table(rows: list[Row]) -> str:
    out = []
    out.append("| Variant | Arch | n_chunks | Mean KLD ± 95% CI | p99 KLD | Notes |")
    out.append("|---|---|---:|---|---:|---|")
    for r in sorted(rows, key=lambda r: (r.variant, r.arch)):
        ci = f"{r.mean_kld:.4f} (CI {r.mean_kld_ci_lo:.4f}–{r.mean_kld_ci_hi:.4f})"
        out.append(f"| {r.variant} | {r.arch} | {r.n_chunks} | {ci} | {r.p99_kld:.3f} | {r.notes} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result-dir", required=True, help="directory of *.kldseq files")
    ap.add_argument("--out-md", default="result-table.md")
    ap.add_argument("--out-json", default="result-data.json")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    files = sorted(result_dir.glob("*.kldseq"))
    if not files:
        print(f"no *.kldseq files in {result_dir}", file=sys.stderr)
        return 1

    rows = [reduce_one(f) for f in files]
    md = render_markdown_table(rows)
    Path(args.out_md).write_text(md + "\n")
    Path(args.out_json).write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(md)
    print()
    print(f"wrote {args.out_md}, {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
