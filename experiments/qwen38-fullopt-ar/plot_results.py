#!/usr/bin/env python3
"""Render the Qwen3.8-27B pure-AR prefill and decode scaling figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    HERE / "results" / "w7900_qwen38_fullopt_ar_gpu1_20260818"
)

PREFILL_COLOR = "#087E8B"
PREFILL_POINT_COLOR = "#78BEC7"
DECODE_COLOR = "#D1495B"
DECODE_POINT_COLOR = "#E9A0AA"
JITTER = np.array([-0.12, -0.06, 0.0, 0.06, 0.12])


def configure_style() -> str:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_name = "Arial"
    except ValueError:
        font_name = "Liberation Sans"

    sns.set_theme(context="paper", style="white", font=font_name)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    return font_name


def finish_axes(ax: plt.Axes) -> None:
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#333333", direction="out")
    ax.grid(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def plot_prefill(data: pd.DataFrame, output_dir: Path) -> None:
    data = data[data["workload"] == "prefill"].copy()
    token_counts = np.sort(data["prefill_tokens"].unique())
    summary = data.groupby("prefill_tokens", sort=True)["prefill_tok_s"].median()

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for x in token_counts:
        samples = data.loc[data["prefill_tokens"] == x, "prefill_tok_s"].to_numpy()
        x_jittered = x * np.power(2.0, JITTER[: len(samples)])
        ax.scatter(
            x_jittered,
            samples,
            s=18,
            color=PREFILL_POINT_COLOR,
            alpha=0.72,
            linewidth=0,
            zorder=2,
        )

    medians = summary.reindex(token_counts).to_numpy()
    ax.plot(
        token_counts,
        medians,
        color=PREFILL_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    for x, y in zip(token_counts, medians):
        ax.annotate(
            f"{y:,.0f}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=PREFILL_COLOR,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(token_counts)
    ax.set_xticklabels([f"{x:,}" for x in token_counts])
    ax.set_xlim(48, 10_800)
    ax.set_ylim(300, 1_350)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Prefill throughput (tokens/s)")
    finish_axes(ax)
    save_figure(fig, output_dir, "qwen38_prefill_scaling")


def plot_decode(data: pd.DataFrame, output_dir: Path) -> None:
    contexts = np.sort(data["context_tokens"].unique())
    positions = np.arange(len(contexts), dtype=float)
    summary = data.groupby("context_tokens", sort=True)["gen_tok_s"].median()

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for position, context in zip(positions, contexts):
        samples = data.loc[data["context_tokens"] == context, "gen_tok_s"].to_numpy()
        ax.scatter(
            position + JITTER[: len(samples)],
            samples,
            s=18,
            color=DECODE_POINT_COLOR,
            alpha=0.72,
            linewidth=0,
            zorder=2,
        )

    medians = summary.reindex(contexts).to_numpy()
    ax.plot(
        positions,
        medians,
        color=DECODE_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    for x, y in zip(positions, medians):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=DECODE_COLOR,
        )

    labels = ["64", "64K", "128K", "192K"]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, len(contexts) - 0.65)
    ax.set_ylim(10, 38)
    ax.set_xlabel("Starting context (tokens)")
    ax.set_ylabel("Decode throughput (tokens/s)")
    finish_axes(ax)
    save_figure(fig, output_dir, "qwen38_decode_scaling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "figures").resolve()
    font_name = configure_style()

    prefill = pd.read_csv(results_dir / "prefill.tsv", sep="\t")
    decode = pd.read_csv(results_dir / "decode.tsv", sep="\t")
    plot_prefill(prefill, output_dir)
    plot_decode(decode, output_dir)
    print(f"font={font_name}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
