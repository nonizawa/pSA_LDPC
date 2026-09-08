#!/usr/bin/env python3
"""Regenerate Main Fig. 4's acquisition/retention summary.

The script reads the copied, validated condition summaries.  It does not
recompute trajectories or statistics.  The upper row reports acquisition from
the default all-zero initialization.  The lower row reports the separate
correct-start retention intervention.  A first-passage value is never imputed
for a method with no reaching trajectories.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = ROOT / "data" / "processed" / "manuscript"
FIGURES = ROOT / "figures" / "regenerated"

SIZES = ["N192_M96", "N288_M144"]
METHODS = ["pSA", "additive", "gain_only", "shuffled"]
LABELS = {
    "pSA": "pSA",
    "additive": "additive",
    "gain_only": "gain-only",
    "shuffled": "shuffled",
}
COLORS = {
    "pSA": "#4C78A8",
    "additive": "#E45756",
    "gain_only": "#F2CF5B",
    "shuffled": "#B279A2",
}
MARKERS = {"pSA": "o", "additive": "s", "gain_only": "^", "shuffled": "D"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select(rows: list[dict[str, str]], size: str, method: str) -> dict[str, str]:
    matches = [row for row in rows if row["size"] == size and row["variant"] == method]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {(size, method)}, found {len(matches)}")
    return matches[0]


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def validate(acquisition: list[dict[str, str]], retention: list[dict[str, str]]) -> None:
    expected = {(size, method) for size in SIZES for method in METHODS}
    for name, rows in [("acquisition", acquisition), ("retention", retention)]:
        observed = {(row["size"], row["variant"]) for row in rows}
        if observed != expected:
            raise ValueError(f"Unexpected {name} condition set: {observed}")

    for size in SIZES:
        for method in METHODS:
            natural = select(acquisition, size, method)
            challenge = select(retention, size, method)
            if int(natural["reached_correct_n"]) != 200:
                raise ValueError(f"Unexpected acquisition count for {(size, method)}")
            if int(challenge["correct_residence_fraction_n"]) != 200:
                raise ValueError(f"Unexpected correct-start count for {(size, method)}")
            reached = value(natural, "reached_correct_mean")
            fpt_n = int(natural["first_correct_cycle_n"])
            fpt = value(natural, "first_correct_cycle_mean")
            if reached == 0.0 and (fpt_n != 0 or math.isfinite(fpt)):
                raise ValueError(f"Nonreacher has a defined FPT for {(size, method)}")
            if reached > 0.0 and (fpt_n == 0 or not math.isfinite(fpt)):
                raise ValueError(f"Reacher lacks an FPT for {(size, method)}")


def draw_points(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    field: str,
    *,
    ci: bool = False,
) -> None:
    centers = np.arange(len(SIZES), dtype=float)
    offsets = dict(zip(METHODS, np.linspace(-0.24, 0.24, len(METHODS))))
    for method in METHODS:
        xs = centers + offsets[method]
        ys = np.asarray([value(select(rows, size, method), field) for size in SIZES])
        kwargs = dict(
            linestyle="none",
            marker=MARKERS[method],
            markersize=6.2,
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.45,
            color=COLORS[method],
            label=LABELS[method],
            clip_on=False,
        )
        if ci:
            lows = np.asarray(
                [value(select(rows, size, method), field.replace("_mean", "_ci_low")) for size in SIZES]
            )
            highs = np.asarray(
                [value(select(rows, size, method), field.replace("_mean", "_ci_high")) for size in SIZES]
            )
            ax.errorbar(xs, ys, yerr=np.vstack([ys - lows, highs - ys]), capsize=2.5, **kwargs)
        else:
            ax.plot(xs, ys, **kwargs)
    ax.set_xticks(centers, [size.split("_")[0] for size in SIZES])
    ax.grid(axis="y", alpha=0.22)


def make_figure() -> None:
    acquisition = read_csv(SOURCE_DATA / "acquisition_natural_initialization_summary.csv")
    retention = read_csv(SOURCE_DATA / "post_acquisition_stability_summary.csv")
    validate(acquisition, retention)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9))

    reach_ax, fpt_ax, residence_ax, escape_ax = axes.flat

    draw_points(reach_ax, acquisition, "reached_correct_mean", ci=True)
    reach_ax.set_title("(a) Default start: correct reach")
    reach_ax.set_ylabel("Reach probability")
    reach_ax.set_ylim(-0.04, 1.06)
    reach_ax.set_xlim(-0.42, 1.42)
    reach_ax.set_yticks(np.linspace(0.0, 1.0, 6))

    centers = np.arange(len(SIZES), dtype=float)
    offsets = dict(zip(METHODS, np.linspace(-0.24, 0.24, len(METHODS))))
    for method in METHODS:
        for size_index, size in enumerate(SIZES):
            row = select(acquisition, size, method)
            x = centers[size_index] + offsets[method]
            fpt_n = int(row["first_correct_cycle_n"])
            if fpt_n == 0:
                fpt_ax.text(
                    x,
                    70,
                    "NR",
                    ha="center",
                    va="center",
                    color=COLORS[method],
                    fontsize=8,
                    fontweight="bold",
                )
                continue
            median = value(row, "first_correct_cycle_median")
            low = value(row, "first_correct_cycle_q25")
            high = value(row, "first_correct_cycle_q75")
            fpt_ax.errorbar(
                x,
                median,
                yerr=[[median - low], [high - median]],
                linestyle="none",
                marker=MARKERS[method],
                markersize=6.2,
                markerfacecolor=COLORS[method],
                markeredgecolor="black",
                markeredgewidth=0.45,
                color=COLORS[method],
                capsize=2.5,
            )
    fpt_ax.set_xticks(centers, [size.split("_")[0] for size in SIZES])
    fpt_ax.set_title("(b) Default start: first passage")
    fpt_ax.set_ylabel("Median cycle (reached only)")
    fpt_ax.set_ylim(0, 1450)
    fpt_ax.set_xlim(-0.42, 1.42)
    fpt_ax.grid(axis="y", alpha=0.22)

    draw_points(residence_ax, retention, "correct_residence_fraction_pooled")
    residence_ax.set_title("(c) Correct-start intervention: retention")
    residence_ax.set_ylabel("Correct-state residence")
    residence_ax.set_ylim(-0.04, 1.06)
    residence_ax.set_xlim(-0.42, 1.42)
    residence_ax.set_yticks(np.linspace(0.0, 1.0, 6))

    draw_points(escape_ax, retention, "correct_escape_probability_pooled")
    escape_ax.set_title("(d) Correct-start intervention: escape")
    escape_ax.set_ylabel("Escape probability / cycle")
    escape_ax.set_ylim(-0.04, 1.06)
    escape_ax.set_xlim(-0.42, 1.42)
    escape_ax.set_yticks(np.linspace(0.0, 1.0, 6))
    for size_index, size in enumerate(SIZES):
        row = select(retention, size, "additive")
        x = centers[size_index] + offsets["additive"]
        y = value(row, "correct_escape_probability_pooled")
        escape_ax.annotate(
            f"{y:.2e}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["additive"],
            fontsize=7,
        )

    handles, legend_labels = reach_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(METHODS),
        frameon=False,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.08, top=0.89, hspace=0.38, wspace=0.28)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "Fig4b_acquisition_retention_summary.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
