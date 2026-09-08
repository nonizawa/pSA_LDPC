#!/usr/bin/env python3
"""Regenerate Supplemental Figs. S5, S8, and S9 from validated summaries.

This script changes only the visual encoding.  It reads the copied
machine-readable summaries and neither recomputes trajectories nor derives new
statistics.  Exact zeros are plotted at zero, undefined quantities are marked
N/A, and symmetric-log panels retain a true linear neighborhood of zero rather
than substituting an arbitrary positive plotting floor.
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

SIZE_ORDER = ["N96_M48", "N192_M96", "N288_M144"]
RETENTION_SIZES = ["N192_M96", "N288_M144"]
METHODS = ["pSA", "additive", "normalized", "gain_only", "shuffled"]
RETENTION_METHODS = ["pSA", "additive", "gain_only", "shuffled"]
INITIALIZATIONS = ["all_zero", "random", "channel_hard"]

LABELS = {
    "pSA": "pSA",
    "additive": "additive",
    "normalized": "normalized",
    "gain_only": "gain-only",
    "shuffled": "shuffled",
}
COLORS = {
    "pSA": "#4C78A8",
    "additive": "#E45756",
    "normalized": "#54A24B",
    "gain_only": "#F2CF5B",
    "shuffled": "#B279A2",
}
MARKERS = {
    "pSA": "o",
    "additive": "s",
    "normalized": "v",
    "gain_only": "^",
    "shuffled": "D",
}
INIT_LABELS = {
    "all_zero": "All-zero",
    "random": "Random",
    "channel_hard": "Channel hard decision",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE_DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def get_row(
    rows: list[dict[str, str]],
    *,
    size: str,
    method: str,
    initialization: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["size"] == size
        and row["variant"] == method
        and (initialization is None or row.get("initialization") == initialization)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for size={size}, method={method}, "
            f"initialization={initialization}; found {len(matches)}"
        )
    return matches[0]


def point_style(method: str, label: bool = True) -> dict[str, object]:
    return {
        "linestyle": "none",
        "marker": MARKERS[method],
        "markersize": 6.0,
        "markerfacecolor": COLORS[method],
        "markeredgecolor": "black",
        "markeredgewidth": 0.45,
        "color": COLORS[method],
        "label": LABELS[method] if label else None,
        "capsize": 2.4,
        "elinewidth": 0.85,
        "clip_on": False,
    }


def asymmetric_errors(mean: float, low: float, high: float) -> np.ndarray:
    # Last-bit differences at a probability boundary are plotting roundoff,
    # not a change to the recorded interval endpoints.
    return np.asarray([[max(0.0, mean - low)], [max(0.0, high - mean)]])


def annotate_zero(ax: plt.Axes, x: float, method: str) -> None:
    ax.annotate(
        "0",
        (x, 0.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=COLORS[method],
        fontsize=7.0,
        fontweight="bold",
    )


def annotate_small(ax: plt.Axes, x: float, y: float, method: str) -> None:
    ax.annotate(
        f"{y:.1e}",
        (x, y),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=COLORS[method],
        fontsize=6.7,
    )


def make_alignment_summary() -> None:
    rows = [
        row
        for row in read_csv("response_alignment_condition_summary.csv")
        if math.isclose(float(row["EbNo_dB"]), 2.5)
    ]
    expected = {(size, method) for size in SIZE_ORDER for method in METHODS}
    observed = {(row["size"], row["variant"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected response-alignment condition set: {observed}")

    # The logger records this diagnostic only when a delayed response source is
    # actually supplied.  pSA and gain-only therefore carry NaN, not zero.
    for size in SIZE_ORDER:
        for method in METHODS:
            value = float(get_row(rows, size=size, method=method)["late_q_used_source_product_mean"])
            if method in {"pSA", "gain_only"} and not math.isnan(value):
                raise ValueError(f"Expected N/A used-product value for {(size, method)}")
            if method not in {"pSA", "gain_only"} and not math.isfinite(value):
                raise ValueError(f"Expected finite used-product value for {(size, method)}")

    specs = [
        ("decoded_BER", "Decoded BER", "log", None),
        ("late_syndrome_weight_fraction", "Late syndrome weight / $M$", "symlog", 1e-3),
        ("late_backflip_rate", "Late correct-to-incorrect rate", "symlog", 1e-4),
        ("late_q_used_source_product", r"Late $\langle q_t q^{\rm used}_{t-1}\rangle$", "symlog", 1e-4),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.45, 5.85))
    centers = np.arange(len(SIZE_ORDER), dtype=float)
    offsets = dict(zip(METHODS, np.linspace(-0.28, 0.28, len(METHODS))))

    for panel_index, (ax, (metric, ylabel, scale, linthresh)) in enumerate(zip(axes.flat, specs)):
        for method in METHODS:
            for size_index, size in enumerate(SIZE_ORDER):
                row = get_row(rows, size=size, method=method)
                mean = float(row[f"{metric}_mean"])
                x = centers[size_index] + offsets[method]
                if math.isnan(mean):
                    ax.annotate(
                        "N/A",
                        (x, 0.0),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        color=COLORS[method],
                        fontsize=6.6,
                        fontweight="bold",
                    )
                    continue
                low = float(row[f"{metric}_ci_low"])
                high = float(row[f"{metric}_ci_high"])
                ax.errorbar(
                    x,
                    mean,
                    yerr=asymmetric_errors(mean, low, high),
                    **point_style(method, label=size_index == 0 and panel_index == 0),
                )
                if mean == 0.0:
                    annotate_zero(ax, x, method)
                elif metric == "late_backflip_rate" and mean < 1e-3:
                    annotate_small(ax, x, mean, method)
                elif metric == "late_q_used_source_product" and abs(mean) < 1e-2:
                    annotate_small(ax, x, mean, method)

        ax.set_xticks(centers, [size.split("_")[0] for size in SIZE_ORDER])
        ax.set_xlim(-0.48, len(SIZE_ORDER) - 0.52)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="both", alpha=0.22)
        ax.set_title(f"({chr(ord('a') + panel_index)})", loc="left")
        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=linthresh, linscale=0.75, base=10)
            if metric == "late_q_used_source_product":
                ax.set_ylim(-4e-4, 1.35)
            else:
                ax.set_ylim(-0.12 * linthresh, 0.75)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(METHODS),
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.89, hspace=0.36, wspace=0.29)
    save(fig, "FigS_response_alignment_summary")


def make_retention_summary() -> None:
    rows = read_csv("post_acquisition_stability_summary.csv")
    expected = {(size, method) for size in RETENTION_SIZES for method in RETENTION_METHODS}
    observed = {(row["size"], row["variant"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected retention condition set: {observed}")

    specs = [
        ("correct_residence_fraction_pooled", "Correct-state residence fraction", "linear", None, False),
        ("correct_escape_probability_pooled", "Escape probability / cycle", "symlog", 1e-5, False),
        ("late_state_BER", "Late state BER", "symlog", 1e-3, True),
        ("late_backflip_rate", "Late bit back-flip rate", "symlog", 1e-4, True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.45, 5.85))
    centers = np.arange(len(RETENTION_SIZES), dtype=float)
    offsets = dict(zip(RETENTION_METHODS, np.linspace(-0.24, 0.24, len(RETENTION_METHODS))))

    exact_zero_count = 0
    very_small_positive_count = 0
    for panel_index, (ax, (metric, ylabel, scale, linthresh, has_interval)) in enumerate(zip(axes.flat, specs)):
        for method in RETENTION_METHODS:
            for size_index, size in enumerate(RETENTION_SIZES):
                row = get_row(rows, size=size, method=method)
                field = f"{metric}_mean" if has_interval else metric
                mean = float(row[field])
                x = centers[size_index] + offsets[method]
                if has_interval:
                    low = float(row[f"{metric}_ci_low"])
                    high = float(row[f"{metric}_ci_high"])
                    error = asymmetric_errors(mean, low, high)
                else:
                    error = None
                ax.errorbar(
                    x,
                    mean,
                    yerr=error,
                    **point_style(method, label=size_index == 0 and panel_index == 0),
                )
                if mean == 0.0:
                    exact_zero_count += 1
                    annotate_zero(ax, x, method)
                elif 0.0 < mean < 1e-3:
                    very_small_positive_count += 1
                    annotate_small(ax, x, mean, method)

        ax.set_xticks(centers, [size.split("_")[0] for size in RETENTION_SIZES])
        ax.set_xlim(-0.43, len(RETENTION_SIZES) - 0.57)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="both", alpha=0.22)
        ax.set_title(f"({chr(ord('a') + panel_index)})", loc="left")
        if scale == "linear":
            ax.set_ylim(-0.04, 1.06)
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
        else:
            ax.set_yscale("symlog", linthresh=linthresh, linscale=0.75, base=10)
            ax.set_ylim(-0.12 * linthresh, 1.4)

    if exact_zero_count != 8:
        raise ValueError(f"Expected 8 exact-zero values in Fig. S8, found {exact_zero_count}")
    if very_small_positive_count != 4:
        raise ValueError(
            f"Expected 4 positive values below 1e-3 in Fig. S8, found {very_small_positive_count}"
        )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Correct-start retention challenge", y=0.955, fontsize=10.0)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(RETENTION_METHODS),
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.86, hspace=0.36, wspace=0.29)
    save(fig, "FigS_post_acquisition_stability")


def make_initialization_acquisition() -> None:
    rows = read_csv("initialization_condition_summary.csv")
    expected = {
        (size, initialization, method)
        for size in RETENTION_SIZES
        for initialization in INITIALIZATIONS
        for method in RETENTION_METHODS
    }
    observed = {(row["size"], row["initialization"], row["variant"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected initialization condition set: {observed}")

    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.75), sharey=True)
    centers = np.arange(len(INITIALIZATIONS), dtype=float)
    offsets = dict(zip(RETENTION_METHODS, np.linspace(-0.24, 0.24, len(RETENTION_METHODS))))
    exact_zero_count = 0

    for ax_index, (ax, size) in enumerate(zip(axes, RETENTION_SIZES)):
        for method in RETENTION_METHODS:
            for init_index, initialization in enumerate(INITIALIZATIONS):
                row = get_row(rows, size=size, method=method, initialization=initialization)
                mean = float(row["correct_acquisition_rate"])
                low = float(row["correct_acquisition_wilson_ci95_low"])
                high = float(row["correct_acquisition_wilson_ci95_high"])
                x = centers[init_index] + offsets[method]
                ax.errorbar(
                    x,
                    mean,
                    yerr=asymmetric_errors(mean, low, high),
                    **point_style(method, label=init_index == 0 and ax_index == 0),
                )
                if mean == 0.0:
                    exact_zero_count += 1

        ax.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
        ax.set_title(size.replace("_", ", "))
        ax.set_xticks(centers, [INIT_LABELS[item] for item in INITIALIZATIONS], rotation=13, ha="right")
        ax.set_xlim(-0.45, len(INITIALIZATIONS) - 0.55)
        ax.set_ylim(-0.045, 1.07)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", alpha=0.22)

    if exact_zero_count != 18:
        raise ValueError(f"Expected 18 exact-zero acquisitions in Fig. S9, found {exact_zero_count}")

    axes[0].set_ylabel("Correct-codeword acquisition probability")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(RETENTION_METHODS),
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.82, wspace=0.10)
    save(fig, "FigS_initialization_acquisition")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    make_alignment_summary()
    make_retention_summary()
    make_initialization_acquisition()


if __name__ == "__main__":
    main()
