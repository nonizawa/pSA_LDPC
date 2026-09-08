#!/usr/bin/env python3
"""Regenerate figures whose labels were updated for publication.

Plotted values are read only from the copied CSV inputs under
``figure_source_data`` and ``source_data``. Internal mode keys identify
validated result rows; only rendered labels change.
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
FIGURES = ROOT / "figures" / "regenerated"
FIGURE_DATA = ROOT / "data" / "processed" / "figure_source_data"
SOURCE_DATA = ROOT / "data" / "processed" / "manuscript"

PAPER_MODES = ["pSA", "lambda_pSA", "tau_pSA"]
MODE_LABELS = {
    "BP": "BP",
    "pSA": "pSA",
    "lambda_pSA": r"additive $\lambda$-pSA",
    "tau_pSA": r"finite-response $\tau$-pSA",
}
MODE_COLORS = {
    "BP": "#222222",
    "pSA": "#4C78A8",
    "lambda_pSA": "#F58518",
    "tau_pSA": "#54A24B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field} in {row}")
    return value


def save_pdf(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)


def load_ldpc_rows(size: str) -> list[dict[str, str]]:
    rows = read_csv(FIGURE_DATA / "ldpc" / size / "final_comparison.csv")
    rows = [row for row in rows if row["mode"] in ["BP", *PAPER_MODES]]
    expected = {(mode, ebno) for mode in ["BP", *PAPER_MODES] for ebno in (2.0, 2.5, 3.0)}
    observed = {(row["mode"], number(row, "EbNo_dB")) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected LDPC condition set for {size}: {observed}")
    return rows


def make_ldpc_figure(sizes: list[str], metric: str, filename: str, log_y: bool) -> None:
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.0 * len(sizes), 4.0), dpi=180, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        rows = load_ldpc_rows(size)
        for mode in ["BP", *PAPER_MODES]:
            selected = sorted(
                [row for row in rows if row["mode"] == mode],
                key=lambda row: number(row, "EbNo_dB"),
            )
            x = [number(row, "EbNo_dB") for row in selected]
            y = [max(number(row, metric), 1e-12) for row in selected]
            ax.plot(
                x,
                y,
                marker="x" if mode == "BP" else "o",
                linestyle="--" if mode == "BP" else "-",
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
        ax.set_title(size)
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel(metric)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_pdf(fig, filename)


def make_maxcut_quality() -> None:
    rows = [row for row in read_csv(SOURCE_DATA / "maxcut_summary.csv") if row["mode"] in PAPER_MODES]
    sizes = sorted({row["size"] for row in rows}, key=lambda value: (int(value.split("_")[0][1:]), value))
    if len(rows) != 3 * len(sizes):
        raise ValueError("MAX-CUT summary does not contain one row per mode and condition")
    x = np.arange(len(sizes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=180)
    for index, mode in enumerate(PAPER_MODES):
        y = [
            number(next(row for row in rows if row["size"] == size and row["mode"] == mode), "normalized_cut")
            for size in sizes
        ]
        ax.bar(x + (index - 1) * width, y, width=width, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    ax.set_xticks(x, sizes)
    ax.set_ylabel("normalized cut")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pdf(fig, "FigS_MAXCUT_quality.pdf")


def make_maxcut_memory_sweep() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted((FIGURE_DATA / "maxcut").glob("*/*/memory_sweep_summary.csv")):
        rows.extend(read_csv(path))
    rows = [row for row in rows if row["mode"] in {"lambda_pSA", "tau_pSA"}]
    if not rows:
        raise ValueError("No MAX-CUT memory-sweep rows found")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for mode in ["lambda_pSA", "tau_pSA"]:
        coefficient = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
        grouped: dict[float, list[float]] = {}
        for row in rows:
            if row["mode"] == mode:
                grouped.setdefault(number(row, coefficient), []).append(number(row, "normalized_cut"))
        x = sorted(grouped)
        y = [float(np.mean(grouped[value])) for value in x]
        ax.plot(x, y, marker="o", color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    ax.set_xlabel("memory coefficient")
    ax.set_ylabel("normalized cut")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pdf(fig, "FigS_MAXCUT_memory_sweep.pdf")


def load_twosat_stats() -> list[dict[str, str]]:
    rows = sorted(
        read_csv(FIGURE_DATA / "twosat" / "initial_joint_optimization_stats.csv"),
        key=lambda row: number(row, "alpha"),
    )
    if len(rows) != 7:
        raise ValueError(f"Expected seven initial 2-SAT density conditions, found {len(rows)}")
    return rows


def make_twosat_mode_distribution() -> None:
    rows = load_twosat_stats()
    labels = [f"{number(row, 'alpha'):.2f}" for row in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=180)
    fields = [
        ("pSA", "pSA_best_mode_fraction"),
        ("lambda_pSA", "lambda_pSA_best_mode_fraction"),
        ("tau_pSA", "tau_pSA_best_mode_fraction"),
    ]
    for mode, field in fields:
        values = np.array([number(row, field) for row in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_xlabel("alpha = M/N")
    ax.set_ylabel("fraction of instances")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pdf(fig, "FigS_2SAT_initial_mode_distribution.pdf")


def make_twosat_response_coefficient() -> None:
    rows = load_twosat_stats()
    x = [number(row, "alpha") for row in rows]
    aggregate = [number(row, "best_response_lambda") for row in rows]
    mean = [number(row, "response_lambda_mean") for row in rows]
    ci95 = [number(row, "response_lambda_ci95") for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.plot(
        x,
        aggregate,
        marker="s",
        linestyle="--",
        color="#333333",
        label=r"aggregate finite-response $\tau$-pSA best",
    )
    ax.errorbar(
        x,
        mean,
        yerr=ci95,
        marker="o",
        color=MODE_COLORS["tau_pSA"],
        capsize=3,
        label=r"per-instance finite-response $\tau$-pSA best",
    )
    ax.set_xlabel("alpha = M/N")
    ax.set_ylabel(r"Selected response coefficient $\rho$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pdf(fig, "FigS_2SAT_initial_rho_vs_alpha.pdf")


def make_post_acquisition_stability() -> None:
    # Keep this legacy entry point consistent with the publication-facing
    # zero/N/A semantics used by the dedicated visual-cleanup generator.
    from make_supplement_visual_cleanup_figures import make_retention_summary

    make_retention_summary()


def make_twosat_matched_kw_slices() -> None:
    aggregate_rows = read_csv(SOURCE_DATA / "matched_kw_response_surface.csv")
    statistics_rows = read_csv(SOURCE_DATA / "matched_kw_2sat_statistics.csv")
    kw_grid = [4.0, 8.0, 12.0, 16.0, 20.0]
    rho_grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12]
    primary_kw = 16.0
    primary_rho = 0.08
    stage_a = [
        row
        for row in aggregate_rows
        if number(row, "alpha") == 1.20 and row["stratum"] == "all"
    ]
    value_map = {
        (number(row, "kw"), number(row, "rho")): number(row, "mean_selected_unsat_rate")
        for row in stage_a
    }
    expected_cells = {(kw, rho) for kw in kw_grid for rho in rho_grid}
    if set(value_map) != expected_cells:
        raise ValueError("Matched-k_w response surface does not contain the expected 30 cells")
    matrix = np.array([[value_map[(kw, rho)] for rho in rho_grid] for kw in kw_grid]) * 1000.0

    curve_cells = sorted(
        [row for row in stage_a if number(row, "kw") == primary_kw],
        key=lambda row: number(row, "rho"),
    )
    curve_stats = {
        number(row, "rho"): row
        for row in statistics_rows
        if number(row, "alpha") == 1.20
        and number(row, "kw") == primary_kw
        and row["metric"] == "all_mean_selected_unsat_rate"
    }

    fig = plt.figure(figsize=(12.2, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], wspace=0.48, hspace=0.30)
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_top = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, 1], sharex=ax_top)

    image = ax_heat.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    ax_heat.set_xticks(np.arange(len(rho_grid)), [f"{rho:.2f}" for rho in rho_grid])
    ax_heat.set_yticks(np.arange(len(kw_grid)), [f"{kw:.0f}" for kw in kw_grid])
    ax_heat.set_xlabel(r"Response coefficient $\rho$")
    ax_heat.set_ylabel(r"Constraint weight $k_w$")
    ax_heat.set_title(r"$N=500$, $\alpha=1.20$: mean unsatisfied-clause rate")
    for y_index in range(len(kw_grid)):
        for x_index in range(len(rho_grid)):
            color = "white" if matrix[y_index, x_index] > np.median(matrix) else "black"
            ax_heat.text(
                x_index,
                y_index,
                f"{matrix[y_index, x_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Mean unsatisfied-clause rate ($\times 10^{-3}$)")

    rhos = np.array([number(row, "rho") for row in curve_cells])
    means = np.array([number(row, "mean_selected_unsat_rate") for row in curve_cells]) * 1000.0
    lows = np.array([number(row, "mean_selected_unsat_rate_ci_low") for row in curve_cells]) * 1000.0
    highs = np.array([number(row, "mean_selected_unsat_rate_ci_high") for row in curve_cells]) * 1000.0
    ax_top.errorbar(
        rhos,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        marker="o",
        color="#1f77b4",
        capsize=3,
    )
    ax_top.axvline(
        primary_rho,
        color="#d62728",
        linestyle="--",
        linewidth=1,
        label=r"Prespecified nonzero $\rho$",
    )
    ax_top.set_title(r"Matched $k_w=16$ response-memory slice")
    ax_top.legend(frameon=False, loc="best")

    difference_rhos = [rho for rho in rho_grid if rho > 0.0]
    difference_means = np.array(
        [number(curve_stats[rho], "mean_difference_treatment_minus_baseline") for rho in difference_rhos]
    ) * 1000.0
    difference_lows = np.array([number(curve_stats[rho], "ci95_low") for rho in difference_rhos]) * 1000.0
    difference_highs = np.array([number(curve_stats[rho], "ci95_high") for rho in difference_rhos]) * 1000.0
    ax_bottom.errorbar(
        difference_rhos,
        difference_means,
        yerr=np.vstack([difference_means - difference_lows, difference_highs - difference_means]),
        marker="o",
        color="#2ca02c",
        capsize=3,
    )
    ax_bottom.axhline(0.0, color="black", linewidth=0.8)
    ax_bottom.axvline(primary_rho, color="#d62728", linestyle="--", linewidth=1)
    ax_bottom.set_xlabel(r"Response coefficient $\rho$")
    ax_bottom.text(0.01, 0.04, "Lower is better", transform=ax_bottom.transAxes, fontsize=8)

    ax_top.set_ylabel(r"Mean rate ($\times 10^{-3}$)", labelpad=4)
    ax_bottom.set_ylabel(r"Paired change vs 0 ($\times 10^{-3}$)", labelpad=4)
    ax_heat.text(-0.10, 1.04, "(a)", transform=ax_heat.transAxes, fontweight="bold")
    ax_top.text(-0.10, 1.08, "(b)", transform=ax_top.transAxes, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.91)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "FigS_2SAT_matched_kw_slices.pdf")
    plt.close(fig)


def main() -> None:
    make_ldpc_figure(["N96_M48"], "BER", "Fig2a_LDPC_BER_N96.pdf", log_y=True)
    make_ldpc_figure(["N192_M96", "N288_M144"], "BER", "Fig2b_LDPC_BER_N192_N288.pdf", log_y=True)
    make_ldpc_figure(["N192_M96", "N288_M144"], "FER", "Fig2c_LDPC_FER_N192_N288.pdf", log_y=False)
    make_maxcut_quality()
    make_maxcut_memory_sweep()
    make_twosat_mode_distribution()
    make_twosat_response_coefficient()
    make_post_acquisition_stability()
    make_twosat_matched_kw_slices()


if __name__ == "__main__":
    main()
