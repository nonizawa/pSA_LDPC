#!/usr/bin/env python3
"""Regenerate manuscript figures whose labels require publication-facing names."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "manuscript"
FIGURES = ROOT / "figures" / "regenerated"
SIZE_ORDER = ["N96_M48", "N192_M96", "N288_M144"]
SIZE_LABEL = {
    "N96_M48": r"$N=96$",
    "N192_M96": r"$N=192$",
    "N288_M144": r"$N=288$",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def public_code_label(row: dict[str, str]) -> str:
    if row.get("realization_type") == "existing":
        return "C00"
    return f"C{int(row['code_id'].split('_')[-1]):02d}"


def make_matched_control_figure() -> None:
    summary = read_csv("matched_causal_control_summary.csv")
    sweeps = read_csv("matched_control_coefficient_sweeps.csv")
    by_size = {row["size"]: row for row in summary}

    colors = {
        "pSA": "#4C78A8",
        "additive": "#E45756",
        "normalized": "#54A24B",
        "gain_only": "#7A5195",
    }
    labels = {
        "pSA": "matched pSA",
        "additive": r"additive $\lambda$-pSA",
        "normalized": "normalized",
        "gain_only": "gain-only",
    }
    fields = {
        "pSA": "same_parameter_pSA_BER",
        "additive": "additive_BER",
        "normalized": "normalized_BER",
        "gain_only": "gain_only_BER",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.05))
    x = np.arange(len(SIZE_ORDER))
    width = 0.19
    for offset_index, variant in enumerate(["pSA", "additive", "normalized", "gain_only"]):
        axes[0].bar(
            x + (offset_index - 1.5) * width,
            [float(by_size[size][fields[variant]]) for size in SIZE_ORDER],
            width=width,
            color=colors[variant],
            edgecolor="black",
            linewidth=0.35,
            label=labels[variant],
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-3, 7e-1)
    axes[0].set_xticks(x, [SIZE_LABEL[size].replace("$", "") for size in SIZE_ORDER])
    axes[0].set_ylabel(r"mean BER over three $E_b/N_0$ values")
    axes[0].set_title("(a) Matched-parameter controls (10,000 trials)", loc="left", fontsize=9)
    axes[0].grid(axis="y", which="both", alpha=0.22)
    axes[0].legend(loc="upper left", frameon=False)

    markers = {"N96_M48": "o", "N192_M96": "s", "N288_M144": "^"}
    line_colors = {"N96_M48": "#4C72B0", "N192_M96": "#DD8452", "N288_M144": "#55A868"}
    for panel_index, parameter_source in enumerate(["lambda", "pSA"], start=1):
        ax = axes[panel_index]
        for size in SIZE_ORDER:
            points = sorted(
                [row for row in sweeps if row["anchor"] == parameter_source and row["size"] == size],
                key=lambda row: float(row["lambda_mem"]),
            )
            ax.plot(
                [float(row["lambda_mem"]) for row in points],
                [float(row["mean_BER"]) for row in points],
                marker=markers[size],
                markersize=3.5,
                linewidth=1.25,
                color=line_colors[size],
                label=SIZE_LABEL[size],
            )
            chosen = [row for row in points if row["selected_for_final_confirmation"].lower() == "true"]
            if chosen:
                ax.scatter(
                    [float(chosen[0]["lambda_mem"])],
                    [float(chosen[0]["mean_BER"])],
                    marker="*",
                    s=60,
                    facecolor="#C44E52",
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=5,
                )
        ax.axvline(0.95, color="#555555", linestyle="--", linewidth=0.9)
        ax.axvspan(0.95, 1.42, color="#777777", alpha=0.08)
        ax.set_yscale("log")
        ax.set_xlim(-0.03, 1.42)
        ax.set_xlabel(r"additive memory coefficient $\lambda$")
        ax.grid(True, which="both", alpha=0.22)
        if parameter_source == "lambda":
            ax.set_ylim(5e-3, 5e-1)
            ax.set_title("(b) Additive-optimized parameters (2,000 trials)", loc="left", fontsize=9)
            ax.legend(loc="upper right", frameon=False)
        else:
            ax.set_ylim(1.5e-2, 7e-1)
            ax.set_title("(c) pSA-optimized parameters (2,000 trials)", loc="left", fontsize=9)
            ax.text(
                0.97,
                0.06,
                r"original bound $\lambda=0.95$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.8,
                color="#555555",
            )
    fig.tight_layout(pad=0.8, w_pad=1.0)
    fig.savefig(FIGURES / "Fig3_matched_causal_controls.pdf", bbox_inches="tight")
    plt.close(fig)


def make_cross_code_ratio_figure() -> None:
    rows = read_csv("cross_code_level_summary.csv")
    colors = {"2.0": "#4C78A8", "2.5": "#F58518", "3.0": "#54A24B", "pooled": "#111111"}
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=False, constrained_layout=True)
    for ax, size in zip(axes, SIZE_ORDER):
        size_rows = [row for row in rows if row["size"] == size]
        identities = {}
        for row in size_rows:
            identities[row["code_id"]] = public_code_label(row)
        code_ids = sorted(identities, key=lambda value: identities[value])
        xmap = {code_id: index for index, code_id in enumerate(code_ids)}
        for scope_value, label in [("2.0", "2.0 dB"), ("2.5", "2.5 dB"), ("3.0", "3.0 dB"), ("pooled", "pooled")]:
            selected = [
                row
                for row in size_rows
                if (scope_value == "pooled" and row["scope"] == "pooled_across_EbNo")
                or row["EbNo_dB"] == scope_value
            ]
            offset = {"2.0": -0.18, "2.5": -0.06, "3.0": 0.06, "pooled": 0.18}[scope_value]
            ax.scatter(
                [xmap[row["code_id"]] + offset for row in selected],
                [float(row["log10_BER_ratio_continuity_corrected"]) for row in selected],
                s=38 if scope_value != "pooled" else 55,
                marker="o" if scope_value != "pooled" else "D",
                color=colors[scope_value],
                label=label,
                alpha=0.9,
            )
        ax.axhline(0.0, color="#777777", linewidth=1, linestyle="--")
        ax.set_ylabel(r"$\log_{10}(\mathrm{BER}_{add}/\mathrm{BER}_{pSA})$")
        ax.set_title(size.replace("_", "/"))
        ax.set_xticks(range(len(code_ids)), [identities[value] for value in code_ids])
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Code realization")
    axes[0].legend(ncol=4, fontsize=8, loc="best")
    fig.savefig(FIGURES / "Fig5_cross_code_BER_ratios.pdf", dpi=300)
    plt.close(fig)


def make_cross_code_mechanism_figure() -> None:
    rows = read_csv("cross_code_mechanism_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = {"pSA": "#9D9D9D", "additive": "#E45756"}
    markers = {"pSA": "o", "additive": "D"}
    x = 0
    ticks: list[float] = []
    labels: list[str] = []
    for size in SIZE_ORDER:
        size_rows = [row for row in rows if row["size"] == size]
        identities = {row["code_id"]: public_code_label({**row, "realization_type": "existing" if row["code_id"].endswith("00") else "new"}) for row in size_rows}
        code_ids = sorted(identities, key=lambda value: identities[value])
        for code_id in code_ids:
            ticks.append(x)
            labels.append(f"{size.split('_')[0][1:]}:{identities[code_id]}")
            for variant in ["pSA", "additive"]:
                row = next(item for item in size_rows if item["code_id"] == code_id and item["variant"] == variant)
                offset = -0.10 if variant == "pSA" else 0.10
                axes[0].scatter(
                    x + offset,
                    float(row["correct_codeword_acquisition_rate"]),
                    color=colors[variant],
                    marker=markers[variant],
                    s=42,
                    label=variant if x == 0 else None,
                )
                axes[1].scatter(
                    x + offset,
                    float(row["mean_late_backflip_rate"]),
                    color=colors[variant],
                    marker=markers[variant],
                    s=42,
                )
            x += 1
        x += 0.5
    axes[0].set_ylabel("Correct-codeword acquisition rate")
    axes[1].set_ylabel("Mean late back-flip rate")
    for ax in axes:
        ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Block length: code realization")
    axes[0].legend()
    fig.savefig(FIGURES / "FigS_cross_code_mechanism.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_matched_control_figure()
    make_cross_code_ratio_figure()
    make_cross_code_mechanism_figure()
