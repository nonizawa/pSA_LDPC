#!/usr/bin/env python3
"""Regenerate study figures from archived CSV summaries only.

This utility calls the plotting functions embedded in the validated study
drivers. It never calls a simulation entry point, optimizer, or statistical
aggregation routine.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "regenerated"
REFERENCE = ROOT / "src" / "reference"
LDPC = ROOT / "src" / "ldpc"
for path in [ROOT, REFERENCE, LDPC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_phase2_ldpc_dynamics as trajectory_driver  # noqa: E402
import run_phase2b_acquisition_retention as acquisition_driver  # noqa: E402
import run_phase3_robustness as cross_code_driver  # noqa: E402
import run_phase4_2sat_matched_kw as twosat_matched_driver  # noqa: E402
from src.studies import run_binary_state_self_feedback as binary_driver  # noqa: E402
from src.studies import run_initialization_robustness as initialization_driver  # noqa: E402
from src.studies import run_psa_specific_fixed_transfer as transfer_driver  # noqa: E402


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def copy_figure(source: Path, target_name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, OUTPUT / target_name)


def plot_trajectory_and_acquisition() -> None:
    trajectory = ROOT / "data" / "processed" / "supplement_figures" / "response_alignment"
    acquisition = ROOT / "data" / "processed" / "supplement_figures" / "acquisition_stability"
    with tempfile.TemporaryDirectory(prefix="memory_pbit_plot_") as temporary:
        work = Path(temporary)
        trajectory_driver._make_figures(
            rows(trajectory / "by_cycle_summary.csv"),
            rows(trajectory / "condition_summary.csv"),
            work,
        )
        copy_figure(work / "Fig_phase2_dynamics_by_size.pdf", "FigS_response_dynamics_by_size.pdf")

        acquisition_driver._make_figures(
            rows(acquisition / "by_cycle_summary.csv"),
            rows(acquisition / "condition_summary.csv"),
            rows(acquisition / "first_passage_curves.csv"),
            work,
        )
        copy_figure(work / "Fig_phase2b_A_channel_acquisition.pdf", "Fig4a_channel_acquisition.pdf")
        copy_figure(work / "Fig_phase2b_C_first_passage_curves.pdf", "FigS_acquisition_first_passage.pdf")


def plot_fixed_transfer() -> None:
    data = ROOT / "data" / "processed" / "cross_code_fixed_transfer"
    with tempfile.TemporaryDirectory(prefix="memory_pbit_transfer_plot_") as temporary:
        transfer_driver.RUN_DIR = Path(temporary)
        transfer_driver.make_figures(
            rows(data / "code_snr_three_way_summary.csv"),
            rows(data / "code_level_pairwise_summary.csv"),
        )
        generated = transfer_driver.RUN_DIR / "figures"
        copy_figure(generated / "Fig_A_three_way_code_BER.pdf", "Fig5_cross_code_three_way_BER.pdf")
        copy_figure(generated / "Fig_B_size_relative_BER_reduction.pdf", "FigS_fixed_transfer_effects.pdf")
        copy_figure(generated / "Fig_C_additive_vs_pSA_specific_log_BER_ratio.pdf", "FigS_fixed_transfer_log_ratio.pdf")

    matched = ROOT / "data" / "processed" / "matched_cross_code_ablation"
    with tempfile.TemporaryDirectory(prefix="memory_pbit_ablation_plot_") as temporary:
        work = Path(temporary)
        cross_code_driver._make_performance_figures(
            rows(matched / "code_level_summary.csv"),
            rows(matched / "size_level_summary.csv"),
            work,
        )
        copy_figure(work / "Fig_B_size_robustness_summary.pdf", "FigS_cross_code_size_summary.pdf")


def plot_initialization() -> None:
    data = ROOT / "data" / "processed" / "initialization_robustness"
    initialization_driver._make_figures(
        rows(data / "condition_summary.csv"),
        rows(data / "first_passage_curves.csv"),
        OUTPUT,
    )
    copy_figure(OUTPUT / "Fig_C_initialization_first_passage.pdf", "FigS_initialization_first_passage.pdf")


def plot_binary_feedback() -> None:
    data = ROOT / "data" / "processed" / "binary_state_feedback"
    required = [
        "best_kappa_selection.csv",
        "ber_optimal_sensitivity_by_ebno.csv",
        "trajectory_by_cycle_summary.csv",
    ]
    with tempfile.TemporaryDirectory(prefix="memory_pbit_binary_plot_") as temporary:
        work = Path(temporary)
        for name in required:
            shutil.copy2(data / name, work / name)
        binary_driver.RUN_DIR = work
        binary_driver.make_figures(
            rows(data / "kappa_sweep.csv"),
            rows(data / "highstat_ber_fer.csv"),
            trajectory_available=True,
        )
        generated = work / "figures"
        copy_figure(generated / "Fig_A_mean_BER_vs_kappa.pdf", "FigS_binary_kappa_sweep.pdf")
        copy_figure(generated / "Fig_B_highstat_matched_comparison.pdf", "FigS_binary_highstat_comparison.pdf")
        copy_figure(generated / "Fig_C_targeted_trajectory_mechanism.pdf", "FigS_binary_trajectory_mechanism.pdf")


def plot_twosat_matched_heatmap() -> None:
    data = ROOT / "data" / "processed" / "supplement_figures" / "twosat_matched_kw"
    with tempfile.TemporaryDirectory(prefix="memory_pbit_twosat_plot_") as temporary:
        work = Path(temporary)
        twosat_matched_driver.make_figures(
            work,
            rows(data / "kw_rho_aggregate_summary.csv"),
            rows(data / "statistics_ci.csv"),
        )
        copy_figure(
            work / "figures" / "Fig_phase4_A_kw_rho_heatmap.pdf",
            "FigS_2SAT_matched_kw_heatmap.pdf",
        )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_trajectory_and_acquisition()
    plot_fixed_transfer()
    plot_initialization()
    plot_binary_feedback()
    plot_twosat_matched_heatmap()
    print("Regenerated archived-study figures without running simulations.")


if __name__ == "__main__":
    main()
