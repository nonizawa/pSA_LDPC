#!/usr/bin/env python3
"""Run a lightweight MAX-CUT boundary experiment."""

from pathlib import Path

from _reference_entry import launch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    launch(
        "run_paper_experiments",
        [
            "--problem", "maxcut-structure", "--size", "N500_ERp0p03", "--mode", "smoke",
            "--n-trials", "2", "--n-cycles", "8", "--search-candidates", "1",
            "--instance-seeds", "0", "--n-workers", "1", "--skip-figures", "--skip-tables",
            "--output-dir", str(ROOT / "outputs" / "maxcut_boundary_quick"),
        ],
    )
