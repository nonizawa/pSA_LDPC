#!/usr/bin/env python3
"""Run a quick or full independently optimized representative LDPC benchmark."""

from pathlib import Path

from _reference_entry import launch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    launch(
        "run_paper_experiments",
        [
            "--problem", "ldpc", "--size", "N96_M48", "--mode", "smoke",
            "--n-trials", "2", "--n-cycles", "8", "--search-candidates", "1",
            "--n-workers", "1", "--skip-figures", "--skip-tables",
            "--output-dir", str(ROOT / "outputs" / "ldpc_benchmark_quick"),
        ],
    )
