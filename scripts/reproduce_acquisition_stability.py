#!/usr/bin/env python3
"""Run acquisition and post-acquisition stability diagnostics."""

from pathlib import Path

from _reference_entry import launch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    launch(
        "run_phase2b_acquisition_retention",
        ["--profile", "smoke", "--run-name", "quick", "--n-workers", "1"],
        [
            f"--source-root={ROOT / 'configs' / 'ldpc' / 'benchmark_sources'}",
            f"--output-dir={ROOT / 'outputs' / 'acquisition_stability'}",
        ],
    )
