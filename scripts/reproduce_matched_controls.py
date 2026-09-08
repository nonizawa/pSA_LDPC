#!/usr/bin/env python3
"""Run the matched causal-control study."""

from pathlib import Path

from _reference_entry import launch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    launch(
        "run_phase1_controls",
        ["--profile", "smoke", "--run-name", "quick", "--n-workers", "1"],
        [
            f"--source-root={ROOT / 'configs' / 'ldpc' / 'benchmark_sources'}",
            f"--output-dir={ROOT / 'outputs' / 'matched_controls'}",
        ],
    )
