#!/usr/bin/env python3
"""Run the matched-k_w random 2-SAT control."""

from pathlib import Path

from _reference_entry import launch

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    launch(
        "run_phase4_2sat_matched_kw",
        ["--stage", "smoke", "--n-trials", "1", "--n-cycles", "4", "--n-workers", "1", "--skip-figures"],
        [f"--output-dir={ROOT / 'outputs' / 'twosat_matched_kw_quick'}"],
    )
